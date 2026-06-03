"""
Train the v2 cached sensor-guided temporal MANO refiner.

This entrypoint trains only `SensorTemporalRefiner` on top of an offline HaMeR
base-prediction cache. It does not run the image backbone during training, so
target frames can be shuffled while each sample still carries its local history
window.

Typical usage:

python scripts/train_sensor_refiner.py \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --base_pred_file /data/hand_data/HO-3D_v3/ho3d_train_hamer_base_cache.npz \
  --output_dir logs/sensor_refiner/ho3d_v3 \
  --history_source base \
  --sensor_mode sensor \
  --window_size 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.datasets.sensor_refiner_dataset import SensorRefinerDataset
from hamer.models.components.sensor_temporal_refiner import SensorTemporalRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cached v2 sensor-guided temporal MANO refiner")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--base_pred_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--history_source", type=str, choices=["base", "gt", "mixed"], default="base")
    parser.add_argument("--sensor_mode", type=str, choices=["sensor", "zero"], default="sensor")
    parser.add_argument("--mixed_gt_prob", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--predict_global_orient", action="store_true")
    parser.add_argument("--predict_cam", action="store_true")
    parser.add_argument("--image_feature_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--smoothness_weight", type=float, default=0.0)
    parser.add_argument("--global_orient_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def compute_smoothness_loss(batch: Dict[str, torch.Tensor], refined_pose: torch.Tensor) -> torch.Tensor:
    pose_window = batch["pose_window"]
    valid_mask = batch["pose_valid_mask"].to(dtype=torch.bool)
    if pose_window.shape[1] < 2:
        return refined_pose.new_zeros(())
    last_two_valid = valid_mask[:, -2:].all(dim=1)
    if not last_two_valid.any():
        return refined_pose.new_zeros(())
    prev2 = pose_window[:, -2, :]
    prev1 = pose_window[:, -1, :]
    accel = refined_pose - 2.0 * prev1 + prev2
    return torch.linalg.norm(accel[last_two_valid], dim=-1).mean()


def compute_loss(
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    smoothness_weight: float,
    global_orient_weight: float,
) -> Dict[str, torch.Tensor]:
    target_pose = batch["target_pose"]
    refined_pose = output["refined_pose"]
    hand_pose_loss = F.mse_loss(refined_pose[:, 3:], target_pose[:, 3:])
    total = hand_pose_loss
    losses = {"loss_hand_pose": hand_pose_loss.detach()}

    if global_orient_weight > 0:
        global_loss = F.mse_loss(refined_pose[:, :3], target_pose[:, :3])
        total = total + float(global_orient_weight) * global_loss
        losses["loss_global_orient"] = global_loss.detach()

    if smoothness_weight > 0:
        smoothness = compute_smoothness_loss(batch, refined_pose)
        total = total + float(smoothness_weight) * smoothness
        losses["loss_smoothness"] = smoothness.detach()

    losses["loss"] = total.detach()
    return {"total": total, "logs": losses}


def recursive_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SensorRefinerDataset(
        dataset_file=args.dataset_file,
        base_pred_file=args.base_pred_file,
        window_size=args.window_size,
        history_source=args.history_source,
        mixed_gt_prob=args.mixed_gt_prob,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = SensorTemporalRefiner(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        predict_global_orient=args.predict_global_orient,
        predict_cam=args.predict_cam,
        image_feature_dim=args.image_feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["num_samples"] = len(dataset)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = recursive_to_device(batch, device)
            sensor_window = batch["sensor_window"]
            if args.sensor_mode == "zero":
                sensor_window = torch.zeros_like(sensor_window)
            output = model(
                base_pose=batch["base_pose"],
                pose_window=batch["pose_window"],
                sensor_window=sensor_window,
                pose_valid_mask=batch["pose_valid_mask"],
                sensor_valid_mask=batch["sensor_valid_mask"],
                base_cam=batch.get("base_cam"),
            )
            loss_dict = compute_loss(
                batch,
                output,
                smoothness_weight=args.smoothness_weight,
                global_orient_weight=args.global_orient_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            optimizer.step()

            global_step += 1
            if global_step % args.log_every == 0:
                logs = {key: float(value.detach().cpu()) for key, value in loss_dict["logs"].items()}
                progress.set_postfix(logs)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(f"Wrote checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
