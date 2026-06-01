"""
Run the v2 sensor-guided temporal MANO refiner on a cached base-prediction file.

Evaluation is sequence-ordered by default. With `--stateful`, previous refined
poses are fed back into the history window, matching the intended online
deployment behavior. Without `--stateful`, the dataset-provided training-style
window is used.

Typical usage:

python scripts/eval_sensor_refiner.py \
  --checkpoint logs/sensor_refiner/ho3d_v3/last.pt \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_evaluation.npz \
  --base_pred_file /data/hand_data/HO-3D_v3/ho3d_eval_hamer_base_cache.npz \
  --output_file results/sensor_refiner/ho3d_eval_refined.npz \
  --stateful
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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
    parser = argparse.ArgumentParser(description="Evaluate cached v2 sensor-guided temporal MANO refiner")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--base_pred_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=None, help="Override checkpoint window size")
    parser.add_argument("--history_source", type=str, choices=["base", "gt", "mixed"], default=None)
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sorted_dataset_indices(dataset: SensorRefinerDataset) -> List[int]:
    return sorted(
        range(len(dataset)),
        key=lambda item_idx: (
            dataset.sequence_keys[int(dataset.sample_indices[item_idx])],
            int(dataset.frame_orders[int(dataset.sample_indices[item_idx])]),
            int(dataset.sample_indices[item_idx]),
        ),
    )


def build_stateful_pose_window(
    dataset: SensorRefinerDataset,
    target_idx: int,
    refined_history: Dict[str, List[torch.Tensor]],
    base_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_key = dataset.sequence_keys[target_idx]
    history = refined_history.setdefault(sequence_key, [])
    pose_slots: List[torch.Tensor] = []
    valid: List[bool] = []
    for offset in range(dataset.window_size - 1, 0, -1):
        hist_pos = len(history) - offset
        if hist_pos < 0:
            pose_slots.append(base_pose.detach().clone())
            valid.append(False)
        else:
            pose_slots.append(history[hist_pos].detach().clone())
            valid.append(True)
    pose_slots.append(base_pose.detach().clone())
    valid.append(True)
    return torch.stack(pose_slots, dim=0), torch.tensor(valid, dtype=torch.bool, device=base_pose.device)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    window_size = int(args.window_size or config.get("window_size", 5))
    history_source = args.history_source or config.get("history_source", "base")

    dataset = SensorRefinerDataset(
        dataset_file=args.dataset_file,
        base_pred_file=args.base_pred_file,
        window_size=window_size,
        history_source=history_source,
    )

    model = SensorTemporalRefiner(
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 2)),
        predict_global_orient=bool(config.get("predict_global_orient", False)),
        predict_cam=bool(config.get("predict_cam", False)),
        image_feature_dim=config.get("image_feature_dim", None),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    refined_pose = np.zeros_like(dataset.base_pose, dtype=np.float32)
    delta_hand_pose = np.zeros((len(dataset), 45), dtype=np.float32)
    sequence_key = []
    frame_order = []
    source_idx = []
    refined_history: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for item_idx in tqdm(sorted_dataset_indices(dataset), desc="Evaluating sensor refiner"):
            sample = dataset[item_idx]
            target_idx = int(sample["idx"])
            base_pose = sample["base_pose"].to(device).unsqueeze(0)
            base_cam = sample["base_cam"].to(device).unsqueeze(0)
            sensor_window = sample["sensor_window"].to(device).unsqueeze(0)
            sensor_valid_mask = sample["sensor_valid_mask"].to(device).unsqueeze(0)

            if args.stateful:
                pose_window_1d, pose_valid_1d = build_stateful_pose_window(
                    dataset,
                    target_idx=target_idx,
                    refined_history=refined_history,
                    base_pose=base_pose[0],
                )
                pose_window = pose_window_1d.unsqueeze(0)
                pose_valid_mask = pose_valid_1d.unsqueeze(0)
            else:
                pose_window = sample["pose_window"].to(device).unsqueeze(0)
                pose_valid_mask = sample["pose_valid_mask"].to(device).unsqueeze(0)

            output = model(
                base_pose=base_pose,
                pose_window=pose_window,
                sensor_window=sensor_window,
                pose_valid_mask=pose_valid_mask,
                sensor_valid_mask=sensor_valid_mask,
                base_cam=base_cam,
            )
            refined = output["refined_pose"][0].detach().cpu()
            refined_pose[target_idx] = refined.numpy().astype(np.float32)
            delta_hand_pose[target_idx] = output["delta_hand_pose"][0].detach().cpu().numpy().astype(np.float32)
            refined_history.setdefault(dataset.sequence_keys[target_idx], []).append(refined.to(device))
            sequence_key.append(dataset.sequence_keys[target_idx])
            frame_order.append(int(dataset.frame_orders[target_idx]))
            source_idx.append(target_idx)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        refined_pose=refined_pose,
        delta_hand_pose=delta_hand_pose,
        base_pose=dataset.base_pose,
        base_cam=dataset.base_cam,
        sequence_key=np.asarray(sequence_key),
        frame_order=np.asarray(frame_order, dtype=np.int64),
        source_idx=np.asarray(source_idx, dtype=np.int64),
        checkpoint=str(args.checkpoint),
        stateful=bool(args.stateful),
    )
    print(f"Wrote refined predictions: {output_path}")


if __name__ == "__main__":
    main()
