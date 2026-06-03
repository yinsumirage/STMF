"""
Cache per-frame base HaMeR predictions for sensor-refiner training.

This script is the bridge between single-frame HaMeR and the v2 temporal
refiner. It runs HaMeR once over a packed NPZ and writes a frame-aligned cache:

- `base_pose`: axis-angle MANO pose, shape `(N, 48)`
- `base_cam`: HaMeR camera parameters, shape `(N, 3)`
- `base_keypoints_3d`: predicted MANO joints, shape `(N, 21, 3)`
- `sequence_key` / `frame_order`: sequence metadata for stateful eval

The cache is a training input, so it must stay frame-aligned with the packed
NPZ. Missing images should fail loudly instead of being silently skipped.

Typical usage:

python scripts/cache_base_hamer_predictions.py \
  --checkpoint /path/to/hamer.ckpt \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3 \
  --output_file /data/hand_data/HO-3D_v3/ho3d_train_hamer_base_cache.npz \
  --split train \
  --batch_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.datasets.image_dataset import ImageDataset
from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from hamer.utils import recursive_to
from hamer.utils.geometry import rotmat_to_aa
from hamer.utils.temporal_eval_utils import parse_sequence_and_frame_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache base HaMeR predictions for v2 sensor-refiner training")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "evaluation"],
        default="train",
        help="Use train for frame-aligned training caches; evaluation applies HO3D official subset filtering.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        default=False,
        help="Diagnostic only. Training caches must cover every packed NPZ frame in order.",
    )
    return parser.parse_args()


def build_pred_pose(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    mano_params = output["pred_mano_params"]
    global_orient = rotmat_to_aa(mano_params["global_orient"].reshape(-1, 3, 3)).reshape(-1, 3)
    hand_pose = rotmat_to_aa(mano_params["hand_pose"].reshape(-1, 3, 3)).reshape(global_orient.shape[0], 45)
    return torch.cat([global_orient, hand_pose], dim=1)


def main() -> None:
    args = parse_args()
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset = ImageDataset(
        model_cfg,
        args.dataset_file,
        args.img_dir,
        train=False,
        rescale_factor=args.rescale_factor,
        skip_missing_images=args.skip_missing_images,
        apply_ho3d_eval_subset=args.split == "evaluation",
    )
    packed_npz = np.load(args.dataset_file, allow_pickle=True)
    packed_len = len(packed_npz["imgname"])
    packed_npz.close()
    if len(dataset) != packed_len:
        raise ValueError(
            "Base prediction cache must be frame-aligned with the packed NPZ. "
            f"ImageDataset length is {len(dataset)}, but packed NPZ length is {packed_len}. "
            "Fix missing images or generate a matching packed NPZ before training the refiner."
        )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    base_pose: List[np.ndarray] = []
    base_cam: List[np.ndarray] = []
    base_keypoints_3d: List[np.ndarray] = []
    base_vertices: List[np.ndarray] = []
    imgname: List[str] = []
    sequence_key: List[str] = []
    frame_order: List[int] = []
    source_idx: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching HaMeR base predictions"):
            batch = recursive_to(batch, device)
            output = model(batch)
            pred_pose = build_pred_pose(output)

            base_pose.append(pred_pose.detach().cpu().numpy().astype(np.float32))
            base_cam.append(output["pred_cam"].detach().cpu().numpy().astype(np.float32))
            base_keypoints_3d.append(output["pred_keypoints_3d"].detach().cpu().numpy().astype(np.float32))
            base_vertices.append(output["pred_vertices"].detach().cpu().numpy().astype(np.float32))

            batch_imgname = batch["imgname_rel"] if "imgname_rel" in batch else batch["imgname"]
            batch_idx = batch["idx"].detach().cpu().numpy().astype(np.int64)
            for name, idx in zip(batch_imgname, batch_idx.tolist()):
                name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                seq_key, frame_idx, _ = parse_sequence_and_frame_order(name)
                imgname.append(name)
                sequence_key.append(seq_key)
                frame_order.append(int(frame_idx))
                source_idx.append(int(idx))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        base_pose=np.concatenate(base_pose, axis=0),
        base_cam=np.concatenate(base_cam, axis=0),
        base_keypoints_3d=np.concatenate(base_keypoints_3d, axis=0),
        base_vertices=np.concatenate(base_vertices, axis=0),
        imgname=np.asarray(imgname),
        sequence_key=np.asarray(sequence_key),
        frame_order=np.asarray(frame_order, dtype=np.int64),
        source_idx=np.asarray(source_idx, dtype=np.int64),
        checkpoint=str(args.checkpoint),
        dataset_file=str(args.dataset_file),
    )
    print(f"Wrote base prediction cache: {output_path}")


if __name__ == "__main__":
    main()
