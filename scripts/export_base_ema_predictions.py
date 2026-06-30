"""
Export a HaMeR+EMA baseline NPZ compatible with eval_sensor_refiner_metrics.py.

This is an image-free baseline: it reads the packed GT NPZ only for sequence
ordering, reads cached HaMeR base predictions, smooths hand pose per sequence,
and writes `refined_pose=EMA(base_pose)`.

Typical usage:

python scripts/export_base_ema_predictions.py \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_evaluation.npz \
  --base_pred_file /data/hand_data/HO-3D_v3/ho3d_evaluation_hamer_base_cache.npz \
  --output_file results/sensor_refiner/ho3d_eval_base_ema_a05.npz \
  --alpha 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.utils.temporal_eval_utils import parse_sequence_and_frame_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached HaMeR+EMA predictions")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--base_pred_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Current-frame EMA weight")
    return parser.parse_args()


def decode_name(name) -> str:
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def sequence_keys_from_imgname(imgname: Iterable) -> List[str]:
    return [parse_sequence_and_frame_order(decode_name(name))[0] for name in imgname]


def apply_hand_pose_ema(base_pose: torch.Tensor, sequence_key: List[str], alpha: float) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    ema_pose = base_pose.clone()
    prev = None
    prev_key = None
    for idx, key in enumerate(sequence_key):
        current = base_pose[idx]
        if prev is None or key != prev_key:
            smoothed = current.clone()
        else:
            smoothed = current.clone()
            smoothed[3:] = float(alpha) * current[3:] + (1.0 - float(alpha)) * prev[3:]
        ema_pose[idx] = smoothed
        prev = smoothed
        prev_key = key
    return ema_pose


def main() -> None:
    args = parse_args()
    dataset_npz = np.load(args.dataset_file, allow_pickle=True)
    cache_npz = np.load(args.base_pred_file, allow_pickle=True)
    imgname = dataset_npz["imgname"]
    base_pose = cache_npz["base_pose"].astype(np.float32)
    base_cam = cache_npz["base_cam"].astype(np.float32)
    if base_pose.shape[0] != len(imgname):
        raise ValueError(f"base_pose length {base_pose.shape[0]} does not match dataset length {len(imgname)}")
    if "imgname" in cache_npz:
        cache_names = np.asarray([decode_name(name) for name in cache_npz["imgname"]])
        gt_names = np.asarray([decode_name(name) for name in imgname])
        if len(cache_names) == len(gt_names) and not np.array_equal(cache_names, gt_names):
            raise ValueError("base prediction cache imgname does not match dataset imgname order")

    sequence_key = sequence_keys_from_imgname(imgname)
    refined_pose = apply_hand_pose_ema(torch.from_numpy(base_pose), sequence_key, args.alpha).numpy().astype(np.float32)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        refined_pose=refined_pose,
        delta_hand_pose=(refined_pose[:, 3:] - base_pose[:, 3:]).astype(np.float32),
        base_pose=base_pose,
        base_cam=base_cam,
        sequence_key=np.asarray(sequence_key),
        source_idx=np.arange(len(imgname), dtype=np.int64),
        stress_mask=np.zeros(len(imgname), dtype=np.bool_),
        ema_alpha=float(args.alpha),
    )
    dataset_npz.close()
    cache_npz.close()
    print(f"Wrote EMA predictions: {output_path}")


if __name__ == "__main__":
    main()
