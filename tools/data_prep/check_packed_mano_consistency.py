"""
Check whether packed HO3D MANO supervision is self-consistent.

This script loads a packed NPZ, reconstructs MANO joints from `hand_pose` and
`betas`, and compares them against the packed `hand_keypoints_3d`.

Why this is useful:
- If `hand_pose` / `betas` and `hand_keypoints_3d` disagree badly, training can
  drift even when the keypoint visualization looks reasonable.
- It also helps diagnose whether the packed keypoint order is already in the
  model/OpenPose order or still in the HO3D official order.

Typical usage:

python tools/data_prep/check_packed_mano_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --num_samples 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hamer.models.mano_wrapper import MANO
from hamer.utils.geometry import aa_to_rotmat


HO3D_OFFICIAL_TO_OPENPOSE = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


def build_mano(mano_model_path: str, gender: str = "neutral", flat_hand_mean: bool = False) -> MANO:
    return MANO(
        model_path=mano_model_path,
        gender=gender,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        num_pca_comps=45,
        is_rhand=True,
    )


def root_relative_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_rr = pred - pred[:, [0], :]
    gt_rr = gt - gt[:, [0], :]
    return np.linalg.norm(pred_rr - gt_rr, axis=-1).mean(axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--mano_model_path", type=str, default=str(REPO_ROOT / "_DATA" / "data" / "mano"))
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.dataset_file, allow_pickle=True)
    hand_pose = data["hand_pose"].astype(np.float32)
    betas = data["betas"].astype(np.float32)
    has_pose = data["has_hand_pose"].astype(np.float32).reshape(-1) > 0
    has_betas = data["has_betas"].astype(np.float32).reshape(-1) > 0
    kps3d = data["hand_keypoints_3d"].astype(np.float32)[..., :3]

    valid = has_pose & has_betas
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No valid samples with both hand_pose and betas were found.")

    rng = np.random.default_rng(args.seed)
    if len(valid_idx) > args.num_samples:
        valid_idx = np.sort(rng.choice(valid_idx, size=args.num_samples, replace=False))

    print(f"Loaded {len(hand_pose)} samples from {args.dataset_file}")
    print(f"Checking {len(valid_idx)} samples with valid MANO supervision")

    hp = torch.from_numpy(hand_pose[valid_idx]).to(args.device)
    bt = torch.from_numpy(betas[valid_idx]).to(args.device)

    packed_as_is = kps3d[valid_idx]
    packed_reordered = packed_as_is[:, HO3D_OFFICIAL_TO_OPENPOSE, :]

    global_orient_rot = aa_to_rotmat(hp[:, :3]).view(-1, 1, 3, 3)
    hand_pose_rot = aa_to_rotmat(hp[:, 3:].reshape(-1, 3)).view(-1, 15, 3, 3)

    print("")
    print("MANO self-consistency check (root-relative MPJPE, mm if source is mm):")
    all_results = []
    for flat_hand_mean in (False, True):
        mano = build_mano(args.mano_model_path, args.gender, flat_hand_mean=flat_hand_mean).to(args.device)
        mano.eval()
        with torch.no_grad():
            out = mano(
                global_orient=global_orient_rot,
                hand_pose=hand_pose_rot,
                betas=bt,
            )
        pred_joints = out.joints.detach().cpu().numpy()

        mpjpe_as_is = root_relative_mpjpe(pred_joints, packed_as_is)
        mpjpe_reordered = root_relative_mpjpe(pred_joints, packed_reordered)
        all_results.append((flat_hand_mean, mpjpe_as_is.mean(), np.median(mpjpe_as_is), mpjpe_reordered.mean(), np.median(mpjpe_reordered)))

        print(f"  flat_hand_mean={flat_hand_mean}")
        print(f"    packed order as-is        mean={mpjpe_as_is.mean():.4f}  median={np.median(mpjpe_as_is):.4f}")
        print(f"    packed order->openpose    mean={mpjpe_reordered.mean():.4f}  median={np.median(mpjpe_reordered):.4f}")

        better_as_is = int((mpjpe_as_is < mpjpe_reordered).sum())
        better_reordered = int((mpjpe_reordered < mpjpe_as_is).sum())
        print(f"    winner counts             as-is={better_as_is}  reordered={better_reordered}")

    print("")
    best = min(all_results, key=lambda x: min(x[1], x[3]))
    best_flat = best[0]
    best_order = "as-is" if best[1] <= best[3] else "reordered"
    print(f"Best matching setting: flat_hand_mean={best_flat}, packed order={best_order}")
    if min(best[1], best[3]) > 20.0:
        print("Interpretation: order alone may not explain the mismatch; packed MANO parameters may be inconsistent with packed keypoints.")


if __name__ == "__main__":
    main()
