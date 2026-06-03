"""
Evaluate cached SensorTemporalRefiner outputs with MANO geometry and temporal metrics.

This script reads the NPZ produced by `scripts/eval_sensor_refiner.py` and the
packed GT NPZ, then compares `base_pose` and `refined_pose` without rerunning
the RGB backbone.

Typical usage:

python scripts/eval_sensor_refiner_metrics.py \
  --checkpoint ./_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --prediction_file results/sensor_refiner/ho3d_train_refined_stateful.npz \
  --output_json results/sensor_refiner/ho3d_train_refined_metrics.json \
  --output_csv results/sensor_refiner/ho3d_train_refined_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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

from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from hamer.utils.geometry import aa_to_rotmat
from hamer.utils.pose_utils import reconstruction_error
from hamer.utils.temporal_eval_utils import compute_temporal_metrics, parse_sequence_and_frame_order, root_align_joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached sensor-refiner predictions")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def decode_name(name) -> str:
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def build_sequence_metadata(imgname: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    sequence_key: List[str] = []
    frame_order: List[int] = []
    for raw_name in imgname:
        seq_key, frame_idx, _ = parse_sequence_and_frame_order(decode_name(raw_name))
        sequence_key.append(seq_key)
        frame_order.append(int(frame_idx))
    return np.asarray(sequence_key), np.asarray(frame_order, dtype=np.int64)


def run_mano(model, pose_aa: torch.Tensor, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = pose_aa.shape[0]
    mano_params = {
        "global_orient": aa_to_rotmat(pose_aa[:, :3].reshape(-1, 3)).reshape(batch_size, 1, 3, 3),
        "hand_pose": aa_to_rotmat(pose_aa[:, 3:].reshape(-1, 3)).reshape(batch_size, 15, 3, 3),
        "betas": betas,
    }
    output = model.mano(**{key: value.float() for key, value in mano_params.items()}, pose2rot=False)
    return output.joints, output.vertices


def mano_arrays(
    model,
    pose: np.ndarray,
    betas: np.ndarray,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    joints: List[np.ndarray] = []
    vertices: List[np.ndarray] = []
    for start in tqdm(range(0, pose.shape[0], batch_size), desc=desc):
        end = min(start + batch_size, pose.shape[0])
        pose_t = torch.from_numpy(pose[start:end]).to(device=device, dtype=torch.float32)
        betas_t = torch.from_numpy(betas[start:end]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            joints_t, vertices_t = run_mano(model, pose_t, betas_t)
        joints.append(joints_t.detach().cpu().numpy().astype(np.float32))
        vertices.append(vertices_t.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(joints, axis=0), np.concatenate(vertices, axis=0)


def compute_frame_errors(pred_joints: np.ndarray, gt_joints: np.ndarray) -> np.ndarray:
    pred = root_align_joints(pred_joints)
    gt = root_align_joints(gt_joints)
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 1000.0


def reconstruction_error_chunked(pred: np.ndarray, gt: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    values = []
    for start in range(0, pred.shape[0], chunk_size):
        end = min(start + chunk_size, pred.shape[0])
        pred_t = torch.from_numpy(pred[start:end]).float()
        gt_t = torch.from_numpy(gt[start:end]).float()
        values.append(reconstruction_error(pred_t, gt_t).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values, axis=0)


def build_records(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    frame_errors: np.ndarray,
    sequence_key: np.ndarray,
    frame_order: np.ndarray,
) -> List[Dict]:
    return [
        {
            "idx": int(idx),
            "sequence_key": str(sequence_key[idx]),
            "frame_order": int(frame_order[idx]),
            "pred_keypoints_3d": pred_joints[idx],
            "gt_keypoints_3d": gt_joints[idx],
            "frame_error": float(frame_errors[idx]),
        }
        for idx in range(pred_joints.shape[0])
    ]


def summarize_prediction(
    label: str,
    pred_pose: np.ndarray,
    pred_joints: np.ndarray,
    pred_vertices: np.ndarray,
    gt_pose: np.ndarray,
    gt_joints: np.ndarray,
    gt_vertices: np.ndarray,
    sequence_key: np.ndarray,
    frame_order: np.ndarray,
    stress_mask: np.ndarray,
) -> Dict[str, float | str]:
    pa_mpjpe = reconstruction_error_chunked(
        root_align_joints(pred_joints),
        root_align_joints(gt_joints),
    ) * 1000.0
    pa_mpvpe = reconstruction_error_chunked(pred_vertices, gt_vertices) * 1000.0
    frame_errors = compute_frame_errors(pred_joints, gt_joints)
    temporal = compute_temporal_metrics(build_records(pred_joints, gt_joints, frame_errors, sequence_key, frame_order))

    hand_pose_diff = pred_pose[:, 3:] - gt_pose[:, 3:]
    full_pose_diff = pred_pose - gt_pose
    row: Dict[str, float | str] = {
        "prediction": label,
        "num_frames": float(pred_pose.shape[0]),
        "PA-MPJPE": float(np.nanmean(pa_mpjpe)),
        "PA-MPVPE": float(np.nanmean(pa_mpvpe)),
        "MPJPE": float(np.nanmean(frame_errors)),
        "PoseRMSE": float(np.sqrt(np.mean(full_pose_diff ** 2))),
        "HandPoseRMSE": float(np.sqrt(np.mean(hand_pose_diff ** 2))),
        **temporal,
    }

    if stress_mask.any():
        row["StressFrameCount"] = float(stress_mask.sum())
        row["Stress_MPJPE"] = float(np.nanmean(frame_errors[stress_mask]))
        row["Stress_PA-MPJPE"] = float(np.nanmean(pa_mpjpe[stress_mask]))
    else:
        row["StressFrameCount"] = 0.0
        row["Stress_MPJPE"] = float("nan")
        row["Stress_PA-MPJPE"] = float("nan")
    return row


def write_csv(path: str, rows: List[Dict[str, float | str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    model, _ = load_hamer(args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    dataset_npz = np.load(args.dataset_file, allow_pickle=True)
    prediction_npz = np.load(args.prediction_file, allow_pickle=True)
    gt_pose = dataset_npz["hand_pose"].astype(np.float32)
    betas = dataset_npz.get("betas", np.zeros((gt_pose.shape[0], 10), dtype=np.float32)).astype(np.float32)
    sequence_key, frame_order = build_sequence_metadata(dataset_npz["imgname"])
    stress_mask = prediction_npz.get("stress_mask", np.zeros(gt_pose.shape[0], dtype=np.bool_)).astype(np.bool_)

    pred_sources = {
        "base": prediction_npz["base_pose"].astype(np.float32),
        "refined": prediction_npz["refined_pose"].astype(np.float32),
    }
    for label, pose in pred_sources.items():
        if pose.shape != gt_pose.shape:
            raise ValueError(f"{label} pose shape {pose.shape} does not match GT pose shape {gt_pose.shape}")
    if stress_mask.shape[0] != gt_pose.shape[0]:
        raise ValueError(f"stress_mask length {stress_mask.shape[0]} does not match GT length {gt_pose.shape[0]}")

    gt_joints, gt_vertices = mano_arrays(model, gt_pose, betas, args.batch_size, device, "GT MANO")
    rows = []
    for label, pose in pred_sources.items():
        pred_joints, pred_vertices = mano_arrays(model, pose, betas, args.batch_size, device, f"{label} MANO")
        rows.append(
            summarize_prediction(
                label=label,
                pred_pose=pose,
                pred_joints=pred_joints,
                pred_vertices=pred_vertices,
                gt_pose=gt_pose,
                gt_joints=gt_joints,
                gt_vertices=gt_vertices,
                sequence_key=sequence_key,
                frame_order=frame_order,
                stress_mask=stress_mask,
            )
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(json.dumps(rows, indent=2))

    dataset_npz.close()
    prediction_npz.close()


if __name__ == "__main__":
    main()
