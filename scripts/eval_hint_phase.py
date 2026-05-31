"""
HInt evaluation with phase-bucket reporting.

Typical usage:

conda run -n STMF python scripts/eval_hint_phase.py \
  --base_hamer \
  --checkpoint /path/to/hamer.ckpt \
  --dataset EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC

conda run -n STMF python scripts/eval_hint_phase.py \
  --checkpoint /path/to/stmf.ckpt \
  --dataset EGO4D-TEST-ALL,EGO4D-TEST-OCC \
  --window_size 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN

from hamer.datasets import create_dataset
from hamer.models import load_hamer, load_stmf
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, rotmat_to_aa, perspective_projection
from hamer.utils.temporal_eval_utils import parse_sequence_and_frame_order


def build_pose_axis_angle(output: Dict) -> torch.Tensor:
    if 'pred_pose' in output:
        return output['pred_pose']
    pred_mano_params = output['pred_mano_params']
    global_orient = rotmat_to_aa(pred_mano_params['global_orient'].reshape(-1, 3, 3)).reshape(pred_mano_params['global_orient'].shape[0], 3)
    hand_pose = rotmat_to_aa(pred_mano_params['hand_pose'].reshape(-1, 3, 3)).reshape(pred_mano_params['hand_pose'].shape[0], 45)
    return torch.cat([global_orient, hand_pose], dim=1)


def rebuild_output_with_smoothed_params(model, output: Dict, pose_aa: torch.Tensor, betas: torch.Tensor, pred_cam: torch.Tensor) -> Dict:
    batch_size = pose_aa.shape[0]
    device = pose_aa.device
    dtype = pose_aa.dtype

    pred_mano_params = {
        'global_orient': aa_to_rotmat(pose_aa[:, :3].reshape(-1, 3)).reshape(batch_size, 1, 3, 3),
        'hand_pose': aa_to_rotmat(pose_aa[:, 3:].reshape(-1, 3)).reshape(batch_size, 15, 3, 3),
        'betas': betas.reshape(batch_size, -1),
    }
    focal_length = model.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
    pred_cam_t = torch.stack(
        [
            pred_cam[:, 1],
            pred_cam[:, 2],
            2 * focal_length[:, 0] / (model.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    mano_output = model.mano(**{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False)
    pred_keypoints_3d = mano_output.joints
    pred_vertices = mano_output.vertices
    pred_keypoints_2d = perspective_projection(
        pred_keypoints_3d,
        translation=pred_cam_t,
        focal_length=focal_length / model.cfg.MODEL.IMAGE_SIZE,
    )

    updated = dict(output)
    updated['pred_cam'] = pred_cam
    updated['pred_pose'] = pose_aa
    updated['pred_cam_t'] = pred_cam_t
    updated['pred_mano_params'] = {k: v.clone() for k, v in pred_mano_params.items()}
    updated['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
    updated['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
    updated['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
    updated['focal_length'] = focal_length
    return updated


def maybe_apply_ema(model, output: Dict, sequence_key: str, ema_decay: float, ema_cache: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
    if ema_decay <= 0:
        return output

    pose_aa = build_pose_axis_angle(output)
    betas = output['pred_mano_params']['betas'].reshape(pose_aa.shape[0], -1)
    pred_cam = output['pred_cam'].reshape(pose_aa.shape[0], -1)

    cache = ema_cache.get(sequence_key)
    if cache is None:
        ema_pose = pose_aa
        ema_betas = betas
        ema_cam = pred_cam
    else:
        cache_pose = cache['pose'].to(device=pose_aa.device, dtype=pose_aa.dtype)
        cache_betas = cache['betas'].to(device=betas.device, dtype=betas.dtype)
        cache_cam = cache['cam'].to(device=pred_cam.device, dtype=pred_cam.dtype)
        ema_pose = ema_decay * cache_pose + (1.0 - ema_decay) * pose_aa
        ema_betas = ema_decay * cache_betas + (1.0 - ema_decay) * betas
        ema_cam = ema_decay * cache_cam + (1.0 - ema_decay) * pred_cam

    ema_cache[sequence_key] = {
        'pose': ema_pose.detach().cpu(),
        'betas': ema_betas.detach().cpu(),
        'cam': ema_cam.detach().cpu(),
    }
    return rebuild_output_with_smoothed_params(model, output, ema_pose, ema_betas, ema_cam)


def load_dataset_cfg(dataset_name: str) -> CN:
    cfg = CN(new_allowed=True)
    config_file = Path(__file__).resolve().parents[1] / 'hamer' / 'configs' / 'datasets_eval.yaml'
    cfg.merge_from_file(str(config_file))
    return cfg[dataset_name]


def build_dataset(model_cfg, dataset_name: str):
    dataset_cfg = load_dataset_cfg(dataset_name)
    return create_dataset(model_cfg, dataset_cfg, train=False, rescale_factor=2)


def prepare_stmf_batch(sample: Dict, window_size: int, pose_history: List[torch.Tensor], prev_betas: Optional[torch.Tensor]) -> Dict:
    device = sample['img'].device
    dtype = sample['img'].dtype
    history_len = max(0, window_size - 1)

    pose_seq = torch.zeros(1, window_size, 48, device=device, dtype=dtype)
    pose_valid_mask = torch.zeros(1, history_len, device=device, dtype=torch.bool)
    if history_len > 0 and pose_history:
        history = pose_history[-history_len:]
        history_tensor = torch.stack([item.to(device=device, dtype=dtype) for item in history], dim=0)
        pose_seq[0, history_len - len(history):history_len, :] = history_tensor
        pose_valid_mask[0, history_len - len(history):history_len] = True

    sensor_seq = torch.zeros(1, window_size, 5, device=device, dtype=dtype)
    sensor_valid_mask = torch.zeros(1, window_size, device=device, dtype=torch.bool)
    img = sample['img'].unsqueeze(1)

    batch = dict(sample)
    batch['img'] = img
    batch['pose_seq'] = pose_seq
    batch['pose_valid_mask'] = pose_valid_mask
    batch['sensor_seq'] = sensor_seq
    batch['sensor_valid_mask'] = sensor_valid_mask
    batch['temporal_indices'] = torch.full((1, window_size), int(sample['idx'].item()), device=device, dtype=torch.long)
    sequence_key = sample.get('sequence_key', ['seq_0'])
    if isinstance(sequence_key, list):
        sequence_key = sequence_key[0]
    batch['sequence_key'] = [str(sequence_key)]
    batch['frame_order'] = torch.tensor([int(sample['frame_order'])], device=device)
    if prev_betas is not None:
        batch['prev_betas'] = prev_betas.to(device=device, dtype=dtype).unsqueeze(0)
        batch['has_prev_betas'] = torch.ones(1, device=device, dtype=dtype)
    else:
        batch['prev_betas'] = torch.zeros(1, 10, device=device, dtype=dtype)
        batch['has_prev_betas'] = torch.zeros(1, device=device, dtype=dtype)
    return batch


def project_to_original_image(sample: Dict, output: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pred_keypoints_2d = output['pred_keypoints_2d'].detach().cpu().numpy()[0].copy()
    right = float(sample['right'].detach().cpu().item())
    pred_keypoints_2d[:, 0] = (2.0 * right - 1.0) * pred_keypoints_2d[:, 0]
    box_size = float(sample['box_size'].detach().cpu().item())
    box_center = sample['box_center'].detach().cpu().numpy()[0]
    bbox_expand_factor = float(sample['bbox_expand_factor'].detach().cpu().item())
    scale = box_size / bbox_expand_factor
    pred_keypoints_2d = pred_keypoints_2d * box_size + box_center[None]

    gt_keypoints = sample['orig_keypoints_2d'].detach().cpu().numpy()[0]
    conf = gt_keypoints[:, 2]
    return pred_keypoints_2d, gt_keypoints[:, :2], conf, scale


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray, conf: np.ndarray, scale: float, thresholds: List[float]) -> Dict[str, float]:
    valid = conf > 0.5
    if valid.sum() == 0:
        metrics = {'mode_kpl2': float('nan')}
        metrics.update({f'PCK@{thr:.2f}': float('nan') for thr in thresholds})
        return metrics

    per_joint_l2 = np.sum((pred - gt) ** 2, axis=1)
    metrics = {'mode_kpl2': float(per_joint_l2[valid].mean())}
    dist = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    for thr in thresholds:
        metrics[f'PCK@{thr:.2f}'] = float((dist[valid] <= scale * thr).mean())
    return metrics


def safe_mean(values: List[float]) -> float:
    if not values:
        return float('nan')
    return float(np.nanmean(np.asarray(values, dtype=np.float32)))


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate HInt splits and phase buckets')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--results_folder', type=str, default='results_hint_phase')
    parser.add_argument('--dataset', type=str, default='EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC')
    parser.add_argument('--base_hamer', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--thresholds', type=str, default='0.05,0.10,0.15')
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)
    thresholds = [float(item) for item in args.thresholds.split(',') if item.strip()]

    model, model_cfg = load_hamer(args.checkpoint) if args.base_hamer else load_stmf(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    summary_rows: List[Dict[str, object]] = []
    phase_rows: List[Dict[str, object]] = []

    for dataset_name in [item.strip() for item in args.dataset.split(',') if item.strip()]:
        dataset = build_dataset(model_cfg, dataset_name)
        if len(dataset) == 0:
            print(json.dumps({
                'dataset': dataset_name,
                'warning': 'dataset is empty after missing-image filtering; skipping split',
            }, indent=2))
            continue
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        ema_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        pose_history_cache: Dict[str, List[torch.Tensor]] = defaultdict(list)
        beta_cache: Dict[str, torch.Tensor] = {}

        sample_rows: List[Dict[str, object]] = []
        for batch in tqdm(dataloader, desc=dataset_name):
            sample = recursive_to(batch, device)
            imgname_rel = sample.get('imgname_rel', [''])[0]
            if isinstance(imgname_rel, torch.Tensor):
                imgname_rel = str(imgname_rel.item())
            sequence_key, frame_order, phase_name = parse_sequence_and_frame_order(str(imgname_rel))
            sample['sequence_key'] = [sequence_key]
            sample['frame_order'] = torch.tensor([frame_order], device=device)

            if args.base_hamer:
                model_batch = sample
            else:
                history = pose_history_cache[sequence_key]
                model_batch = prepare_stmf_batch(sample, window_size=args.window_size, pose_history=history, prev_betas=beta_cache.get(sequence_key))

            with torch.no_grad():
                output = model(model_batch)
            output = maybe_apply_ema(model, output, sequence_key, args.ema_decay, ema_cache)

            if not args.base_hamer:
                pred_pose = build_pose_axis_angle(output)[0].detach().cpu()
                pose_history_cache[sequence_key].append(pred_pose)
                pose_history_cache[sequence_key] = pose_history_cache[sequence_key][-(args.window_size - 1):]
                beta_cache[sequence_key] = output['pred_mano_params']['betas'][0].detach().cpu()

            pred, gt, conf, scale = project_to_original_image(sample, output)
            metrics = compute_sample_metrics(pred, gt, conf, scale, thresholds=thresholds)
            sample_rows.append({
                'dataset': dataset_name,
                'sequence_key': sequence_key,
                'frame_order': frame_order,
                'phase_name': phase_name,
                **metrics,
            })

        summary_row = {
            'dataset': dataset_name,
            'model': 'hamer' if args.base_hamer else 'stmf',
            'ema_decay': args.ema_decay,
        }
        for metric_name in ['mode_kpl2', *[f'PCK@{thr:.2f}' for thr in thresholds]]:
            summary_row[metric_name] = safe_mean([float(row[metric_name]) for row in sample_rows])
        summary_rows.append(summary_row)
        print(json.dumps(summary_row, indent=2))

        grouped_phase: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in sample_rows:
            if row['phase_name'] is not None:
                grouped_phase[str(row['phase_name'])].append(row)
        for phase_name, rows in grouped_phase.items():
            phase_row = {
                'dataset': dataset_name,
                'phase_name': phase_name,
                'model': 'hamer' if args.base_hamer else 'stmf',
                'ema_decay': args.ema_decay,
                'sample_count': len(rows),
            }
            for metric_name in ['mode_kpl2', *[f'PCK@{thr:.2f}' for thr in thresholds]]:
                phase_row[metric_name] = safe_mean([float(row[metric_name]) for row in rows])
            phase_rows.append(phase_row)

    save_csv(os.path.join(args.results_folder, 'hint_summary.csv'), summary_rows)
    save_csv(os.path.join(args.results_folder, 'hint_phase_profile.csv'), phase_rows)


if __name__ == '__main__':
    main()
