"""
Temporal evaluation for HaMeR / STMF baselines.

Typical usage:

Evaluate stateful STMF on HO3D:

conda run -n STMF python scripts/eval_temporal_metrics.py \
  --checkpoint /path/to/stmf.ckpt \
  --dataset HO3D-VAL \
  --window_size 5 \
  --results_folder results_temporal

Evaluate HaMeR + EMA baseline:

conda run -n STMF python scripts/eval_temporal_metrics.py \
  --base_hamer \
  --ema_decay 0.8 \
  --checkpoint /path/to/hamer.ckpt \
  --dataset HO3D-VAL \
  --results_folder results_temporal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN

from hamer.datasets import create_dataset
from hamer.models import load_hamer, load_stmf
from hamer.utils import recursive_to
from hamer.utils.geometry import aa_to_rotmat, rotmat_to_aa, perspective_projection
from hamer.utils.pose_utils import reconstruction_error
from hamer.utils.render_openpose import render_openpose
from hamer.utils.temporal_eval_utils import (
    build_blackout_schedule,
    compute_recovery_metrics,
    compute_temporal_metrics,
    group_records_by_sequence,
)


def extract_sample(batch, sample_idx: int):
    if isinstance(batch, dict):
        return {k: extract_sample(v, sample_idx) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch[sample_idx:sample_idx + 1]
    if isinstance(batch, list):
        return batch[sample_idx]
    return batch


def prepare_base_hamer_batch(batch: Dict) -> Dict:
    if 'img' in batch and isinstance(batch['img'], torch.Tensor) and batch['img'].dim() == 5:
        batch = dict(batch)
        batch['img'] = batch['img'][:, -1, ...]
    return batch


def inject_stateful_history(sample: Dict, pose_cache: Dict[int, torch.Tensor], beta_cache: Dict[str, torch.Tensor]) -> Dict:
    if 'pose_seq' not in sample or 'temporal_indices' not in sample:
        return sample

    pose_seq = sample['pose_seq'].clone()
    temporal_indices = sample['temporal_indices'][0].tolist()
    zero_pose = torch.zeros_like(pose_seq[0, 0])
    history = []
    for hist_idx in temporal_indices[:-1]:
        cached_pose = pose_cache.get(int(hist_idx))
        history.append(cached_pose.to(device=pose_seq.device, dtype=pose_seq.dtype) if cached_pose is not None else zero_pose)

    if history:
        pose_seq[0, :-1, :] = torch.stack(history, dim=0)
    sample['pose_seq'] = pose_seq
    if pose_seq.shape[1] > 1:
        sample['prev_pose'] = pose_seq[:, -2, :]

    sequence_key = get_sequence_key(sample)
    cached_beta = beta_cache.get(sequence_key)
    if cached_beta is not None:
        sample['prev_betas'] = cached_beta.to(device=pose_seq.device, dtype=pose_seq.dtype).unsqueeze(0)
        sample['has_prev_betas'] = torch.ones(1, device=pose_seq.device, dtype=pose_seq.dtype)
    return sample


def get_sequence_key(sample: Dict) -> str:
    sequence_key = sample.get('sequence_key')
    if isinstance(sequence_key, list):
        sequence_key = sequence_key[0]
    if sequence_key is None:
        idx = sample.get('idx')
        if isinstance(idx, torch.Tensor):
            return f"idx_{int(idx.item())}"
        return "seq_0"
    return str(sequence_key)


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


def load_dataset_cfg(dataset_name: str, config_name: str) -> CN:
    cfg = CN(new_allowed=True)
    config_file = Path(__file__).resolve().parents[1] / 'hamer' / 'configs' / config_name
    cfg.merge_from_file(str(config_file))
    return cfg[dataset_name]


def build_dataset(model_cfg, dataset_name: str, args) -> torch.utils.data.Dataset:
    dataset_cfg = load_dataset_cfg(dataset_name, args.dataset_config_name).clone()
    if args.dataset_file is not None:
        dataset_cfg.defrost()
        dataset_cfg.DATASET_FILE = args.dataset_file
        dataset_cfg.freeze()
    if args.img_dir is not None:
        dataset_cfg.defrost()
        dataset_cfg.IMG_DIR = args.img_dir
        dataset_cfg.freeze()

    return create_dataset(
        model_cfg,
        dataset_cfg,
        train=False,
        rescale_factor=-1,
        window_size=args.window_size,
    )


def build_sample_meta(dataset) -> List[Dict]:
    meta = []
    if hasattr(dataset, 'valid_indices') and hasattr(dataset, 'sequence_keys'):
        for logical_idx, seq_idx in enumerate(dataset.valid_indices):
            target_idx = int(seq_idx[-1])
            meta.append({
                'dataset_idx': logical_idx,
                'sequence_key': str(dataset.sequence_keys[target_idx]),
                'frame_order': int(dataset.frame_orders[target_idx]) if hasattr(dataset, 'frame_orders') else target_idx,
            })
        return meta

    for idx in range(len(dataset)):
        imgname = dataset.imgname[idx]
        if isinstance(imgname, bytes):
            imgname = imgname.decode('utf-8')
        meta.append({
            'dataset_idx': idx,
            'sequence_key': f"idx_{idx}",
            'frame_order': idx,
        })
    return meta


def apply_blackout_to_sample(sample: Dict, should_blackout: bool) -> Dict:
    if not should_blackout:
        return sample
    updated = dict(sample)
    img = sample['img'].clone()
    if img.dim() == 5:
        img[:, -1, ...] = 0.0
    elif img.dim() == 4:
        img[:, ...] = 0.0
    updated['img'] = img
    return updated


def compute_frame_error_mm(pred_keypoints_3d: np.ndarray, gt_keypoints_3d: np.ndarray) -> float:
    pred = pred_keypoints_3d - pred_keypoints_3d[:1]
    gt = gt_keypoints_3d - gt_keypoints_3d[:1]
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def compute_pa_metrics(model, sample: Dict, output: Dict) -> Dict[str, float]:
    gt_keypoints_3d = sample['keypoints_3d'][0, :, :3]
    pred_keypoints_3d = output['pred_keypoints_3d'][0]
    gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:1]
    pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:1]
    pa_mpjpe = float(reconstruction_error(pred_keypoints_3d[None], gt_keypoints_3d[None])[0] * 1000.0)

    has_betas = sample['has_mano_params']['betas'][0] > 0
    has_pose = sample['has_mano_params']['hand_pose'][0] > 0
    if not bool(has_betas and has_pose):
        return {'PA-MPJPE': pa_mpjpe, 'PA-MPVPE': float('nan')}

    gt_mano = sample['mano_params']
    gt_mano_params = {
        'global_orient': aa_to_rotmat(gt_mano['global_orient'].reshape(-1, 3)).reshape(1, 1, 3, 3),
        'hand_pose': aa_to_rotmat(gt_mano['hand_pose'].reshape(-1, 3)).reshape(1, 15, 3, 3),
        'betas': gt_mano['betas'].reshape(1, -1),
    }
    gt_vertices = model.mano(**{k: v.float() for k, v in gt_mano_params.items()}, pose2rot=False).vertices[0]
    pred_vertices = output['pred_vertices'][0]
    pa_mpvpe = float(reconstruction_error(pred_vertices[None], gt_vertices[None])[0] * 1000.0)
    return {'PA-MPJPE': pa_mpjpe, 'PA-MPVPE': pa_mpvpe}


def tensor_image_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.clip((img * std + mean) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def patch_kps_to_pixels(pred_keypoints_2d: np.ndarray, image_size: int) -> np.ndarray:
    coords = pred_keypoints_2d.copy()
    coords[:, 0] = (coords[:, 0] + 0.5) * image_size
    coords[:, 1] = (coords[:, 1] + 0.5) * image_size
    return np.concatenate([coords, np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)


def save_sequence_video(output_path: str, frames: List[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def run_sequence_eval(model, dataset, device, args, corruption_len: int = 0, save_video_path: Optional[str] = None) -> Tuple[List[Dict], Dict[str, float]]:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    sample_meta = build_sample_meta(dataset)
    grouped_meta = group_records_by_sequence(sample_meta)
    blackout_schedule = build_blackout_schedule(grouped_meta, blackout_len=corruption_len) if corruption_len > 0 else {}

    pose_cache: Dict[int, torch.Tensor] = {}
    beta_cache: Dict[str, torch.Tensor] = {}
    ema_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    records: List[Dict] = []
    video_frames: List[np.ndarray] = []
    video_sequence_key = next(iter(blackout_schedule.keys()), None)

    pa_mpjpe_values = []
    pa_mpvpe_values = []
    cursor = 0

    for batch in tqdm(dataloader, desc='temporal-eval'):
        batch_size = batch['img'].shape[0]
        for sample_idx in range(batch_size):
            meta = sample_meta[cursor]
            cursor += 1

            sample = extract_sample(batch, sample_idx)
            sample = recursive_to(sample, device)
            sequence_key = str(meta['sequence_key'])
            frame_order = int(meta['frame_order'])

            should_blackout = False
            if sequence_key in blackout_schedule:
                start, end = blackout_schedule[sequence_key]
                seq_records = grouped_meta[sequence_key]
                local_pos = next(i for i, item in enumerate(seq_records) if item['dataset_idx'] == meta['dataset_idx'])
                should_blackout = start <= local_pos < end
            sample = apply_blackout_to_sample(sample, should_blackout)

            if args.base_hamer:
                model_batch = prepare_base_hamer_batch(sample)
            else:
                model_batch = sample
                if not args.stateless:
                    model_batch = inject_stateful_history(model_batch, pose_cache, beta_cache)

            with torch.no_grad():
                output = model(model_batch)

            output = maybe_apply_ema(model, output, sequence_key, args.ema_decay, ema_cache)

            pred_pose = build_pose_axis_angle(output)
            pose_cache[int(sample['idx'].item())] = pred_pose[0].detach().cpu()
            beta_cache[sequence_key] = output['pred_mano_params']['betas'][0].detach().cpu()

            pa_metrics = compute_pa_metrics(model, model_batch, output)
            pa_mpjpe_values.append(pa_metrics['PA-MPJPE'])
            if not np.isnan(pa_metrics['PA-MPVPE']):
                pa_mpvpe_values.append(pa_metrics['PA-MPVPE'])

            pred_keypoints_3d = output['pred_keypoints_3d'][0].detach().cpu().numpy()
            gt_keypoints_3d = model_batch['keypoints_3d'][0, :, :3].detach().cpu().numpy()
            frame_error = compute_frame_error_mm(pred_keypoints_3d, gt_keypoints_3d)
            records.append({
                'idx': int(model_batch['idx'].item()),
                'sequence_key': sequence_key,
                'frame_order': frame_order,
                'pred_keypoints_3d': pred_keypoints_3d,
                'gt_keypoints_3d': gt_keypoints_3d,
                'frame_error': frame_error,
            })

            if save_video_path is not None and video_sequence_key is not None and sequence_key == video_sequence_key:
                img_tensor = model_batch['img'][0, -1] if model_batch['img'].dim() == 5 else model_batch['img'][0]
                frame_bgr = tensor_image_to_bgr(img_tensor)
                pred_kps_px = patch_kps_to_pixels(output['pred_keypoints_2d'][0].detach().cpu().numpy(), frame_bgr.shape[0])
                video_frames.append(render_openpose(frame_bgr, pred_kps_px))

    if save_video_path is not None:
        save_sequence_video(save_video_path, video_frames, fps=10)

    metrics = compute_temporal_metrics(records)
    metrics['PA-MPJPE'] = float(np.nanmean(np.asarray(pa_mpjpe_values, dtype=np.float32)))
    metrics['PA-MPVPE'] = float(np.nanmean(np.asarray(pa_mpvpe_values, dtype=np.float32))) if pa_mpvpe_values else float('nan')
    return records, metrics


def save_metrics_csv(output_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Temporal evaluation for HaMeR / STMF')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--results_folder', type=str, default='results_temporal')
    parser.add_argument('--dataset', type=str, default='HO3D-VAL')
    parser.add_argument('--dataset_config_name', type=str, default='datasets_stmf.yaml')
    parser.add_argument('--dataset_file', type=str, default=None)
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--base_hamer', action='store_true')
    parser.add_argument('--stateless', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--blackout_lengths', type=str, default='1,3')
    parser.add_argument('--save_video_dir', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)
    model, model_cfg = load_hamer(args.checkpoint) if args.base_hamer else load_stmf(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    rows = []
    blackout_lengths = [int(item) for item in args.blackout_lengths.split(',') if item.strip()]

    for dataset_name in [item.strip() for item in args.dataset.split(',') if item.strip()]:
        dataset = build_dataset(model_cfg, dataset_name, args)
        clean_video = None
        if args.save_video_dir:
            clean_video = os.path.join(args.save_video_dir, f'{dataset_name.lower()}_clean.mp4')
        clean_records, clean_metrics = run_sequence_eval(model, dataset, device, args, corruption_len=0, save_video_path=clean_video)
        row = {
            'dataset': dataset_name,
            'model': 'hamer' if args.base_hamer else 'stmf',
            'ema_decay': args.ema_decay,
            'stateful': not args.stateless,
            **clean_metrics,
        }

        for blackout_len in blackout_lengths:
            video_path = None
            if args.save_video_dir:
                video_path = os.path.join(args.save_video_dir, f'{dataset_name.lower()}_blackout{blackout_len}.mp4')
            corrupt_records, _ = run_sequence_eval(model, dataset, device, args, corruption_len=blackout_len, save_video_path=video_path)
            row.update(compute_recovery_metrics(clean_records, corrupt_records, blackout_len=blackout_len))

        rows.append(row)
        print(json.dumps(row, indent=2))

    save_metrics_csv(os.path.join(args.results_folder, 'temporal_metrics.csv'), rows)


if __name__ == '__main__':
    main()
