"""
Utilities for sequence parsing and temporal evaluation.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


PHASE_NAME_TO_ORDER = {
    'pre_45': -45,
    'pre_30': -30,
    'pre_15': -15,
    'pre_frame': -1,
    'contact_frame': 0,
    'pnr_frame': 1,
    'post_frame': 15,
}


def parse_sequence_and_frame_order(image_name: str) -> Tuple[str, int, str | None]:
    norm_name = image_name.replace('\\', '/')
    parts = norm_name.split('/')
    base_name = os.path.basename(norm_name)
    stem, _ = os.path.splitext(base_name)

    if len(parts) >= 3 and parts[-2] == 'rgb':
        return parts[-3], _extract_last_number(parts[-1]), None

    capture_idx = next((i for i, part in enumerate(parts) if part.startswith('Capture')), None)
    if capture_idx is not None and capture_idx + 2 < len(parts):
        capture = parts[capture_idx]
        seq_name = parts[capture_idx + 1]
        camera = parts[capture_idx + 2]
        return f"{capture}/{seq_name}/{camera}", _extract_last_number(parts[-1]), None

    event_match = re.match(
        r'(.+?)_(pre_45|pre_30|pre_15|pre_frame|contact_frame|pnr_frame|post_frame)_(l|r)$',
        stem,
    )
    if event_match:
        seq_root, phase_name, side = event_match.groups()
        return f"{seq_root}_{side}", PHASE_NAME_TO_ORDER[phase_name], phase_name

    frame_match = re.match(r'(.+?)_frame_(\d+)_(l|r)$', stem)
    if frame_match:
        seq_root, frame_idx, side = frame_match.groups()
        return f"{seq_root}_{side}", int(frame_idx), None

    return "seq_0", _extract_last_number(stem), None


def _extract_last_number(value: str) -> int:
    numbers = re.findall(r'(\d+)', value)
    if not numbers:
        return 0
    return int(numbers[-1])


def group_records_by_sequence(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[str(record['sequence_key'])].append(record)
    for sequence_key in grouped:
        grouped[sequence_key].sort(key=lambda item: (int(item.get('frame_order', 0)), int(item.get('idx', 0))))
    return grouped


def root_align_joints(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    return joints - joints[..., :1, :]


def compute_temporal_metrics(records: Iterable[Dict]) -> Dict[str, float]:
    grouped = group_records_by_sequence(records)
    velocity_errors = []
    acceleration_errors = []
    pred_jitters = []

    for seq_records in grouped.values():
        if len(seq_records) < 2:
            continue

        pred = root_align_joints(np.stack([item['pred_keypoints_3d'] for item in seq_records], axis=0))
        gt = root_align_joints(np.stack([item['gt_keypoints_3d'] for item in seq_records], axis=0))

        pred_vel = pred[1:] - pred[:-1]
        gt_vel = gt[1:] - gt[:-1]
        velocity_errors.append(np.linalg.norm(pred_vel - gt_vel, axis=-1).mean() * 1000.0)

        if len(seq_records) >= 3:
            pred_acc = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
            gt_acc = gt[2:] - 2.0 * gt[1:-1] + gt[:-2]
            acceleration_errors.append(np.linalg.norm(pred_acc - gt_acc, axis=-1).mean() * 1000.0)
            pred_jitters.append(np.linalg.norm(pred_acc, axis=-1).mean() * 1000.0)

    return {
        'MPJVE': _safe_mean(velocity_errors),
        'MPJAE': _safe_mean(acceleration_errors),
        'PredJitter': _safe_mean(pred_jitters),
        'TemporalSeqCount': float(sum(len(v) >= 2 for v in grouped.values())),
    }


def build_blackout_schedule(grouped_records: Dict[str, List[Dict]], blackout_len: int) -> Dict[str, Tuple[int, int]]:
    schedule = {}
    for sequence_key, seq_records in grouped_records.items():
        seq_len = len(seq_records)
        if seq_len < blackout_len + 3:
            continue
        start = max(1, (seq_len // 2) - (blackout_len // 2))
        end = min(seq_len, start + blackout_len)
        schedule[sequence_key] = (start, end)
    return schedule


def compute_recovery_metrics(
    clean_records: Iterable[Dict],
    corrupt_records: Iterable[Dict],
    blackout_len: int,
    threshold_ratio: float = 1.1,
) -> Dict[str, float]:
    clean_grouped = group_records_by_sequence(clean_records)
    corrupt_grouped = group_records_by_sequence(corrupt_records)
    schedule = build_blackout_schedule(corrupt_grouped, blackout_len=blackout_len)

    peak_errors = []
    recovery_frames = []

    for sequence_key, (start, end) in schedule.items():
        clean_seq = clean_grouped.get(sequence_key)
        corrupt_seq = corrupt_grouped.get(sequence_key)
        if clean_seq is None or corrupt_seq is None:
            continue
        if len(clean_seq) != len(corrupt_seq):
            continue

        clean_err = np.asarray([item['frame_error'] for item in clean_seq], dtype=np.float32)
        corrupt_err = np.asarray([item['frame_error'] for item in corrupt_seq], dtype=np.float32)
        peak_errors.append(float(corrupt_err[start:end].mean()))

        threshold = clean_err * float(threshold_ratio)
        recovery = len(corrupt_seq) - end
        for rel_idx, frame_idx in enumerate(range(end, len(corrupt_seq))):
            if corrupt_err[frame_idx] <= threshold[frame_idx]:
                recovery = rel_idx
                break
        recovery_frames.append(float(recovery))

    return {
        f'Blackout{blackout_len}_PeakError': _safe_mean(peak_errors),
        f'Blackout{blackout_len}_RecoveryFrames@10%': _safe_mean(recovery_frames),
        f'Blackout{blackout_len}_SeqCount': float(len(recovery_frames)),
    }


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float('nan')
    return float(np.mean(np.asarray(values, dtype=np.float32)))
