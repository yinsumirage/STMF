"""
Shared helpers for pseudo-sensor generation and augmentation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# HaMeR/MANO internal joint order used by this repo.
# Sensor computation expects the official MANO finger-grouped order:
# wrist, thumb, index, middle, ring, pinky.
MODEL_TO_OFFICIAL = np.array(
    [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20],
    dtype=np.int64,
)

FINGER_CHAINS = (
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
)


def _as_batch_joints(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim == 2:
        joints = joints[None, ...]
    if joints.ndim != 3 or joints.shape[-2:] != (21, 3):
        raise ValueError(f"Expected joints with shape (21, 3) or (N, 21, 3), got {joints.shape}")
    return joints


def compute_pseudo_sensor_from_model_joints(
    joints: np.ndarray,
    fist_ratio: float = 0.5,
) -> np.ndarray:
    """
    Compute 5D normalized pseudo-sensor values from joints in the repo's model order.
    """
    joints = _as_batch_joints(joints)
    joints = joints[:, MODEL_TO_OFFICIAL, :]

    current_dists = []
    lmax_values = []
    for chain in FINGER_CHAINS:
        finger_joints = joints[:, chain, :]
        current_dists.append(np.linalg.norm(finger_joints[:, -1] - finger_joints[:, 0], axis=-1))
        bone_lengths = np.linalg.norm(finger_joints[:, 1:] - finger_joints[:, :-1], axis=-1)
        lmax_values.append(bone_lengths.sum(axis=-1))

    current_dists = np.stack(current_dists, axis=1)
    lmax_values = np.stack(lmax_values, axis=1)
    lmin_values = lmax_values * float(fist_ratio)
    denom = np.clip(lmax_values - lmin_values, a_min=1e-6, a_max=None)
    sensors = (current_dists - lmin_values) / denom
    sensors = np.clip(sensors, 0.0, 1.0).astype(np.float32)
    if sensors.shape[0] == 1:
        return sensors[0]
    return sensors


def apply_sensor_augmentations(
    sensor_seq: np.ndarray,
    valid_mask: Optional[np.ndarray],
    noise_std: float = 0.0,
    sensor_dropout: float = 0.0,
    channel_dropout: float = 0.0,
    temporal_dropout: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Augment a pseudo/real sensor sequence while preserving invalid history slots.
    """
    sensor_seq = np.asarray(sensor_seq, dtype=np.float32).copy()
    if sensor_seq.ndim != 2 or sensor_seq.shape[1] != 5:
        raise ValueError(f"Expected sensor_seq shape (T, 5), got {sensor_seq.shape}")

    if rng is None:
        rng = np.random.RandomState()

    if valid_mask is None:
        valid_mask = np.ones(sensor_seq.shape[0], dtype=np.bool_)
    else:
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)

    valid_steps = np.where(valid_mask)[0]
    if valid_steps.size == 0:
        return sensor_seq

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=sensor_seq.shape).astype(np.float32)
        sensor_seq[valid_mask] += noise[valid_mask]

    if temporal_dropout > 0:
        dropped_steps = rng.rand(sensor_seq.shape[0]) < temporal_dropout
        dropped_steps &= valid_mask
        sensor_seq[dropped_steps] = 0.0

    if channel_dropout > 0:
        dropped_channels = rng.rand(sensor_seq.shape[0], sensor_seq.shape[1]) < channel_dropout
        dropped_channels &= valid_mask[:, None]
        sensor_seq[dropped_channels] = 0.0

    if sensor_dropout > 0 and rng.rand() < sensor_dropout:
        sensor_seq[valid_mask] = 0.0

    sensor_seq[~valid_mask] = 0.0
    return np.clip(sensor_seq, 0.0, 1.0).astype(np.float32)
