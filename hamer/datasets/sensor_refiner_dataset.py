"""
Dataset for cached sensor-guided temporal MANO refinement.

This dataset is intentionally image-free: it consumes a packed GT NPZ and an
offline HaMeR base-prediction cache, then builds target-frame training samples
with temporal pose/sensor windows. That keeps v2 refiner training shuffleable
while preserving sequence-local history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hamer.utils.sensor_utils import compute_pseudo_sensor_from_model_joints
from hamer.utils.temporal_eval_utils import parse_sequence_and_frame_order


HistorySource = Literal["base", "gt", "mixed"]


@dataclass(frozen=True)
class SequenceInfo:
    key: str
    ordered_indices: Tuple[int, ...]


class SensorRefinerDataset(Dataset):
    """
    Build cached temporal samples for ``SensorTemporalRefiner``.

    Required GT NPZ keys:
        ``imgname``, ``hand_pose``, ``hand_keypoints_3d``

    Required base cache NPZ keys:
        ``base_pose`` with shape ``(N, 48)``
        ``base_cam`` with shape ``(N, 3)``

    Optional cache keys:
        ``imgname`` for alignment sanity check
    """

    def __init__(
        self,
        dataset_file: str,
        base_pred_file: str,
        window_size: int = 5,
        history_source: HistorySource = "base",
        mixed_gt_prob: float = 0.5,
        sensor_fist_ratio: float = 0.5,
        seed: int = 12345,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if history_source not in {"base", "gt", "mixed"}:
            raise ValueError(f"Unsupported history_source: {history_source}")

        self.dataset_file = dataset_file
        self.base_pred_file = base_pred_file
        self.window_size = int(window_size)
        self.history_source: HistorySource = history_source
        self.mixed_gt_prob = float(mixed_gt_prob)
        self.rng = np.random.RandomState(seed)

        loaded_gt = np.load(dataset_file, allow_pickle=True)
        self.gt = {key: loaded_gt[key] for key in loaded_gt.files}
        loaded_gt.close()
        loaded_cache = np.load(base_pred_file, allow_pickle=True)
        self.cache = {key: loaded_cache[key] for key in loaded_cache.files}
        loaded_cache.close()

        self.imgname = self.gt["imgname"]
        self.gt_pose = self.gt["hand_pose"].astype(np.float32)
        self.base_pose = self.cache["base_pose"].astype(np.float32)
        self.base_cam = self.cache["base_cam"].astype(np.float32)
        self.keypoints_3d = self.gt["hand_keypoints_3d"].astype(np.float32)
        self.keypoints_2d = self.gt.get("hand_keypoints_2d", np.zeros((len(self.imgname), 21, 3), dtype=np.float32)).astype(np.float32)
        self.betas = self.gt.get("betas", np.zeros((len(self.imgname), 10), dtype=np.float32)).astype(np.float32)

        self._validate_shapes()
        self.sequence_keys, self.frame_orders = self._parse_sequence_metadata()
        self.sequence_infos, self.index_to_order_position = self._build_sequence_index()
        self.sensor_values = self._build_sensor_values(sensor_fist_ratio=sensor_fist_ratio)
        self.sample_indices = np.arange(len(self.imgname), dtype=np.int64)

    def _validate_shapes(self) -> None:
        n = len(self.imgname)
        if self.gt_pose.shape != (n, 48):
            raise ValueError(f"GT hand_pose must have shape ({n}, 48), got {self.gt_pose.shape}")
        if self.base_pose.shape != (n, 48):
            raise ValueError(f"base_pose must have shape ({n}, 48), got {self.base_pose.shape}")
        if self.base_cam.shape != (n, 3):
            raise ValueError(f"base_cam must have shape ({n}, 3), got {self.base_cam.shape}")
        if self.keypoints_3d.shape[0] != n or self.keypoints_3d.shape[1] != 21 or self.keypoints_3d.shape[2] < 3:
            raise ValueError(f"hand_keypoints_3d must have shape ({n}, 21, >=3), got {self.keypoints_3d.shape}")
        if "imgname" in self.cache and len(self.cache["imgname"]) == n:
            gt_names = np.asarray([self._decode_name(name) for name in self.imgname])
            cache_names = np.asarray([self._decode_name(name) for name in self.cache["imgname"]])
            if not np.array_equal(gt_names, cache_names):
                raise ValueError("base prediction cache imgname does not match dataset imgname order")

    def _parse_sequence_metadata(self) -> Tuple[List[str], np.ndarray]:
        sequence_keys: List[str] = []
        frame_orders: List[int] = []
        for raw_name in self.imgname:
            sequence_key, frame_order, _ = parse_sequence_and_frame_order(self._decode_name(raw_name))
            sequence_keys.append(sequence_key)
            frame_orders.append(int(frame_order))
        return sequence_keys, np.asarray(frame_orders, dtype=np.int64)

    def _build_sequence_index(self) -> Tuple[Dict[str, SequenceInfo], Dict[int, Tuple[str, int]]]:
        grouped: Dict[str, List[int]] = {}
        for idx, sequence_key in enumerate(self.sequence_keys):
            grouped.setdefault(sequence_key, []).append(idx)

        sequence_infos: Dict[str, SequenceInfo] = {}
        index_to_order_position: Dict[int, Tuple[str, int]] = {}
        for sequence_key, indices in grouped.items():
            ordered = tuple(sorted(indices, key=lambda idx: (int(self.frame_orders[idx]), int(idx))))
            sequence_infos[sequence_key] = SequenceInfo(sequence_key, ordered)
            for order_pos, idx in enumerate(ordered):
                index_to_order_position[int(idx)] = (sequence_key, int(order_pos))
        return sequence_infos, index_to_order_position

    def _build_sensor_values(self, sensor_fist_ratio: float) -> np.ndarray:
        if "sensor" in self.gt:
            sensor = self.gt["sensor"].astype(np.float32)
            if sensor.shape != (len(self.imgname), 5):
                raise ValueError(f"sensor must have shape ({len(self.imgname)}, 5), got {sensor.shape}")
            return np.clip(sensor, 0.0, 1.0).astype(np.float32)
        joints = self.keypoints_3d[:, :, :3]
        return compute_pseudo_sensor_from_model_joints(joints, fist_ratio=sensor_fist_ratio).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def __getitem__(self, item_idx: int) -> Dict:
        target_idx = int(self.sample_indices[item_idx])
        history_indices, valid_mask = self._window_indices(target_idx)
        pose_source = self._resolve_history_source()
        pose_bank = self.gt_pose if pose_source == "gt" else self.base_pose

        pose_window = pose_bank[history_indices].astype(np.float32).copy()
        sensor_window = self.sensor_values[history_indices].astype(np.float32).copy()
        sensor_window[~valid_mask] = 0.0

        return {
            "idx": target_idx,
            "sequence_key": self.sequence_keys[target_idx],
            "frame_order": int(self.frame_orders[target_idx]),
            "history_source": pose_source,
            "base_pose": torch.from_numpy(self.base_pose[target_idx].copy()),
            "base_cam": torch.from_numpy(self.base_cam[target_idx].copy()),
            "target_pose": torch.from_numpy(self.gt_pose[target_idx].copy()),
            "target_hand_pose": torch.from_numpy(self.gt_pose[target_idx, 3:].copy()),
            "pose_window": torch.from_numpy(pose_window),
            "sensor_window": torch.from_numpy(sensor_window),
            "pose_valid_mask": torch.from_numpy(valid_mask.copy()),
            "sensor_valid_mask": torch.from_numpy(valid_mask.copy()),
            "keypoints_2d": torch.from_numpy(self.keypoints_2d[target_idx].copy()),
            "keypoints_3d": torch.from_numpy(self.keypoints_3d[target_idx].copy()),
            "betas": torch.from_numpy(self.betas[target_idx].copy()),
        }

    def _window_indices(self, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sequence_key, order_pos = self.index_to_order_position[target_idx]
        ordered = self.sequence_infos[sequence_key].ordered_indices
        first_idx = int(ordered[0])
        indices: List[int] = []
        valid: List[bool] = []
        for offset in range(self.window_size - 1, -1, -1):
            pos = order_pos - offset
            if pos < 0:
                indices.append(first_idx)
                valid.append(False)
            else:
                indices.append(int(ordered[pos]))
                valid.append(True)
        return np.asarray(indices, dtype=np.int64), np.asarray(valid, dtype=np.bool_)

    def _resolve_history_source(self) -> Literal["base", "gt"]:
        if self.history_source == "mixed":
            return "gt" if self.rng.rand() < self.mixed_gt_prob else "base"
        return self.history_source

    @staticmethod
    def _decode_name(name) -> str:
        return name.decode("utf-8") if isinstance(name, bytes) else str(name)
