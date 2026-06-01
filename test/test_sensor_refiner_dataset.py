import tempfile
from pathlib import Path

import numpy as np
import torch

from hamer.datasets.sensor_refiner_dataset import SensorRefinerDataset


def _write_npz(path: Path, **arrays):
    np.savez(path, **arrays)


def _make_joints(num_samples: int) -> np.ndarray:
    joints = np.zeros((num_samples, 21, 4), dtype=np.float32)
    joints[..., 3] = 1.0
    for sample_idx in range(num_samples):
        for joint_idx in range(21):
            joints[sample_idx, joint_idx, 0] = float(joint_idx + sample_idx)
    return joints


def test_sensor_refiner_dataset_builds_left_padded_base_history_window():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        dataset_file = root / "dataset.npz"
        cache_file = root / "base_cache.npz"
        imgname = np.asarray(
            [
                "train/S1/rgb/0000.jpg",
                "train/S1/rgb/0001.jpg",
                "train/S1/rgb/0002.jpg",
                "train/S2/rgb/0000.jpg",
            ]
        )
        gt_pose = np.arange(4 * 48, dtype=np.float32).reshape(4, 48)
        base_pose = gt_pose + 1000.0
        base_cam = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        _write_npz(
            dataset_file,
            imgname=imgname,
            hand_pose=gt_pose,
            has_hand_pose=np.ones(4, dtype=np.float32),
            betas=np.zeros((4, 10), dtype=np.float32),
            has_betas=np.ones(4, dtype=np.float32),
            hand_keypoints_3d=_make_joints(4),
        )
        _write_npz(cache_file, base_pose=base_pose, base_cam=base_cam, imgname=imgname)

        dataset = SensorRefinerDataset(
            dataset_file=str(dataset_file),
            base_pred_file=str(cache_file),
            window_size=3,
            history_source="base",
        )

        sample = dataset[1]

        assert sample["idx"] == 1
        torch.testing.assert_close(sample["base_pose"], torch.from_numpy(base_pose[1]))
        torch.testing.assert_close(sample["target_pose"], torch.from_numpy(gt_pose[1]))
        torch.testing.assert_close(sample["pose_window"][0], torch.from_numpy(base_pose[0]))
        torch.testing.assert_close(sample["pose_window"][1], torch.from_numpy(base_pose[0]))
        torch.testing.assert_close(sample["pose_window"][2], torch.from_numpy(base_pose[1]))
        assert sample["pose_valid_mask"].tolist() == [False, True, True]
        assert sample["sensor_valid_mask"].tolist() == [False, True, True]
        assert sample["sensor_window"].shape == (3, 5)
        assert sample["sequence_key"] == "S1"
        assert sample["frame_order"] == 1


def test_sensor_refiner_dataset_can_use_gt_history_for_teacher_forcing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        dataset_file = root / "dataset.npz"
        cache_file = root / "base_cache.npz"
        imgname = np.asarray(["train/S1/rgb/0000.jpg", "train/S1/rgb/0001.jpg"])
        gt_pose = np.arange(2 * 48, dtype=np.float32).reshape(2, 48)
        base_pose = gt_pose + 1000.0
        _write_npz(
            dataset_file,
            imgname=imgname,
            hand_pose=gt_pose,
            has_hand_pose=np.ones(2, dtype=np.float32),
            betas=np.zeros((2, 10), dtype=np.float32),
            has_betas=np.ones(2, dtype=np.float32),
            hand_keypoints_3d=_make_joints(2),
        )
        _write_npz(cache_file, base_pose=base_pose, base_cam=np.zeros((2, 3), dtype=np.float32), imgname=imgname)

        dataset = SensorRefinerDataset(
            dataset_file=str(dataset_file),
            base_pred_file=str(cache_file),
            window_size=2,
            history_source="gt",
        )

        sample = dataset[1]

        torch.testing.assert_close(sample["pose_window"][0], torch.from_numpy(gt_pose[0]))
        torch.testing.assert_close(sample["pose_window"][1], torch.from_numpy(gt_pose[1]))


if __name__ == "__main__":
    for test_name, test_fn in sorted(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
