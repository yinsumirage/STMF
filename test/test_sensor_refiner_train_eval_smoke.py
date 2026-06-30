import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import torch

from hamer.datasets.sensor_refiner_dataset import SensorRefinerDataset
from hamer.models.components.sensor_temporal_refiner import SensorTemporalRefiner


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_sensor_refiner = _load_script_module("train_sensor_refiner", "scripts/train_sensor_refiner.py")
eval_sensor_refiner = _load_script_module("eval_sensor_refiner", "scripts/eval_sensor_refiner.py")


def _make_fake_files(root: Path):
    imgname = np.asarray([
        "train/S1/rgb/0000.jpg",
        "train/S1/rgb/0001.jpg",
        "train/S1/rgb/0002.jpg",
        "train/S2/rgb/0000.jpg",
    ])
    target_pose = np.random.randn(4, 48).astype(np.float32)
    base_pose = target_pose + np.random.randn(4, 48).astype(np.float32) * 0.1
    joints = np.zeros((4, 21, 4), dtype=np.float32)
    joints[..., 3] = 1.0
    for sample_idx in range(4):
        for joint_idx in range(21):
            joints[sample_idx, joint_idx, 0] = float(sample_idx + joint_idx)

    dataset_file = root / "dataset.npz"
    cache_file = root / "cache.npz"
    np.savez(
        dataset_file,
        imgname=imgname,
        hand_pose=target_pose,
        has_hand_pose=np.ones(4, dtype=np.float32),
        betas=np.zeros((4, 10), dtype=np.float32),
        has_betas=np.ones(4, dtype=np.float32),
        hand_keypoints_3d=joints,
    )
    np.savez(cache_file, imgname=imgname, base_pose=base_pose, base_cam=np.zeros((4, 3), dtype=np.float32))
    return dataset_file, cache_file


def test_sensor_refiner_one_step_train_save_load_and_stateful_window():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_file, cache_file = _make_fake_files(Path(tmp_dir))
        dataset = SensorRefinerDataset(
            dataset_file=str(dataset_file),
            base_pred_file=str(cache_file),
            window_size=3,
            history_source="base",
        )
        batch = dataset[1]
        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim > 0 else value
            for key, value in batch.items()
        }

        model = SensorTemporalRefiner(hidden_dim=32, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        output = model(
            base_pose=batch["base_pose"],
            pose_window=batch["pose_window"],
            sensor_window=batch["sensor_window"],
            pose_valid_mask=batch["pose_valid_mask"],
            sensor_valid_mask=batch["sensor_valid_mask"],
        )
        loss_dict = train_sensor_refiner.compute_loss(batch, output, smoothness_weight=0.1, global_orient_weight=0.0)
        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        optimizer.step()

        ckpt = Path(tmp_dir) / "last.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": {"hidden_dim": 32, "num_layers": 1}}, ckpt)
        loaded = torch.load(ckpt, map_location="cpu")
        reloaded = SensorTemporalRefiner(hidden_dim=32, num_layers=1)
        reloaded.load_state_dict(loaded["model_state_dict"], strict=True)

        refined_history = {"S1": [torch.zeros(48), torch.ones(48)]}
        pose_window, valid_mask = eval_sensor_refiner.build_stateful_pose_window(
            dataset,
            target_idx=2,
            refined_history=refined_history,
            base_pose=torch.full((48,), 2.0),
        )
        assert pose_window.shape == (3, 48)
        assert valid_mask.tolist() == [True, True, True]
        torch.testing.assert_close(pose_window[0], torch.zeros(48))
        torch.testing.assert_close(pose_window[1], torch.ones(48))
        torch.testing.assert_close(pose_window[2], torch.full((48,), 2.0))


def test_training_base_pose_noise_keeps_global_orientation_clean():
    base_pose = torch.zeros(4, 48)
    noisy = train_sensor_refiner.apply_base_pose_noise(base_pose.clone(), noise_std=0.1)

    torch.testing.assert_close(noisy[:, :3], base_pose[:, :3])
    assert torch.count_nonzero(noisy[:, 3:]) > 0


if __name__ == "__main__":
    for test_name, test_fn in sorted(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
