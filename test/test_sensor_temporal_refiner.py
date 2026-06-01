import numpy as np
import torch

from hamer.models.components.sensor_temporal_refiner import SensorTemporalRefiner
from hamer.utils.sensor_utils import MODEL_TO_OFFICIAL, compute_pseudo_sensor_from_model_joints


def test_sensor_temporal_refiner_defaults_to_identity_hand_pose_refinement():
    refiner = SensorTemporalRefiner(
        pose_dim=48,
        sensor_dim=5,
        hidden_dim=32,
        num_layers=1,
    )
    base_pose = torch.randn(2, 48)
    pose_window = torch.randn(2, 4, 48)
    sensor_window = torch.rand(2, 4, 5)

    output = refiner(
        base_pose=base_pose,
        pose_window=pose_window,
        sensor_window=sensor_window,
    )

    assert set(output.keys()) == {"delta_hand_pose", "refined_pose"}
    assert output["delta_hand_pose"].shape == (2, 45)
    assert output["refined_pose"].shape == (2, 48)
    torch.testing.assert_close(output["delta_hand_pose"], torch.zeros(2, 45))
    torch.testing.assert_close(output["refined_pose"], base_pose)


def test_sensor_temporal_refiner_optional_global_and_camera_ablation_heads_are_identity_initialized():
    refiner = SensorTemporalRefiner(
        pose_dim=48,
        sensor_dim=5,
        hidden_dim=32,
        num_layers=1,
        predict_global_orient=True,
        predict_cam=True,
        image_feature_dim=16,
    )
    base_pose = torch.randn(3, 48)
    base_cam = torch.randn(3, 3)
    pose_window = torch.randn(3, 2, 48)
    sensor_window = torch.rand(3, 2, 5)
    image_feature = torch.randn(3, 16)

    output = refiner(
        base_pose=base_pose,
        pose_window=pose_window,
        sensor_window=sensor_window,
        base_cam=base_cam,
        image_feature=image_feature,
    )

    assert output["delta_global_orient"].shape == (3, 3)
    assert output["delta_cam"].shape == (3, 3)
    assert output["refined_cam"].shape == (3, 3)
    torch.testing.assert_close(output["delta_global_orient"], torch.zeros(3, 3))
    torch.testing.assert_close(output["delta_cam"], torch.zeros(3, 3))
    torch.testing.assert_close(output["refined_pose"], base_pose)
    torch.testing.assert_close(output["refined_cam"], base_cam)


def test_pseudo_sensor_from_model_joints_returns_normalized_open_hand_values():
    official_joints = np.zeros((21, 3), dtype=np.float32)
    for chain in ([0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]):
        for offset, joint_idx in enumerate(chain):
            official_joints[joint_idx, 0] = float(offset)

    model_order_joints = np.zeros_like(official_joints)
    model_order_joints[MODEL_TO_OFFICIAL] = official_joints

    sensor = compute_pseudo_sensor_from_model_joints(model_order_joints, fist_ratio=0.5)

    assert sensor.shape == (5,)
    np.testing.assert_allclose(sensor, np.ones(5, dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    for test_name, test_fn in sorted(globals().items()):
        if test_name.startswith("test_"):
            test_fn()
