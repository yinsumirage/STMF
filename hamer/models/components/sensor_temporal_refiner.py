"""
Sensor-guided temporal MANO pose refiner.

This is the v2 refiner interface for the new mainline:
- keep the RGB backbone/base HaMeR prediction as the anchor
- use previous MANO poses and 5D normalized pull-sensor windows as temporal priors
- default to refining only MANO hand pose, not global wrist orientation or camera

The module is intentionally independent from ``STMF_HAMER`` so it can be tested
and iterated without disturbing the v1 baseline.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class SensorTemporalRefiner(nn.Module):
    """
    Lightweight residual refiner for sensor-guided temporal MANO recovery.

    Args:
        pose_dim: Full MANO pose dimension, expected as 48 = global_orient 3 + hand_pose 45.
        sensor_dim: Pull-sensor dimension. The current project standard is 5 normalized channels.
        hidden_dim: Internal sequence feature dimension.
        num_layers: Number of GRU layers for temporal aggregation.
        predict_global_orient: Optional ablation head for 3D wrist orientation residuals.
        predict_cam: Optional ablation head for camera residuals.
    """

    def __init__(
        self,
        pose_dim: int = 48,
        sensor_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2,
        predict_global_orient: bool = False,
        predict_cam: bool = False,
    ) -> None:
        super().__init__()
        if pose_dim < 4:
            raise ValueError(f"pose_dim must contain global orient and hand pose, got {pose_dim}")
        if sensor_dim <= 0:
            raise ValueError(f"sensor_dim must be positive, got {sensor_dim}")

        self.pose_dim = int(pose_dim)
        self.hand_pose_dim = self.pose_dim - 3
        self.sensor_dim = int(sensor_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_global_orient = bool(predict_global_orient)
        self.predict_cam = bool(predict_cam)

        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.sensor_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.base_pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.image_encoder = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

        self.temporal_encoder = nn.GRU(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.delta_hand_pose_head = nn.Linear(self.hidden_dim, self.hand_pose_dim)
        self.delta_global_orient_head = nn.Linear(self.hidden_dim, 3) if self.predict_global_orient else None
        self.delta_cam_head = nn.Linear(self.hidden_dim, 3) if self.predict_cam else None

        self._zero_initialize_residual_heads()

    def _zero_initialize_residual_heads(self) -> None:
        for head in (self.delta_hand_pose_head, self.delta_global_orient_head, self.delta_cam_head):
            if head is None:
                continue
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        *,
        base_pose: torch.Tensor,
        pose_window: torch.Tensor,
        sensor_window: torch.Tensor,
        pose_valid_mask: Optional[torch.Tensor] = None,
        sensor_valid_mask: Optional[torch.Tensor] = None,
        image_feature: Optional[torch.Tensor] = None,
        base_cam: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine a base MANO pose with temporal pose and sensor context.

        Shapes:
            base_pose: ``(B, 48)``
            pose_window: ``(B, T, 48)``
            sensor_window: ``(B, T, 5)``
            pose_valid_mask / sensor_valid_mask: optional ``(B, T)`` boolean masks
            image_feature: optional ``(B, D)`` feature from the current frame
            base_cam: optional ``(B, 3)`` camera parameters for the camera ablation head
        """
        self._validate_inputs(base_pose, pose_window, sensor_window, base_cam)
        pose_valid_mask = self._default_mask(pose_window, pose_valid_mask)
        sensor_valid_mask = self._default_mask(sensor_window, sensor_valid_mask)
        valid_mask = pose_valid_mask & sensor_valid_mask

        pose_tokens = self.pose_encoder(pose_window)
        sensor_tokens = self.sensor_encoder(sensor_window)
        temporal_tokens = torch.cat([pose_tokens, sensor_tokens], dim=-1)

        encoded, _ = self.temporal_encoder(temporal_tokens)
        temporal_summary = self._masked_mean(encoded, valid_mask)
        base_summary = self.base_pose_encoder(base_pose)
        if image_feature is not None:
            base_summary = base_summary + self.image_encoder(image_feature)

        fused = self.fusion(torch.cat([temporal_summary, base_summary], dim=-1))
        delta_hand_pose = self.delta_hand_pose_head(fused)

        refined_pose = base_pose.clone()
        refined_pose[:, 3:] = refined_pose[:, 3:] + delta_hand_pose

        output: Dict[str, torch.Tensor] = {
            "delta_hand_pose": delta_hand_pose,
            "refined_pose": refined_pose,
        }

        if self.delta_global_orient_head is not None:
            delta_global_orient = self.delta_global_orient_head(fused)
            refined_pose[:, :3] = base_pose[:, :3] + delta_global_orient
            output["delta_global_orient"] = delta_global_orient
            output["refined_pose"] = refined_pose

        if self.delta_cam_head is not None:
            delta_cam = self.delta_cam_head(fused)
            output["delta_cam"] = delta_cam
            if base_cam is not None:
                output["refined_cam"] = base_cam + delta_cam

        return output

    def _validate_inputs(
        self,
        base_pose: torch.Tensor,
        pose_window: torch.Tensor,
        sensor_window: torch.Tensor,
        base_cam: Optional[torch.Tensor],
    ) -> None:
        if base_pose.ndim != 2 or base_pose.shape[1] != self.pose_dim:
            raise ValueError(f"base_pose must have shape (B, {self.pose_dim}), got {tuple(base_pose.shape)}")
        if pose_window.ndim != 3 or pose_window.shape[0] != base_pose.shape[0] or pose_window.shape[2] != self.pose_dim:
            raise ValueError(f"pose_window must have shape (B, T, {self.pose_dim}), got {tuple(pose_window.shape)}")
        if sensor_window.ndim != 3 or sensor_window.shape[:2] != pose_window.shape[:2] or sensor_window.shape[2] != self.sensor_dim:
            raise ValueError(f"sensor_window must have shape (B, T, {self.sensor_dim}), got {tuple(sensor_window.shape)}")
        if self.predict_cam and base_cam is not None and (base_cam.ndim != 2 or base_cam.shape != (base_pose.shape[0], 3)):
            raise ValueError(f"base_cam must have shape (B, 3), got {tuple(base_cam.shape)}")

    @staticmethod
    def _default_mask(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        return mask.to(device=values.device, dtype=torch.bool)

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
