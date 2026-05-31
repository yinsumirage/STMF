"""
Inspect HO3D projection consistency on packed samples.

This tool is meant to answer a very specific debugging question:

- Do packed HO3D 2D keypoints agree with
  1) packed 3D joints projected with the HO3D/OpenGL coordinate conversion?
  2) meta `handJoints3D` projected the same way?
  3) GT MANO parameters projected with / without `handTrans`?
- What happens if we remove that coordinate conversion?
- Is the remaining discrepancy coming from `handTrans`, MANO reconstruction, or the
  packed NPZ itself?

Typical usage:

1. Inspect one packed HO3D train sample by index:
python tools/data_prep/inspect_ho3d_projection_consistency.py   --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz   --base_dir /data/hand_data/HO-3D_v3   --out_dir ./inspect_projection_debug   --indices 11903,23807
conda run -n STMF python tools/data_prep/inspect_ho3d_projection_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --out_dir /home/mirage/STMF/_DATA/HO-3D_v3/inspect_projection_debug \
  --indices 0

2. Inspect a few evenly spaced samples:
conda run -n STMF python tools/data_prep/inspect_ho3d_projection_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --out_dir /home/mirage/STMF/_DATA/HO-3D_v3/inspect_projection_debug \
  --num_samples 8

3. Inspect one random sample:
conda run -n STMF python tools/data_prep/inspect_ho3d_projection_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --out_dir /home/mirage/STMF/_DATA/HO-3D_v3/inspect_projection_debug \
  --num_samples 1 \
  --selection random \
  --seed 42

4. Additionally fit the best possible HaMeR-style `pred_cam_t` on the deterministic
   patch-space sample using GT MANO local joints:
conda run -n STMF python tools/data_prep/inspect_ho3d_projection_consistency.py \
  --dataset_file /home/mirage/STMF/_DATA/HO-3D_v3/ho3d_train.npz \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --out_dir /home/mirage/STMF/_DATA/HO-3D_v3/inspect_projection_debug \
  --indices 11903,23807 \
  --fit_patch_pred_cam_t

Outputs:
- Per-sample contact sheets with multiple projection variants.
- `summary.json` containing per-sample reprojection errors and key meta values.
- If `--fit_patch_pred_cam_t` is enabled:
  - extra patch-space contact sheets
  - fitted `cam_t`
  - patch-space 2D error statistics for `MANO local`, `coord_change(MANO local)`,
    and `coord_change(MANO+handTrans)`
  - signed-`tz` diagnostics to test whether the remaining hard-case residual is caused
    by HaMeR's positive-depth camera parameterization
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.models.mano_wrapper import MANO
from hamer.datasets.image_dataset import ImageDataset
from hamer.datasets.utils import gen_trans_from_patch_cv
from hamer.utils.geometry import aa_to_rotmat, perspective_projection
from hamer.utils.render_openpose import render_openpose
from scripts.train import apply_runtime_overrides

HO3D_OFFICIAL_TO_OPENPOSE = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


def parse_indices(indices: Optional[str]) -> Optional[List[int]]:
    if not indices:
        return None
    return [int(x.strip()) for x in indices.split(",") if x.strip()]


def build_cfg(experiment: str) -> DictConfig:
    overrides = [
        f"experiment={experiment}",
        "strategy=auto",
        "run_validation=false",
        "num_sanity_val_steps=0",
        "log_lr=false",
    ]
    with hydra.initialize_config_dir(version_base="1.2", config_dir=str(root / "hamer" / "configs_hydra")):
        cfg = hydra.compose(config_name="train.yaml", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.extras.enforce_tags = False
    cfg.extras.print_config = False
    apply_runtime_overrides(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def choose_indices(total: int, num_samples: int, selection: str = "spread", seed: int = 42) -> List[int]:
    if total <= 0:
        return []
    if num_samples >= total:
        return list(range(total))
    if selection == "first":
        return list(range(num_samples))
    if selection == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(total, size=num_samples, replace=False).tolist())
    return sorted(set(np.linspace(0, total - 1, num=num_samples, dtype=int).tolist()))


def resolve_image_path(base_dir: str, rel_path: str) -> str:
    path = os.path.join(base_dir, rel_path)
    if os.path.exists(path):
        return path

    stem = os.path.splitext(path)[0]
    for ext in (".jpg", ".png", ".jpeg"):
        alt = stem + ext
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"Image not found for sample: {rel_path}")


def resolve_meta_path(base_dir: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/")
    stem = os.path.splitext(rel)[0]
    if "/rgb/" not in stem:
        raise ValueError(f"Cannot infer HO3D meta path from: {rel_path}")
    meta_rel = stem.replace("/rgb/", "/meta/") + ".pkl"
    meta_path = os.path.join(base_dir, meta_rel)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found for sample: {meta_rel}")
    return meta_path


class InspectImageDataset(ImageDataset):
    def _apply_ho3d_official_subset_filter(self, dataset_file: str) -> None:
        # This diagnostic inspects arbitrary train samples by raw index.
        # We do not want the evaluation whitelist to silently remap or drop indices.
        return


def draw_bbox(img: np.ndarray, center: np.ndarray, scale: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        w = h = float(scale[0])
    else:
        w, h = float(scale[0]), float(scale[1])
    cx, cy = float(center[0]), float(center[1])
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 3, color, -1)
    return out


def add_caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_contact_sheet(panels: List[np.ndarray], ncols: int, pad: int = 8, bg=(20, 20, 20)) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to render")
    h, w = panels[0].shape[:2]
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    sheet = np.full((nrows * h + (nrows + 1) * pad, ncols * w + (ncols + 1) * pad, 3), bg, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // ncols
        c = idx % ncols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = panel
    return sheet


def render_skeleton_panel(img_bgr: np.ndarray, kps_2d: np.ndarray, center: np.ndarray, scale: np.ndarray, title: str) -> np.ndarray:
    panel = draw_bbox(img_bgr, center, scale)
    rendered = render_openpose(panel, kps_2d.copy())
    rendered = np.ascontiguousarray(rendered.astype(np.uint8))
    return add_caption(rendered, title)


def denormalize_patch_image(img_tensor: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    img = img_tensor.astype(np.float32).copy()
    for c in range(min(img.shape[0], 3)):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def patch_coords_to_pixels(keypoints_2d: np.ndarray, image_size: int) -> np.ndarray:
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
    out = keypoints_2d.copy()
    out[:, :2] = image_size * (out[:, :2] + 0.5)
    return out


def patch_pixels_to_normalized(keypoints_2d_px: np.ndarray, image_size: int) -> np.ndarray:
    keypoints_2d_px = np.asarray(keypoints_2d_px, dtype=np.float32)
    out = keypoints_2d_px.copy()
    out[:, :2] = out[:, :2] / float(image_size) - 0.5
    return out


def project_points(cam_mat: np.ndarray, pts3d: np.ndarray, apply_coord_change: bool) -> np.ndarray:
    pts3d = np.asarray(pts3d, dtype=np.float32)
    coord_change = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    if apply_coord_change:
        pts3d = pts3d @ coord_change.T
    proj = pts3d @ np.asarray(cam_mat, dtype=np.float32).T
    xy = np.stack([proj[:, 0] / proj[:, 2], proj[:, 1] / proj[:, 2]], axis=1)
    return np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)


def apply_affine_to_points(points_2d: np.ndarray, trans: np.ndarray) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    out = points_2d.copy()
    ones = np.ones((points_2d.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points_2d[:, :2], ones], axis=1)
    mapped = homo @ trans.T
    out[:, :2] = mapped
    return out


def affine_to_homogeneous(trans: np.ndarray) -> np.ndarray:
    trans = np.asarray(trans, dtype=np.float32)
    out = np.eye(3, dtype=np.float32)
    out[:2, :] = trans
    return out


def decompose_patch_cam_mat(cam_mat: np.ndarray) -> Dict[str, float]:
    cam_mat = np.asarray(cam_mat, dtype=np.float32)
    return {
        "fx": float(cam_mat[0, 0]),
        "fy": float(cam_mat[1, 1]),
        "cx": float(cam_mat[0, 2]),
        "cy": float(cam_mat[1, 2]),
        "skew_xy": float(cam_mat[0, 1]),
        "skew_yx": float(cam_mat[1, 0]),
    }


def apply_coord_change(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    coord_change = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    return points @ coord_change.T


def reprojection_error(pred_kps: np.ndarray, gt_kps: np.ndarray) -> float:
    pred_kps = np.asarray(pred_kps, dtype=np.float32)
    gt_kps = np.asarray(gt_kps, dtype=np.float32)
    conf = gt_kps[:, 2] > 0.0
    if conf.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(pred_kps[conf, :2] - gt_kps[conf, :2], axis=1).mean())


def build_mano(mano_model_path: str) -> MANO:
    return MANO(
        model_path=mano_model_path,
        gender="neutral",
        use_pca=False,
        flat_hand_mean=False,
        num_pca_comps=45,
        is_rhand=True,
    )


def reconstruct_mano_joints(meta: Dict, mano: MANO) -> Tuple[np.ndarray, np.ndarray]:
    hand_pose = torch.from_numpy(np.asarray(meta["handPose"], dtype=np.float32)).unsqueeze(0)
    betas = torch.from_numpy(np.asarray(meta["handBeta"], dtype=np.float32)).unsqueeze(0)
    hand_trans = torch.from_numpy(np.asarray(meta["handTrans"], dtype=np.float32)).unsqueeze(0)

    global_orient = aa_to_rotmat(hand_pose[:, :3]).view(-1, 1, 3, 3)
    pose = aa_to_rotmat(hand_pose[:, 3:].reshape(-1, 3)).view(-1, 15, 3, 3)

    with torch.no_grad():
        mano_out = mano(global_orient=global_orient, hand_pose=pose, betas=betas)
    joints_local = mano_out.joints[0].cpu().numpy().astype(np.float32)
    joints_with_trans = joints_local + hand_trans.cpu().numpy().astype(np.float32)
    return joints_local, joints_with_trans


def per_joint_errors(pred_kps: np.ndarray, gt_kps: np.ndarray) -> List[float]:
    pred_kps = np.asarray(pred_kps, dtype=np.float32)
    gt_kps = np.asarray(gt_kps, dtype=np.float32)
    conf = gt_kps[:, 2] > 0.0
    errs = np.linalg.norm(pred_kps[:, :2] - gt_kps[:, :2], axis=1)
    out = []
    for idx, err in enumerate(errs.tolist()):
        out.append(float(err) if bool(conf[idx]) else float("nan"))
    return out


def root_relative_3d_error_mm(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    pred_rr = pred_xyz - pred_xyz[[0]]
    gt_rr = gt_xyz - gt_xyz[[0]]
    return float(np.linalg.norm(pred_rr - gt_rr, axis=1).mean())


def fit_patch_pred_cam_t(
    points_3d: np.ndarray,
    gt_keypoints_2d: np.ndarray,
    image_size: int,
    focal_length: float,
    num_iters: int = 2000,
    lr: float = 0.05,
    free_tz: bool = False,
) -> Dict[str, object]:
    device = torch.device("cpu")
    points = torch.from_numpy(np.asarray(points_3d, dtype=np.float32)).unsqueeze(0).to(device)
    gt = torch.from_numpy(np.asarray(gt_keypoints_2d, dtype=np.float32)).unsqueeze(0).to(device)
    conf = gt[..., 2:3]
    target_xy = gt[..., :2]
    focal = torch.tensor([[float(focal_length), float(focal_length)]], dtype=torch.float32, device=device)

    init_z = 2.0 * float(focal_length) / float(image_size)
    init_raw_z = float(init_z) if free_tz else float(np.log(np.expm1(init_z)))
    cam_raw = torch.nn.Parameter(torch.tensor([[0.0, 0.0, init_raw_z]], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([cam_raw], lr=lr)

    def unpack_cam(raw: torch.Tensor) -> torch.Tensor:
        tx_ty = raw[:, :2]
        tz = raw[:, 2:3] if free_tz else (torch.nn.functional.softplus(raw[:, 2:3]) + 1e-6)
        return torch.cat([tx_ty, tz], dim=1)

    best = None
    for _ in range(num_iters):
        optimizer.zero_grad()
        cam_t = unpack_cam(cam_raw)
        pred_xy = perspective_projection(points, translation=cam_t, focal_length=focal / float(image_size))
        diff = (pred_xy - target_xy) * conf
        loss = (diff ** 2).sum() / conf.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            l1_sum = torch.abs(diff).sum()
            if best is None or float(l1_sum.item()) < best["l1_sum"]:
                best = {
                    "cam_t": cam_t.detach().cpu().numpy()[0].astype(np.float32),
                    "pred_xy": pred_xy.detach().cpu().numpy()[0].astype(np.float32),
                    "l1_sum": float(l1_sum.item()),
                }

    pred_xy = best["pred_xy"]
    pred_kps = np.concatenate([pred_xy, np.ones((pred_xy.shape[0], 1), dtype=np.float32)], axis=1)
    gt_np = np.asarray(gt_keypoints_2d, dtype=np.float32)
    valid = gt_np[:, 2] > 0
    mean_norm = float(np.linalg.norm(pred_xy[valid] - gt_np[valid, :2], axis=1).mean()) if valid.any() else float("nan")
    pred_px = patch_coords_to_pixels(pred_kps, image_size)
    gt_px = patch_coords_to_pixels(gt_np, image_size)
    mean_px = float(np.linalg.norm(pred_px[valid, :2] - gt_px[valid, :2], axis=1).mean()) if valid.any() else float("nan")
    per_joint_px = per_joint_errors(pred_px, gt_px)
    return {
        "cam_t": best["cam_t"].tolist(),
        "l1_sum": best["l1_sum"],
        "mean_norm_error": mean_norm,
        "mean_px_error": mean_px,
        "pred_keypoints_2d": pred_kps,
        "pred_keypoints_2d_px": pred_px,
        "gt_keypoints_2d_px": gt_px,
        "per_joint_px": per_joint_px,
    }


def fit_patch_camera_family(
    points_3d: np.ndarray,
    gt_keypoints_2d: np.ndarray,
    image_size: int,
    focal_length: float,
    num_iters: int = 3000,
    lr: float = 0.03,
    free_tz: bool = False,
) -> Dict[str, object]:
    device = torch.device("cpu")
    points = torch.from_numpy(np.asarray(points_3d, dtype=np.float32)).unsqueeze(0).to(device)
    gt = torch.from_numpy(np.asarray(gt_keypoints_2d, dtype=np.float32)).unsqueeze(0).to(device)
    conf = gt[..., 2:3]
    target_xy = gt[..., :2]
    base_focal = torch.tensor([[float(focal_length), float(focal_length)]], dtype=torch.float32, device=device)

    init_z = 2.0 * float(focal_length) / float(image_size)
    init_raw_z = float(init_z) if free_tz else float(np.log(np.expm1(init_z)))
    cam_raw = torch.nn.Parameter(torch.tensor([[0.0, 0.0, init_raw_z]], dtype=torch.float32, device=device))
    intrinsics_raw = torch.nn.Parameter(torch.zeros((1, 4), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([cam_raw, intrinsics_raw], lr=lr)

    def unpack_cam(raw: torch.Tensor) -> torch.Tensor:
        tx_ty = raw[:, :2]
        tz = raw[:, 2:3] if free_tz else (torch.nn.functional.softplus(raw[:, 2:3]) + 1e-6)
        return torch.cat([tx_ty, tz], dim=1)

    best = None
    for _ in range(num_iters):
        optimizer.zero_grad()
        cam_t = unpack_cam(cam_raw)
        delta_log_f = torch.tanh(intrinsics_raw[:, :2])
        center = 0.5 * torch.tanh(intrinsics_raw[:, 2:4])
        focal = base_focal * torch.exp(delta_log_f)
        pred_xy = perspective_projection(
            points,
            translation=cam_t,
            focal_length=focal / float(image_size),
            camera_center=center,
        )
        diff = (pred_xy - target_xy) * conf
        loss = torch.abs(diff).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if best is None or float(loss.item()) < best["l1_sum"]:
                best = {
                    "cam_t": cam_t.detach().cpu().numpy()[0].astype(np.float32),
                    "focal_length": focal.detach().cpu().numpy()[0].astype(np.float32),
                    "camera_center": center.detach().cpu().numpy()[0].astype(np.float32),
                    "pred_xy": pred_xy.detach().cpu().numpy()[0].astype(np.float32),
                    "l1_sum": float(loss.item()),
                }

    pred_xy = best["pred_xy"]
    pred_kps = np.concatenate([pred_xy, np.ones((pred_xy.shape[0], 1), dtype=np.float32)], axis=1)
    gt_np = np.asarray(gt_keypoints_2d, dtype=np.float32)
    valid = gt_np[:, 2] > 0
    mean_norm = float(np.linalg.norm(pred_xy[valid] - gt_np[valid, :2], axis=1).mean()) if valid.any() else float("nan")
    pred_px = patch_coords_to_pixels(pred_kps, image_size)
    gt_px = patch_coords_to_pixels(gt_np, image_size)
    mean_px = float(np.linalg.norm(pred_px[valid, :2] - gt_px[valid, :2], axis=1).mean()) if valid.any() else float("nan")
    per_joint_px = per_joint_errors(pred_px, gt_px)
    return {
        "cam_t": best["cam_t"].tolist(),
        "focal_length": best["focal_length"].tolist(),
        "camera_center": best["camera_center"].tolist(),
        "l1_sum": best["l1_sum"],
        "mean_norm_error": mean_norm,
        "mean_px_error": mean_px,
        "pred_keypoints_2d": pred_kps,
        "pred_keypoints_2d_px": pred_px,
        "gt_keypoints_2d_px": gt_px,
        "per_joint_px": per_joint_px,
    }


def inspect_patch_projection_fit(
    patch_dataset: ImageDataset,
    idx: int,
    mano_local_joints3d: np.ndarray,
    mano_joints3d: np.ndarray,
    packed_kps3d: np.ndarray,
    cam_mat_raw: np.ndarray,
    out_dir: str,
    image_size: int,
    focal_length: float,
) -> Dict[str, object]:
    item = patch_dataset[idx]
    patch_img_bgr = denormalize_patch_image(item["img"], patch_dataset.mean, patch_dataset.std)
    gt_patch_kps = item["keypoints_2d"].astype(np.float32)
    gt_patch_kps_px = patch_coords_to_pixels(gt_patch_kps, image_size)
    box_center = np.asarray(item["box_center"], dtype=np.float32)
    box_size = float(item["box_size"])
    trans = gen_trans_from_patch_cv(
        float(box_center[0]),
        float(box_center[1]),
        box_size,
        box_size,
        float(image_size),
        float(image_size),
        1.0,
        0.0,
    )
    trans_h = affine_to_homogeneous(trans)
    patch_cam_mat = trans_h @ np.asarray(cam_mat_raw, dtype=np.float32)
    patch_cam_params = decompose_patch_cam_mat(patch_cam_mat)

    local_fit = fit_patch_pred_cam_t(
        points_3d=mano_local_joints3d,
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
    )
    local_coord_fit = fit_patch_pred_cam_t(
        points_3d=apply_coord_change(mano_local_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
    )
    local_coord_camk_fit = fit_patch_camera_family(
        points_3d=apply_coord_change(mano_local_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
    )
    local_coord_freez_fit = fit_patch_pred_cam_t(
        points_3d=apply_coord_change(mano_local_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
        free_tz=True,
    )
    local_coord_camk_freez_fit = fit_patch_camera_family(
        points_3d=apply_coord_change(mano_local_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
        free_tz=True,
    )
    mano_coord_fit = fit_patch_pred_cam_t(
        points_3d=apply_coord_change(mano_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
    )
    mano_coord_camk_fit = fit_patch_camera_family(
        points_3d=apply_coord_change(mano_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
    )
    mano_coord_freez_fit = fit_patch_pred_cam_t(
        points_3d=apply_coord_change(mano_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
        free_tz=True,
    )
    mano_coord_camk_freez_fit = fit_patch_camera_family(
        points_3d=apply_coord_change(mano_joints3d),
        gt_keypoints_2d=gt_patch_kps,
        image_size=image_size,
        focal_length=focal_length,
        free_tz=True,
    )

    packed_raw_proj = project_points(cam_mat_raw, packed_kps3d, apply_coord_change=True)
    packed_exact_patch_px = apply_affine_to_points(packed_raw_proj, trans)
    packed_exact_patch_norm = patch_pixels_to_normalized(packed_exact_patch_px, image_size)

    mano_raw_proj = project_points(cam_mat_raw, mano_joints3d, apply_coord_change=True)
    mano_exact_patch_px = apply_affine_to_points(mano_raw_proj, trans)
    mano_exact_patch_norm = patch_pixels_to_normalized(mano_exact_patch_px, image_size)

    packed_patch_proj_via_k = project_points(patch_cam_mat, packed_kps3d, apply_coord_change=True)
    mano_patch_proj_via_k = project_points(patch_cam_mat, mano_joints3d, apply_coord_change=True)

    exact_metrics = {
        "packed_exact_crop_l1": float(np.abs(packed_exact_patch_norm[:, :2] - gt_patch_kps[:, :2]).sum()),
        "packed_exact_crop_px": reprojection_error(packed_exact_patch_px, gt_patch_kps_px),
        "packed_patchK_l1": float(np.abs(packed_patch_proj_via_k[:, :2] / float(image_size) - 0.5 - gt_patch_kps[:, :2]).sum()),
        "packed_patchK_px": reprojection_error(packed_patch_proj_via_k, gt_patch_kps_px),
        "mano_exact_crop_l1": float(np.abs(mano_exact_patch_norm[:, :2] - gt_patch_kps[:, :2]).sum()),
        "mano_exact_crop_px": reprojection_error(mano_exact_patch_px, gt_patch_kps_px),
        "mano_patchK_l1": float(np.abs(mano_patch_proj_via_k[:, :2] / float(image_size) - 0.5 - gt_patch_kps[:, :2]).sum()),
        "mano_patchK_px": reprojection_error(mano_patch_proj_via_k, gt_patch_kps_px),
    }

    panels = [
        add_caption(patch_img_bgr, f"patch idx={idx}"),
        add_caption(
            render_openpose(patch_img_bgr.copy(), gt_patch_kps_px.copy()),
            "patch GT 2D",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), packed_exact_patch_px.copy()),
            f"exact crop(packed3D) {exact_metrics['packed_exact_crop_px']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_exact_patch_px.copy()),
            f"exact crop(MANO+trans) {exact_metrics['mano_exact_crop_px']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), packed_patch_proj_via_k.copy()),
            f"patchK(packed3D) {exact_metrics['packed_patchK_px']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_patch_proj_via_k.copy()),
            f"patchK(MANO+trans) {exact_metrics['mano_patchK_px']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), local_fit["pred_keypoints_2d_px"].copy()),
            f"fit cam_t on MANO local {local_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), local_coord_fit["pred_keypoints_2d_px"].copy()),
            f"fit cam_t on coord_change(MANO local) {local_coord_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), local_coord_camk_fit["pred_keypoints_2d_px"].copy()),
            f"fit cam+K on coord_change(MANO local) {local_coord_camk_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), local_coord_freez_fit["pred_keypoints_2d_px"].copy()),
            f"fit free-z cam_t on coord_change(MANO local) {local_coord_freez_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), local_coord_camk_freez_fit["pred_keypoints_2d_px"].copy()),
            f"fit free-z cam+K on coord_change(MANO local) {local_coord_camk_freez_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_coord_fit["pred_keypoints_2d_px"].copy()),
            f"fit cam_t on coord_change(MANO+trans) {mano_coord_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_coord_camk_fit["pred_keypoints_2d_px"].copy()),
            f"fit cam+K on coord_change(MANO+trans) {mano_coord_camk_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_coord_freez_fit["pred_keypoints_2d_px"].copy()),
            f"fit free-z cam_t on coord_change(MANO+trans) {mano_coord_freez_fit['mean_px_error']:.2f}px",
        ),
        add_caption(
            render_openpose(patch_img_bgr.copy(), mano_coord_camk_freez_fit["pred_keypoints_2d_px"].copy()),
            f"fit free-z cam+K on coord_change(MANO+trans) {mano_coord_camk_freez_fit['mean_px_error']:.2f}px",
        ),
    ]
    sheet = build_contact_sheet(panels, ncols=3, pad=8)
    out_file = os.path.join(out_dir, f"{idx:06d}_patch_fit.jpg")
    cv2.imwrite(out_file, sheet)

    return {
        "output_file": out_file,
        "gt_patch_keypoints_2d_px": gt_patch_kps_px.tolist(),
        "patch_cam_mat": patch_cam_mat.tolist(),
        "patch_cam_params": patch_cam_params,
        "exact_patch_projection": exact_metrics,
        "mano_local_fit": {
            "cam_t": local_fit["cam_t"],
            "loss_keypoints_2d": local_fit["l1_sum"],
            "mean_patch_px_error": local_fit["mean_px_error"],
            "mean_patch_norm_error": local_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(mano_local_joints3d, packed_kps3d),
            "per_joint_px": local_fit["per_joint_px"],
        },
        "mano_local_coord_change_fit": {
            "cam_t": local_coord_fit["cam_t"],
            "loss_keypoints_2d": local_coord_fit["l1_sum"],
            "mean_patch_px_error": local_coord_fit["mean_px_error"],
            "mean_patch_norm_error": local_coord_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_local_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": local_coord_fit["per_joint_px"],
        },
        "mano_local_coord_change_camk_fit": {
            "cam_t": local_coord_camk_fit["cam_t"],
            "focal_length": local_coord_camk_fit["focal_length"],
            "camera_center": local_coord_camk_fit["camera_center"],
            "loss_keypoints_2d": local_coord_camk_fit["l1_sum"],
            "mean_patch_px_error": local_coord_camk_fit["mean_px_error"],
            "mean_patch_norm_error": local_coord_camk_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_local_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": local_coord_camk_fit["per_joint_px"],
        },
        "mano_local_coord_change_free_tz_fit": {
            "cam_t": local_coord_freez_fit["cam_t"],
            "loss_keypoints_2d": local_coord_freez_fit["l1_sum"],
            "mean_patch_px_error": local_coord_freez_fit["mean_px_error"],
            "mean_patch_norm_error": local_coord_freez_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_local_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": local_coord_freez_fit["per_joint_px"],
        },
        "mano_local_coord_change_camk_free_tz_fit": {
            "cam_t": local_coord_camk_freez_fit["cam_t"],
            "focal_length": local_coord_camk_freez_fit["focal_length"],
            "camera_center": local_coord_camk_freez_fit["camera_center"],
            "loss_keypoints_2d": local_coord_camk_freez_fit["l1_sum"],
            "mean_patch_px_error": local_coord_camk_freez_fit["mean_px_error"],
            "mean_patch_norm_error": local_coord_camk_freez_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_local_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": local_coord_camk_freez_fit["per_joint_px"],
        },
        "mano_with_trans_coord_change_fit": {
            "cam_t": mano_coord_fit["cam_t"],
            "loss_keypoints_2d": mano_coord_fit["l1_sum"],
            "mean_patch_px_error": mano_coord_fit["mean_px_error"],
            "mean_patch_norm_error": mano_coord_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": mano_coord_fit["per_joint_px"],
        },
        "mano_with_trans_coord_change_camk_fit": {
            "cam_t": mano_coord_camk_fit["cam_t"],
            "focal_length": mano_coord_camk_fit["focal_length"],
            "camera_center": mano_coord_camk_fit["camera_center"],
            "loss_keypoints_2d": mano_coord_camk_fit["l1_sum"],
            "mean_patch_px_error": mano_coord_camk_fit["mean_px_error"],
            "mean_patch_norm_error": mano_coord_camk_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": mano_coord_camk_fit["per_joint_px"],
        },
        "mano_with_trans_coord_change_free_tz_fit": {
            "cam_t": mano_coord_freez_fit["cam_t"],
            "loss_keypoints_2d": mano_coord_freez_fit["l1_sum"],
            "mean_patch_px_error": mano_coord_freez_fit["mean_px_error"],
            "mean_patch_norm_error": mano_coord_freez_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": mano_coord_freez_fit["per_joint_px"],
        },
        "mano_with_trans_coord_change_camk_free_tz_fit": {
            "cam_t": mano_coord_camk_freez_fit["cam_t"],
            "focal_length": mano_coord_camk_freez_fit["focal_length"],
            "camera_center": mano_coord_camk_freez_fit["camera_center"],
            "loss_keypoints_2d": mano_coord_camk_freez_fit["l1_sum"],
            "mean_patch_px_error": mano_coord_camk_freez_fit["mean_px_error"],
            "mean_patch_norm_error": mano_coord_camk_freez_fit["mean_norm_error"],
            "rootrel_3d_vs_packed_mm": root_relative_3d_error_mm(apply_coord_change(mano_joints3d), apply_coord_change(packed_kps3d)),
            "per_joint_px": mano_coord_camk_freez_fit["per_joint_px"],
        },
    }


def inspect_sample(
    dataset: Dict[str, np.ndarray],
    base_dir: str,
    idx: int,
    mano: MANO,
    out_dir: str,
    patch_dataset: Optional[ImageDataset] = None,
    image_size: int = 256,
    focal_length: float = 5000.0,
) -> Dict[str, object]:
    rel_path = dataset["imgname"][idx]
    if isinstance(rel_path, bytes):
        rel_path = rel_path.decode("utf-8")

    image_path = resolve_image_path(base_dir, rel_path)
    meta_path = resolve_meta_path(base_dir, rel_path)
    meta = pickle.load(open(meta_path, "rb"))

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    center = dataset["center"][idx].astype(np.float32)
    scale = dataset["scale"][idx].astype(np.float32)
    gt_kps2d = dataset["hand_keypoints_2d"][idx].astype(np.float32)
    packed_kps3d = dataset["hand_keypoints_3d"][idx].astype(np.float32)[..., :3]
    meta_kps3d = np.asarray(meta.get("handJoints3D"), dtype=np.float32)
    meta_kps3d_reordered = meta_kps3d[HO3D_OFFICIAL_TO_OPENPOSE]
    hand_trans = np.asarray(meta.get("handTrans"), dtype=np.float32)

    cam_mat = np.asarray(meta["camMat"], dtype=np.float32)
    mano_joints3d_local, mano_joints3d = reconstruct_mano_joints(meta, mano)

    packed_proj_with = project_points(cam_mat, packed_kps3d, apply_coord_change=True)
    packed_proj_without = project_points(cam_mat, packed_kps3d, apply_coord_change=False)
    meta_proj_with = project_points(cam_mat, meta_kps3d, apply_coord_change=True)
    meta_proj_without = project_points(cam_mat, meta_kps3d, apply_coord_change=False)
    meta_reordered_proj_with = project_points(cam_mat, meta_kps3d_reordered, apply_coord_change=True)
    meta_reordered_proj_without = project_points(cam_mat, meta_kps3d_reordered, apply_coord_change=False)
    mano_local_proj_with = project_points(cam_mat, mano_joints3d_local, apply_coord_change=True)
    mano_local_proj_without = project_points(cam_mat, mano_joints3d_local, apply_coord_change=False)
    mano_proj_with = project_points(cam_mat, mano_joints3d, apply_coord_change=True)
    mano_proj_without = project_points(cam_mat, mano_joints3d, apply_coord_change=False)

    metrics = {
        "packed3d_with_coord_change_px": reprojection_error(packed_proj_with, gt_kps2d),
        "packed3d_without_coord_change_px": reprojection_error(packed_proj_without, gt_kps2d),
        "meta_joints_with_coord_change_px": reprojection_error(meta_proj_with, gt_kps2d),
        "meta_joints_without_coord_change_px": reprojection_error(meta_proj_without, gt_kps2d),
        "meta_joints_reordered_with_coord_change_px": reprojection_error(meta_reordered_proj_with, gt_kps2d),
        "meta_joints_reordered_without_coord_change_px": reprojection_error(meta_reordered_proj_without, gt_kps2d),
        "mano_local_with_coord_change_px": reprojection_error(mano_local_proj_with, gt_kps2d),
        "mano_local_without_coord_change_px": reprojection_error(mano_local_proj_without, gt_kps2d),
        "mano_with_coord_change_px": reprojection_error(mano_proj_with, gt_kps2d),
        "mano_without_coord_change_px": reprojection_error(mano_proj_without, gt_kps2d),
        "packed3d_vs_meta3d_mm": float(np.linalg.norm(packed_kps3d - meta_kps3d, axis=1).mean()),
        "packed3d_vs_meta3d_reordered_mm": float(np.linalg.norm(packed_kps3d - meta_kps3d_reordered, axis=1).mean()),
        "mano_local_vs_meta3d_mm": float(np.linalg.norm(mano_joints3d_local - meta_kps3d, axis=1).mean()),
        "mano_with_trans_vs_meta3d_mm": float(np.linalg.norm(mano_joints3d - meta_kps3d, axis=1).mean()),
        "hand_trans_norm_mm": float(np.linalg.norm(hand_trans)),
    }

    panels = [
        add_caption(draw_bbox(img_bgr, center, scale), f"idx={idx} raw+bbox"),
        render_skeleton_panel(img_bgr, gt_kps2d, center, scale, "packed GT 2D"),
        render_skeleton_panel(
            img_bgr,
            packed_proj_with,
            center,
            scale,
            f"packed 3D -> proj (coord_change) {metrics['packed3d_with_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            packed_proj_without,
            center,
            scale,
            f"packed 3D -> proj (no change) {metrics['packed3d_without_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            meta_proj_with,
            center,
            scale,
            f"meta joints -> proj (coord_change) {metrics['meta_joints_with_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            meta_proj_without,
            center,
            scale,
            f"meta joints -> proj (no change) {metrics['meta_joints_without_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            meta_reordered_proj_with,
            center,
            scale,
            f"meta reordered -> proj (coord_change) {metrics['meta_joints_reordered_with_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            meta_reordered_proj_without,
            center,
            scale,
            f"meta reordered -> proj (no change) {metrics['meta_joints_reordered_without_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            mano_local_proj_with,
            center,
            scale,
            f"MANO local -> proj (coord_change) {metrics['mano_local_with_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            mano_local_proj_without,
            center,
            scale,
            f"MANO local -> proj (no change) {metrics['mano_local_without_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            mano_proj_with,
            center,
            scale,
            f"MANO+trans -> proj (coord_change) {metrics['mano_with_coord_change_px']:.2f}px",
        ),
        render_skeleton_panel(
            img_bgr,
            mano_proj_without,
            center,
            scale,
            f"MANO+trans -> proj (no change) {metrics['mano_without_coord_change_px']:.2f}px",
        ),
    ]

    sheet = build_contact_sheet(panels, ncols=3, pad=8)
    out_file = os.path.join(out_dir, f"{idx:06d}_{Path(rel_path).stem}.jpg")
    cv2.imwrite(out_file, sheet)

    per_joint = {
        "packed3d_with_coord_change_px": per_joint_errors(packed_proj_with, gt_kps2d),
        "meta_joints_with_coord_change_px": per_joint_errors(meta_proj_with, gt_kps2d),
        "meta_joints_reordered_with_coord_change_px": per_joint_errors(meta_reordered_proj_with, gt_kps2d),
        "mano_local_with_coord_change_px": per_joint_errors(mano_local_proj_with, gt_kps2d),
        "mano_with_coord_change_px": per_joint_errors(mano_proj_with, gt_kps2d),
    }

    patch_fit = None
    if patch_dataset is not None:
        patch_fit = inspect_patch_projection_fit(
            patch_dataset=patch_dataset,
            idx=idx,
            mano_local_joints3d=mano_joints3d_local,
            mano_joints3d=mano_joints3d,
            packed_kps3d=packed_kps3d,
            cam_mat_raw=cam_mat,
            out_dir=out_dir,
            image_size=image_size,
            focal_length=focal_length,
        )

    return {
        "index": int(idx),
        "imgname_rel": rel_path,
        "image_path": image_path,
        "meta_path": meta_path,
        "center": center.tolist(),
        "scale": scale.reshape(-1).tolist(),
        "hand_trans": hand_trans.reshape(-1).tolist(),
        "cam_mat": cam_mat.tolist(),
        "metrics": metrics,
        "per_joint_px": per_joint,
        "patch_fit": patch_fit,
        "output_file": out_file,
    }


def iter_indices(total: int, args: argparse.Namespace) -> Iterable[int]:
    explicit = parse_indices(args.indices)
    if explicit is not None:
        return explicit
    return choose_indices(total, args.num_samples, selection=args.selection, seed=args.seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Packed HO3D NPZ")
    parser.add_argument("--base_dir", type=str, required=True, help="HO3D root directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save comparison panels")
    parser.add_argument("--mano_model_path", type=str, default=str(root / "_DATA" / "data" / "mano"))
    parser.add_argument("--experiment", type=str, default="hamer_ho3d_pose_only_finetune", help="Hydra experiment used to build the deterministic patch-space dataset")
    parser.add_argument(
        "--fit_patch_pred_cam_t",
        action="store_true",
        help="Also fit patch-space camera families, including signed-tz diagnostics, on the deterministic sample",
    )
    parser.add_argument("--num_samples", type=int, default=8, help="Number of evenly spaced samples when --indices is not given")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated sample indices")
    parser.add_argument("--selection", type=str, default="spread", choices=["spread", "first", "random"], help="How to choose samples when --indices is not given")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --selection=random")
    args = parser.parse_args()

    dataset = np.load(args.dataset_file, allow_pickle=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    mano = build_mano(args.mano_model_path)
    mano.eval()
    patch_dataset = None
    image_size = 256
    focal_length = 5000.0
    if args.fit_patch_pred_cam_t:
        cfg = build_cfg(args.experiment)
        image_size = int(cfg.MODEL.IMAGE_SIZE)
        focal_length = float(cfg.EXTRA.FOCAL_LENGTH)
        patch_dataset = InspectImageDataset(
            cfg,
            dataset_file=args.dataset_file,
            img_dir=args.base_dir,
            train=False,
        )

    results = []
    selected = list(iter_indices(len(dataset["imgname"]), args))
    for idx in selected:
        result = inspect_sample(
            dataset,
            args.base_dir,
            idx,
            mano,
            args.out_dir,
            patch_dataset=patch_dataset,
            image_size=image_size,
            focal_length=focal_length,
        )
        results.append(result)
        print(f"[saved] idx={idx} -> {result['output_file']}")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value:.4f}px")
        if result["patch_fit"] is not None:
            exact = result["patch_fit"]["exact_patch_projection"]
            patch_cam_params = result["patch_fit"]["patch_cam_params"]
            local_fit = result["patch_fit"]["mano_local_fit"]
            local_coord_fit = result["patch_fit"]["mano_local_coord_change_fit"]
            local_coord_camk_fit = result["patch_fit"]["mano_local_coord_change_camk_fit"]
            local_coord_freez_fit = result["patch_fit"]["mano_local_coord_change_free_tz_fit"]
            local_coord_camk_freez_fit = result["patch_fit"]["mano_local_coord_change_camk_free_tz_fit"]
            mano_with_trans_coord_fit = result["patch_fit"]["mano_with_trans_coord_change_fit"]
            mano_with_trans_coord_camk_fit = result["patch_fit"]["mano_with_trans_coord_change_camk_fit"]
            mano_with_trans_coord_freez_fit = result["patch_fit"]["mano_with_trans_coord_change_free_tz_fit"]
            mano_with_trans_coord_camk_freez_fit = result["patch_fit"]["mano_with_trans_coord_change_camk_free_tz_fit"]
            print(f"  patch_exact_packed_px: {exact['packed_exact_crop_px']:.4f}px")
            print(f"  patch_exact_mano_px: {exact['mano_exact_crop_px']:.4f}px")
            print(f"  patch_patchK_packed_px: {exact['packed_patchK_px']:.4f}px")
            print(f"  patch_patchK_mano_px: {exact['mano_patchK_px']:.4f}px")
            print(
                "  patchK_params:"
                f" fx={patch_cam_params['fx']:.4f}"
                f" fy={patch_cam_params['fy']:.4f}"
                f" cx={patch_cam_params['cx']:.4f}"
                f" cy={patch_cam_params['cy']:.4f}"
                f" skew_xy={patch_cam_params['skew_xy']:.4f}"
                f" skew_yx={patch_cam_params['skew_yx']:.4f}"
            )
            print(f"  patch_fit_mano_local_l1: {local_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_local_px: {local_fit['mean_patch_px_error']:.4f}px")
            print(f"  patch_fit_mano_local_coord_l1: {local_coord_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_local_coord_px: {local_coord_fit['mean_patch_px_error']:.4f}px")
            print(f"  patch_fit_mano_local_coord_camk_l1: {local_coord_camk_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_local_coord_camk_px: {local_coord_camk_fit['mean_patch_px_error']:.4f}px")
            print(
                "  patch_fit_mano_local_coord_camk_params:"
                f" fx={local_coord_camk_fit['focal_length'][0]:.4f}"
                f" fy={local_coord_camk_fit['focal_length'][1]:.4f}"
                f" cx={local_coord_camk_fit['camera_center'][0]:.4f}"
                f" cy={local_coord_camk_fit['camera_center'][1]:.4f}"
            )
            print(f"  patch_fit_mano_local_coord_free_tz_l1: {local_coord_freez_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_local_coord_free_tz_px: {local_coord_freez_fit['mean_patch_px_error']:.4f}px")
            print(f"  patch_fit_mano_local_coord_camk_free_tz_l1: {local_coord_camk_freez_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_local_coord_camk_free_tz_px: {local_coord_camk_freez_fit['mean_patch_px_error']:.4f}px")
            print(
                "  patch_fit_mano_local_coord_camk_free_tz_params:"
                f" fx={local_coord_camk_freez_fit['focal_length'][0]:.4f}"
                f" fy={local_coord_camk_freez_fit['focal_length'][1]:.4f}"
                f" cx={local_coord_camk_freez_fit['camera_center'][0]:.4f}"
                f" cy={local_coord_camk_freez_fit['camera_center'][1]:.4f}"
            )
            print(f"  patch_fit_mano_with_trans_coord_l1: {mano_with_trans_coord_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_with_trans_coord_px: {mano_with_trans_coord_fit['mean_patch_px_error']:.4f}px")
            print(f"  patch_fit_mano_with_trans_coord_camk_l1: {mano_with_trans_coord_camk_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_with_trans_coord_camk_px: {mano_with_trans_coord_camk_fit['mean_patch_px_error']:.4f}px")
            print(
                "  patch_fit_mano_with_trans_coord_camk_params:"
                f" fx={mano_with_trans_coord_camk_fit['focal_length'][0]:.4f}"
                f" fy={mano_with_trans_coord_camk_fit['focal_length'][1]:.4f}"
                f" cx={mano_with_trans_coord_camk_fit['camera_center'][0]:.4f}"
                f" cy={mano_with_trans_coord_camk_fit['camera_center'][1]:.4f}"
            )
            print(f"  patch_fit_mano_with_trans_coord_free_tz_l1: {mano_with_trans_coord_freez_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_with_trans_coord_free_tz_px: {mano_with_trans_coord_freez_fit['mean_patch_px_error']:.4f}px")
            print(f"  patch_fit_mano_with_trans_coord_camk_free_tz_l1: {mano_with_trans_coord_camk_freez_fit['loss_keypoints_2d']:.4f}")
            print(f"  patch_fit_mano_with_trans_coord_camk_free_tz_px: {mano_with_trans_coord_camk_freez_fit['mean_patch_px_error']:.4f}px")
            print(
                "  patch_fit_mano_with_trans_coord_camk_free_tz_params:"
                f" fx={mano_with_trans_coord_camk_freez_fit['focal_length'][0]:.4f}"
                f" fy={mano_with_trans_coord_camk_freez_fit['focal_length'][1]:.4f}"
                f" cx={mano_with_trans_coord_camk_freez_fit['camera_center'][0]:.4f}"
                f" cy={mano_with_trans_coord_camk_freez_fit['camera_center'][1]:.4f}"
            )

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
