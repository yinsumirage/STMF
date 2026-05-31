"""
Plain HaMeR HO3D diagnostic-only overfit entrypoint.

This file is not a main training entrypoint for the new sensor-guided temporal
MANO refinement line. It is kept to preserve the HO3D protocol analysis and
negative-result evidence around GT coord recipes, patchK projection, and camera
residual diagnostics.

This script is intentionally isolated from `scripts/train.py`:
- it trains on a deterministic NPZ subset instead of the WebDataset tar pipeline
- it is meant for 1-sample / 8-sample overfit checks
- it logs a fixed visualization batch at step 0 and during training
- it logs both raw loss values and weighted loss contributions on the same batch

Typical usage:

1. Overfit a single HO3D train sample with the current pose-only recipe:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 1 \
  --max_steps 500

2. Overfit eight evenly spaced samples and inspect the fixed visualizations:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 8 \
  --selection spread \
  --loss_recipe 2d_3d \
  --max_steps 1000

3. Compare whether `3D` or `MANO` supervision is the one pulling the hand into a flipped solution:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 1 \
  --loss_recipe 2d_mano \
  --max_steps 3000

4. Test whether the flipped solution is specifically tied to the current GT 3D frame:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 1 \
  --loss_recipe 2d_3d \
  --gt_coord_recipe flip_gt_keypoints_3d \
  --max_steps 3000

5. Test whether the whole visible GT 3D supervision package should be interpreted in the flipped frame:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 1 \
  --loss_recipe full \
  --gt_coord_recipe flip_gt_keypoints_3d_mano \
  --max_steps 3000

6. Test whether the missing rigid alignment mainly comes from `global_orient` rather than local finger pose:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --num_samples 1 \
  --loss_recipe full \
  --gt_coord_recipe flip_gt_keypoints_3d_global_orient \
  --unfreeze_camera_head \
  --max_steps 3000

7. Diagnostic-only: test whether a tiny residual patch-camera head (`delta_f, delta_cx, delta_cy`)
   can pull back hard cases that the fixed patch camera cannot:
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --indices 11903 \
  --num_samples 1 \
  --loss_recipe full \
  --gt_coord_recipe flip_gt_keypoints_3d_global_orient \
  --unfreeze_camera_head \
  --camera_residual_mode focal_center \
  --max_steps 3000

8. Diagnostic-only: if shared focal is still too restrictive, try a 4-parameter patch-camera residual
   (`delta_fx, delta_fy, delta_cx, delta_cy`):
conda run -n STMF python scripts/train_overfit.py \
  --experiment hamer_ho3d_pose_only_finetune \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3/ \
  --checkpoint /home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  --indices 11903 \
  --num_samples 1 \
  --loss_recipe full \
  --gt_coord_recipe flip_gt_keypoints_3d_global_orient \
  --unfreeze_camera_head \
  --camera_residual_mode focal_xy_center \
  --max_steps 3000

Notes:
- By default this uses deterministic crops (`train=False`) so the same sample stays comparable.
- `--loss_recipe` provides a few common diagnosis presets such as
  `2d_only`, `3d_only`, `mano_only`, `2d_3d`, and `2d_mano`.
- `--ho3d_coord_change_before_projection` is a temporary diagnostic flag:
  it applies the HO3D/OpenGL `diag(1, -1, -1)` conversion only before the 2D projection
  inside this overfit script, without changing the 3D or MANO supervision.
- `--gt_coord_recipe` is a separate temporary diagnostic flag:
  it modifies the GT tensors before loss computation, without changing model predictions.
- `fixed/predictions` is logged to TensorBoard and also saved to disk as PNG snapshots.
- The selected sample indices and image paths are saved to `selected_samples.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import hydra
import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid, save_image

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.datasets.image_dataset import ImageDataset
from hamer.datasets.utils import gen_trans_from_patch_cv
from hamer.models.hamer import HAMER
from hamer.utils.geometry import aa_to_rotmat, perspective_projection, rotmat_to_aa
from hamer.utils import recursive_to
from hamer.utils.misc import close_loggers
from hamer.utils.pylogger import get_pylogger
from hamer.utils.render_openpose import render_openpose
from scripts.train import apply_runtime_overrides, freeze_modules

log = get_pylogger(__name__)


LOSS_NAME_TO_WEIGHT_KEY = {
    "loss_keypoints_2d": "KEYPOINTS_2D",
    "loss_keypoints_3d": "KEYPOINTS_3D",
    "loss_global_orient": "GLOBAL_ORIENT",
    "loss_hand_pose": "HAND_POSE",
    "loss_betas": "BETAS",
    "loss_gen": "ADVERSARIAL",
}


LOSS_RECIPE_TO_ZERO_KEYS = {
    "full": [],
    "2d_only": ["KEYPOINTS_3D", "GLOBAL_ORIENT", "HAND_POSE", "BETAS"],
    "3d_only": ["KEYPOINTS_2D", "GLOBAL_ORIENT", "HAND_POSE", "BETAS"],
    "mano_only": ["KEYPOINTS_2D", "KEYPOINTS_3D"],
    "2d_3d": ["GLOBAL_ORIENT", "HAND_POSE", "BETAS"],
    "2d_mano": ["KEYPOINTS_3D"],
}


GT_COORD_RECIPE_CHOICES = [
    "none",
    "flip_gt_keypoints_3d",
    "flip_gt_keypoints_3d_global_orient",
    "flip_gt_keypoints_3d_mano",
]

PROJECTION_MODE_CHOICES = [
    "default",
    "ho3d_patchK",
]


def resolve_ho3d_meta_path(img_dir: str, image_file_rel: str) -> Optional[str]:
    rel = image_file_rel.replace("\\", "/")
    stem = os.path.splitext(rel)[0]
    if "/rgb/" not in stem:
        return None
    meta_rel = stem.replace("/rgb/", "/meta/") + ".pkl"
    meta_path = os.path.join(img_dir, meta_rel)
    if os.path.exists(meta_path):
        return meta_path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny NPZ subset for loss debugging")
    parser.add_argument("--experiment", type=str, default="hamer_ho3d_pose_only_finetune", help="Hydra experiment config name under hamer/configs_hydra/experiment/")
    parser.add_argument("--dataset_file", type=str, required=True, help="Packed NPZ used for the overfit subset")
    parser.add_argument("--img_dir", type=str, required=True, help="Image root used by imgname in the NPZ")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base checkpoint used to initialize HaMeR")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to overfit when --indices is not given")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated dataset indices to overfit")
    parser.add_argument("--selection", type=str, default="spread", choices=["first", "random", "spread"], help="How to choose indices when --indices is not given")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for subset selection")
    parser.add_argument("--max_steps", type=int, default=500, help="Number of optimizer steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size. Defaults to the subset size")
    parser.add_argument("--lr", type=float, default=None, help="Optional learning-rate override")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers for the tiny subset")
    parser.add_argument("--viz_steps", type=int, default=50, help="How often to log the fixed diagnostic batch")
    parser.add_argument("--vis_max_samples", type=int, default=8, help="How many fixed samples to render in the diagnostic grid")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the tiny subset each epoch")
    parser.add_argument("--use_train_aug", action="store_true", help="Enable training augmentation instead of deterministic crops")
    parser.add_argument("--loss_recipe", type=str, default="full", choices=sorted(LOSS_RECIPE_TO_ZERO_KEYS.keys()), help="Convenience preset for common overfit loss ablations")
    parser.add_argument("--unfreeze_camera_head", action="store_true", help="Convenience flag to set freeze_camera_head=False for overfit diagnostics")
    parser.add_argument("--camera_residual_mode", type=str, default="none", choices=["none", "focal_center", "focal_xy_center"], help="Optional tiny residual patch-camera head used only in this overfit diagnostic")
    parser.add_argument("--camera_residual_hidden_dim", type=int, default=256, help="Hidden dim for the residual patch-camera MLP")
    parser.add_argument("--projection_mode", type=str, default="default", choices=PROJECTION_MODE_CHOICES, help="How to project predicted 3D joints to patch 2D inside this overfit diagnostic")
    parser.add_argument("--ho3d_coord_change_before_projection", action="store_true", help="Apply HO3D/OpenGL coord change before 2D projection, only inside this overfit tool")
    parser.add_argument("--gt_coord_recipe", type=str, default="none", choices=GT_COORD_RECIPE_CHOICES, help="Apply a temporary coord-change to GT supervision tensors before loss computation")
    parser.add_argument("--accelerator", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices passed to Lightning")
    parser.add_argument("--precision", type=int, default=32, help="Lightning precision")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Lightning logging frequency")
    parser.add_argument("--grad_clip_val", type=float, default=None, help="Optional gradient clipping value")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override, repeatable, e.g. --override freeze_backbone=False")
    parser.add_argument("--output_root", type=str, default="logs/overfit/runs", help="Root directory for this diagnostic run")
    return parser.parse_args()


def parse_indices(indices: Optional[str]) -> Optional[List[int]]:
    if not indices:
        return None
    return [int(x.strip()) for x in indices.split(",") if x.strip()]


def choose_indices(total: int, args: argparse.Namespace) -> List[int]:
    explicit = parse_indices(args.indices)
    if explicit is not None:
        return explicit

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if total <= 0:
        raise ValueError("Dataset is empty")

    num = min(args.num_samples, total)
    if args.selection == "first":
        return list(range(num))
    if args.selection == "spread":
        if num == total:
            return list(range(total))
        return sorted(set(torch.linspace(0, total - 1, steps=num).round().to(dtype=torch.int64).tolist()))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return sorted(torch.randperm(total, generator=generator)[:num].tolist())


def build_cfg(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"experiment={args.experiment}",
        "strategy=auto",
        "run_validation=false",
        "num_sanity_val_steps=0",
        "log_lr=false",
    ]
    overrides.extend(args.override)

    with hydra.initialize_config_dir(version_base="1.2", config_dir=str(root / "hamer" / "configs_hydra")):
        cfg = hydra.compose(config_name="train.yaml", overrides=overrides)

    OmegaConf.set_struct(cfg, False)
    cfg.extras.enforce_tags = False
    cfg.extras.print_config = False
    cfg.task_name = "overfit"
    cfg.checkpoint = args.checkpoint
    cfg.max_steps = args.max_steps
    cfg.limit_val_batches = 0
    cfg.limit_test_batches = 0
    cfg.run_validation = False
    cfg.num_sanity_val_steps = 0
    cfg.accelerator = args.accelerator
    cfg.devices = args.devices
    cfg.precision = args.precision
    cfg.log_every_n_steps = args.log_every_n_steps
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    if args.grad_clip_val is not None:
        cfg.TRAIN.GRAD_CLIP_VAL = float(args.grad_clip_val)
    apply_runtime_overrides(cfg)
    OmegaConf.set_struct(cfg, False)
    for weight_key in LOSS_RECIPE_TO_ZERO_KEYS[args.loss_recipe]:
        cfg.LOSS_WEIGHTS[weight_key] = 0.0
    if args.unfreeze_camera_head:
        cfg.freeze_camera_head = False
    if cfg.get("MODEL", None) and cfg.MODEL.get("BACKBONE", None):
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
    cfg.loss_recipe = args.loss_recipe
    cfg.camera_residual_mode = str(args.camera_residual_mode)
    cfg.camera_residual_hidden_dim = int(args.camera_residual_hidden_dim)
    cfg.projection_mode = str(args.projection_mode)
    cfg.ho3d_coord_change_before_projection = bool(args.ho3d_coord_change_before_projection)
    cfg.gt_coord_recipe = str(args.gt_coord_recipe)
    exp_suffix = f"{args.loss_recipe}_{args.num_samples}samples"
    if args.unfreeze_camera_head:
        exp_suffix += "_camfree"
    if args.camera_residual_mode != "none":
        exp_suffix += f"_camres-{args.camera_residual_mode}"
    if args.projection_mode != "default":
        exp_suffix += f"_proj-{args.projection_mode}"
    if args.ho3d_coord_change_before_projection:
        exp_suffix += "_coordchange"
    if args.gt_coord_recipe != "none":
        exp_suffix += f"_{args.gt_coord_recipe}"
    cfg.exp_name = f"{cfg.exp_name}_overfit_{exp_suffix}"
    OmegaConf.set_struct(cfg, True)
    return cfg


def resolve_output_dir(args: argparse.Namespace, cfg: DictConfig) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_root) / cfg.exp_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


class NPZOverfitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        dataset_file: str,
        img_dir: str,
        indices: Sequence[int],
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        use_train_aug: bool,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_file = dataset_file
        self.img_dir = img_dir
        self.indices = list(indices)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.use_train_aug = use_train_aug
        self.dataset: Optional[ImageDataset] = None
        self.subset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.dataset = OverfitImageDataset(
                self.cfg,
                dataset_file=self.dataset_file,
                img_dir=self.img_dir,
                train=self.use_train_aug,
            )
            self.subset = torch.utils.data.Subset(self.dataset, self.indices)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.subset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=False,
        )


class OverfitImageDataset(ImageDataset):
    def _apply_ho3d_official_subset_filter(self, dataset_file: str) -> None:
        # Overfit diagnostics often use HO3D train NPZ with deterministic crops.
        # In that case we do not want the evaluation whitelist to zero out the train set.
        return

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        meta_path = resolve_ho3d_meta_path(self.img_dir, item["imgname_rel"])
        if meta_path is not None:
            with open(meta_path, "rb") as f:
                anno = pickle.load(f, encoding="latin1")
            if "camMat" in anno and anno["camMat"] is not None:
                item["cam_mat_raw"] = np.asarray(anno["camMat"], dtype=np.float32)
        return item


class OverfitHAMER(HAMER):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.camera_residual_mode = str(cfg.get("camera_residual_mode", "none"))
        if self.camera_residual_mode == "none":
            self.camera_residual_pool = None
            self.camera_residual_head = None
        else:
            in_dim = int(getattr(self.backbone, "embed_dim", 1280))
            hidden_dim = int(cfg.get("camera_residual_hidden_dim", 256))
            out_dim = 3 if self.camera_residual_mode == "focal_center" else 4
            self.camera_residual_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.camera_residual_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
            nn.init.zeros_(self.camera_residual_head[-1].weight)
            nn.init.zeros_(self.camera_residual_head[-1].bias)

    @staticmethod
    def _coord_change_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            device=device,
            dtype=dtype,
        )

    @classmethod
    def _apply_coord_change_to_points(cls, points: torch.Tensor) -> torch.Tensor:
        coord_change = cls._coord_change_matrix(points.device, points.dtype)
        transformed_xyz = torch.einsum("ij,...j->...i", coord_change, points[..., :3])
        transformed = points.clone()
        transformed[..., :3] = transformed_xyz
        return transformed

    @classmethod
    def _apply_coord_change_to_global_orient_axis_angle(cls, aa: torch.Tensor) -> torch.Tensor:
        coord_change = cls._coord_change_matrix(aa.device, aa.dtype)
        rotmat = aa_to_rotmat(aa.reshape(-1, 3))
        transformed_rotmat = coord_change.unsqueeze(0) @ rotmat
        return rotmat_to_aa(transformed_rotmat).reshape_as(aa)

    @classmethod
    def _apply_coord_change_to_local_axis_angle(cls, aa: torch.Tensor) -> torch.Tensor:
        coord_change = cls._coord_change_matrix(aa.device, aa.dtype)
        rotmat = aa_to_rotmat(aa.reshape(-1, 3))
        transformed_rotmat = coord_change.unsqueeze(0) @ rotmat @ coord_change.unsqueeze(0)
        return rotmat_to_aa(transformed_rotmat).reshape_as(aa)

    @staticmethod
    def _affine_to_homogeneous(trans: np.ndarray) -> np.ndarray:
        out = np.eye(3, dtype=np.float32)
        out[:2, :] = trans
        return out

    def _build_patch_cam_mats(self, batch: Dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if "cam_mat_raw" not in batch:
            raise KeyError("projection_mode=ho3d_patchK requires `cam_mat_raw` in the batch")
        batch_size = batch["img"].shape[0]
        image_size = float(self.cfg.MODEL.IMAGE_SIZE)
        mats = []
        for i in range(batch_size):
            center = batch["box_center"][i].detach().cpu().numpy().astype(np.float32)
            box_size = float(batch["box_size"][i].detach().cpu().item())
            cam_mat_raw = batch["cam_mat_raw"][i].detach().cpu().numpy().astype(np.float32)
            trans = gen_trans_from_patch_cv(
                float(center[0]),
                float(center[1]),
                box_size,
                box_size,
                image_size,
                image_size,
                1.0,
                0.0,
            )
            patch_cam = self._affine_to_homogeneous(trans) @ cam_mat_raw
            mats.append(torch.from_numpy(patch_cam))
        return torch.stack(mats, dim=0).to(device=device, dtype=dtype)

    def _project_with_patch_cam(
        self,
        points_3d: torch.Tensor,
        translation: torch.Tensor,
        patch_cam_mat: torch.Tensor,
    ) -> torch.Tensor:
        points_cam = points_3d + translation.unsqueeze(1)
        coord_change = self._coord_change_matrix(points_cam.device, points_cam.dtype)
        points_cam = torch.einsum("ij,bkj->bki", coord_change, points_cam)
        proj = torch.einsum("bij,bkj->bki", patch_cam_mat, points_cam)
        xy_px = proj[:, :, :2] / proj[:, :, 2:3]
        return xy_px / float(self.cfg.MODEL.IMAGE_SIZE) - 0.5

    @staticmethod
    def _patch_cam_params_from_mat(patch_cam_mat: torch.Tensor, image_size: int) -> Dict[str, torch.Tensor]:
        focal_length = torch.stack([patch_cam_mat[:, 0, 0], patch_cam_mat[:, 1, 1]], dim=1)
        camera_center_px = torch.stack([patch_cam_mat[:, 0, 2], patch_cam_mat[:, 1, 2]], dim=1)
        camera_center = camera_center_px / float(image_size) - 0.5
        return {
            "focal_length": focal_length,
            "camera_center": camera_center,
        }

    def _render_keypoint_diagnostics(self, batch: Dict, output: Dict, num_images: int) -> torch.Tensor:
        images = batch["img"][:num_images].detach().cpu().numpy()
        pred_keypoints_2d = output["pred_keypoints_2d"][:num_images].detach().cpu().numpy()
        gt_keypoints_2d = batch["keypoints_2d"][:num_images].detach().cpu().numpy()
        mean = (255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN, dtype=np.float32)).reshape(3, 1, 1)
        std = (255.0 * np.array(self.cfg.MODEL.IMAGE_STD, dtype=np.float32)).reshape(3, 1, 1)

        panels = []
        for image, pred_kps, gt_kps in zip(images, pred_keypoints_2d, gt_keypoints_2d):
            img = image.astype(np.float32) * std + mean
            img = np.clip(img, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            pred_px = np.concatenate(
                [self.cfg.MODEL.IMAGE_SIZE * (pred_kps + 0.5), np.ones((pred_kps.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            gt_px = gt_kps.copy()
            gt_px[:, :2] = self.cfg.MODEL.IMAGE_SIZE * (gt_px[:, :2] + 0.5)

            raw_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(render_openpose(img_bgr.copy(), pred_px.copy()).astype(np.uint8), cv2.COLOR_BGR2RGB)
            gt_rgb = cv2.cvtColor(render_openpose(img_bgr.copy(), gt_px.copy()).astype(np.uint8), cv2.COLOR_BGR2RGB)

            panels.extend(
                [
                    torch.from_numpy(raw_rgb).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(pred_rgb).permute(2, 0, 1).float() / 255.0,
                    torch.from_numpy(gt_rgb).permute(2, 0, 1).float() / 255.0,
                ]
            )
        return make_grid(panels, nrow=3, padding=2)

    def _transform_gt_batch_for_loss(self, batch: Dict) -> Dict:
        recipe = str(self.cfg.get("gt_coord_recipe", "none"))
        if recipe == "none":
            return batch

        transformed_batch = dict(batch)

        if recipe in {"flip_gt_keypoints_3d", "flip_gt_keypoints_3d_global_orient", "flip_gt_keypoints_3d_mano"}:
            transformed_batch["keypoints_3d"] = self._apply_coord_change_to_points(batch["keypoints_3d"])

        if recipe in {"flip_gt_keypoints_3d_global_orient", "flip_gt_keypoints_3d_mano"}:
            transformed_mano_params = dict(batch["mano_params"])
            transformed_mano_params["global_orient"] = self._apply_coord_change_to_global_orient_axis_angle(batch["mano_params"]["global_orient"])
            if recipe == "flip_gt_keypoints_3d_mano":
                transformed_mano_params["hand_pose"] = self._apply_coord_change_to_local_axis_angle(batch["mano_params"]["hand_pose"].reshape(-1, 3)).reshape_as(batch["mano_params"]["hand_pose"])
            transformed_batch["mano_params"] = transformed_mano_params

        return transformed_batch

    def get_parameters(self):
        all_params = list(super().get_parameters())
        if self.camera_residual_head is not None:
            all_params += list(self.camera_residual_head.parameters())
        return all_params

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        x = batch["img"]
        batch_size = x.shape[0]
        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        output = {}
        output["pred_cam"] = pred_cam
        output["pred_mano_params"] = {k: v.clone() for k, v in pred_mano_params.items()}

        device = pred_mano_params["hand_pose"].device
        dtype = pred_mano_params["hand_pose"].dtype
        projection_mode = str(self.cfg.get("projection_mode", "default"))
        patch_cam_mat = None
        if projection_mode == "ho3d_patchK":
            patch_cam_mat = self._build_patch_cam_mats(batch, device=device, dtype=dtype)
            patch_cam_params = self._patch_cam_params_from_mat(patch_cam_mat, self.cfg.MODEL.IMAGE_SIZE)
            focal_length = patch_cam_params["focal_length"]
            camera_center = patch_cam_params["camera_center"]
        else:
            focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
            camera_center = torch.zeros(batch_size, 2, device=device, dtype=dtype)

        focal_for_depth = focal_length.mean(dim=1)
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_for_depth / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        if self.camera_residual_head is not None:
            residual_raw = self.camera_residual_head(self.camera_residual_pool(conditioning_feats))
            if self.camera_residual_mode == "focal_center":
                delta_log_f = torch.tanh(residual_raw[:, :1])
                delta_center = 0.25 * torch.tanh(residual_raw[:, 1:3])
                focal_length = focal_length * torch.exp(delta_log_f)
            elif self.camera_residual_mode == "focal_xy_center":
                delta_log_f = torch.tanh(residual_raw[:, :2])
                delta_center = 0.25 * torch.tanh(residual_raw[:, 2:4])
                focal_length = focal_length * torch.exp(delta_log_f)
            else:
                raise ValueError(f"Unknown camera_residual_mode: {self.camera_residual_mode}")
            camera_center = camera_center + delta_center
            if projection_mode == "ho3d_patchK":
                patch_cam_mat = patch_cam_mat.clone()
                patch_cam_mat[:, 0, 0] = focal_length[:, 0]
                patch_cam_mat[:, 1, 1] = focal_length[:, 1]
                patch_cam_mat[:, 0, 2] = self.cfg.MODEL.IMAGE_SIZE * (camera_center[:, 0] + 0.5)
                patch_cam_mat[:, 1, 2] = self.cfg.MODEL.IMAGE_SIZE * (camera_center[:, 1] + 0.5)
            output["camera_residual_raw"] = residual_raw

        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length
        output["camera_center"] = camera_center

        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)

        if projection_mode == "ho3d_patchK":
            pred_keypoints_2d = self._project_with_patch_cam(
                pred_keypoints_3d,
                pred_cam_t.reshape(-1, 3),
                patch_cam_mat,
            )
            output["patch_cam_mat"] = patch_cam_mat
        elif bool(self.cfg.get("ho3d_coord_change_before_projection", False)):
            coord_change = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                device=pred_keypoints_3d.device,
                dtype=pred_keypoints_3d.dtype,
            )
            pred_keypoints_3d_for_proj = torch.einsum("ij,bkj->bki", coord_change, pred_keypoints_3d)
            pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d_for_proj,
                translation=pred_cam_t.reshape(-1, 3),
                focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
                camera_center=camera_center.reshape(-1, 2),
            )
        else:
            pred_keypoints_3d_for_proj = pred_keypoints_3d
            pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d_for_proj,
                translation=pred_cam_t.reshape(-1, 3),
                focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
                camera_center=camera_center.reshape(-1, 2),
            )
        output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(pred_keypoints_2d.shape[0], -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        transformed_batch = self._transform_gt_batch_for_loss(batch)
        return super().compute_loss(transformed_batch, output, train=train)

    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> torch.Tensor:
        if str(self.cfg.get("projection_mode", "default")) == "ho3d_patchK":
            mode = "train" if train else "val"
            losses = output["losses"]
            if write_to_summary_writer:
                summary_writer = self.logger.experiment
                for loss_name, val in losses.items():
                    summary_writer.add_scalar(mode + "/" + loss_name, val.detach().item(), step_count)
            batch_size = batch["img"].shape[0]
            num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)
            predictions = self._render_keypoint_diagnostics(batch, output, num_images)
            if write_to_summary_writer:
                self.logger.experiment.add_image(f"{mode}/predictions", predictions, step_count)
            return predictions
        return super().tensorboard_logging(batch, output, step_count, train=train, write_to_summary_writer=write_to_summary_writer)

    def training_step(self, joint_batch, batch_idx):
        output = super().training_step(joint_batch, batch_idx)

        if isinstance(joint_batch, dict) and "img" in joint_batch and isinstance(joint_batch["img"], dict):
            batch = joint_batch["img"]
        else:
            batch = joint_batch
        batch_size = batch["img"].shape[0]

        for loss_name, loss_value in output["losses"].items():
            weight_key = LOSS_NAME_TO_WEIGHT_KEY.get(loss_name)
            if weight_key is None:
                continue
            weight = float(self.cfg.LOSS_WEIGHTS.get(weight_key, 0.0))
            self.log(
                f"train_weighted/{loss_name}",
                loss_value * weight,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=batch_size,
            )

        if "camera_center" in output:
            self.log("train_camera/center_x", output["camera_center"][:, 0].mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
            self.log("train_camera/center_y", output["camera_center"][:, 1].mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
        if "focal_length" in output:
            self.log("train_camera/focal_x", output["focal_length"][:, 0].mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
            self.log("train_camera/focal_y", output["focal_length"][:, 1].mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
            self.log("train_camera/focal_mean", output["focal_length"].mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)

        return output


class FixedSubsetDiagnostics(Callback):
    def __init__(
        self,
        dataset: ImageDataset,
        indices: Sequence[int],
        viz_steps: int,
        output_dir: str,
        vis_max_samples: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = list(indices)
        self.viz_steps = max(1, int(viz_steps))
        self.output_dir = Path(output_dir)
        self.vis_max_samples = max(1, int(vis_max_samples))
        self.fixed_batch = None
        self.predictions_dir = self.output_dir / "fixed_predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, trainer: Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self.fixed_batch is not None:
            return
        subset = torch.utils.data.Subset(self.dataset, self.indices[: self.vis_max_samples])
        loader = torch.utils.data.DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0, drop_last=False)
        self.fixed_batch = next(iter(loader))

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self._run_diagnostics(trainer, pl_module)

    def on_train_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.viz_steps == 0:
            self._run_diagnostics(trainer, pl_module)

    @pl.utilities.rank_zero.rank_zero_only
    def _run_diagnostics(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if self.fixed_batch is None or trainer.logger is None:
            return

        writer = trainer.logger.experiment
        step = trainer.global_step
        model_was_training = pl_module.training
        batch = recursive_to(deepcopy(self.fixed_batch), pl_module.device)

        pl_module.eval()
        with torch.no_grad():
            output = pl_module.forward_step(batch, train=False)
            total_loss = pl_module.compute_loss(batch, output, train=False)
            predictions = pl_module.tensorboard_logging(
                batch,
                output,
                step_count=step,
                train=False,
                write_to_summary_writer=False,
            )

        writer.add_scalar("fixed/loss_total", float(total_loss.detach().item()), step)
        manual_total = 0.0
        for loss_name, loss_value in output["losses"].items():
            raw = float(loss_value.detach().item())
            writer.add_scalar(f"fixed_raw/{loss_name}", raw, step)
            weight_key = LOSS_NAME_TO_WEIGHT_KEY.get(loss_name)
            if weight_key is None:
                continue
            weight = float(pl_module.cfg.LOSS_WEIGHTS.get(weight_key, 0.0))
            weighted = raw * weight
            manual_total += weighted
            writer.add_scalar(f"fixed_weighted/{loss_name}", weighted, step)

        writer.add_scalar("fixed_weighted/manual_total", manual_total, step)
        if "camera_center" in output:
            writer.add_scalar("fixed_camera/center_x", float(output["camera_center"][:, 0].mean().detach().item()), step)
            writer.add_scalar("fixed_camera/center_y", float(output["camera_center"][:, 1].mean().detach().item()), step)
        if "focal_length" in output:
            writer.add_scalar("fixed_camera/focal_x", float(output["focal_length"][:, 0].mean().detach().item()), step)
            writer.add_scalar("fixed_camera/focal_y", float(output["focal_length"][:, 1].mean().detach().item()), step)
            writer.add_scalar("fixed_camera/focal_mean", float(output["focal_length"].mean().detach().item()), step)
        writer.add_image("fixed/predictions", predictions, step)
        save_image(predictions, self.predictions_dir / f"step_{step:06d}.png")

        if model_was_training:
            pl_module.train()
            if hasattr(pl_module, "_keep_frozen_modules_in_eval"):
                pl_module._keep_frozen_modules_in_eval()


def save_selected_samples(dataset: ImageDataset, indices: Sequence[int], output_dir: str) -> None:
    items = []
    for idx in indices:
        raw = dataset.imgname[idx]
        rel = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        items.append(
            {
                "index": int(idx),
                "imgname_rel": rel,
                "personid": int(dataset.personid[idx]),
            }
        )
    with open(Path(output_dir) / "selected_samples.json", "w") as f:
        json.dump(items, f, indent=2)


def load_checkpoint_into_model(model: HAMER, checkpoint_path: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        state_dict_to_load = state_dict["state_dict"]
    else:
        state_dict_to_load = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict_to_load, strict=False)


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    cfg = build_cfg(args)
    output_dir = resolve_output_dir(args, cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.paths.output_dir = output_dir
    OmegaConf.set_struct(cfg, True)

    all_dataset = OverfitImageDataset(cfg, dataset_file=args.dataset_file, img_dir=args.img_dir, train=args.use_train_aug)
    selected_indices = choose_indices(len(all_dataset), args)
    if not selected_indices:
        raise RuntimeError("No samples were selected for the overfit subset")

    batch_size = args.batch_size or len(selected_indices)
    batch_size = max(1, min(batch_size, len(selected_indices)))

    datamodule = NPZOverfitDataModule(
        cfg=cfg,
        dataset_file=args.dataset_file,
        img_dir=args.img_dir,
        indices=selected_indices,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        use_train_aug=args.use_train_aug,
    )
    datamodule.setup()
    save_selected_samples(datamodule.dataset, selected_indices, output_dir)

    model = OverfitHAMER(cfg)
    load_checkpoint_into_model(model, args.checkpoint)
    freeze_modules(model, cfg)

    with open(Path(output_dir) / "model_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=False))
    with open(Path(output_dir) / "cli_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger = TensorBoardLogger(os.path.join(output_dir, "tensorboard"), name="", version="")
    callbacks: List[Callback] = [
        FixedSubsetDiagnostics(
            dataset=datamodule.dataset,
            indices=selected_indices,
            viz_steps=args.viz_steps,
            output_dir=output_dir,
            vis_max_samples=min(args.vis_max_samples, len(selected_indices)),
        )
    ]

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="auto",
        max_epochs=10_000,
        max_steps=args.max_steps,
        limit_val_batches=0,
        limit_test_batches=0,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=[logger],
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=False,
    )

    logger.log_hyperparams(
        {
            "experiment": args.experiment,
            "checkpoint": args.checkpoint,
            "dataset_file": args.dataset_file,
            "img_dir": args.img_dir,
            "selected_indices": selected_indices,
            "num_samples": len(selected_indices),
            "max_steps": args.max_steps,
            "batch_size": batch_size,
            "lr": float(cfg.TRAIN.LR),
            "loss_recipe": args.loss_recipe,
            "use_train_aug": bool(args.use_train_aug),
            "unfreeze_camera_head": bool(args.unfreeze_camera_head),
            "camera_residual_mode": str(args.camera_residual_mode),
            "projection_mode": str(args.projection_mode),
            "ho3d_coord_change_before_projection": bool(args.ho3d_coord_change_before_projection),
            "gt_coord_recipe": str(args.gt_coord_recipe),
        }
    )

    log.info("Selected subset indices: %s", selected_indices)
    log.info("Output dir: %s", output_dir)
    trainer.fit(model, datamodule=datamodule)
    close_loggers()


if __name__ == "__main__":
    main()
