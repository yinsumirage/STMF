"""
PyTorch Lightning datamodule for STMF.

Used by:
- `scripts/train_stmf.py`

This datamodule bypasses the original HaMeR WebDataset tar pipeline and instead
loads sequence-aware NPZ datasets through `TemporalImageDataset`.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from ..datasets.temporal_dataset import TemporalImageDataset


class STMFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for STMF training.
    Bypasses the original WebDataset pipeline in favor of sequence-aware NPZ datasets.
    """

    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.repo_root = Path(__file__).resolve().parents[2]

    def _resolve_path(self, path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return str(path)

        candidates = []
        if not path.is_absolute():
            candidates.append(self.repo_root / path)
        else:
            raw = str(path)
            if '/_DATA/' in raw:
                candidates.append(self.repo_root / '_DATA' / raw.split('/_DATA/', 1)[1])
            if '/STMF/' in raw:
                candidates.append(self.repo_root / raw.split('/STMF/', 1)[1])

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(path)

    @staticmethod
    def _stage_matches(stage: Optional[object], *tags: str) -> bool:
        if stage is None:
            return True
        stage_name = str(getattr(stage, 'value', stage)).lower()
        return any(tag in stage_name for tag in tags)

    def _build_train_dataset(self) -> None:
        if self.train_dataset is not None:
            return

        datasets_to_concat = []
        for ds_name, ds_info in self.dataset_cfg.items():
            if 'TRAIN' not in ds_name:
                continue
            if type(ds_info) != CfgNode or 'DATASET_FILE' not in ds_info:
                continue

            dataset_file = self._resolve_path(ds_info.DATASET_FILE)
            img_dir = self._resolve_path(ds_info.IMG_DIR)
            print(f"Loading {ds_name} STMF Temporal Dataset from: {dataset_file}")
            if not os.path.exists(dataset_file):
                print(f"Skipping {ds_name}: missing dataset file {dataset_file}")
                continue
            if not os.path.exists(img_dir):
                print(f"Skipping {ds_name}: missing image dir {img_dir}")
                continue

            ds = TemporalImageDataset(
                cfg=self.cfg,
                dataset_file=dataset_file,
                img_dir=img_dir,
                train=True,
                window_size=self.cfg.TRAIN.get('WINDOW_SIZE', 5),
            )
            datasets_to_concat.append(ds)

        if datasets_to_concat:
            self.train_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)

    def _build_val_dataset(self) -> None:
        if self.val_dataset is not None:
            return

        datasets_to_concat = []
        for ds_name, ds_info in self.dataset_cfg.items():
            if 'VAL' not in ds_name:
                continue
            if type(ds_info) != CfgNode or 'DATASET_FILE' not in ds_info:
                continue

            dataset_file = self._resolve_path(ds_info.DATASET_FILE)
            img_dir = self._resolve_path(ds_info.IMG_DIR)
            print(f"Loading {ds_name} STMF Evaluation Dataset from: {dataset_file}")
            if not os.path.exists(dataset_file):
                print(f"Skipping {ds_name}: missing dataset file {dataset_file}")
                continue
            if not os.path.exists(img_dir):
                print(f"Skipping {ds_name}: missing image dir {img_dir}")
                continue

            ds = TemporalImageDataset(
                cfg=self.cfg,
                dataset_file=dataset_file,
                img_dir=img_dir,
                train=False,
                window_size=self.cfg.TRAIN.get('WINDOW_SIZE', 5),
            )
            datasets_to_concat.append(ds)

        if datasets_to_concat:
            self.val_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build datasets lazily. Lightning may pass enum-like stage objects rather than plain strings,
        so stage handling needs to be permissive.
        """
        if self._stage_matches(stage, 'fit', 'train'):
            self._build_train_dataset()

        if self._stage_matches(stage, 'fit', 'validate', 'val', 'test'):
            self._build_val_dataset()

    def train_dataloader(self) -> Dict:
        """
        Returns dictionary containing our temporal image dataloader.
        """
        if self.train_dataset is None:
            self.setup('fit')
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is not initialized. Check dataset paths in "
                "hamer/configs/datasets_stmf.yaml."
            )

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            drop_last=True,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            shuffle=True,
        )
        return {'img': train_dataloader}

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            self.setup('validate')
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            drop_last=False,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        return self.val_dataloader()
