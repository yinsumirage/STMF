from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from ..datasets.temporal_dataset import TemporalImageDataset

class STMFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for STMF training. 
    Bypasses WebDataset architecture in favor of Sequential npz loaded datasets.
    """
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup sequence-aware temporal datasets dynamically parsed from dataset_cfg.
        """
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                datasets_to_concat = []
                for ds_name, ds_info in self.dataset_cfg.items():
                    if 'TRAIN' not in ds_name:
                        continue
                    if type(ds_info) != CfgNode or 'DATASET_FILE' not in ds_info:
                        continue
                    
                    print(f"Loading {ds_name} STMF Temporal Dataset from: {ds_info.DATASET_FILE}")
                    ds = TemporalImageDataset(
                        cfg=self.cfg,
                        dataset_file=ds_info.DATASET_FILE,
                        img_dir=ds_info.IMG_DIR,
                        train=True,
                        window_size=self.cfg.TRAIN.get('WINDOW_SIZE', 5)
                    )
                    datasets_to_concat.append(ds)
                
                if datasets_to_concat:
                    self.train_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
        
        if stage in ['fit', 'test'] or stage is None:
            if self.val_dataset is None:
                datasets_to_concat = []
                for ds_name, ds_info in self.dataset_cfg.items():
                    if 'VAL' not in ds_name:
                        continue
                    if type(ds_info) != CfgNode or 'DATASET_FILE' not in ds_info:
                        continue
                    
                    print(f"Loading {ds_name} STMF Evaluation Dataset from: {ds_info.DATASET_FILE}")
                    ds = TemporalImageDataset(
                        cfg=self.cfg,
                        dataset_file=ds_info.DATASET_FILE,
                        img_dir=ds_info.IMG_DIR,
                        train=False,
                        window_size=self.cfg.TRAIN.get('WINDOW_SIZE', 5)
                    )
                    datasets_to_concat.append(ds)
                
                if datasets_to_concat:
                    self.val_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)

    def train_dataloader(self) -> Dict:
        """
        Returns dictionary containing our temporal image dataloader.
        """
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            drop_last=True, 
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            shuffle=True # Shuffle *sequences*, frames within sequences remain ordered
        )
        
        # We don't need adversarial mocap prior for the specialized STMF fusion since we have physical constraints
        return {'img': train_dataloader}

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is not None:
            return torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=self.cfg.TRAIN.BATCH_SIZE, 
                drop_last=False, 
                num_workers=self.cfg.GENERAL.NUM_WORKERS,
                shuffle=False
            )
        return None

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        return self.val_dataloader()
