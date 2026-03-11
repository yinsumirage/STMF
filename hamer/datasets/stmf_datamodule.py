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
        if self.train_dataset is None:
            datasets_to_concat = []
            
            # Iterate through all configured datasets in the YAML
            for ds_name, ds_info in self.dataset_cfg.items():
                if type(ds_info) != CfgNode or 'DATASET_FILE' not in ds_info:
                    continue # Skip non-dict config items

                # Extract configs
                dataset_file = ds_info.DATASET_FILE
                img_dir = ds_info.IMG_DIR
                
                print(f"Loading {ds_name} STMF Temporal Dataset from: {dataset_file}")
                ds = TemporalImageDataset(
                    cfg=self.cfg,
                    dataset_file=dataset_file,
                    img_dir=img_dir,
                    train=True,
                    window_size=3
                )
                datasets_to_concat.append(ds)

            if len(datasets_to_concat) > 0:
                self.train_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
            else:
                print("WARNING: No datasets loaded into STMF Data Module!")

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
            val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=self.cfg.TRAIN.BATCH_SIZE, 
                drop_last=False, 
                num_workers=self.cfg.GENERAL.NUM_WORKERS
            )
            return val_dataloader
        return None 
