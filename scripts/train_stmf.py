from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from yacs.config import CfgNode, CfgNode as CN
from hamer.configs import dataset_config
from hamer.datasets.stmf_datamodule import STMFDataModule

# We load the HaMeR checkpoint config and inject our STMF configs
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.models.stmf import STMF_HAMER
from hamer.utils.pylogger import get_pylogger

import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(rootdir, 'model_config.yaml'), 'w') as f:
        f.write(model_cfg.dump())
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataloder config
    dataset_cfg = dataset_config(name='datasets_stmf.yaml')

    # Base checkpoint to load ViT weights
    checkpoint = cfg.get('checkpoint') or DEFAULT_CHECKPOINT
    
    log.info(f"Loading Base HaMeR configurations from: {checkpoint}")
    _, model_cfg = load_hamer(checkpoint)

    # Inject STMF-specific configurations
    model_cfg.defrost()
    model_cfg.TRAIN = CN()
    model_cfg.TRAIN.BATCH_SIZE = cfg.get('batch_size', 4) # Reduced batch size due to sliding window holding T frames
    model_cfg.TRAIN.LR = 1e-4
    model_cfg.TRAIN.WEIGHT_DECAY = 1e-4
    model_cfg.GENERAL = CN()
    model_cfg.GENERAL.NUM_WORKERS = getattr(cfg, 'num_workers', 0)
    model_cfg.GENERAL.PREFETCH_FACTOR = 2
    
    if not hasattr(model_cfg, 'LOSS_WEIGHTS'):
        model_cfg.LOSS_WEIGHTS = CN()
    model_cfg.LOSS_WEIGHTS.SMOOTHNESS = 10.0
    model_cfg.LOSS_WEIGHTS.FK_SENSOR = 50.0
    model_cfg.freeze()

    # Output directory mapping
    output_dir = getattr(cfg.paths, 'output_dir', './logs/stmf_training')

    save_configs(model_cfg, dataset_cfg, output_dir)

    log.info(f"Initializing STMF Data Module...")
    datamodule = STMFDataModule(model_cfg, dataset_cfg)

    log.info(f"Initializing STMF Model over HaMeR Backbone...")
    model = STMF_HAMER(model_cfg, init_renderer=False)
    
    # We transfer visual backbone weights from original haMeR model
    state_dict = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict_to_load = state_dict['state_dict']
    else:
        state_dict_to_load = state_dict['model']
        
    log.info("Loading pre-trained frozen backbone weights...")
    model.load_state_dict(state_dict_to_load, strict=False) # strict=False because STMF adds new modules

    logger = TensorBoardLogger(os.path.join(output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'), 
        every_n_train_steps=1000, 
        save_last=True,
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    # Allow adding new keys to the config for easy override
    OmegaConf.set_struct(cfg, False)

    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=cfg.get('epochs', 100),
        limit_train_batches=cfg.get('limit_train_batches', 1.0),
        limit_val_batches=cfg.get('limit_val_batches', 0), # Default to 0 for STMF
        callbacks=callbacks,
        logger=[logger],
        precision=cfg.get('precision', 32),
        log_every_n_steps=cfg.get('log_every_n_steps', 1),
    )

    log.info("Starting Temporal Model training!")
    trainer.fit(model, datamodule=datamodule)
    log.info("Training process completed.")


@hydra.main(version_base="1.2", config_path=str(root/"hamer/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
