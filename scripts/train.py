"""
Original HaMeR training entrypoint.

Typical usage for HO3D-v3 webdataset finetuning:

conda run -n STMF python scripts/train.py \
  experiment=hamer_vit_transformer \
  data=ho3d_only \
  dataset_config_name=datasets_tar_ho3d_v3.yaml \
  checkpoint=/path/to/hamer.ckpt \
  LOSS_WEIGHTS.ADVERSARIAL=0 \
  trainer.devices='[0,1]'

Notes:
- This script still uses the original HaMeR WebDataset tar pipeline.
- `dataset_config_name` selects which tar metadata yaml under `hamer/configs/` to read.
- `checkpoint` loads base model weights for finetuning.
- When `LOSS_WEIGHTS.ADVERSARIAL=0`, mocap is no longer required.
- `data=ho3d_only` switches the training mix to a single HO3D dataset defined in
  `hamer/configs_hydra/data/ho3d_only.yaml`.
- Use `ckpt_path=/path/to/last.ckpt` only when resuming a previous Lightning run.
"""

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
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
#from pytorch_lightning.trainingtype import DDPPlugin

from yacs.config import CfgNode
from hamer.configs import dataset_config
from hamer.datasets import HAMERDataModule
from hamer.models.hamer import HAMER
from hamer.utils.pylogger import get_pylogger
from hamer.utils.misc import task_wrapper, log_hyperparameters

# HACK reset the signal handling so the lightning is free to set it
# Based on https://github.com/facebookincubator/submitit/issues/1709#issuecomment-1246758283
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # Load dataset config
    dataset_cfg_name = cfg.get('dataset_config_name', 'datasets_tar.yaml')
    dataset_cfg = dataset_config(dataset_cfg_name)

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = HAMERDataModule(cfg, dataset_cfg)

    # Setup model
    model = HAMER(cfg)
    checkpoint = cfg.get('checkpoint', None)
    if checkpoint:
        log.info(f'Loading initialization checkpoint from {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict_to_load = state_dict['state_dict']
        else:
            state_dict_to_load = state_dict.get('model', state_dict)
        model.load_state_dict(state_dict_to_load, strict=False)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    rich_callback = pl.callbacks.RichProgressBar()
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
        # rich_callback
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
        #plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else DDPPlugin(find_unused_parameters=False)), # Submitit uses SIGUSR2
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None), # Submitit uses SIGUSR2
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path', None))
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path=str(root/"hamer/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
