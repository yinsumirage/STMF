"""
Original HaMeR training entrypoint.

Typical usage for conservative HO3D-v3 webdataset finetuning:

conda run -n STMF python scripts/train.py \
  experiment=hamer_ho3d_finetune \
  data=ho3d_only \
  dataset_config_name=/data/hand_data/HO-3D_v3/datasets_tar_ho3d_v3.yaml \
  checkpoint=/home/user/code/STMF/_DATA/hamer_ckpts/checkpoints/hamer.ckpt \
  max_steps=5000

Useful overrides:
- `freeze_backbone=true|false`: freeze the ViT backbone during finetuning.
- `freeze_mano_transformer=true|false`: freeze the shared MANO decoder transformer.
- `freeze_camera_head=true|false`: freeze the camera readout head.
- `freeze_shape_head=true|false`: freeze the betas readout head.
- `batch_size=16`: overrides `TRAIN.BATCH_SIZE`.
- `lr=1e-5`: overrides `TRAIN.LR`.
- `checkpoint_step_frequency=500`: save checkpoints more often for short runs.
- `max_steps=2000`: safer than long epoch-based runs when validating a new setup.

Notes:
- This script still uses the original HaMeR WebDataset tar pipeline.
- `dataset_config_name` selects which tar metadata yaml under `hamer/configs/` to read.
- `checkpoint` loads base model weights for finetuning.
- When `LOSS_WEIGHTS.ADVERSARIAL=0`, mocap is no longer required.
- `data=ho3d_only` switches the training mix to a single HO3D dataset defined in
  `hamer/configs_hydra/data/ho3d_only.yaml`.
- Use `ckpt_path=/path/to/last.ckpt` when resuming a previous Lightning run.
- By default, `ckpt_path` is treated as a safe weight-initialization fallback.
- When either `checkpoint` or `ckpt_path` is provided, the standalone
  `MODEL.BACKBONE.PRETRAINED_WEIGHTS` file is disabled before model construction,
  because the checkpoint already contains backbone weights.
- `log_lr=false` keeps TensorBoard clean by disabling the learning-rate curve.
- `run_validation=false` / `limit_val_batches=0` matches the current STMF-style setup
  when you only want to watch training loss.
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
torch.set_float32_matmul_precision('high')
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning_fabric.utilities.cloud_io import _load as pl_load
#from pytorch_lightning.trainingtype import DDPPlugin

from yacs.config import CfgNode

if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([CfgNode, DictConfig, ListConfig, ContainerMetadata])

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


class UnsafeResumeCheckpointIO(TorchCheckpointIO):
    """Checkpoint IO that disables weights_only for trusted local resume checkpoints.

    PyTorch 2.6+ defaults torch.load(..., weights_only=True), which breaks exact
    Lightning resume when checkpoints contain OmegaConf/YACS objects. For our
    own locally-produced checkpoints we prefer exact state restoration here.
    """

    def load_checkpoint(self, path, map_location=None, weights_only: bool = False):
        del weights_only
        return pl_load(path, map_location=map_location, weights_only=False)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'), resolve=True)
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())


def apply_runtime_overrides(cfg: DictConfig) -> None:
    """Map top-level convenience args onto the original HaMeR config tree."""
    OmegaConf.set_struct(cfg, False)

    if cfg.get('batch_size', None) is not None:
        cfg.TRAIN.BATCH_SIZE = int(cfg.batch_size)
    if cfg.get('num_workers', None) is not None:
        cfg.GENERAL.NUM_WORKERS = int(cfg.num_workers)
    if cfg.get('prefetch_factor', None) is not None:
        cfg.GENERAL.PREFETCH_FACTOR = int(cfg.prefetch_factor)
    if cfg.get('lr', None) is not None:
        cfg.TRAIN.LR = float(cfg.lr)
    if cfg.get('weight_decay', None) is not None:
        cfg.TRAIN.WEIGHT_DECAY = float(cfg.weight_decay)
    if cfg.get('log_steps', None) is not None:
        cfg.GENERAL.LOG_STEPS = int(cfg.log_steps)
    if cfg.get('checkpoint_step_frequency', None) is not None:
        cfg.GENERAL.CHECKPOINT_STEPS = int(cfg.checkpoint_step_frequency)
    if cfg.get('checkpoint_save_top_k', None) is not None:
        cfg.GENERAL.CHECKPOINT_SAVE_TOP_K = int(cfg.checkpoint_save_top_k)

    OmegaConf.set_struct(cfg, True)


def freeze_modules(model: HAMER, cfg: DictConfig) -> None:
    """Freeze selected modules for safer finetuning."""
    freeze_backbone = bool(cfg.get('freeze_backbone', False))
    freeze_mano_head = bool(cfg.get('freeze_mano_head', False))
    freeze_mano_transformer = bool(cfg.get('freeze_mano_transformer', False))
    freeze_camera_head = bool(cfg.get('freeze_camera_head', False))
    freeze_shape_head = bool(cfg.get('freeze_shape_head', False))

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        log.info("Freezing backbone parameters for conservative finetuning.")

    if freeze_mano_head:
        for param in model.mano_head.parameters():
            param.requires_grad = False
        log.info("Freezing MANO head parameters.")
    else:
        if freeze_mano_transformer:
            for param in model.mano_head.transformer.parameters():
                param.requires_grad = False
            log.info("Freezing MANO transformer decoder.")
        if freeze_camera_head:
            for param in model.mano_head.deccam.parameters():
                param.requires_grad = False
            log.info("Freezing MANO camera head.")
        if freeze_shape_head:
            for param in model.mano_head.decshape.parameters():
                param.requires_grad = False
            log.info("Freezing MANO shape head.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Trainable params after freezing: {trainable:,}; frozen params: {frozen:,}")


def resolve_output_dir(cfg: DictConfig) -> str:
    """Choose a stable output dir, reusing the original run dir on exact resume."""
    resume_ckpt = cfg.get('ckpt_path', None)
    default_output_dir = str(cfg.paths.output_dir)

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.name.endswith('.ckpt') and ckpt_path.parent.name == 'checkpoints':
            resumed_output_dir = str(ckpt_path.parent.parent)
            if resumed_output_dir != default_output_dir:
                log.info(
                    "Exact resume requested: reusing original run output_dir from checkpoint: "
                    f"{resumed_output_dir}"
                )
            return resumed_output_dir

    return default_output_dir

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    apply_runtime_overrides(cfg)

    # Load dataset config
    dataset_cfg_name = cfg.get('dataset_config_name', 'datasets_tar.yaml')
    dataset_cfg = dataset_config(dataset_cfg_name)

    # When initializing from a model checkpoint or resuming a Lightning run, the
    # checkpoint already contains backbone weights. Disable the standalone
    # PRETRAINED_WEIGHTS path before constructing HAMER so resume works even when
    # vitpose_backbone.pth is not present locally.
    checkpoint = cfg.get('checkpoint', None)
    resume_ckpt = cfg.get('ckpt_path', None)
    init_source = checkpoint or resume_ckpt
    if init_source and cfg.get('MODEL', None) and cfg.MODEL.get('BACKBONE', None):
        pretrained_path = cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None)
        if pretrained_path:
            log.info(
                "Disabling BACKBONE.PRETRAINED_WEIGHTS because checkpoint-based "
                f"initialization is provided: {init_source}"
            )
            OmegaConf.set_struct(cfg, False)
            cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
            OmegaConf.set_struct(cfg, True)

    output_dir = resolve_output_dir(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.paths.output_dir = output_dir
    OmegaConf.set_struct(cfg, True)

    # Save configs
    save_configs(cfg, dataset_cfg, output_dir)

    # Setup training and validation datasets
    datamodule = HAMERDataModule(cfg, dataset_cfg)

    # Setup model
    model = HAMER(cfg)
    if checkpoint:
        log.info(f'Loading initialization checkpoint from {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict_to_load = state_dict['state_dict']
        else:
            state_dict_to_load = state_dict.get('model', state_dict)
        model.load_state_dict(state_dict_to_load, strict=False)

    freeze_modules(model, cfg)

    # Setup Tensorboard logger
    logger = TensorBoardLogger(
        os.path.join(output_dir, 'tensorboard'),
        name=cfg.get('exp_name', ''),
        version=cfg.get('run_version', None),
        default_hp_metric=False,
    )
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        filename='step_{step}',
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    callbacks = [checkpoint_callback]
    if cfg.get('log_lr', False):
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))

    log.info("Instantiating PyTorch Lightning trainer")
    run_validation = bool(cfg.get('run_validation', True))
    limit_val_batches = cfg.get('limit_val_batches', 1.0 if run_validation else 0.0)
    num_sanity_val_steps = cfg.get('num_sanity_val_steps', 2 if run_validation else 0)
    plugins = []
    if cfg.get('launcher', None) is not None:
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR2))
    if resume_ckpt:
        log.info("Enabling exact Lightning checkpoint resume with custom CheckpointIO (weights_only=False).")
        plugins.append(UnsafeResumeCheckpointIO())

    trainer: Trainer = Trainer(
        accelerator=cfg.get('accelerator', 'gpu'),
        devices=cfg.get('devices', 1),
        strategy=cfg.get('strategy', 'auto'),
        max_epochs=cfg.get('epochs', 100),
        max_steps=cfg.get('max_steps', -1) if cfg.get('max_steps', None) is not None else -1,
        limit_train_batches=cfg.get('limit_train_batches', 1.0),
        limit_val_batches=limit_val_batches,
        limit_test_batches=cfg.get('limit_test_batches', 1.0),
        num_sanity_val_steps=num_sanity_val_steps,
        check_val_every_n_epoch=1 if run_validation and limit_val_batches else None,
        callbacks=callbacks,
        logger=loggers,
        precision=cfg.get('precision', 32),
        log_every_n_steps=cfg.get('log_every_n_steps', 1),
        plugins=plugins or None,
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
