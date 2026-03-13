import torch
from hamer.configs import dataset_config
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.stmf_datamodule import STMFDataModule
from yacs.config import CfgNode as CN

def check_batch():
    # Load config
    dataset_cfg = dataset_config(name='datasets_stmf.yaml')
    _, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    
    # Matching train_stmf.py logic
    model_cfg.defrost()
    model_cfg.TRAIN = CN()
    model_cfg.TRAIN.BATCH_SIZE = 4
    model_cfg.TRAIN.LR = 1e-4
    model_cfg.TRAIN.WEIGHT_DECAY = 1e-4
    model_cfg.GENERAL = CN()
    model_cfg.GENERAL.NUM_WORKERS = 0
    model_cfg.GENERAL.LOG_STEPS = 10
    model_cfg.freeze()

    print("Initializing DataModule...")
    datamodule = STMFDataModule(model_cfg, dataset_cfg)
    datamodule.setup(stage='fit')
    
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    
    print("\n--- Batch Structure Check ---")
    print(f"Batch type: {type(batch)}")
    if isinstance(batch, dict):
        print(f"Batch keys: {batch.keys()}")
        if 'img' in batch:
            print(f"img shape: {batch['img'].shape} (type: {type(batch['img'])})")
        if 'keypoints_2d' in batch:
            kps_2d = batch['keypoints_2d']
            print(f"keypoints_2d shape: {kps_2d.shape}")
            print(f"keypoints_2d confidence mean: {torch.mean(kps_2d[..., -1])}")
            print(f"keypoints_2d nonzero count: {torch.count_nonzero(kps_2d)}")
        
        # Test the suspicious logic in stmf.py
        suspicious_batch = batch['img'] if 'img' in batch else batch
        print(f"\nSuspicious logic 'batch = joint_batch['img']' results in type: {type(suspicious_batch)}")
        try:
            test_access = suspicious_batch['keypoints_2d']
            print("Access successful (This means suspicious_batch is a dict)")
        except Exception as e:
            print(f"Access FAILED: {e} (This means suspicious_batch is likely a tensor)")

if __name__ == "__main__":
    check_batch()
