import torch
from yacs.config import CfgNode as CN
from hamer.datasets.temporal_dataset import TemporalImageDataset

def test_dataset():
    print("Testing TemporalImageDataset...")
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.IMAGE_SIZE = 256
    cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    cfg.MANO = CN()
    cfg.MANO.NUM_HAND_JOINTS = 15
    cfg.DATASETS = CN()
    cfg.DATASETS.CONFIG = CN()
    cfg.DATASETS.CONFIG.SCALE_FACTOR = 0.3
    cfg.DATASETS.CONFIG.ROT_FACTOR = 45
    cfg.DATASETS.CONFIG.TRANS_FACTOR = 0.02
    cfg.DATASETS.CONFIG.COLOR_SCALE = 0.2
    cfg.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
    cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.1
    cfg.DATASETS.CONFIG.FLIP_AUG_RATE = 0.0
    cfg.DATASETS.CONFIG.DO_FLIP = False
    dataset_file = '/home/mirage/STMF/_DATA/HO-3D_v3/ho3d_evaluation.npz'
    img_dir = '/home/mirage/STMF/_DATA/HO-3D_v3/'

    try:
        ds = TemporalImageDataset(cfg, dataset_file, img_dir, train=False, seq_len=3, stride=1)
        print(f"Dataset length (sequences): {len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print(f"img shape: {item['img'].shape}")
            print(f"sensor shape: {item['sensor'].shape}")
            print(f"prev_pose shape: {item['prev_pose'].shape}")
            print(f"temporal_indices: {item['temporal_indices']}")
            print(f"sensor_valid_mask: {item['sensor_valid_mask']}")
            print(f"pose_valid_mask: {item['pose_valid_mask']}")
            print(f"sequence_key: {item['sequence_key']}")
            assert item['sensor_valid_mask'][-1] == True
            assert item['pose_valid_mask'].shape[0] == item['pose_seq'].shape[0] - 1
            print("Successfully loaded window!")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == '__main__':
    test_dataset()
