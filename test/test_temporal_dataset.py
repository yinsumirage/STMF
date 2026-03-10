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
    
    dataset_file = '/home/mirage/STMF/_DATA/hamer_evaluation_data/freihand_val.npz'
    img_dir = '/home/mirage/STMF/_DATA/FreiHAND_pub_v2/evaluation/rgb'

    try:
        ds = TemporalImageDataset(cfg, dataset_file, img_dir, train=False, seq_len=3, stride=1)
        print(f"Dataset length (sequences): {len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print(f"img shape: {item['img'].shape}")
            print(f"sensor shape: {item['sensor'].shape}")
            print(f"prev_pose shape: {item['prev_pose'].shape}")
            print("Successfully loaded window!")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == '__main__':
    test_dataset()
