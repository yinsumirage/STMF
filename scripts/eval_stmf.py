"""
python scripts/eval_stmf.py \
    --checkpoint /home/mirage/STMF/logs/train/runs/你的日期/checkpoints/last.ckpt \
    --dataset FREIHAND-VAL,HO3D-VAL \
    --batch_size 64 \
    --window_size 5 \
    --results_folder results_stmf_final
"""

import argparse
import os
import json
import torch
import traceback
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from filelock import FileLock
from tqdm import tqdm
from yacs.config import CfgNode

# Add yacs.config.CfgNode to torch safe globals for PyTorch 2.6+
from yacs.config import CfgNode as CN
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([CN])

from hamer.configs import dataset_eval_config, get_config
from hamer.datasets import create_dataset
from hamer.utils import Evaluator, recursive_to
from hamer.models import HAMER, load_stmf, DEFAULT_CHECKPOINT
from hamer.models.stmf import STMF_HAMER


def extract_sample(batch, sample_idx: int):
    if isinstance(batch, dict):
        return {k: extract_sample(v, sample_idx) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch[sample_idx:sample_idx + 1]
    if isinstance(batch, list):
        return batch[sample_idx]
    return batch


def inject_stateful_history(sample: Dict, pose_cache: Dict[int, torch.Tensor], beta_cache: Dict[str, torch.Tensor]) -> Dict:
    if 'pose_seq' not in sample or 'temporal_indices' not in sample:
        return sample

    pose_seq = sample['pose_seq'].clone()
    temporal_indices = sample['temporal_indices'][0].tolist()
    zero_pose = torch.zeros_like(pose_seq[0, 0])
    history = []
    for hist_idx in temporal_indices[:-1]:
        cached_pose = pose_cache.get(int(hist_idx))
        history.append(cached_pose.to(device=pose_seq.device, dtype=pose_seq.dtype) if cached_pose is not None else zero_pose)

    if history:
        pose_seq[0, :-1, :] = torch.stack(history, dim=0)
    sample['pose_seq'] = pose_seq
    if pose_seq.shape[1] > 1:
        sample['prev_pose'] = pose_seq[:, -2, :]

    sequence_key = sample.get('sequence_key')
    if sequence_key is not None:
        if isinstance(sequence_key, list):
            sequence_key = sequence_key[0]
        cached_beta = beta_cache.get(str(sequence_key))
        if cached_beta is not None:
            sample['prev_betas'] = cached_beta.to(device=pose_seq.device, dtype=pose_seq.dtype).unsqueeze(0)
            sample['has_prev_betas'] = torch.ones(1, device=pose_seq.device, dtype=pose_seq.dtype)
    return sample


def main():
    parser = argparse.ArgumentParser(description='Evaluate STMF model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to STMF model checkpoint')
    parser.add_argument('--results_folder', type=str, default='results_stmf', help='Path to results folder.')
    parser.add_argument('--dataset', type=str, default='FREIHAND-VAL,HO3D-VAL', help='Dataset to evaluate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers used for data loading')
    parser.add_argument('--window_size', type=int, default=5, help='Temporal window size (seq_len)')
    parser.add_argument('--log_freq', type=int, default=10, help='How often to log results')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='Shuffle the dataset during evaluation')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--stateless', dest='stateless', action='store_true', default=False, help='Disable autoregressive pose history and use dataset pose_seq as-is')

    args = parser.parse_args()
    os.makedirs(args.results_folder, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, model_cfg = load_stmf(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load stmf eval config
    stmf_eval_cfg = CN(new_allowed=True)
    stmf_eval_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../hamer/configs/datasets_stmf.yaml')
    stmf_eval_cfg.merge_from_file(stmf_eval_config_file)

    print('Evaluating on datasets: {}'.format(args.dataset), flush=True)
    for dataset_name in args.dataset.split(','):
        if dataset_name not in stmf_eval_cfg:
            print(f"Dataset {dataset_name} not found in stmf_eval_cfg. Skipping.")
            continue
            
        dataset_cfg = stmf_eval_cfg[dataset_name]
        run_eval(model, model_cfg, dataset_cfg, dataset_name, device, args)

def run_eval(model, model_cfg, dataset_cfg, dataset_name, device, args):
    # Setup metrics and predictions
    # For FreiHAND and HO3D Eval, we primarily care about vertices and 3d keypoints
    metrics = None
    preds = ['vertices', 'keypoints_3d']
    rescale_factor = -1 # Matches original eval.py for these datasets

    # Create dataset - Ensure it uses TemporalImageDataset for STMF
    print(f"Creating dataset {dataset_name}...")
    dataset = create_dataset(model_cfg, dataset_cfg, train=False, rescale_factor=rescale_factor, window_size=args.window_size)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # Setup evaluator object
    evaluator = Evaluator(
        dataset_length=len(dataset),
        dataset=dataset_name,
        keypoint_list=dataset_cfg.get('KEYPOINT_LIST', [0]), 
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND, 
        metrics=metrics,
        preds=preds,
    )

    # Inference loop
    try:
        pose_cache = {}
        beta_cache = {}
        for i, batch in enumerate(tqdm(dataloader, desc=dataset_name)):
            batch_size = batch['img'].shape[0]
            for sample_idx in range(batch_size):
                sample = extract_sample(batch, sample_idx)
                sample = recursive_to(sample, device)
                if not args.stateless:
                    sample = inject_stateful_history(sample, pose_cache, beta_cache)

                with torch.no_grad():
                    out = model(sample)

                evaluator(out, sample)
                if 'pred_pose' in out and 'idx' in sample:
                    pose_cache[int(sample['idx'].item())] = out['pred_pose'][0].detach().cpu()
                if 'pred_mano_params' in out and 'betas' in out['pred_mano_params'] and 'sequence_key' in sample:
                    sequence_key = sample['sequence_key']
                    if isinstance(sequence_key, list):
                        sequence_key = sequence_key[0]
                    beta_cache[str(sequence_key)] = out['pred_mano_params']['betas'][0].detach().cpu()

            if i % args.log_freq == args.log_freq - 1:
                evaluator.log()
        evaluator.log()
        error = None
    except (Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        error = repr(e)
        i = 0

    # Save results
    if preds is not None:
        results_json = os.path.join(args.results_folder, '%s.json' % dataset_name.lower())
        preds_dict = evaluator.get_preds_dict()
        save_preds_result(results_json, preds_dict)

def save_preds_result(pred_out_path: str, preds_dict: Dict) -> None:
    xyz_pred_list = preds_dict['keypoints_3d']
    verts_pred_list = preds_dict['vertices']
    
    # Convert to lists for JSON serialization
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    with open(pred_out_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print(f'Dumped {len(xyz_pred_list)} joints and {len(verts_pred_list)} verts predictions to {pred_out_path}')

if __name__ == '__main__':
    main()
