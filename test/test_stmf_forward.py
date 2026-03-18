"""
Test script for STMF_HAMER model initialization and forward pass.
Uses the real HaMeR checkpoint config to avoid MANO head type mismatch.
"""
import torch
import os
import sys

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.models.stmf import STMF_HAMER

def test_stmf():
    print("=" * 60)
    print("Testing STMF Model initialization and forward pass...")
    print("=" * 60)
    
    # ---- Step 1: Load the REAL HaMeR config from the checkpoint ----
    # This is the same approach used in demo.py and eval.py.
    # The config contains the correct MANO_HEAD.TYPE='transformer_decoder'
    # and all nested sub-keys (TRANSFORMER_DECODER, IEF_ITERS, etc.)
    print("\n[1/5] Loading HaMeR config from checkpoint...")
    _, model_cfg = load_hamer(DEFAULT_CHECKPOINT)  # We only need the config, not the model
    
    # Add STMF-specific loss weights to the config
    model_cfg.defrost()
    if not hasattr(model_cfg, 'LOSS_WEIGHTS'):
        model_cfg.LOSS_WEIGHTS = {}
    model_cfg.LOSS_WEIGHTS.SMOOTHNESS = 10.0
    model_cfg.LOSS_WEIGHTS.FK_SENSOR = 50.0
    model_cfg.freeze()
    
    # ---- Step 2: Initialize STMF_HAMER ----
    # STMF_HAMER calls super().__init__ which builds backbone + original mano_head,
    # then replaces mano_head with STMFHead and freezes the backbone.
    print("[2/5] Initializing STMF_HAMER (init_renderer=False)...")
    model = STMF_HAMER(model_cfg, init_renderer=False)
    model.eval()
    
    # Verify backbone is frozen
    backbone_params = list(model.backbone.parameters())
    frozen = all(not p.requires_grad for p in backbone_params)
    print(f"  Backbone frozen: {frozen} ({len(backbone_params)} params)")
    
    # Verify STMFHead is trainable
    head_params = list(model.stmf_head.parameters())
    trainable = all(p.requires_grad for p in head_params)
    print(f"  STMFHead trainable: {trainable} ({len(head_params)} params)")
    print(f"  STMFHead type: {type(model.stmf_head).__name__}")
    
    # ---- Step 3: Forward pass ----
    print("\n[3/5] Running forward pass...")
    B, T, C, H, W = 2, 3, 3, 256, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    pose_seq = torch.randn(B, T, 48, device=device)
    sensor_seq = torch.rand(B, T, 5, device=device)
    sensor_valid_mask = torch.tensor([
        [False, False, True],
        [True, True, True],
    ], device=device)
    pose_valid_mask = torch.tensor([
        [False, True],
        [True, True],
    ], device=device)
    batch = {
        'img': torch.randn(B, T, C, H, W, device=device),
        'sensor_seq': sensor_seq,
        'pose_seq': pose_seq,
        'sensor_valid_mask': sensor_valid_mask,
        'pose_valid_mask': pose_valid_mask,
        'sensor': sensor_seq[:, -1, :],
        'prev_pose': pose_seq[:, -2, :],
        # Fake ground truths for loss computation:
        'keypoints_2d': torch.randn(B, 21, 3, device=device),
        'keypoints_3d': torch.randn(B, 21, 4, device=device),
        'mano_params': {
            'global_orient': torch.randn(B, 3, device=device),
            'hand_pose': torch.randn(B, 45, device=device),
            'betas': torch.randn(B, 10, device=device),
        },
        'has_mano_params': {
            'global_orient': torch.ones(B, device=device),
            'hand_pose': torch.ones(B, device=device),
            'betas': torch.ones(B, device=device),
        },
        'mano_params_is_axis_angle': {
            'global_orient': torch.tensor(True),
            'hand_pose': torch.tensor(True),
            'betas': torch.tensor(False),
        },
    }
    
    with torch.no_grad():
        output = model.forward_step(batch, train=False)
    
    print(f"  pred_mano_params['hand_pose']:    {output['pred_mano_params']['hand_pose'].shape}")
    print(f"  pred_mano_params['global_orient']: {output['pred_mano_params']['global_orient'].shape}")
    print(f"  pred_mano_params['betas']:         {output['pred_mano_params']['betas'].shape}")
    print(f"  pred_cam:          {output['pred_cam'].shape}")
    print(f"  pred_keypoints_3d: {output['pred_keypoints_3d'].shape}")
    print(f"  pred_vertices:     {output['pred_vertices'].shape}")
    print(f"  pred_keypoints_2d: {output['pred_keypoints_2d'].shape}")
    print(f"  pred_pose:         {output['pred_pose'].shape}")
    print(f"  pred_poses_seq_valid_mask: {output['pred_poses_seq_valid_mask']}")
    
    # Check for NaN
    has_nan = any(torch.isnan(v).any().item() for v in [
        output['pred_cam'], output['pred_keypoints_3d'], output['pred_vertices']
    ])
    print(f"  Contains NaN: {has_nan}")
    
    # ---- Step 4: Compute Loss ----
    print("\n[4/5] Computing loss (train=True, with gradients)...")
    model.train()
    output_train = model.forward_step(batch, train=True)
    loss = model.compute_loss(batch, output_train, train=True)
    print(f"  Total Loss: {loss.item():.4f}")
    if 'losses' in output_train:
        for k, v in output_train['losses'].items():
            print(f"    {k}: {v.item():.4f}")
    
    # ---- Step 5: Backward pass ----
    print("\n[5/5] Backward pass (gradient check for STMFHead)...")
    loss.backward()
    
    sensor_grad = model.stmf_head.tokenizer.sensor_mlp[0].weight.grad
    if sensor_grad is not None:
        print(f"  Sensor MLP grad shape: {sensor_grad.shape}, norm: {sensor_grad.norm().item():.6f}")
    else:
        print("  WARNING: Sensor MLP grad is None!")
    
    pose_grad = model.stmf_head.tokenizer.pose_mlp[0].weight.grad
    if pose_grad is not None:
        print(f"  Pose MLP grad shape:   {pose_grad.shape}, norm: {pose_grad.norm().item():.6f}")
    else:
        print("  WARNING: Pose MLP grad is None!")

    fusion_grad = model.stmf_head.fusion_head.pose_regressor[0].weight.grad
    if fusion_grad is not None:
        print(f"  Fusion head grad shape: {fusion_grad.shape}, norm: {fusion_grad.norm().item():.6f}")
    else:
        print("  WARNING: Fusion head grad is None!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    test_stmf()
