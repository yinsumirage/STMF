import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, Optional

from .hamer import HAMER

class TemporalPositionalEncoding(nn.Module):
    """
    Inject temporal positional information into visual Patch Tokens within a sliding window.
    Uses standard absolute sinusoidal and cosine functions as proposed by Vaswani et al.
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super(TemporalPositionalEncoding, self).__init__()
        
        # Initialize a (max_len, d_model) zero matrix
        pe = torch.zeros(max_len, d_model)
        # Create sequence of positions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute attenuation term in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Even indices get sine, odd indices get cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Reshape to (1, max_len, 1, d_model) for broadcasting over Batch and Num_Patches
        pe = pe.unsqueeze(0).unsqueeze(2)
        # Register as buffer so it moves to device automatically but isn't updated by optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual feature tensor of shape (Batch, Temporal_Window, Num_Patches, d_model)
        Returns:
            Tensor with temporal positional encoding added.
        """
        seq_len = x.size(1)
        # Add sliced PE to visual tokens (utilizing PyTorch auto-broadcasting)
        x = x + self.pe[:, :seq_len, :, :]
        return x


class SequencePositionalEncoding(nn.Module):
    """
    Absolute sinusoidal positional encoding for token sequences shaped (B, T, D).
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class ModalityTokenization(nn.Module):
    """
    Multi-modal Tokenization Module:
    1. Maps low-dim continuous physical sensor data and historical kinematic parameters to Transformer feature space.
    2. Processes temporal window visual features, injecting time context and flattening into a memory sequence.
    """
    def __init__(self, sensor_dim: int = 5, pose_dim: int = 48, d_model: int = 1024):
        super(ModalityTokenization, self).__init__()
        
        self.d_model = d_model
        
        # MLP network mapping 5-dim fingertip wire-pull physical data
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, d_model)
        )
        
        # MLP network mapping the previous frame's 48-dim MANO kinematic pose
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, d_model)
        )
        
        self.sensor_temporal_pe = SequencePositionalEncoding(d_model=d_model)
        self.pose_temporal_pe = SequencePositionalEncoding(d_model=d_model)
        self.sensor_modality_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pose_modality_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self,
        sensor_seq: torch.Tensor,
        pose_seq: torch.Tensor,
        visual_tokens: torch.Tensor,
        sensor_valid_mask: Optional[torch.Tensor] = None,
        pose_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sensor_seq: (Batch, T, 5) - Temporal sequence of normalized physical distances.
            pose_seq: (Batch, T-1, 48) - History of kinematic poses.
            visual_tokens: (Batch, Num_Patches, d_model) - Current frame visual tokens.
        """
        batch_size, T, _ = sensor_seq.shape
        if sensor_valid_mask is None:
            sensor_valid_mask = torch.ones(batch_size, T, device=sensor_seq.device, dtype=torch.bool)
        else:
            sensor_valid_mask = sensor_valid_mask.to(device=sensor_seq.device, dtype=torch.bool)
        
        # 1. Map each timestep of sensor and pose to tokens
        # sensor_seq: (B, T, 5) -> (B*T, 5) -> MLP -> (B*T, d_model) -> (B, T, d_model)
        t_sensors = self.sensor_mlp(sensor_seq.reshape(-1, 5)).view(batch_size, T, -1)
        t_sensors = self.sensor_temporal_pe(t_sensors) + self.sensor_modality_embedding
        
        # pose_seq: (B, T-1, 48) -> (B*(T-1), 48) -> MLP -> (B*(T-1), d_model) -> (B, T-1, d_model)
        if pose_seq.shape[1] > 0:
            if pose_valid_mask is None:
                pose_valid_mask = torch.ones(batch_size, pose_seq.shape[1], device=pose_seq.device, dtype=torch.bool)
            else:
                pose_valid_mask = pose_valid_mask.to(device=pose_seq.device, dtype=torch.bool)
            t_poses = self.pose_mlp(pose_seq.reshape(-1, 48)).view(batch_size, -1, self.d_model)
            t_poses = self.pose_temporal_pe(t_poses) + self.pose_modality_embedding
            # Concatenate History
            query_tokens = torch.cat([t_sensors, t_poses], dim=1) # (Batch, T + T-1, d_model)
            query_valid_mask = torch.cat([sensor_valid_mask, pose_valid_mask], dim=1)
        else:
            query_tokens = t_sensors
            query_valid_mask = sensor_valid_mask
        
        # 2. Current-frame visual memory bank
        memory_tokens = visual_tokens
        
        return query_tokens, memory_tokens, query_valid_mask

class CrossModalFusionHead(nn.Module):
    """
    Cross-Modal Transformer Decoder to reconcile visual ambiguities against strong physical priors.
    """
    def __init__(self, d_model: int = 1024, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048):
        super(CrossModalFusionHead, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final regressions from fused tokens to MANO space
        # We use mean pooling over the query sequence, so input is d_model
        self.pose_regressor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 48)  # Regress to 48-dim pose parameters
        )
        self.cam_regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 3)   # Scale, tx, ty
        )

        # Zero-initialize the last layer of each regressor so we start as an identity refinement
        nn.init.zeros_(self.pose_regressor[-1].weight)
        nn.init.zeros_(self.pose_regressor[-1].bias)
        nn.init.zeros_(self.cam_regressor[-1].weight)
        nn.init.zeros_(self.cam_regressor[-1].bias)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        query_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_tokens: (Batch, Sequence=2, d_model)
            memory_tokens: (Batch, Sequence=T*196, d_model)
            
        Returns:
            pred_pose: (Batch, 48)
            pred_cam: (Batch, 3) 
        """
        # TransformerDecoder requires: tgt (query), memory (encoded src)
        query_padding_mask = None
        if query_valid_mask is not None:
            query_padding_mask = ~query_valid_mask.to(device=query_tokens.device, dtype=torch.bool)
        fused_tokens = self.transformer_decoder(
            tgt=query_tokens,
            memory=memory_tokens,
            tgt_key_padding_mask=query_padding_mask,
        )
        
        # Pull meaningful information via global average pooling over the fused token sequence
        if query_valid_mask is None:
            fused_flat = fused_tokens.mean(dim=1)
        else:
            weights = query_valid_mask.to(device=fused_tokens.device, dtype=fused_tokens.dtype).unsqueeze(-1)
            fused_flat = (fused_tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        
        pred_pose = self.pose_regressor(fused_flat)
        pred_cam = self.cam_regressor(fused_flat)
        
        return pred_pose, pred_cam

class STMFHead(nn.Module):
    """
    The wrapper bounding tokenization and cross-modal fusion.
    Intended to replace `mano_head` for the STMF temporal model.
    """
    def __init__(self, sensor_dim: int = 5, pose_dim: int = 48, d_model: int = 1024, nhead: int = 8, num_layers: int = 3):
        super(STMFHead, self).__init__()
        self.tokenizer = ModalityTokenization(sensor_dim, pose_dim, d_model)
        self.fusion_head = CrossModalFusionHead(d_model, nhead, num_layers)
        
    def forward(
        self,
        visual_tokens: torch.Tensor,
        sensor_seq: torch.Tensor,
        pose_seq: torch.Tensor,
        sensor_valid_mask: Optional[torch.Tensor] = None,
        pose_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, torch.Tensor]:
        """
        visual_tokens: (Batch, Num_Patches, d_model)
        sensor_seq: (Batch, T, 5)
        pose_seq: (Batch, T-1, 48)
        """
        query_tokens, memory_tokens, query_valid_mask = self.tokenizer(
            sensor_seq,
            pose_seq,
            visual_tokens,
            sensor_valid_mask=sensor_valid_mask,
            pose_valid_mask=pose_valid_mask,
        )
        pred_pose, pred_cam = self.fusion_head(query_tokens, memory_tokens, query_valid_mask=query_valid_mask)
        
        # Structure as expected by the rest of the code
        # HaMeR typically regresses global_orient (3) + hand_pose (45)
        pred_mano_params = {
            'global_orient': pred_pose[:, :3],
            'hand_pose': pred_pose[:, 3:],
        }
        
        return pred_mano_params, pred_cam

from ..utils.geometry import perspective_projection, aa_to_rotmat, rotmat_to_aa

class TemporalSmoothnessLoss(nn.Module):
    """
    Penalizes the magnitude of the 2nd derivative of the predicted pose across the time window.
    Forces output trajectory to be smooth and continuous.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, poses: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        poses: (B, T, 48) or similar shape
        """
        if poses.size(1) < 3:
            return poses.new_zeros(())
            
        # 2nd derivative (acceleration): theta_t - 2 * theta_{t-1} + theta_{t-2}
        accel = poses[:, 2:] - 2 * poses[:, 1:-1] + poses[:, :-2]
        accel_norm = torch.norm(accel, p=2, dim=2)
        if valid_mask is None:
            loss = accel_norm.mean()
        else:
            valid_mask = valid_mask.to(device=poses.device, dtype=torch.bool)
            triplet_valid = valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]
            if triplet_valid.any():
                weights = triplet_valid.to(dtype=accel_norm.dtype)
                loss = (accel_norm * weights).sum() / weights.sum().clamp_min(1.0)
            else:
                loss = poses.new_zeros(())
        return self.weight * loss

class FKSensorLoss(nn.Module):
    """
    Forward Kinematics Sensor Loss.
    Computes L2 difference between MANO generated 3D wrist-to-fingertip distances 
    and ground truth 1D string pull sensor data.
    """
    def __init__(self, weight=1.0, fist_ratio: float = 0.5):
        super().__init__()
        self.weight = weight
        self.fist_ratio = fist_ratio
        self.official_joint_reorder = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
        self.finger_chains = (
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
        )

    def forward(self, pred_keypoints_3d: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        pred_keypoints_3d: (B, 21, 3) predicted MANO joints in HaMeR/OpenPose order
        sensor_data: (B, 5) normalized sensor targets in [0, 1]
        """
        joints = pred_keypoints_3d[:, self.official_joint_reorder, :]
        current_dists = []
        lmax_values = []
        for chain in self.finger_chains:
            finger_joints = joints[:, chain, :]
            current_dists.append(torch.norm(finger_joints[:, -1] - finger_joints[:, 0], dim=-1))
            bone_lengths = torch.norm(finger_joints[:, 1:] - finger_joints[:, :-1], dim=-1)
            lmax_values.append(bone_lengths.sum(dim=-1))

        current_dists = torch.stack(current_dists, dim=1)
        lmax_values = torch.stack(lmax_values, dim=1)
        lmin_values = lmax_values * self.fist_ratio
        pred_sensor = (current_dists - lmin_values) / (lmax_values - lmin_values + 1e-6)
        pred_sensor = pred_sensor.clamp(0.0, 1.0)

        loss = torch.nn.functional.mse_loss(pred_sensor, sensor_data)
        return self.weight * loss

class STMF_HAMER(HAMER):
    """
    STMF model built on top of HaMeR backbone.
    Override to handle temporal sliding windows and multi-modal input.
    """
    def __init__(self, cfg, init_renderer: bool = True):
        super(STMF_HAMER, self).__init__(cfg, init_renderer)
        
        # Freeze the visual backbone AND original mano_head as specified in the residual proposal
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.mano_head.parameters():
            param.requires_grad = False
            
        d_model = cfg.MODEL.BACKBONE.get('OUT_CHANNELS', 1024) 
        if hasattr(self.backbone, 'embed_dim'):
            d_model = self.backbone.embed_dim
        
        self.stmf_head = STMFHead(
            sensor_dim=5,
            pose_dim=48,
            d_model=d_model,
            nhead=8,
            num_layers=3
        )
        
        # STMF Specific Losses
        self.smoothness_loss = TemporalSmoothnessLoss(weight=cfg.LOSS_WEIGHTS.get('SMOOTHNESS', 10.0))
        self.fk_sensor_loss = FKSensorLoss(weight=cfg.LOSS_WEIGHTS.get('FK_SENSOR', 50.0))
        self.beta_momentum = float(cfg.MODEL.get('BETA_MOMENTUM', 0.9))

    def training_step(self, joint_batch: Dict, batch_idx: int):
        """
        Override standard HaMeR training step to handle manual optimization and detailed logging.
        """
        if 'mocap' in joint_batch:
            return super().training_step(joint_batch, batch_idx)
            
        # Access the data dictionary for both forward pass and losses
        if 'img' in joint_batch and isinstance(joint_batch['img'], dict):
            data_batch = joint_batch['img']
        else:
            data_batch = joint_batch
            
        # Access optimizer (manual optimization mode)
        opts = self.optimizers(use_pl_optimizer=True)
        if isinstance(opts, list):
            optimizer = opts[0]
        else:
            optimizer = opts
        
        output = self.forward_step(data_batch, train=True)
        loss = self.compute_loss(data_batch, output, train=True)

        # # DEBUG: Save data to file for analysis
        # if not hasattr(self, 'debug_saved'):
        #     save_data = {
        #         'joint_batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in joint_batch.items()} if isinstance(joint_batch, dict) else joint_batch,
        #         'data_batch': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()} if isinstance(data_batch, dict) else data_batch,
        #         'output': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in output.items()} if isinstance(output, dict) else output
        #     }
        #     save_path = '/home/mirage/STMF/tools/debug_training_data.pt'
        #     torch.save(save_data, save_path)
        #     print("\n" + "="*50)
        #     print(f"DEBUG: Saved training variables to {save_path}")
        #     print("="*50 + "\n")
        #     self.debug_saved = True

        # Optimization step
        optimizer.zero_grad()
        self.manual_backward(loss)
        
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        optimizer.step()

        # Detailed Logging
        losses = output.get('losses', {})
        for loss_name, val in losses.items():
            self.log(f'train/{loss_name}', val, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0 and self.mesh_renderer is not None:
            self.tensorboard_logging(data_batch, output, self.global_step, train=True)

        # Main loss to progress bar
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=data_batch.get('img').shape[0] if isinstance(data_batch.get('img'), torch.Tensor) else None)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Validation step for STMF, ensuring metrics are logged per epoch.
        """
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        
        losses = output.get('losses', {})
        for loss_name, val in losses.items():
            self.log(f'val/{loss_name}', val, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if batch_idx == 0 and self.mesh_renderer is not None:
            self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """
        Test step, reusing validation logic.
        """
        return self.validation_step(batch, batch_idx)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the STMF network.
        Expected batch:
            - 'img': (B, T, C, H, W)
            - 'sensor_seq': (B, T, 5)
            - 'pose_seq': (B, T, 48), where the last frame corresponds to the current timestep target
        """
        x = batch['img']
        if len(x.shape) == 4:
            # Fallback if dataloader is providing standard (B, C, H, W). We pretend T=1.
            x = x.unsqueeze(1)
            
        B, T, C, H, W = x.shape

        with torch.no_grad():
            self.backbone.eval()
            self.mano_head.eval()
            # If using ViT backbone, aspect ratio crops are standard in HaMeR:
            # We crop the 256x256 image to 256x192 if necessary, depending on config.
            # Hamer: x[:,:,:,32:-32]
            current_feats = self.backbone(x[:, -1, :, :, 32:-32])
            base_mano_target, base_cam_target, _ = self.mano_head(current_feats)

        _, D, Hp, Wp = current_feats.shape
        visual_tokens = current_feats.flatten(2).transpose(1, 2)
        
        # Prepare Temporal Modal Sequences for STMF Head
        if 'sensor_seq' in batch:
            sensor_seq = batch['sensor_seq']
        else:
            sensor_seq = torch.zeros(B, T, 5, device=x.device, dtype=x.dtype)
        sensor_valid_mask = batch.get('sensor_valid_mask')
        if sensor_valid_mask is None:
            sensor_valid_mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
        else:
            sensor_valid_mask = sensor_valid_mask.to(device=x.device, dtype=torch.bool)
            
        if 'pose_seq' in batch:
            # History is previous T-1 frames
            pose_seq = batch['pose_seq'][:, :-1, :]
        else:
            pose_seq = torch.zeros(B, T-1, 48, device=x.device, dtype=x.dtype)
        pose_valid_mask = batch.get('pose_valid_mask')
        if pose_valid_mask is None:
            if pose_seq.shape[1] > 0:
                pose_valid_mask = torch.ones(B, pose_seq.shape[1], device=x.device, dtype=torch.bool)
            else:
                pose_valid_mask = torch.zeros(B, 0, device=x.device, dtype=torch.bool)
        else:
            pose_valid_mask = pose_valid_mask.to(device=x.device, dtype=torch.bool)

        delta_mano_params, delta_cam = self.stmf_head(
            visual_tokens,
            sensor_seq,
            pose_seq,
            sensor_valid_mask=sensor_valid_mask,
            pose_valid_mask=pose_valid_mask,
        )

        # Convert delta axis-angles to rotation matrices
        go_aa_delta = delta_mano_params['global_orient']  # (B, 3)
        hp_aa_delta = delta_mano_params['hand_pose']      # (B, 45)
        
        go_rotmat_delta = aa_to_rotmat(go_aa_delta.reshape(-1, 3)).reshape(B, 1, 3, 3)
        hp_rotmat_delta = aa_to_rotmat(hp_aa_delta.reshape(-1, 3)).reshape(B, 15, 3, 3)

        # Residual Refinement formulation: composing rotations R_final = R_delta @ R_base
        pred_mano_params = {}
        pred_mano_params['global_orient'] = torch.matmul(go_rotmat_delta, base_mano_target['global_orient'].view(B, 1, 3, 3))
        pred_mano_params['hand_pose'] = torch.matmul(hp_rotmat_delta, base_mano_target['hand_pose'].view(B, 15, 3, 3))
        pred_mano_params['betas'] = self._stabilize_betas(base_mano_target['betas'].view(B, -1), batch)
        pred_cam = base_cam_target.view(B, -1) + delta_cam

        # Build output dictionary (same as HaMeR to preserve loss functions and rendering hooks)
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k, v in pred_mano_params.items()}

        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(B, 2, device=x.device, dtype=x.dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices and joints
        # The hand poses in pred_mano_params are already valid rotation matrices:
        # global_orient (B, 1, 3, 3), hand_pose (B, 15, 3, 3). No aa_to_rotmat needed!
        
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(B, -1)
        
        mano_output = self.mano(**{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(B, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(B, -1, 3)
        
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(B, -1, 2)

        pred_pose = torch.cat([
            rotmat_to_aa(pred_mano_params['global_orient'].reshape(-1, 3, 3)).reshape(B, 3),
            rotmat_to_aa(pred_mano_params['hand_pose'].reshape(-1, 3, 3)).reshape(B, 45),
        ], dim=1)
        output['pred_pose'] = pred_pose

        if pose_seq.shape[1] >= 2:
            output['pred_poses_seq'] = torch.cat([pose_seq[:, -2:, :], pred_pose.unsqueeze(1)], dim=1)
            output['pred_poses_seq_valid_mask'] = torch.cat([
                pose_valid_mask[:, -2:],
                torch.ones(B, 1, device=x.device, dtype=torch.bool),
            ], dim=1)

        return output

    def _stabilize_betas(self, base_betas: torch.Tensor, batch: Dict) -> torch.Tensor:
        prev_betas = batch.get('prev_betas')
        has_prev_betas = batch.get('has_prev_betas')
        if prev_betas is None or has_prev_betas is None:
            return base_betas

        prev_betas = prev_betas.to(device=base_betas.device, dtype=base_betas.dtype)
        has_prev_betas = has_prev_betas.to(device=base_betas.device, dtype=base_betas.dtype).view(-1, 1)
        blended = self.beta_momentum * prev_betas + (1.0 - self.beta_momentum) * base_betas
        return has_prev_betas * blended + (1.0 - has_prev_betas) * base_betas

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Extends original compute_loss to include Physical Constraints and Smoothness.
        """
        # The output['pred_mano_params'] are already valid rotation matrices
        # so we can directly call super() which translates GT axis-angle to rotmat to match.
        
        # Original Baseline Loss computation
        base_loss = super().compute_loss(batch, output, train)
        losses = output['losses']
        
        pred_keypoints_3d = output['pred_keypoints_3d']

        # Temporary diagnostic prints for zero-loss issue
        if self.global_step > 0 and self.global_step % 10 == 0:
            print(f"\n--- [LOSS DIAGNOSTIC Step {self.global_step}] ---")
            print(f"  Base Total Loss: {base_loss.item():.6f}")
            print(f"  Weight 2D: {self.cfg.LOSS_WEIGHTS.get('KEYPOINTS_2D', 'NA')}")
            print(f"  Weight 3D: {self.cfg.LOSS_WEIGHTS.get('KEYPOINTS_3D', 'NA')}")
            for k, v in losses.items():
                print(f"  Raw {k}: {v.item():.6f}")
        
        if 'sensor' in batch: # (B, 5)
            sensor_data = batch['sensor']
            loss_fk = self.fk_sensor_loss(pred_keypoints_3d, sensor_data)
            base_loss += loss_fk
            losses['loss_fk_sensor'] = loss_fk.detach()
            
        # Note: True temporal smoothness requires predicting poses for the entire window
        # If the batch supplies sequence poses or the model predicts sequentially, apply it:
        if 'pred_poses_seq' in output:
            loss_smooth = self.smoothness_loss(
                output['pred_poses_seq'],
                output.get('pred_poses_seq_valid_mask'),
            )
            base_loss += loss_smooth
            losses['loss_smoothness'] = loss_smooth.detach()
            
        output['losses'] = losses
        return base_loss

    def get_parameters(self):
        # We only return the tunable parameters (STMF Head), since Backbone & original MANO head are FROZEN.
        return list(self.stmf_head.parameters())
