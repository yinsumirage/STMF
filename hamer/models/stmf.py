import torch
import torch.nn as nn
import math
from typing import Tuple, Dict

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
        
        self.temporal_pe = TemporalPositionalEncoding(d_model=d_model)

    def forward(self, sensor_data: torch.Tensor, prev_pose: torch.Tensor, visual_buffer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensor_data: (Batch, 5) - Current timestep's normalized physical distances for 5 fingers.
            prev_pose: (Batch, 48) - t-1 frame's continuous kinematic MANO parameters.
            visual_buffer: (Batch, T, 196, d_model) - Temporal sequence of patch features extracted by frozen ViT.
            
        Returns:
            query_tokens: (Batch, 2, d_model) - Multi-modal query vectors to actively interrogate the visual memory bank.
            memory_tokens: (Batch, T * 196, d_model) - Global visual feature sequence bank infused with spatio-temporal info.
        """
        batch_size = sensor_data.size(0)
        
        # 1. Dimensionality mapping and alignment of continuous modalities
        # Map via MLP, then unsqueeze(1) to add Sequence dimension, forming standard Token shape (Batch, 1, d_model)
        t_sensor = self.sensor_mlp(sensor_data).unsqueeze(1)
        t_pose = self.pose_mlp(prev_pose).unsqueeze(1)
        
        # Concatenate Physical Token and Kinematic Token in the sequence dimension
        # Output shape: (Batch, 2, d_model)
        query_tokens = torch.cat([t_sensor, t_pose], dim=1)
        
        # 2. Temporal processing and flattening of visual features
        # Sequence shape should be (Batch, T, Num_Patches, d_model)
        visual_buffer_pe = self.temporal_pe(visual_buffer)
        
        # Flatten the temporal steps (T) and spatial patches dimension into a long sequence
        # Output shape: (Batch, T * Num_Patches, d_model)
        memory_tokens = visual_buffer_pe.reshape(batch_size, -1, self.d_model)
        
        return query_tokens, memory_tokens

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
        # We'll use the two query tokens (sensor & prev_pose) concatenated
        self.pose_regressor = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Linear(512, 48)  # Regress to 48-dim pose parameters
        )
        self.shape_regressor = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 10)  # Regress to 10-dim betas
        )
        self.cam_regressor = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 3)   # Scale, tx, ty
        )

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query_tokens: (Batch, Sequence=2, d_model)
            memory_tokens: (Batch, Sequence=T*196, d_model)
            
        Returns:
            pred_pose: (Batch, 48)
            pred_shape: (Batch, 10)
            pred_cam: (Batch, 3) 
        """
        # TransformerDecoder requires: tgt (query), memory (encoded src)
        fused_tokens = self.transformer_decoder(tgt=query_tokens, memory=memory_tokens)
        
        # Flatten the fused sequence of length 2
        batch_size = fused_tokens.size(0)
        fused_flat = fused_tokens.reshape(batch_size, -1)
        
        pred_pose = self.pose_regressor(fused_flat)
        pred_shape = self.shape_regressor(fused_flat)
        pred_cam = self.cam_regressor(fused_flat)
        
        return pred_pose, pred_shape, pred_cam

class STMFHead(nn.Module):
    """
    The wrapper bounding tokenization and cross-modal fusion.
    Intended to replace `mano_head` for the STMF temporal model.
    """
    def __init__(self, sensor_dim: int = 5, pose_dim: int = 48, d_model: int = 1024, nhead: int = 8, num_layers: int = 3):
        super(STMFHead, self).__init__()
        self.tokenizer = ModalityTokenization(sensor_dim, pose_dim, d_model)
        self.fusion_head = CrossModalFusionHead(d_model, nhead, num_layers)
        
    def forward(self, visual_buffer: torch.Tensor, sensor_data: torch.Tensor, prev_pose: torch.Tensor) -> Tuple[Dict, torch.Tensor]:
        """
        visual_buffer: (Batch, T, 196, d_model)
        sensor_data: (Batch, 5)
        prev_pose: (Batch, 48)
        """
        query_tokens, memory_tokens = self.tokenizer(sensor_data, prev_pose, visual_buffer)
        pred_pose, pred_shape, pred_cam = self.fusion_head(query_tokens, memory_tokens)
        
        # Structure as expected by the rest of the code
        # HaMeR typically regresses global_orient (3) + hand_pose (45)
        pred_mano_params = {
            'global_orient': pred_pose[:, :3],
            'hand_pose': pred_pose[:, 3:],
            'betas': pred_shape
        }
        
        return pred_mano_params, pred_cam

from ..utils.geometry import perspective_projection, aa_to_rotmat

class TemporalSmoothnessLoss(nn.Module):
    """
    Penalizes the magnitude of the 2nd derivative of the predicted pose across the time window.
    Forces output trajectory to be smooth and continuous.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        """
        poses: (B, T, 48) or similar shape
        """
        if poses.size(1) < 3:
            return torch.tensor(0.0, device=poses.device, requires_grad=True)
            
        # 2nd derivative (acceleration): theta_t - 2 * theta_{t-1} + theta_{t-2}
        accel = poses[:, 2:] - 2 * poses[:, 1:-1] + poses[:, :-2]
        loss = torch.norm(accel, p=2, dim=2).mean()
        return self.weight * loss

class FKSensorLoss(nn.Module):
    """
    Forward Kinematics Sensor Loss.
    Computes L2 difference between MANO generated 3D wrist-to-fingertip distances 
    and ground truth 1D string pull sensor data.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        # Standard MANO topology approximate fingertip indices
        self.fingertip_indices = [745, 317, 444, 556, 673]
        self.wrist_index = 0

    def forward(self, pred_vertices: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        pred_vertices: (B, 778, 3) predicted mesh
        sensor_data: (B, 5) ground truth pull distance
        """
        wrist = pred_vertices[:, self.wrist_index, :] # (B, 3)
        fingertips = pred_vertices[:, self.fingertip_indices, :] # (B, 5, 3)
        
        # Calculate 3D Euclidean distances
        pred_dists = torch.norm(fingertips - wrist.unsqueeze(1), dim=-1) # (B, 5)
        
        # Loss between predicted mesh distances and ground truth sensor stretches
        loss = torch.nn.functional.mse_loss(pred_dists, sensor_data)
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
            num_layers=6 # Increased from 3 to 6 for better capacity
        )
        
        # STMF Specific Losses
        self.smoothness_loss = TemporalSmoothnessLoss(weight=cfg.LOSS_WEIGHTS.get('SMOOTHNESS', 10.0))
        self.fk_sensor_loss = FKSensorLoss(weight=cfg.LOSS_WEIGHTS.get('FK_SENSOR', 50.0))

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the STMF network.
        Expected batch:
            - 'img': (B, T, C, H, W)
            - 'sensor': (B, 5)  (At current timestep t)
            - 'prev_pose': (B, 48) (Either ground truth for teaching forcing, or previous prediction)
        """
        x = batch['img']
        if len(x.shape) == 4:
            # Fallback if dataloader is providing standard (B, C, H, W). We pretend T=1.
            x = x.unsqueeze(1)
            
        B, T, C, H, W = x.shape
        
        # Process the temporal window through the frozen backbone
        # Flatten B and T to process everything in parallel through ViT
        x_flat = x.view(B * T, C, H, W)
        
        with torch.no_grad():
            self.backbone.eval()
            self.mano_head.eval()
            # If using ViT backbone, aspect ratio crops are standard in HaMeR:
            # We crop the 256x256 image to 256x192 if necessary, depending on config.
            # Hamer: x[:,:,:,32:-32]
            conditioning_feats = self.backbone(x_flat[:, :, :, 32:-32])  
            # conditioning_feats shape: (B*T, D_model, H_patch, W_patch) e.g., (B*T, 1024, 16, 12) -> size=192 patches
            
            # Predict base pose using original frozen mano_head
            base_mano_params, base_cam, _ = self.mano_head(conditioning_feats)
        
        # Original mano_head outputs shape (B*T, ...). We extract the base pose for the target frame (last frame in window).
        target_idx = -1 
        base_mano_target = {k: v.view(B, T, -1)[:, target_idx, :] for k, v in base_mano_params.items()}
        base_cam_target = base_cam.view(B, T, -1)[:, target_idx, :]

        _, D, Hp, Wp = conditioning_feats.shape
        
        # Reshape to visual_buffer shape expected by tokenizer: (B, T, Num_Patches, d_model)
        visual_buffer = conditioning_feats.flatten(2).transpose(1, 2)  # (B*T, Hp*Wp, D)
        visual_buffer = visual_buffer.view(B, T, Hp*Wp, D)
        
        # Handle sensor and prev_pose gracefully for test cases where they might be missing
        if 'sensor' in batch:
            sensor_data = batch['sensor']
        else:
            sensor_data = torch.zeros(B, 5, device=x.device, dtype=x.dtype)
            
        # --- Modality Dropout: 10% chance to drop sensor data during training ---
        if train and torch.rand(1).item() < 0.1:
            sensor_data = torch.zeros_like(sensor_data)
            
        if 'prev_pose' in batch:
            prev_pose = batch['prev_pose']
        else:
            prev_pose = torch.zeros(B, 48, device=x.device, dtype=x.dtype)

        delta_mano_params, delta_cam = self.stmf_head(visual_buffer, sensor_data, prev_pose)

        # Convert delta axis-angles to rotation matrices
        go_aa_delta = delta_mano_params['global_orient']  # (B, 3)
        hp_aa_delta = delta_mano_params['hand_pose']      # (B, 45)
        
        go_rotmat_delta = aa_to_rotmat(go_aa_delta.reshape(-1, 3)).reshape(B, 1, 3, 3)
        hp_rotmat_delta = aa_to_rotmat(hp_aa_delta.reshape(-1, 3)).reshape(B, 15, 3, 3)

        # Residual Refinement formulation: composing rotations R_final = R_delta @ R_base
        pred_mano_params = {}
        pred_mano_params['global_orient'] = torch.matmul(go_rotmat_delta, base_mano_target['global_orient'].view(B, 1, 3, 3))
        pred_mano_params['hand_pose'] = torch.matmul(hp_rotmat_delta, base_mano_target['hand_pose'].view(B, 15, 3, 3))
        pred_mano_params['betas'] = base_mano_target['betas'].view(B, -1) + delta_mano_params['betas'].view(B, -1)
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
        
        # Save historical trajectory for Temporal Smoothness Loss
        if 'poses_seq' in batch:
            # Emulate having predicted T poses for smoothness:
            pass # Implementation relies on sequence tracking
            
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Extends original compute_loss to include Physical Constraints and Smoothness.
        """
        # The output['pred_mano_params'] are already valid rotation matrices
        # so we can directly call super() which translates GT axis-angle to rotmat to match.
        
        # Original Baseline Loss computation
        base_loss = super().compute_loss(batch, output, train)
        losses = output['losses']
        
        pred_vertices = output['pred_vertices']
        B = pred_vertices.size(0)
        
        if 'sensor' in batch: # (B, 5)
            sensor_data = batch['sensor']
            loss_fk = self.fk_sensor_loss(pred_vertices, sensor_data)
            base_loss += loss_fk
            losses['loss_fk_sensor'] = loss_fk.detach()
            
        # Note: True temporal smoothness requires predicting poses for the entire window
        # If the batch supplies sequence poses or the model predicts sequentially, apply it:
        if 'pred_poses_seq' in output:
            loss_smooth = self.smoothness_loss(output['pred_poses_seq'])
            base_loss += loss_smooth
            losses['loss_smoothness'] = loss_smooth.detach()
            
        output['losses'] = losses
        return base_loss

    def get_parameters(self):
        # We only return the tunable parameters (STMF Head), since Backbone & original MANO head are FROZEN.
        return list(self.stmf_head.parameters())
