import os
import copy
import json
import torch
import numpy as np
from typing import Dict

from .image_dataset import ImageDataset


class TemporalImageDataset(ImageDataset):
    """
    Extends original HaMeR ImageDataset to support Temporal Sliding Windows 
    and 5D physical sensor inputs for the STMF architecture.
    """
    def __init__(self,
                 cfg,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 rescale_factor: int = 2,
                 prune: Dict = {},
                 seq_len: int = 3,           # Length of the temporal window
                 stride: int = 1,            # Step size for sliding window
                 **kwargs):
                 
        # 1. Initialize the parent (loads all base annotations correctly)
        super().__init__(cfg, dataset_file, img_dir, train, rescale_factor, prune, **kwargs)
        
        self.seq_len = seq_len
        self.stride = stride
        
        # 2. Try to load 5-finger physical distance sensor data
        # Prioritize reading from the NPZ file if available
        if 'sensor' in self.data:
            self.sensor_data_source = self.data['sensor'].astype(np.float32)
            print(f"Loaded sensor data from NPZ: {self.sensor_data_source.shape}")
        else:
            # Fallback to loading from JSON files
            self.sensor_data_source = None
            base_dir = os.path.dirname(os.path.normpath(img_dir))
            
            # Determine split name roughly based on dataset_file naming
            split_prefix = 'training' if 'train' in dataset_file.lower() else 'evaluation'
            
            sensor_json_path = os.path.join(base_dir, f'{split_prefix}_finger_distances.json')
            if os.path.exists(sensor_json_path):
                with open(sensor_json_path, 'r') as f:
                    raw_sensor = json.load(f)
                
                # Extract only the "normalized" values into an (N, 5) numpy array
                self.sensor_data_source = np.array(
                    [item['normalized'] for item in raw_sensor], dtype=np.float32
                )
                
                if len(self.sensor_data_source) != len(self.center):
                    print(f"Warning: TemporalImageDataset sensor count ({len(self.sensor_data_source)}) "
                          f"!= image count ({len(self.center)}). Padding with zeros.")
                    pad_len = max(0, len(self.center) - len(self.sensor_data_source))
                    if pad_len > 0:
                        pad = np.zeros((pad_len, 5), dtype=np.float32)
                        self.sensor_data_source = np.vstack([self.sensor_data_source, pad])
            else:
                # If no sensor data is provided natively by the dataset, fallback to zeros
                self.sensor_data_source = np.zeros((len(self.center), 5), dtype=np.float32)

        # 3. Build sequence indices
        # We assume dataset_file is sorted temporally (like FreiHAND video sequences).
        # Fallback to single-frame duplication if the dataset is purely spatial images.
        self.valid_indices = self._build_sequences()


    def _build_sequences(self):
        """
        Create a list of temporal arrays. 
        Each element is an array of indices `[t-N, ... t]` pointing to `self.data`.
        """
        valid_seqs = []
        n_samples = len(self.center)
        
        # Simple sliding window approach.
        # Note: In a real video dataset, we should check `video_id` or `person_id` boundaries.
        # Since FreiHAND is one continuous capture per subject but concatenated, 
        # we will let the window slide through. Hard breaks can be implemented via personid.
        for i in range(0, n_samples - self.seq_len + 1, self.stride):
            
            # Check if all frames in this window belong to the same person/sequence
            person_ids = self.personid[i : i + self.seq_len]
            if len(np.unique(person_ids)) == 1:
                valid_seqs.append(np.arange(i, i + self.seq_len))
                
        # If no valid sequences were found (e.g., sequence length too long for tiny dataset),
        # fallback to single frame duplicated.
        if len(valid_seqs) == 0:
            for i in range(n_samples):
                valid_seqs.append(np.full(self.seq_len, i))
                
        return valid_seqs

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Aggregates `seq_len` frames into a temporal batch item.
        Returns shapes like (T, C, H, W) for images.
        """
        seq_idx = self.valid_indices[idx]
        
        # We construct the temporal window by calling parent's __getitem__
        # for every frame in the sequence
        frames = []
        for i in seq_idx:
            frames.append(super().__getitem__(i)) # Call parent's item fetcher
            
        # Target frame (usually the last frame in the window `t`)
        # provides the primary ground truths for rendering
        target_idx_in_seq = -1
        target_global_idx = seq_idx[target_idx_in_seq]
        target_frame = frames[target_idx_in_seq]
        
        # Stack temporal features
        item = {}
        
        # Image: [T, 3, 256, 256]
        item['img'] = torch.stack([torch.tensor(f['img']) if not isinstance(f['img'], torch.Tensor) else f['img'] for f in frames])
        
        # Sensor data for the current TARGET frame `t`: [5]
        item['sensor'] = self.sensor_data_source[target_global_idx]
        
        # Ground truths are passed exactly as expected by HaMeR loss calculators
        # but we also construct temporal variations if required by Smoothness loss.
        # By default, HaMeR expects these at the target frame level:
        for k in ['keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d', 'idx']:
            item[k] = target_frame[k]
            
        # MANO parameters
        item['mano_params'] = target_frame['mano_params']
        item['has_mano_params'] = target_frame['has_mano_params']
        item['mano_params_is_axis_angle'] = target_frame['mano_params_is_axis_angle']
        
        # Bounding Box info
        for k in ['box_center', 'box_size', 'img_size', '_scale', 'bbox_expand_factor']:
            if k in target_frame:
                item[k] = target_frame[k]
                
        item['imgname'] = target_frame['imgname']
        item['right'] = target_frame['right']
        
        # To provide Prev Pose (t-1) for STMF Kinematic MLP:
        # If sequence length > 1, the previous pose is retrieved from the second to last frame.
        if self.seq_len > 1:
            prev_frame = frames[-2]
            prev_hand_pose = prev_frame['mano_params']['hand_pose'] # (45,)
            prev_global_orient = prev_frame['mano_params']['global_orient'] # (3,)
            # Reconstruct (48,) continuous pose parameter
            # We assume STMF tokenizer expects flattened axis-angle
            if isinstance(prev_hand_pose, torch.Tensor):
                item['prev_pose'] = torch.cat([prev_global_orient, prev_hand_pose], dim=0)
            else:
                item['prev_pose'] = np.concatenate([prev_global_orient, prev_hand_pose], axis=0).astype(np.float32)
        else:
            # Fallback to zero if T=1
            item['prev_pose'] = np.zeros(48, dtype=np.float32)

        return item
