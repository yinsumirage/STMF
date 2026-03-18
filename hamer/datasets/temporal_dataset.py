import os
import copy
import json
import torch
import numpy as np
from typing import Dict, List, Tuple

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
        
        # Handle both seq_len and window_size for compatibility
        self.seq_len = kwargs.get('window_size', seq_len)
        self.stride = stride
        self.train = train
        self.sequence_keys = self._resolve_sequence_keys()
        self.sequence_start = self._compute_sequence_starts()
        
        # 2. Try to load 5-finger physical distance sensor data
        if 'sensor' in self.data:
            self.sensor_data_source = self.data['sensor'].astype(np.float32)
            print(f"Loaded sensor data from NPZ: {self.sensor_data_source.shape}")
        else:
            self.sensor_data_source = np.zeros((len(self.center), 5), dtype=np.float32)

        # 3. HO3D-specific Filter: Use official evaluation.txt to match the 20137 count
        self.valid_subset_mask = None
        if not train and ('HO3D' in str(dataset_file).upper() or 'HO-3D' in str(dataset_file).upper()):
            # Look for evaluation.txt in the same directory as images
            subset_file = os.path.join(img_dir, 'evaluation.txt')
            if os.path.exists(subset_file):
                print(f"Applying official whitelist: {subset_file}")
                with open(subset_file, 'r') as f:
                    whitelist = set([line.strip() for line in f.readlines() if line.strip()])
                
                # Match whitelisted "Folder/Frame" against NPZ paths
                self.valid_subset_mask = []
                for name in self.imgname:
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    # Convert 'evaluation/SM1/rgb/0000.jpg' -> 'SM1/0000'
                    parts = name.replace('.jpg', '').split('/')
                    if len(parts) >= 4:
                        short_name = f"{parts[-3]}/{parts[-1]}"
                        self.valid_subset_mask.append(short_name in whitelist)
                    else:
                        self.valid_subset_mask.append(True)
                self.valid_subset_mask = np.array(self.valid_subset_mask)
                print(f"Matched {self.valid_subset_mask.sum()} entries from whitelist.")

        # 4. Build sequence indices
        self.valid_indices = self._build_sequences()


    def _build_sequences(self):
        """
        Create a list of temporal arrays of length `seq_len`.
        """
        valid_seqs = []
        n_samples = len(self.center)

        for t in range(n_samples):
            # FILTER: If we have a whitelist mask (for HO3D), only include valid frames as TARGETS
            if self.valid_subset_mask is not None and not self.valid_subset_mask[t]:
                continue

            # Target frame is `t`. We look back to find `seq_len - 1` previous frames.
            seq = []
            start_of_person_seq = self.sequence_start[t]

            for offset in range(self.seq_len - 1, -1, -1):
                prev_idx = t - offset * self.stride

                # Boundary check and left padding within the same sequence.
                clamped_idx = max(prev_idx, start_of_person_seq)
                seq.append(clamped_idx)

            valid_seqs.append(np.array(seq))

        return valid_seqs

    def _resolve_sequence_keys(self) -> List[str]:
        """
        Recover stable sequence identifiers.
        Priority:
        1. Existing personid from NPZ.
        2. Sequence-like prefix parsed from imgname.
        3. Fallback to a single sequence.
        """
        derived_ids, derived_keys = self._derive_personids_from_imgname()
        personid = np.asarray(self.personid).reshape(-1)

        if derived_ids is not None:
            loaded_unique = len(np.unique(personid))
            derived_unique = len(np.unique(derived_ids))
            # Old exports sometimes leave personid all-zero even when imgname spans multiple sequences.
            if loaded_unique <= 1 < derived_unique:
                self.personid = derived_ids
                return derived_keys

        if len(np.unique(personid)) > 1:
            return [f"pid_{int(pid)}" for pid in personid.tolist()]

        if derived_keys is not None:
            return derived_keys

        return ['seq_0'] * len(self.imgname)

    def _derive_personids_from_imgname(self) -> Tuple[np.ndarray, List[str]]:
        """
        Infer sequence ids from paths like:
          train/ABF10/rgb/0001.jpg -> ABF10
          evaluation/SM1/rgb/0001.jpg -> SM1
        """
        sequence_names = []
        for raw_name in self.imgname:
            name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)
            parts = name.replace('\\', '/').split('/')
            if len(parts) >= 4 and parts[2] == 'rgb':
                sequence_names.append(parts[1])
            else:
                sequence_names.append('seq_0')

        unique_names = {name: idx for idx, name in enumerate(dict.fromkeys(sequence_names))}
        derived_ids = np.array([unique_names[name] for name in sequence_names], dtype=np.int32)
        return derived_ids, sequence_names

    def _compute_sequence_starts(self) -> np.ndarray:
        sequence_start = np.zeros(len(self.center), dtype=np.int64)
        current_start = 0
        for idx in range(len(self.center)):
            if idx > 0 and self.personid[idx] != self.personid[idx - 1]:
                current_start = idx
            sequence_start[idx] = current_start
        return sequence_start

    def _build_temporal_valid_masks(self, target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build validity masks for the temporal window.
        Valid means the slot corresponds to a real in-sequence timestep rather than
        left padding introduced to keep the window length fixed.
        """
        start_of_sequence = int(self.sequence_start[target_idx])
        sensor_valid_mask = []
        for offset in range(self.seq_len - 1, -1, -1):
            hist_idx = target_idx - offset * self.stride
            sensor_valid_mask.append(hist_idx >= start_of_sequence)

        sensor_valid_mask = np.asarray(sensor_valid_mask, dtype=np.bool_)
        pose_valid_mask = sensor_valid_mask[:-1].copy()
        return sensor_valid_mask, pose_valid_mask

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        """ Aggregates `seq_len` frames into a temporal batch item. """
        seq_idx = self.valid_indices[idx]
        
        frames = []
        for i in seq_idx:
            frames.append(super().__getitem__(i))
            
        target_frame = frames[-1]
        target_idx = int(seq_idx[-1])
        sensor_valid_mask, pose_valid_mask = self._build_temporal_valid_masks(target_idx)
        item = {}
        
        # Image: [T, 3, 256, 256]
        item['img'] = torch.stack([torch.tensor(f['img']) if not isinstance(f['img'], torch.Tensor) else f['img'] for f in frames])
        
        # Ground truths
        for k in ['keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d', 'idx']:
            if k in target_frame:
                item[k] = target_frame[k]
        item['mano_params'] = target_frame['mano_params']
        item['has_mano_params'] = target_frame['has_mano_params']
        item['mano_params_is_axis_angle'] = target_frame['mano_params_is_axis_angle']
        
        for k in ['box_center', 'box_size', 'img_size', '_scale', 'bbox_expand_factor']:
            if k in target_frame:
                item[k] = target_frame[k]
        item['imgname'] = target_frame['imgname']
        item['right'] = target_frame['right']
        item['personid'] = int(self.personid[target_idx])
        item['sequence_key'] = self.sequence_keys[target_idx]
        item['temporal_indices'] = np.array(seq_idx, dtype=np.int64)
        item['sensor_valid_mask'] = sensor_valid_mask
        item['pose_valid_mask'] = pose_valid_mask

        # SEQUENCES
        item['sensor_seq'] = np.stack([self.sensor_data_source[idx] for idx in seq_idx]).astype(np.float32)
        pose_seq = []
        for f in frames:
            p_hand = f['mano_params']['hand_pose']
            p_orient = f['mano_params']['global_orient']
            if isinstance(p_hand, torch.Tensor):
                p_full = torch.cat([p_orient, p_hand], dim=0)
            else:
                p_full = np.concatenate([p_orient, p_hand], axis=0).astype(np.float32)
            pose_seq.append(p_full)
        item['pose_seq'] = np.stack(pose_seq) if not isinstance(pose_seq[0], torch.Tensor) else torch.stack(pose_seq)
        
        if self.train:
            noise = np.random.normal(0, 0.02, size=item['pose_seq'].shape).astype(np.float32)
            item['pose_seq'] = item['pose_seq'] + noise

        has_prev_step = bool(pose_valid_mask[-1]) if pose_valid_mask.size > 0 else False
        if self.seq_len > 1 and has_prev_step:
            prev_betas = frames[-2]['mano_params']['betas']
            has_prev_betas = frames[-2]['has_mano_params']['betas']
            prev_pose = item['pose_seq'][-2]
        else:
            prev_betas = np.zeros(10, dtype=np.float32)
            has_prev_betas = np.float32(0.0)
            prev_pose = np.zeros(48, dtype=np.float32)

        item['prev_betas'] = prev_betas
        item['has_prev_betas'] = has_prev_betas

        # Legacy
        item['sensor'] = item['sensor_seq'][-1]
        item['prev_pose'] = prev_pose
        return item
