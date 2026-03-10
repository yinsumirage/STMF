"""
MANO Hand Processor Core Implementation
"""

import numpy as np
from typing import Tuple, List, Optional


class MANOHandProcessor:
    """
    MANO手部模型处理器，实现design.md中描述的功能
    使用正确的关节索引（来自design.md）
    """
    
    def __init__(self):
        """初始化处理器"""
        # 根据design.md定义的正确关节索引
        # 腕关节: 0
        # 大拇指: [0, 1, 2, 3, 4]
        # 食指: [0, 5, 6, 7, 8]  
        # 中指: [0, 9, 10, 11, 12]
        # 无名指: [0, 13, 14, 15, 16]
        # 小指: [0, 17, 18, 19, 20]
        self.finger_joints = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }
        
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    def validate_joints(self, joints: np.ndarray) -> bool:
        """验证关节数据格式"""
        if joints.shape != (21, 3):
            raise ValueError(f"Expected joints shape (21, 3), got {joints.shape}")
        return True
    
    def compute_bone_lengths(self, joints: np.ndarray) -> dict:
        """
        计算各手指的骨长（不受姿态影响）
        返回每根手指各段的长度
        """
        self.validate_joints(joints)
        
        bone_lengths = {}
        
        for finger in self.finger_names:
            joint_indices = self.finger_joints[finger]
            segments = []
            for i in range(len(joint_indices) - 1):
                start_idx = joint_indices[i]
                end_idx = joint_indices[i + 1]
                segment_length = np.linalg.norm(joints[end_idx] - joints[start_idx])
                segments.append(segment_length)
            bone_lengths[finger] = np.array(segments)
            
        return bone_lengths
    
    def compute_lmax_from_bones(self, bone_lengths: dict) -> np.ndarray:
        """
        从骨长计算L_max（完全伸展时的最大长度）
        L_max = 各段骨长之和（三角不等式等号成立时）
        """
        lmax = np.zeros(5)
        for i, finger in enumerate(self.finger_names):
            lmax[i] = np.sum(bone_lengths[finger])
        return lmax
    
    def compute_lmax_direct(self, joints: np.ndarray) -> np.ndarray:
        """
        直接从关节位置计算L_max
        这假设输入的joints已经是完全伸展姿态
        """
        bone_lengths = self.compute_bone_lengths(joints)
        return self.compute_lmax_from_bones(bone_lengths)
    
    def compute_current_distances(self, joints: np.ndarray) -> np.ndarray:
        """
        计算当前姿态下腕部到各指尖的直线距离
        """
        self.validate_joints(joints)
        
        distances = np.zeros(5)
        for i, finger in enumerate(self.finger_names):
            joint_indices = self.finger_joints[finger]
            wrist_idx = joint_indices[0]  # 0
            tip_idx = joint_indices[-1]   # 4, 8, 12, 16, 20
            distance = np.linalg.norm(joints[tip_idx] - joints[wrist_idx])
            distances[i] = distance
            
        return distances
    
    def estimate_lmin_from_lmax(self, lmax: np.ndarray, 
                               fist_ratio: float = 0.5) -> np.ndarray:
        """
        基于L_max估算L_min
        fist_ratio: 握拳时长度与完全伸展长度的比例（经验值0.4-0.6）
        """
        return lmax * fist_ratio
    
    def compute_lmin_from_fist_pose(self, fist_joints: np.ndarray) -> np.ndarray:
        """
        从真实的握拳姿态计算L_min
        如果有真实的握拳姿态数据，使用此方法
        """
        return self.compute_current_distances(fist_joints)
    
    def normalize_distances(self, current_distances: np.ndarray,
                          lmin: np.ndarray, lmax: np.ndarray) -> np.ndarray:
        """
        Min-Max归一化：v_sensor = (D_current - L_min) / (L_max - L_min)
        """
        # 避免除零
        denominator = lmax - lmin
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        
        normalized = (current_distances - lmin) / denominator
        return np.clip(normalized, 0.0, 1.0)
    
    def process_hand_frame(self, joints: np.ndarray, 
                          lmin_method: str = 'estimate',
                          fist_joints: Optional[np.ndarray] = None,
                          fist_ratio: float = 0.5) -> dict:
        """
        处理单帧手部数据
        
        Args:
            joints: 当前帧的21个关节3D坐标 (21, 3)
            lmin_method: 'estimate' 或 'fist_pose'
            fist_joints: 真实握拳姿态的关节坐标（如果lmin_method='fist_pose'）
            fist_ratio: 估算L_min时使用的比例
            
        Returns:
            包含所有计算结果的字典
        """
        self.validate_joints(joints)
        
        # 计算骨长（这是手的固有属性，不受姿态影响）
        bone_lengths = self.compute_bone_lengths(joints)
        
        # 计算L_max
        lmax = self.compute_lmax_from_bones(bone_lengths)
        
        # 计算L_min
        if lmin_method == 'fist_pose' and fist_joints is not None:
            self.validate_joints(fist_joints)
            lmin = self.compute_lmin_from_fist_pose(fist_joints)
        else:
            lmin = self.estimate_lmin_from_lmax(lmax, fist_ratio)
        
        # 计算当前距离
        current_distances = self.compute_current_distances(joints)
        
        # 归一化
        normalized = self.normalize_distances(current_distances, lmin, lmax)
        
        return {
            'bone_lengths': bone_lengths,
            'lmax': lmax,
            'lmin': lmin,
            'current_distances': current_distances,
            'normalized_sensor_values': normalized
        }