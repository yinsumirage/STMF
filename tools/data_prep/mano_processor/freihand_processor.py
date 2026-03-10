"""
FreiHAND数据集处理器
一键化生成虚拟拉线传感器数据
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional
from .core import MANOHandProcessor


class FreiHANDProcessor:
    """
    FreiHAND数据集处理器，用于生成虚拟拉线传感器数据
    """
    
    def __init__(self, 
                 mano_json_path: str,
                 output_dir: str,
                 fist_ratio: float = 0.5,
                 use_real_fist: bool = False):
        """
        初始化处理器
        
        Args:
            mano_json_path: FreiHAND的MANO参数JSON文件路径
            output_dir: 输出目录
            fist_ratio: 握拳比例（用于估算L_min）
            use_real_fist: 是否使用真实握拳姿态（需要额外的握拳数据）
        """
        self.mano_json_path = mano_json_path
        self.output_dir = output_dir
        self.fist_ratio = fist_ratio
        self.use_real_fist = use_real_fist
        self.processor = MANOHandProcessor()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载MANO参数
        self.mano_params = self._load_mano_params()
    
    def _load_mano_params(self) -> List[Dict]:
        """加载MANO参数"""
        with open(self.mano_json_path, 'r') as f:
            params = json.load(f)
        return params
    
    def extract_3d_joints_from_mano(self, mano_param: List[float]) -> np.ndarray:
        """
        从MANO参数提取3D关节坐标
        注意：这需要完整的MANO模型，但FreiHAND通常直接提供3D关节
        
        对于FreiHAND，建议直接使用提供的3D关节坐标
        """
        # 这里假设输入已经是21个3D关节坐标
        # 如果是MANO参数(θ, β)，需要完整的MANO前向运动学
        joints = np.array(mano_param).reshape(21, 3)
        return joints
    
    def process_single_frame(self, joints_3d: np.ndarray) -> Dict:
        """
        处理单帧数据
        
        Args:
            joints_3d: 21个3D关节坐标 (21, 3)
            
        Returns:
            处理结果字典
        """
        if self.use_real_fist:
            # 需要提供真实的握拳姿态
            raise NotImplementedError("Real fist pose not implemented")
        else:
            # 使用估算方法
            result = self.processor.process_hand_frame(
                joints_3d,
                lmin_method='estimate',
                fist_ratio=self.fist_ratio
            )
        
        return {
            'lmax': result['lmax'].tolist(),
            'lmin': result['lmin'].tolist(),
            'current_distances': result['current_distances'].tolist(),
            'normalized_sensor_values': result['normalized_sensor_values'].tolist()
        }
    
    def process_dataset(self, 
                      joints_3d_list: List[np.ndarray],
                      save_intermediate: bool = True) -> List[Dict]:
        """
        处理整个数据集
        
        Args:
            joints_3d_list: 3D关节坐标列表
            save_intermediate: 是否保存中间结果
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i, joints_3d in enumerate(joints_3d_list):
            try:
                result = self.process_single_frame(joints_3d)
                results.append(result)
                
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(joints_3d_list)} frames")
                    
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                # 添加默认值
                results.append({
                    'lmax': [0.0] * 5,
                    'lmin': [0.0] * 5,
                    'current_distances': [0.0] * 5,
                    'normalized_sensor_values': [0.0] * 5
                })
        
        # 保存结果
        if save_intermediate:
            output_path = os.path.join(self.output_dir, 'sensor_data.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return results


def create_sample_freihand_data():
    """创建示例FreiHAND数据用于测试"""
    # 模拟FreiHAND的3D关节数据
    np.random.seed(42)
    num_samples = 100
    joints_3d_list = []
    
    for i in range(num_samples):
        # 创建合理的手部关节位置
        joints = np.random.rand(21, 3) * 0.1
        joints[0] = [0.0, 0.0, 0.0]  # 腕关节在原点
        joints_3d_list.append(joints)
    
    return joints_3d_list


def main():
    """主函数：演示如何使用"""
    print("=== FreiHAND Data Processor ===")
    
    # 创建示例数据
    print("Creating sample data...")
    joints_3d_list = create_sample_freihand_data()
    
    # 处理数据
    print("Processing data...")
    processor = FreiHANDProcessor(
        mano_json_path="dummy.json",  # 占位符
        output_dir="output",
        fist_ratio=0.5
    )
    
    results = processor.process_dataset(joints_3d_list)
    
    print(f"Processed {len(results)} frames")
    print("Sample result:")
    print(f"  L_max: {results[0]['lmax']}")
    print(f"  Normalized: {results[0]['normalized_sensor_values']}")
    
    print("\n=== Processing completed! ===")


if __name__ == "__main__":
    main()