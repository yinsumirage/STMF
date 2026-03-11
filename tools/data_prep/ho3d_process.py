"""
HO-3D 数据集预处理脚本
======================
处理 HO-3D 数据集的训练和评估 split，计算五指距离并提取关键点、相机参数等，
最终导出为 .npz 文件供模型 DataLoader 使用。

用法:
    python tools/data_prep/ho3d_process.py \
        --base_dir /path/to/HO-3D_v3 \
        --split both \
        --output_dir /optional/output/dir

参数说明:
    --base_dir:   HO-3D 数据集根目录。
                  注意对于 evaluation split，此目录下必须包含 evaluation_xyz.json 和 evaluation.txt。
    --split:      处理子集，选项: [training, evaluation, both]，默认 both
    --output_dir: 结果保存目录，默认保存在 base_dir 下。

输出结果:
    - ho3d_train.npz: 训练集数据，包含 'sensor' 键 (五指归一化距离)。
    - ho3d_evaluate.npz: 评估集数据，包含 'sensor' 键。
"""
import os
import glob
import json
import pickle
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mano_processor.core import MANOHandProcessor

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]], axis=1)
    return proj_pts

def compute_bbox(keypoints_2d, padding=0.25):
    """
    Computes a square bounding box given 2d keypoints.
    Returns (center_x, center_y), scale
    scale is defined such that bounding_box_size = scale * 200
    """
    min_x = np.min(keypoints_2d[:, 0])
    max_x = np.max(keypoints_2d[:, 0])
    min_y = np.min(keypoints_2d[:, 1])
    max_y = np.max(keypoints_2d[:, 1])

    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    size = max(max_x - min_x, max_y - min_y)
    size = size * (1.0 + padding)
    scale = size / 200.0
    return center, scale

def process_ho3d_split(base_dir, split_name, output_dir, fist_ratio=0.45):
    print(f"Processing HO-3D {split_name} split...")
    split_dir = os.path.join(base_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"Error: {split_dir} does not exist.")
        return

    sequences = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    npz_data = {
        'imgname': [], 'center': [], 'scale': [],
        'hand_pose': [], 'betas': [],
        'has_hand_pose': [], 'has_betas': [], 'right': [],
        'keypoints_2d': [], 'keypoints_3d': [], 'personid': [],
        'sensor': [] # Added sensor field
    }
    
    processor = MANOHandProcessor()
    
    # --- EVALUATION DATA LOADING ---
    eval_joints_map = {}
    is_eval = split_name == 'evaluate'
    if is_eval:
        xyz_json = os.path.join(base_dir, 'evaluation_xyz.json')
        txt_path = os.path.join(base_dir, 'evaluation.txt')
        if os.path.exists(xyz_json) and os.path.exists(txt_path):
            print("  Loading evaluation joints...")
            with open(xyz_json, 'r') as f:
                all_joints = json.load(f)
            with open(txt_path, 'r') as f:
                all_paths = [line.strip() for line in f.readlines()]
            
            for path, joints in zip(all_paths, all_joints):
                # path is like "SM1/0000"
                eval_joints_map[path] = np.array(joints, dtype=np.float32)
            print(f"  Loaded {len(eval_joints_map)} evaluation joints.")
    
    person_id_map = {}
    valid_count = 0
    error_count = 0
    
    for seq_idx, seq in enumerate(sequences):
        person_id_map[seq] = seq_idx
        seq_dir = os.path.join(split_dir, seq)
        meta_dir = os.path.join(seq_dir, 'meta')
        rgb_dir = os.path.join(seq_dir, 'rgb')
        
        if not os.path.exists(meta_dir) or not os.path.exists(rgb_dir):
            continue
            
        pkl_files = sorted(glob.glob(os.path.join(meta_dir, '*.pkl')))
        print(f"  Sequence {seq}: {len(pkl_files)} frames")
        
        for pkl_file in tqdm(pkl_files, leave=False):
            frame_id = os.path.splitext(os.path.basename(pkl_file))[0]
            
            # Read imgname relative path
            img_rel_path = f"{split_name}/{seq}/rgb/{frame_id}.jpg"
            if not os.path.exists(os.path.join(base_dir, img_rel_path)):
                img_rel_path = f"{split_name}/{seq}/rgb/{frame_id}.png"
                if not os.path.exists(os.path.join(base_dir, img_rel_path)):
                    continue

            try:
                with open(pkl_file, 'rb') as f:
                    anno = pickle.load(f, encoding='latin1')
            except Exception as e:
                error_count += 1
                continue

            # Need 3D joints and poses
            
            # --- EVALUATION SPLIT PROCESSING ---
            if is_eval:
                cam_mat = anno['camMat']
                
                # Try to get real joints from the map
                frame_path = f"{seq}/{frame_id}"
                if frame_path in eval_joints_map:
                    joints_3d = eval_joints_map[frame_path]
                else:
                    # Fallback or error
                    joints_3d = None

                if joints_3d is not None:
                    # Project to 2D
                    pts_2d = project_3D_points(cam_mat, joints_3d, is_OpenGL_coords=True)
                    # Build kps
                    kps_2d = np.concatenate([pts_2d, np.ones((21, 1))], axis=1)
                    kps_3d = np.concatenate([joints_3d, np.ones((21, 1))], axis=1)
                else:
                    # Zero out GT requirements if not found
                    kps_2d = np.zeros((21, 3))
                    kps_3d = np.zeros((21, 4))

                # Bounding box logic
                if 'handBoundingBox' in anno and anno['handBoundingBox'] is not None:
                    bbox = anno['handBoundingBox']
                    # [u1, v1, u2, v2] -> (min_x, min_y, max_x, max_y)
                    center_x = (bbox[0] + bbox[2]) / 2.0
                    center_y = (bbox[1] + bbox[3]) / 2.0
                    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.25
                    center = np.array([center_x, center_y])
                    scale = size / 200.0
                elif joints_3d is not None:
                    center, scale = compute_bbox(pts_2d, padding=0.25)
                else:
                    center_x, center_y = 320.0, 240.0
                    center = np.array([center_x, center_y])
                    scale = 2.0

                hand_pose = np.zeros(48, dtype=np.float32)
                betas = np.zeros(10, dtype=np.float32)
                has_pose = 0.0
                has_betas = 0.0
                
                # Compute Sensor Results for Eval
                if joints_3d is not None:
                    try:
                        result = processor.process_hand_frame(
                            joints_3d, lmin_method='estimate', fist_ratio=fist_ratio
                        )
                        sensor_res = result['normalized_sensor_values'].astype(np.float32)
                    except Exception as e:
                        sensor_res = np.zeros(5, dtype=np.float32)
                else:
                    sensor_res = np.zeros(5, dtype=np.float32)

            # --- TRAIN SPLIT PROCESSING ---
            else:
                if 'handJoints3D' not in anno or anno['handJoints3D'] is None:
                    continue
                    
                joints_3d = anno['handJoints3D'] # (21, 3)
                if joints_3d.shape[0] != 21:
                    continue

                cam_mat = anno['camMat']

                # Project to 2D
                pts_2d = project_3D_points(cam_mat, joints_3d, is_OpenGL_coords=True)
                
                # Calculate bbox
                center, scale = compute_bbox(pts_2d, padding=0.25)

                # Build 21x3 (x,y,conf) and 21x4 (x,y,z,conf)
                kps_2d = np.concatenate([pts_2d, np.ones((21, 1))], axis=1) # 21x3
                kps_3d = np.concatenate([joints_3d, np.ones((21, 1))], axis=1) # 21x4
                
                # MANO parameters
                has_pose = 1.0 if 'handPose' in anno and anno['handPose'] is not None else 0.0
                has_betas = 1.0 if 'handBeta' in anno and anno['handBeta'] is not None else 0.0
                
                if has_pose:
                    # global_orient (3) + hand_pose (45)
                    hand_pose = anno['handPose'].flatten()
                    if hand_pose.shape[0] != 48:
                        has_pose = 0.0
                        hand_pose = np.zeros(48, dtype=np.float32)
                else:
                    hand_pose = np.zeros(48, dtype=np.float32)
                    
                if has_betas:
                    betas = anno['handBeta'].flatten()
                else:
                    betas = np.zeros(10, dtype=np.float32)

                # Compute Finger Distances
                try:
                    result = processor.process_hand_frame(
                        joints_3d, lmin_method='estimate', fist_ratio=fist_ratio
                    )
                    sensor_res = result['normalized_sensor_values'].astype(np.float32)
                except Exception as e:
                    error_count += 1
                    continue

            # Append to lists
            npz_data['imgname'].append(img_rel_path)
            npz_data['center'].append(center)
            npz_data['scale'].append(np.array([scale]))
            npz_data['hand_pose'].append(hand_pose)
            npz_data['betas'].append(betas)
            npz_data['has_hand_pose'].append(has_pose)
            npz_data['has_betas'].append(has_betas)
            npz_data['right'].append(1.0)  # HO-3D is right hand
            npz_data['keypoints_2d'].append(kps_2d)
            npz_data['keypoints_3d'].append(kps_3d)
            npz_data['personid'].append(seq_idx)
            npz_data['sensor'].append(sensor_res)
            
            valid_count += 1

    print(f"Processed {valid_count} valid frames (Errors: {error_count})")
    
    # Save NPZ
    os.makedirs(output_dir, exist_ok=True)
    npz_out_path = os.path.join(output_dir, f'ho3d_{split_name}.npz')
    
    # Convert lists to numpy arrays
    final_npz = {}
    for k, v in npz_data.items():
        if k == 'imgname':
            final_npz[k] = np.array(v)
        else:
            final_npz[k] = np.stack(v).astype(np.float32 if k != 'personid' else np.int32)
            
    np.savez(npz_out_path, **final_npz)
    print(f"Saved npz to {npz_out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/mirage/STMF/_DATA/HO-3D_v3')
    parser.add_argument('--split', type=str, choices=['training', 'evaluation', 'both'], default='both')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.base_dir

    if args.split in ['training', 'both']:
        process_ho3d_split(args.base_dir, 'train', args.output_dir)
    if args.split in ['evaluation', 'both']:
        process_ho3d_split(args.base_dir, 'evaluate', args.output_dir)