"""
FreiHAND NPZ 导出脚本
=====================
将 FreiHAND 数据集及其扩展的五指距离标注打包成 .npz 文件，供 STMF 模型使用。

用法:
    python tools/data_prep/freihand_npz_exporter.py \
        --freihand_dir /path/to/FreiHAND_pub_v2 \
        --split both

参数说明:
    --freihand_dir: FreiHAND 根目录。导出前请确保已运行 freihand_process.py 生成了 
                    [split]_finger_distances.json。
    --split:        导出选项: [training, evaluation, both]，默认 both
    --output_dir:   NPZ 输出目录，默认保存在 freihand_dir 下。

输出结果:
    - freihand_training.npz: 训练集打包数据，包含图像路径、关键点、MANO 参数以及 'sensor' 归一化距离。
    - freihand_evaluation.npz: 评估集打包数据。
"""
import os
import json
import numpy as np
import argparse
from tqdm import tqdm

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
    """
    min_x = np.min(keypoints_2d[:, 0])
    max_x = np.max(keypoints_2d[:, 0])
    min_y = np.min(keypoints_2d[:, 1])
    max_y = np.max(keypoints_2d[:, 1])

    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    size = max(max_x - min_x, max_y - min_y)
    size = size * (1.0 + padding)
    #scale = size / 200.0
    return center, size

def export_freihand_split(freihand_dir, split_name, output_dir):
    print(f"Exporting FreiHAND {split_name} split to NPZ...")
    
    # Load JSON annotations
    try:
        with open(os.path.join(freihand_dir, f'{split_name}_xyz.json'), 'r') as f:
            xyz = json.load(f)
        with open(os.path.join(freihand_dir, f'{split_name}_K.json'), 'r') as f:
            K_data = json.load(f)
        # Evaluation might not have mano.json
        mano = None
        mano_path = os.path.join(freihand_dir, f'{split_name}_mano.json')
        if os.path.exists(mano_path):
            with open(mano_path, 'r') as f:
                mano = json.load(f)
        
        # Load finger distances if they exist
        finger_distances = None
        fd_path = os.path.join(freihand_dir, f'{split_name}_finger_distances.json')
        if os.path.exists(fd_path):
            print(f"  Loading finger distances from {fd_path}...")
            with open(fd_path, 'r') as f:
                finger_distances = json.load(f)
    except FileNotFoundError as e:
        print(f"Could not load annotations for {split_name}: {e}")
        return

    n_base = len(xyz)
    print(f"  Base {split_name} samples: {n_base}")

    # FreiHAND training has 4 sets of backgrounds (green screen + 3 natural backgrounds) => 4 * 32560 = 130240 images
    # FreiHAND evaluation only has 1 set of images (no background variations) => 3960 images
    # We rely on the physical image count in the directory to find the multiplier
    rgb_dir = os.path.join(freihand_dir, split_name if split_name != 'training' else 'training/rgb')
    if split_name == 'evaluation':
        rgb_dir = os.path.join(freihand_dir, 'evaluation/rgb')
    
    # Quick fix for FreiHAND's nested structure
    if not os.path.exists(rgb_dir):
        alt_rgb = os.path.join(freihand_dir, f'{split_name}') 
        if os.path.exists(os.path.join(alt_rgb, '00000000.jpg')):
            rgb_dir = alt_rgb

    try:
        img_count = len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
        multiplier = max(1, img_count // n_base)
        print(f"  Found {img_count} images. Multiplier = {multiplier}.")
    except Exception as e:
        print(f"  Could not count images: {e}. Assuming multiplier = 1.")
        multiplier = 1

    npz_data = {
        'imgname': [], 'center': [], 'scale': [],
        'hand_pose': [], 'betas': [],
        'has_hand_pose': [], 'has_betas': [], 'right': [],
        'hand_keypoints_2d': [], 'hand_keypoints_3d': [], 'personid': [],
        'sensor': [] # Added sensor field
    }

    person_id_counter = 0

    for m in range(multiplier):
        for idx in tqdm(range(n_base), desc=f"Pass {m+1}/{multiplier}"):
            # 1. Image name
            actual_idx = idx + (m * n_base)
            
            # Form relative path similar to how dataset is saved
            folder = 'training/rgb' if split_name == 'training' else 'evaluation/rgb'
            imgname = f"{folder}/{actual_idx:08d}.jpg"
            if not os.path.exists(os.path.join(freihand_dir, imgname)):
                # Adjust for simple folder structure if needed
                folder = split_name
                imgname = f"{folder}/{actual_idx:08d}.jpg"
            
            # 2. Keypoints 3D and Camera
            joints_3d = np.array(xyz[idx], dtype=np.float32) # (21, 3)
            cam_mat = np.array(K_data[idx], dtype=np.float32) # (3, 3)

            # FreiHAND uses OpenGL coords natively, project expecting OpenGL coordinates = False
            try:
                # K * XYZ
                pts_2d = (cam_mat @ joints_3d.T).T
                pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
            except:
                pts_2d = project_3D_points(cam_mat, joints_3d, is_OpenGL_coords=False)

            center, scale = compute_bbox(pts_2d, padding=0.25)
            
            kps_2d = np.concatenate([pts_2d, np.ones((21, 1))], axis=1) # 21x3
            kps_3d = np.concatenate([joints_3d, np.ones((21, 1))], axis=1) # 21x4

            # 3. MANO params
            has_pose = 0.0
            has_betas = 0.0
            hand_pose = np.zeros(48, dtype=np.float32)
            betas = np.zeros(10, dtype=np.float32)

            if mano is not None and len(mano) > idx:
                has_pose = 1.0
                has_betas = 1.0
                
                mano_params = np.array(mano[idx], dtype=np.float32)
                if len(mano_params) >= 58:
                    hand_pose = mano_params[:48]
                    betas = mano_params[48:58]
            
            # 4. Finger Distances
            if finger_distances is not None and len(finger_distances) > idx:
                # Use pre-computed normalized values
                sensor_res = np.array(finger_distances[idx]['normalized'], dtype=np.float32)
            else:
                sensor_res = np.zeros(5, dtype=np.float32)

            # Sequence enforcement
            personid = person_id_counter
            person_id_counter += 1

            npz_data['imgname'].append(imgname)
            npz_data['center'].append(center)
            npz_data['scale'].append(np.array([scale]))
            npz_data['hand_pose'].append(hand_pose)
            npz_data['betas'].append(betas)
            npz_data['has_hand_pose'].append(has_pose)
            npz_data['has_betas'].append(has_betas)
            npz_data['right'].append(1.0) # FreiHAND is 100% right hand
            npz_data['hand_keypoints_2d'].append(kps_2d)
            npz_data['hand_keypoints_3d'].append(kps_3d)
            npz_data['personid'].append(personid)
            npz_data['sensor'].append(sensor_res)

    # Save to NPZ
    os.makedirs(output_dir, exist_ok=True)
    out_name = f'freihand_{split_name}.npz'
    npz_out_path = os.path.join(output_dir, out_name)
    
    final_npz = {}
    for k, v in npz_data.items():
        if k == 'imgname':
            final_npz[k] = np.array(v)
        else:
            final_npz[k] = np.stack(v).astype(np.float32 if k != 'personid' else np.int32)
            
    np.savez(npz_out_path, **final_npz)
    print(f"Saved NPZ to {npz_out_path} ({len(npz_data['imgname'])} items)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freihand_dir', type=str, default='/home/mirage/STMF/_DATA/FreiHAND_pub_v2')
    parser.add_argument('--split', type=str, choices=['training', 'evaluation', 'both'], default='both')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.freihand_dir

    if args.split in ['training', 'both']:
        export_freihand_split(args.freihand_dir, 'training', args.output_dir)
    if args.split in ['evaluation', 'both']:
        export_freihand_split(args.freihand_dir, 'evaluation', args.output_dir)

