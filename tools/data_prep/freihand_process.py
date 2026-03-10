"""
FreiHAND 数据集扩展脚本
======================
从 FreiHAND 的 3D 关节标注中生成五指归一化距离标注，
输出 training_finger_distances.json 作为扩展数据。

用法:
    python freihand_process.py [--freihand_dir PATH] [--split both] [--fist_ratio 0.45]
                              [--viz_samples 10] [--viz_output PATH]

输出:
    FreiHAND_pub_v2/training_finger_distances.json  -- 训练集五指标注
    FreiHAND_pub_v2/evaluation_finger_distances.json -- 评估集五指标注
    (可选) viz_output/ -- 可视化抽样示例
"""

import json
import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from mano_processor.core import MANOHandProcessor


# ======================== FreiHAND 关节定义 ========================
# 标准 MANO 21 关节:
#   0: Wrist
#   1-4:  Thumb  (CMC, MCP, IP, Tip)
#   5-8:  Index  (MCP, PIP, DIP, Tip)
#   9-12: Middle (MCP, PIP, DIP, Tip)
#  13-16: Ring   (MCP, PIP, DIP, Tip)
#  17-20: Pinky  (MCP, PIP, DIP, Tip)

FINGERTIP_INDICES = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_COLORS = {
    'thumb':  (255, 50, 50),
    'index':  (50, 255, 50),
    'middle': (50, 50, 255),
    'ring':   (255, 255, 50),
    'pinky':  (255, 50, 255),
}
FINGER_CHAINS = {
    'thumb':  [0, 1, 2, 3, 4],
    'index':  [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring':   [0, 13, 14, 15, 16],
    'pinky':  [0, 17, 18, 19, 20],
}


# ======================== 数据加载 ========================

def load_freihand_annotations(freihand_dir: str, prefix: str = 'training') -> dict:
    """
    加载 FreiHAND 各种标注文件

    Returns:
        dict with keys: 'xyz', 'mano', 'K', 'scale'
    """
    annotations = {}

    files = {
        'xyz':   f'{prefix}_xyz.json',
        'mano':  f'{prefix}_mano.json',
        'K':     f'{prefix}_K.json',
        'scale': f'{prefix}_scale.json',
    }

    for key, filename in files.items():
        path = os.path.join(freihand_dir, filename)
        if os.path.exists(path):
            print(f"  Loading {filename}...", end='', flush=True)
            with open(path, 'r') as f:
                annotations[key] = json.load(f)
            print(f" ({len(annotations[key])} items)")
        else:
            print(f"  Warning: {filename} not found, skipping")
            annotations[key] = None

    return annotations


# ======================== 五指距离计算 ========================

def compute_all_finger_distances(
    xyz_data: List,
    processor: MANOHandProcessor,
    fist_ratio: float = 0.45,
) -> List[Dict]:
    """
    对全量 FreiHAND 3D 关节数据计算五指归一化距离。

    每个样本输出:
    {
        "normalized": [5 floats],      # 归一化距离 (0=握拳, 1=张开)
        "distances":  [5 floats],      # 当前欧氏距离 (指尖->腕)
        "lmax":       [5 floats],      # 最大距离 (完全伸展, 由骨长之和得到)
        "lmin":       [5 floats],      # 最小距离 (由 fist_ratio 估算)
    }
    """
    n = len(xyz_data)
    results = []
    errors = 0
    t0 = time.time()

    for i in range(n):
        try:
            joints = np.array(xyz_data[i], dtype=np.float64)
            result = processor.process_hand_frame(
                joints, lmin_method='estimate', fist_ratio=fist_ratio
            )
            results.append({
                'normalized': result['normalized_sensor_values'].tolist(),
                'distances':  result['current_distances'].tolist(),
                'lmax':       result['lmax'].tolist(),
                'lmin':       result['lmin'].tolist(),
            })
        except Exception as e:
            errors += 1
            results.append({
                'normalized': [0.0] * 5,
                'distances':  [0.0] * 5,
                'lmax':       [0.0] * 5,
                'lmin':       [0.0] * 5,
                'error':      str(e),
            })

        if (i + 1) % 5000 == 0 or i == n - 1:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (n - i - 1) / speed if speed > 0 else 0
            print(f"  [{i+1:6d}/{n}]  {speed:.0f} samples/s  "
                  f"ETA: {eta:.1f}s  errors: {errors}")

    return results


def compute_global_statistics(results: List[Dict]) -> Dict:
    """计算全局统计信息"""
    all_norm = np.array([r['normalized'] for r in results if 'error' not in r])
    all_dist = np.array([r['distances'] for r in results if 'error' not in r])
    all_lmax = np.array([r['lmax'] for r in results if 'error' not in r])

    stats = {}
    for i, name in enumerate(FINGER_NAMES):
        stats[name] = {
            'norm_mean': float(all_norm[:, i].mean()),
            'norm_std':  float(all_norm[:, i].std()),
            'norm_min':  float(all_norm[:, i].min()),
            'norm_max':  float(all_norm[:, i].max()),
            'dist_mean': float(all_dist[:, i].mean()),
            'lmax_mean': float(all_lmax[:, i].mean()),
        }

    stats['total_samples'] = len(results)
    stats['error_count'] = sum(1 for r in results if 'error' in r)
    return stats


# ======================== 可视化 (可选) ========================

def project_3d_to_2d(joints_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """将3D关节坐标投影到2D图像平面"""
    z = joints_3d[:, 2:3]
    z = np.where(z < 1e-6, 1e-6, z)
    xy = joints_3d[:, :2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = xy[:, 0] * fx + cx
    v = xy[:, 1] * fy + cy
    return np.stack([u, v], axis=1)


def visualize_samples(
    freihand_dir: str,
    xyz_data: List,
    K_data: List,
    results: List[Dict],
    output_dir: str,
    sample_indices: Optional[List[int]] = None,
    num_random: int = 10,
):
    """
    抽样可视化，在图片上绘制骨架和五指归一化值。
    不修改原始训练图片——输出到独立目录。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  Warning: PIL not available, skipping visualization")
        return

    os.makedirs(output_dir, exist_ok=True)
    n = len(xyz_data)

    if sample_indices is None:
        rng = np.random.RandomState(42)
        sample_indices = sorted(rng.choice(n, size=min(num_random, n), replace=False))

    # Identify if processing training or evaluation for the RGB path
    if 'eval' in output_dir or 'evaluation' in output_dir:
        rgb_dir = os.path.join(freihand_dir, "evaluation", "rgb")
    else:
        rgb_dir = os.path.join(freihand_dir, "training", "rgb")

    for idx in sample_indices:
        img_path = os.path.join(rgb_dir, f"{idx:08d}.jpg")
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        joints_3d = np.array(xyz_data[idx])
        K = np.array(K_data[idx])
        joints_2d = project_3d_to_2d(joints_3d, K)
        normalized = results[idx]['normalized']

        # 绘制骨架 (轻微半透明线条)
        for finger_name, chain in FINGER_CHAINS.items():
            color = FINGER_COLORS[finger_name]
            for j in range(len(chain) - 1):
                p1 = tuple(joints_2d[chain[j]].astype(int))
                p2 = tuple(joints_2d[chain[j+1]].astype(int))
                draw.line([p1, p2], fill=color, width=2)

        # 绘制关节点
        for j in range(21):
            x, y = joints_2d[j]
            r = 3
            draw.ellipse([x-r, y-r, x+r, y+r],
                         fill=(255, 255, 255), outline=(0, 0, 0))

        # 在指尖标注归一化值
        for i, finger in enumerate(FINGER_NAMES):
            tip_idx = FINGERTIP_INDICES[finger]
            tx, ty = joints_2d[tip_idx]
            label = f"{normalized[i]:.2f}"
            color = FINGER_COLORS[finger]
            bbox = draw.textbbox((tx + 8, ty - 8), label)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2],
                           fill=(0, 0, 0))
            draw.text((tx + 8, ty - 8), label, fill=color)

        # 底部信息面板
        panel_h = 100
        out_img = Image.new('RGB', (img.width, img.height + panel_h), (30, 30, 30))
        out_img.paste(img, (0, 0))
        info_draw = ImageDraw.Draw(out_img)

        y_off = img.height + 5
        info_draw.text((10, y_off), f"Sample #{idx}", fill=(255, 255, 255))
        y_off += 18
        for i, finger in enumerate(FINGER_NAMES):
            dist = results[idx]['distances'][i]
            lmax = results[idx]['lmax'][i]
            text = f"{finger:7s}: norm={normalized[i]:.3f}  dist={dist:.4f}  lmax={lmax:.4f}"
            info_draw.text((10, y_off), text, fill=FINGER_COLORS[finger])
            y_off += 14

        out_img.save(os.path.join(output_dir, f"viz_{idx:05d}.png"))

    print(f"  Saved {len(sample_indices)} visualizations to {output_dir}/")


# ======================== 主函数 ========================

def process_split(split_name: str, args):
    freihand_dir = args.freihand_dir
    print("=" * 60)
    print(f"  Processing split: {split_name.upper()}")
    print("=" * 60)

    # 1. 加载标注
    print(f"[1/4] Loading {split_name} annotations...")
    annot = load_freihand_annotations(freihand_dir, prefix=split_name)

    if annot['xyz'] is None:
        print(f"ERROR: {split_name}_xyz.json not found. Cannot proceed.")
        return

    n = len(annot['xyz'])
    print(f"  Total samples: {n}")
    print()

    # 2. 计算五指距离
    print("[2/4] Computing 5-finger distances for all samples...")
    processor = MANOHandProcessor()
    t0 = time.time()
    results = compute_all_finger_distances(annot['xyz'], processor, args.fist_ratio)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({n/elapsed:.0f} samples/s)")
    print()

    # 3. 保存结果
    print("[3/4] Saving extended annotations...")
    output_path = os.path.join(freihand_dir, f'{split_name}_finger_distances.json')
    with open(output_path, 'w') as f:
        json.dump(results, f)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size:.1f} MB")

    # 统计
    stats = compute_global_statistics(results)
    stats_path = os.path.join(freihand_dir, f'{split_name}_finger_distances_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")
    print()

    print("  Global statistics:")
    print(f"  {'finger':>8s}  {'mean':>6s}  {'std':>6s}  {'min':>6s}  {'max':>6s}")
    print(f"  {'-'*40}")
    for name in FINGER_NAMES:
        s = stats[name]
        print(f"  {name:>8s}  {s['norm_mean']:.3f}  {s['norm_std']:.3f}  "
              f"{s['norm_min']:.3f}  {s['norm_max']:.3f}")
    print(f"  Errors: {stats['error_count']}/{stats['total_samples']}")
    print()

    # 4. 可视化
    if args.viz_samples > 0 and annot['K'] is not None:
        print(f"[4/4] Generating {args.viz_samples} visualization samples...")
        split_viz_output = os.path.join(args.viz_output, split_name)
        visualize_samples(
            freihand_dir, annot['xyz'], annot['K'],
            results, split_viz_output, num_random=args.viz_samples
        )
    else:
        print("[4/4] Skipping visualization.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="FreiHAND 数据集扩展：生成五指归一化距离标注"
    )
    parser.add_argument(
        '--freihand_dir', type=str,
        default=os.path.join(os.path.dirname(__file__), 'FreiHAND_pub_v2'),
        help='FreiHAND 数据集根目录'
    )
    parser.add_argument(
        '--split', type=str, choices=['train', 'eval', 'both'], default='both',
        help='选择处理的子集: train(处理training_*), eval(处理evaluation_*), both(默认)'
    )
    parser.add_argument(
        '--fist_ratio', type=float, default=0.45,
        help='握拳时指尖-腕距离与完全伸展的比例 (L_min = ratio * L_max)'
    )
    parser.add_argument(
        '--viz_samples', type=int, default=20,
        help='可视化抽样数量 (0=不可视化)'
    )
    parser.add_argument(
        '--viz_output', type=str, default=None,
        help='可视化输出目录 (默认: freihand_dir 同级的 viz_output/)'
    )
    args = parser.parse_args()

    freihand_dir = args.freihand_dir
    if args.viz_output is None:
        args.viz_output = os.path.join(os.path.dirname(freihand_dir), 'viz_output')

    print("=" * 60)
    print("  FreiHAND Dataset Extension: 5-Finger Distance Annotation")
    print("=" * 60)
    print(f"  FreiHAND dir:  {freihand_dir}")
    print(f"  Split:         {args.split}")
    print(f"  Fist ratio:    {args.fist_ratio}")
    print(f"  Viz samples:   {args.viz_samples}")
    print()

    if args.split in ['train', 'both']:
        process_split('training', args)
        
    if args.split in ['eval', 'both']:
        process_split('evaluation', args)

    print("=" * 60)
    print("  Extension complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
