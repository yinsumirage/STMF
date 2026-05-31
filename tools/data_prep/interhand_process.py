"""
InterHand2.6M -> STMF NPZ exporter.

Typical usage:

Export train/val single-hand tracks from the 30fps release:

conda run -n STMF python tools/data_prep/interhand_process.py \
  --base_dir /path/to/InterHand2.6M_30fps \
  --split train,val \
  --hand_filter single \
  --output_dir /path/to/InterHand2.6M_30fps

Export single-hand tracks and split interacting frames into per-hand samples:

conda run -n STMF python tools/data_prep/interhand_process.py \
  --base_dir /path/to/InterHand2.6M_30fps \
  --split train \
  --hand_filter split \
  --max_samples 50000

Notes:
- Exports the same NPZ fields used by `TemporalImageDataset`.
- Pseudo sensor values are computed from 3D keypoints and stored in `sensor`.
- MANO parameters are copied from `*_MANO_NeuralAnnot.json` when available.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


INTERHAND_TO_MODEL_ORDER = np.array(
    [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16],
    dtype=np.int64,
)

RIGHT_OFFSET = 0
LEFT_OFFSET = 21


def camera_to_pixel(cam_coord: np.ndarray, focal: np.ndarray, princpt: np.ndarray) -> np.ndarray:
    z = np.clip(cam_coord[:, 2:3], a_min=1e-6, a_max=None)
    xy = cam_coord[:, :2] / z
    return np.stack(
        [
            xy[:, 0] * float(focal[0]) + float(princpt[0]),
            xy[:, 1] * float(focal[1]) + float(princpt[1]),
        ],
        axis=1,
    ).astype(np.float32)


def world_to_camera(world_coord: np.ndarray, campos: np.ndarray, camrot: np.ndarray) -> np.ndarray:
    world_coord = np.asarray(world_coord, dtype=np.float32)
    campos = np.asarray(campos, dtype=np.float32)
    camrot = np.asarray(camrot, dtype=np.float32)
    return ((camrot @ (world_coord - campos).T).T).astype(np.float32)


def compute_bbox(keypoints_2d: np.ndarray, conf: np.ndarray, padding: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    valid = conf > 0.5
    if valid.sum() < 4:
        return np.array([128.0, 128.0], dtype=np.float32), np.array([256.0, 256.0], dtype=np.float32)

    xy = keypoints_2d[valid, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    size = np.maximum(max_xy - min_xy, 1.0)
    center = (min_xy + max_xy) * 0.5
    scale = size * (1.0 + float(padding) * 2.0)
    return center.astype(np.float32), scale.astype(np.float32)


def compute_pseudo_sensor_from_model_joints(joints: np.ndarray, fist_ratio: float = 0.5) -> np.ndarray:
    from hamer.utils.sensor_utils import compute_pseudo_sensor_from_model_joints as compute_sensor

    return compute_sensor(joints, fist_ratio=fist_ratio)


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='InterHand2.6M root containing images/ and annotations/')
    parser.add_argument('--split', type=str, default='train,val', help='Comma separated splits, e.g. train,val')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for NPZ files')
    parser.add_argument('--hand_filter', type=str, choices=['single', 'split'], default='single', help='single: keep right/left only; split: also split interacting samples into per-hand exports')
    parser.add_argument('--fist_ratio', type=float, default=0.5, help='Pseudo-sensor fist ratio')
    parser.add_argument('--padding', type=float, default=0.25, help='BBox padding factor')
    parser.add_argument('--min_visible_joints', type=int, default=4, help='Minimum number of image-visible joints to keep a hand sample')
    parser.add_argument('--max_samples', type=int, default=0, help='Optional cap per split; 0 means all')
    return parser.parse_args()


def get_annotation_paths(base_dir: str, split: str) -> Dict[str, str]:
    anno_dir = os.path.join(base_dir, 'annotations', split)
    return {
        'data': os.path.join(anno_dir, f'InterHand2.6M_{split}_data.json'),
        'camera': os.path.join(anno_dir, f'InterHand2.6M_{split}_camera.json'),
        'joint_3d': os.path.join(anno_dir, f'InterHand2.6M_{split}_joint_3d.json'),
        'mano': os.path.join(anno_dir, f'InterHand2.6M_{split}_MANO_NeuralAnnot.json'),
    }


def normalize_valid_mask(valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid)
    if valid.ndim > 1:
        valid = valid.any(axis=-1)
    return valid.astype(np.float32).reshape(-1)


def hand_candidates(hand_type: str, hand_filter: str) -> Iterable[str]:
    if hand_type in ('right', 'left'):
        yield hand_type
        return
    if hand_type == 'interacting' and hand_filter == 'split':
        yield 'right'
        yield 'left'


def extract_hand_slice(array_42: np.ndarray, hand_side: str) -> np.ndarray:
    start = RIGHT_OFFSET if hand_side == 'right' else LEFT_OFFSET
    return array_42[start:start + 21]


def reorder_interhand_hand(joints_or_conf: np.ndarray) -> np.ndarray:
    return np.asarray(joints_or_conf)[INTERHAND_TO_MODEL_ORDER]


def extract_mano_params(
    mano_frame: Optional[Dict],
    hand_side: str,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if mano_frame is None or hand_side not in mano_frame or mano_frame[hand_side] is None:
        return (
            np.zeros(48, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
            0.0,
            0.0,
        )

    mano_info = mano_frame[hand_side]
    pose = np.asarray(mano_info.get('pose', []), dtype=np.float32).reshape(-1)
    betas = np.asarray(mano_info.get('shape', []), dtype=np.float32).reshape(-1)
    if pose.shape[0] != 48:
        pose = np.zeros(48, dtype=np.float32)
        has_pose = 0.0
    else:
        has_pose = 1.0
    if betas.shape[0] != 10:
        betas = np.zeros(10, dtype=np.float32)
        has_betas = 0.0
    else:
        has_betas = 1.0
    return pose, betas, has_pose, has_betas


def process_split(args: argparse.Namespace, split: str) -> str:
    paths = get_annotation_paths(args.base_dir, split)
    for key, path in paths.items():
        if key == 'mano':
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required {key} annotation: {path}")

    print(f"Loading InterHand2.6M {split} annotations...")
    data_json = load_json(paths['data'])
    camera_json = load_json(paths['camera'])
    joint_json = load_json(paths['joint_3d'])
    mano_json = load_json(paths['mano']) if os.path.exists(paths['mano']) else {}

    images_by_id = {int(item['id']): item for item in data_json['images']}
    records = []

    iterator = tqdm(data_json['annotations'], desc=f"{split}", leave=False)
    for ann in iterator:
        image_info = images_by_id[int(ann['image_id'])]
        capture = str(image_info['capture'])
        camera_name = str(image_info['camera'])
        frame_idx = str(image_info['frame_idx'])

        camera_info = camera_json[capture]
        focal = np.asarray(camera_info['focal'][camera_name], dtype=np.float32)
        princpt = np.asarray(camera_info['princpt'][camera_name], dtype=np.float32)
        campos = np.asarray(camera_info['campos'][camera_name], dtype=np.float32)
        camrot = np.asarray(camera_info['camrot'][camera_name], dtype=np.float32)

        world_info = joint_json[capture][frame_idx]
        world_coord = np.asarray(world_info['world_coord'], dtype=np.float32).reshape(-1, 3)
        cam_coord = world_to_camera(world_coord, campos=campos, camrot=camrot)

        image_valid = normalize_valid_mask(ann['joint_valid'])
        world_valid = normalize_valid_mask(world_info['joint_valid'])
        mano_frame = mano_json.get(capture, {}).get(frame_idx)

        for hand_side in hand_candidates(str(ann['hand_type']), args.hand_filter):
            hand_cam_coord = extract_hand_slice(cam_coord, hand_side) / 1000.0
            hand_image_valid = extract_hand_slice(image_valid, hand_side)
            hand_world_valid = extract_hand_slice(world_valid, hand_side)
            if int((hand_image_valid > 0.5).sum()) < int(args.min_visible_joints):
                continue

            hand_cam_coord = reorder_interhand_hand(hand_cam_coord)
            hand_image_valid = reorder_interhand_hand(hand_image_valid)
            hand_world_valid = reorder_interhand_hand(hand_world_valid)

            keypoints_2d_xy = camera_to_pixel(hand_cam_coord * 1000.0, focal=focal, princpt=princpt)
            center, scale = compute_bbox(keypoints_2d_xy, hand_image_valid, padding=args.padding)

            keypoints_2d = np.concatenate([keypoints_2d_xy, hand_image_valid[:, None]], axis=1).astype(np.float32)
            keypoints_3d = np.concatenate([hand_cam_coord, hand_world_valid[:, None]], axis=1).astype(np.float32)
            sensor = compute_pseudo_sensor_from_model_joints(hand_cam_coord, fist_ratio=args.fist_ratio).astype(np.float32)
            hand_pose, betas, has_pose, has_betas = extract_mano_params(mano_frame, hand_side)

            sequence_key = f"Capture{capture}/{image_info['seq_name']}/{camera_name}_{hand_side}"
            frame_order = int(image_info['frame_idx'])
            img_rel_path = os.path.join('images', split, image_info['file_name']).replace('\\', '/')

            records.append({
                'imgname': img_rel_path,
                'center': center,
                'scale': scale,
                'hand_pose': hand_pose,
                'betas': betas,
                'has_hand_pose': np.float32(has_pose),
                'has_betas': np.float32(has_betas),
                'right': np.float32(hand_side == 'right'),
                'hand_keypoints_2d': keypoints_2d,
                'hand_keypoints_3d': keypoints_3d,
                'sensor': sensor,
                'personid': sequence_key,
                'frame_order': frame_order,
                'extra_info': {
                    'capture': int(capture),
                    'camera': camera_name,
                    'seq_name': image_info['seq_name'],
                    'frame_idx': int(image_info['frame_idx']),
                    'hand_side': hand_side,
                },
            })

        if args.max_samples > 0 and len(records) >= int(args.max_samples):
            break

    records.sort(key=lambda item: (item['personid'], item['frame_order'], item['imgname']))
    personid_map = {name: idx for idx, name in enumerate(dict.fromkeys(item['personid'] for item in records))}

    npz_data = defaultdict(list)
    for item in records:
        for key, value in item.items():
            if key == 'personid':
                npz_data[key].append(personid_map[value])
            elif key == 'frame_order':
                continue
            else:
                npz_data[key].append(value)

    final_npz = {}
    for key, values in npz_data.items():
        if key == 'imgname':
            final_npz[key] = np.asarray(values)
        elif key == 'extra_info':
            final_npz[key] = np.asarray(values, dtype=object)
        elif key == 'personid':
            final_npz[key] = np.asarray(values, dtype=np.int32)
        else:
            final_npz[key] = np.stack(values).astype(np.float32)

    output_dir = args.output_dir or args.base_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'interhand_{split}.npz')
    np.savez(output_path, **final_npz)
    print(f"Saved {len(records)} samples to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    for split in [part.strip() for part in args.split.split(',') if part.strip()]:
        process_split(args, split)


if __name__ == '__main__':
    main()
