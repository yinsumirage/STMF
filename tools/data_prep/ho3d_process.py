"""
HO-3D 数据集预处理脚本
======================
处理 HO-3D 数据集的训练和评估 split，计算五指距离并提取关键点、相机参数等，
最终导出为 .npz 文件供模型 DataLoader 使用。

用法:
python tools/data_prep/ho3d_process.py \
  --base_dir /home/mirage/STMF/_DATA/HO-3D_v3 \
  --split both \
  --bbox_source vitpose \
  --body_detector regnety \
  --output_format both

如果要生成 HaMeR 原生训练 tar:
python tools/data_prep/ho3d_process.py \
  --base_dir /data/hand_data/HO-3D_v3 \
  --split both \
  --bbox_source vitpose \
  --body_detector regnety \
  --output_format webdataset

参数说明:
    --base_dir:   HO-3D 数据集根目录。
                  注意对于 evaluation split，此目录下必须包含 evaluation_xyz.json 和 evaluation.txt。
    --split:      处理子集，选项: [training, evaluation, both]，默认 both
    --output_dir: 结果保存目录，默认保存在 base_dir 下。
    --output_format:
                  `npz` 只导出 NPZ；
                  `webdataset` 只导出 HaMeR 原生 tar；
                  `both` 两者都导出。
    --tar_shard_size:
                  每个 webdataset tar shard 的最大样本数。

输出结果:
    - ho3d_train.npz: 训练集数据，包含 'sensor' 键 (五指归一化距离)。
    - ho3d_evaluation.npz: 评估集数据，包含 'sensor' 键。
    - 可选导出 HaMeR 原生 WebDataset tar，用于 `scripts/train.py`。
    - 导出 webdataset 时，会额外生成 `datasets_tar_ho3d_v3.yaml` 供 `scripts/train.py` 使用。

说明:
    - `--bbox_source gt`:
      使用 GT 3D 投影 / 标注 handBoundingBox 生成 bbox。
    - `--bbox_source vitpose`:
      复用 HaMeR demo 的流程，先做人检测，再跑 ViTPose，再从右手关键点生成 hand bbox。
    - `--body_detector`:
      仅在 `bbox_source=vitpose` 时生效，可选 `regnety` 或 `vitdet`。
"""
import os
import glob
import json
import pickle
import textwrap
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import sys
from pathlib import Path

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from mano_processor.core import MANOHandProcessor


def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:, 0] / proj_pts[:, 2], proj_pts[:, 1] / proj_pts[:, 2]], axis=1)
    return proj_pts


def compute_bbox(keypoints_2d, padding=0.25, scale_mult_xy=(1.0, 1.0)):
    """
    Compute center and bbox size in pixel units.
    The saved `scale` must match HaMeR's NPZ convention: bbox width/height in pixels.
    `ImageDataset` will divide by 200 internally when loading.
    """
    min_x = np.min(keypoints_2d[:, 0])
    max_x = np.max(keypoints_2d[:, 0])
    min_y = np.min(keypoints_2d[:, 1])
    max_y = np.max(keypoints_2d[:, 1])

    center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0], dtype=np.float32)
    size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
    size = size * (1.0 + padding)
    size = size * np.asarray(scale_mult_xy, dtype=np.float32)
    return center, size


def bbox_xyxy_to_center_scale(bbox, padding=0.25, scale_mult_xy=(1.0, 1.0)):
    """
    Convert [u1, v1, u2, v2] bbox to center and pixel bbox size.
    """
    bbox = np.asarray(bbox, dtype=np.float32)
    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
    size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32)
    size = size * (1.0 + padding)
    size = size * np.asarray(scale_mult_xy, dtype=np.float32)
    return center, size


def resolve_image_path(base_dir, split_name, seq, frame_id):
    for ext in ('.jpg', '.png', '.jpeg'):
        img_rel_path = f"{split_name}/{seq}/rgb/{frame_id}{ext}"
        img_abs_path = os.path.join(base_dir, img_rel_path)
        if os.path.exists(img_abs_path):
            return img_rel_path, img_abs_path
    return None, None


def build_demo_bbox_detector(body_detector='regnety', device=None):
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from vitpose_model import ViTPoseModel

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        raise ValueError(f'Unsupported body_detector: {body_detector}')

    pose_model = ViTPoseModel(device)
    return detector, pose_model


def detect_right_hand_bbox(
    img_cv2,
    detector,
    pose_model,
    ref_center=None,
    person_score_thresh=0.5,
    hand_kp_thresh=0.5,
    min_valid_hand_kps=4,
    rescale_factor=2.0,
):
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > person_score_thresh)
    if int(valid_idx.sum()) == 0:
        return None

    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    det_results = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
    vitposes_out = pose_model.predict_pose(img_cv2[:, :, ::-1], det_results, box_score_threshold=person_score_thresh)

    candidates = []
    for vitposes in vitposes_out:
        right_hand_keyp = vitposes['keypoints'][-21:]
        valid = right_hand_keyp[:, 2] > hand_kp_thresh
        if int(valid.sum()) < min_valid_hand_kps:
            continue
        xy = right_hand_keyp[valid, :2]
        bbox = np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()], dtype=np.float32)
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
        score = float(right_hand_keyp[valid, 2].mean())
        candidates.append((bbox, center, score))

    if not candidates:
        return None

    if ref_center is not None:
        ref_center = np.asarray(ref_center, dtype=np.float32)
        bbox, center, _ = min(
            candidates,
            key=lambda item: (float(np.sum((item[1] - ref_center) ** 2)), -item[2]),
        )
    else:
        bbox, center, _ = max(candidates, key=lambda item: item[2])

    size = rescale_factor * np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32)
    return center.astype(np.float32), size.astype(np.float32)


def process_ho3d_split(
    base_dir,
    split_name,
    output_dir,
    fist_ratio=0.45,
    scale_mult_xy=(1.0, 1.0),
    bbox_source='gt',
    body_detector='regnety',
    detector_rescale_factor=2.0,
    output_format='npz',
    tar_shard_size=1000,
):
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
        'hand_keypoints_2d': [], 'hand_keypoints_3d': [], 'personid': [],
        'sensor': []
    }

    processor = MANOHandProcessor()
    detector = None
    pose_model = None
    detector_success_count = 0
    detector_fallback_count = 0
    if bbox_source == 'vitpose':
        print(f"  Loading offline bbox pipeline: detector={body_detector}, rescale_factor={detector_rescale_factor}")
        detector, pose_model = build_demo_bbox_detector(body_detector=body_detector)

    eval_joints_map = {}
    is_eval = split_name == 'evaluation'
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
                eval_joints_map[path] = np.array(joints, dtype=np.float32)
            print(f"  Loaded {len(eval_joints_map)} evaluation joints.")

    valid_count = 0
    error_count = 0

    for seq_idx, seq in enumerate(sequences):
        seq_dir = os.path.join(split_dir, seq)
        meta_dir = os.path.join(seq_dir, 'meta')
        rgb_dir = os.path.join(seq_dir, 'rgb')

        if not os.path.exists(meta_dir) or not os.path.exists(rgb_dir):
            continue

        pkl_files = sorted(glob.glob(os.path.join(meta_dir, '*.pkl')))
        print(f"  Sequence {seq}: {len(pkl_files)} frames")

        for pkl_file in tqdm(pkl_files, leave=False):
            frame_id = os.path.splitext(os.path.basename(pkl_file))[0]
            img_rel_path, img_abs_path = resolve_image_path(base_dir, split_name, seq, frame_id)
            if img_abs_path is None:
                continue

            try:
                with open(pkl_file, 'rb') as f:
                    anno = pickle.load(f, encoding='latin1')
            except Exception:
                error_count += 1
                continue

            if is_eval:
                cam_mat = anno['camMat']
                frame_path = f"{seq}/{frame_id}"
                joints_3d = eval_joints_map.get(frame_path)

                if joints_3d is not None:
                    pts_2d = project_3D_points(cam_mat, joints_3d, is_OpenGL_coords=True)
                    kps_2d = np.concatenate([pts_2d, np.ones((21, 1))], axis=1)
                    kps_3d = np.concatenate([joints_3d, np.ones((21, 1))], axis=1)
                else:
                    pts_2d = None
                    kps_2d = np.zeros((21, 3), dtype=np.float32)
                    kps_3d = np.zeros((21, 4), dtype=np.float32)

                if 'handBoundingBox' in anno and anno['handBoundingBox'] is not None:
                    gt_center, gt_scale = bbox_xyxy_to_center_scale(anno['handBoundingBox'], padding=0.25, scale_mult_xy=scale_mult_xy)
                elif pts_2d is not None:
                    gt_center, gt_scale = compute_bbox(pts_2d, padding=0.25, scale_mult_xy=scale_mult_xy)
                else:
                    gt_center = np.array([320.0, 240.0], dtype=np.float32)
                    gt_scale = np.array([400.0, 400.0], dtype=np.float32) * np.asarray(scale_mult_xy, dtype=np.float32)

                if bbox_source == 'vitpose':
                    img_cv2 = cv2.imread(img_abs_path)
                    detected_bbox = None
                    if isinstance(img_cv2, np.ndarray):
                        detected_bbox = detect_right_hand_bbox(
                            img_cv2,
                            detector,
                            pose_model,
                            ref_center=gt_center if pts_2d is not None else None,
                            rescale_factor=detector_rescale_factor,
                        )
                    if detected_bbox is not None:
                        center, scale = detected_bbox
                        detector_success_count += 1
                    else:
                        center, scale = gt_center, gt_scale
                        detector_fallback_count += 1
                else:
                    center, scale = gt_center, gt_scale

                hand_pose = np.zeros(48, dtype=np.float32)
                betas = np.zeros(10, dtype=np.float32)
                has_pose = 0.0
                has_betas = 0.0

                if joints_3d is not None:
                    try:
                        result = processor.process_hand_frame(joints_3d, lmin_method='estimate', fist_ratio=fist_ratio)
                        sensor_res = result['normalized_sensor_values'].astype(np.float32)
                    except Exception:
                        sensor_res = np.zeros(5, dtype=np.float32)
                else:
                    sensor_res = np.zeros(5, dtype=np.float32)

            else:
                if 'handJoints3D' not in anno or anno['handJoints3D'] is None:
                    continue

                joints_3d = anno['handJoints3D']
                if joints_3d.shape[0] != 21:
                    continue

                cam_mat = anno['camMat']
                pts_2d = project_3D_points(cam_mat, joints_3d, is_OpenGL_coords=True)
                gt_center, gt_scale = compute_bbox(pts_2d, padding=0.25, scale_mult_xy=scale_mult_xy)

                if bbox_source == 'vitpose':
                    img_cv2 = cv2.imread(img_abs_path)
                    detected_bbox = None
                    if isinstance(img_cv2, np.ndarray):
                        detected_bbox = detect_right_hand_bbox(
                            img_cv2,
                            detector,
                            pose_model,
                            ref_center=gt_center,
                            rescale_factor=detector_rescale_factor,
                        )
                    if detected_bbox is not None:
                        center, scale = detected_bbox
                        detector_success_count += 1
                    else:
                        center, scale = gt_center, gt_scale
                        detector_fallback_count += 1
                else:
                    center, scale = gt_center, gt_scale

                kps_2d = np.concatenate([pts_2d, np.ones((21, 1))], axis=1)
                kps_3d = np.concatenate([joints_3d, np.ones((21, 1))], axis=1)

                has_pose = 1.0 if 'handPose' in anno and anno['handPose'] is not None else 0.0
                has_betas = 1.0 if 'handBeta' in anno and anno['handBeta'] is not None else 0.0

                if has_pose:
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

                try:
                    result = processor.process_hand_frame(joints_3d, lmin_method='estimate', fist_ratio=fist_ratio)
                    sensor_res = np.asarray(result['normalized_sensor_values'], dtype=np.float32)
                except Exception:
                    sensor_res = np.zeros(5, dtype=np.float32)

            npz_data['imgname'].append(img_rel_path)
            npz_data['center'].append(center)
            npz_data['scale'].append(np.asarray(scale, dtype=np.float32))
            npz_data['hand_pose'].append(hand_pose)
            npz_data['betas'].append(betas)
            npz_data['has_hand_pose'].append(has_pose)
            npz_data['has_betas'].append(has_betas)
            npz_data['right'].append(1.0)
            npz_data['hand_keypoints_2d'].append(kps_2d)
            npz_data['hand_keypoints_3d'].append(kps_3d)
            npz_data['personid'].append(seq_idx)
            npz_data['sensor'].append(sensor_res)
            valid_count += 1

    print(f"Processed {valid_count} valid frames (Errors: {error_count})")
    if bbox_source == 'vitpose':
        print(f"  Detector bbox success: {detector_success_count}")
        print(f"  Detector bbox fallback-to-gt: {detector_fallback_count}")

    final_npz = {}
    for k, v in npz_data.items():
        if k == 'imgname':
            final_npz[k] = np.array(v)
        else:
            dtype = np.int32 if k == 'personid' else np.float32
            final_npz[k] = np.stack(v).astype(dtype)
    os.makedirs(output_dir, exist_ok=True)

    if output_format in ('npz', 'both'):
        npz_out_path = os.path.join(output_dir, f'ho3d_{split_name}.npz')
        np.savez(npz_out_path, **final_npz)
        print(f"Saved npz to {npz_out_path}")

    tar_urls = None
    if output_format in ('webdataset', 'both'):
        tar_urls = write_webdataset_split(
            base_dir=base_dir,
            split_name=split_name,
            output_dir=output_dir,
            final_npz=final_npz,
            shard_size=tar_shard_size,
        )

    return final_npz, tar_urls


def write_webdataset_split(base_dir, split_name, output_dir, final_npz, shard_size=1000):
    import webdataset as wds

    split_tag = 'train' if split_name == 'train' else 'val'
    tar_dir = os.path.join(output_dir, 'dataset_tars', f'ho3d-{split_tag}')
    os.makedirs(tar_dir, exist_ok=True)
    pattern = os.path.join(tar_dir, '%06d.tar')

    print(f"Saving WebDataset shards to {tar_dir} (shard_size={shard_size})")
    with wds.ShardWriter(pattern, maxcount=shard_size) as sink:
        num_samples = len(final_npz['imgname'])
        for idx in tqdm(range(num_samples), desc=f"Writing {split_name} tar", leave=False):
            rel_path = final_npz['imgname'][idx]
            if isinstance(rel_path, bytes):
                rel_path = rel_path.decode('utf-8')
            image_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(image_path):
                continue

            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            ext = Path(image_path).suffix.lower().lstrip('.')
            if ext == 'jpeg':
                ext = 'jpg'
            if ext not in ('jpg', 'png'):
                ext = 'jpg'

            sample = {
                'center': final_npz['center'][idx].astype(np.float32),
                # WebDataset pipeline expects scale already normalized by 200.
                'scale': (final_npz['scale'][idx].astype(np.float32) / 200.0),
                'hand_pose': final_npz['hand_pose'][idx].astype(np.float32),
                'betas': final_npz['betas'][idx].astype(np.float32),
                'has_hand_pose': np.array(final_npz['has_hand_pose'][idx], dtype=np.float32),
                'has_betas': np.array(final_npz['has_betas'][idx], dtype=np.float32),
                'right': np.array(final_npz['right'][idx], dtype=np.float32),
                'keypoints_2d': final_npz['hand_keypoints_2d'][idx].astype(np.float32),
                'keypoints_3d': final_npz['hand_keypoints_3d'][idx].astype(np.float32),
                'sensor': final_npz['sensor'][idx].astype(np.float32),
                'extra_info': {
                    'imgname_rel': rel_path,
                    'personid': int(final_npz['personid'][idx]),
                },
            }

            key = rel_path.replace('/', '__')
            sink.write({
                '__key__': key,
                ext: image_bytes,
                'data.pyd': [sample],
            })

    tar_urls = sorted(glob.glob(os.path.join(tar_dir, '*.tar')))
    print(f"Saved {len(tar_urls)} tar shards to {tar_dir}")
    return tar_urls


def write_webdataset_dataset_config(output_dir, train_urls=None, val_urls=None, train_epoch_size=None):
    config_path = os.path.join(output_dir, 'datasets_tar_ho3d_v3.yaml')

    def emit_urls(urls):
        return '\n'.join([f'    - {url}' for url in urls])

    chunks = []
    if train_urls:
        chunks.append("HO3D-TRAIN:")
        chunks.append("  TYPE: ImageDataset")
        chunks.append("  URLS:")
        chunks.append(emit_urls(train_urls))
        if train_epoch_size is not None:
            chunks.append(f"  epoch_size: {int(train_epoch_size):,}".replace(',', '_'))
    if val_urls:
        if chunks:
            chunks.append("")
        chunks.append("HO3D-VAL:")
        chunks.append("  TYPE: ImageDataset")
        chunks.append("  URLS:")
        chunks.append(emit_urls(val_urls))

    with open(config_path, 'w') as f:
        f.write('\n'.join(chunks) + '\n')

    print(f"Saved dataset config to {config_path}")
    print(textwrap.dedent(f"""
    Training example:
      conda run -n STMF python scripts/train.py \\
        experiment=hamer_vit_transformer \\
        data=ho3d_only \\
        dataset_config_name={config_path} \\
        checkpoint=_DATA/hamer_ckpts/checkpoints/hamer.ckpt \\
        LOSS_WEIGHTS.ADVERSARIAL=0
    """).strip())
    return config_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/mirage/STMF/_DATA/HO-3D_v3')
    parser.add_argument('--split', type=str, choices=['training', 'evaluation', 'both'], default='both')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--bbox_source', type=str, choices=['gt', 'vitpose'], default='gt')
    parser.add_argument('--body_detector', type=str, choices=['vitdet', 'regnety'], default='regnety')
    parser.add_argument('--detector_rescale_factor', type=float, default=2.0, help='Hand bbox expansion factor used by the demo pipeline before cropping')
    parser.add_argument('--scale_mult_x', type=float, default=1.57, help='Extra width multiplier to align HO3D crops with HaMeR bbox convention')
    parser.add_argument('--scale_mult_y', type=float, default=1.55, help='Extra height multiplier to align HO3D crops with HaMeR bbox convention')
    parser.add_argument('--output_format', type=str, choices=['npz', 'webdataset', 'both'], default='npz')
    parser.add_argument('--tar_shard_size', type=int, default=1000, help='Max samples per exported webdataset tar shard')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.base_dir

    scale_mult_xy = (args.scale_mult_x, args.scale_mult_y)

    train_tar_urls = None
    val_tar_urls = None
    train_count = None

    if args.split in ['training', 'both']:
        train_npz, train_tar_urls = process_ho3d_split(
            args.base_dir,
            'train',
            args.output_dir,
            scale_mult_xy=scale_mult_xy,
            bbox_source=args.bbox_source,
            body_detector=args.body_detector,
            detector_rescale_factor=args.detector_rescale_factor,
            output_format=args.output_format,
            tar_shard_size=args.tar_shard_size,
        )
        train_count = len(train_npz['imgname'])
    if args.split in ['evaluation', 'both']:
        _, val_tar_urls = process_ho3d_split(
            args.base_dir,
            'evaluation',
            args.output_dir,
            scale_mult_xy=scale_mult_xy,
            bbox_source=args.bbox_source,
            body_detector=args.body_detector,
            detector_rescale_factor=args.detector_rescale_factor,
            output_format=args.output_format,
            tar_shard_size=args.tar_shard_size,
        )
    if args.output_format in ('webdataset', 'both'):
        write_webdataset_dataset_config(
            output_dir=args.output_dir,
            train_urls=train_tar_urls,
            val_urls=val_tar_urls,
            train_epoch_size=train_count,
        )
