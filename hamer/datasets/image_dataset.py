"""
Shared image dataset loader for both HaMeR and STMF pipelines.

How it is used:
- `scripts/eval.py` / `scripts/eval_stmf.py` read NPZ metadata through this file.
- `TemporalImageDataset` extends this class for STMF sequence windows.

Important conventions:
- NPZ `center` / `scale` follow the HaMeR format.
- `scale` is expected to be bbox width/height in pixels before dividing by 200.
- At evaluation time this loader now tolerates `.png/.jpg/.jpeg` mismatches and can skip missing images.
"""

import copy
import os
import numpy as np
import torch
from typing import Any, Dict, List, Union
from yacs.config import CfgNode
import braceexpand
import cv2

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: Union[str, List[str]]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

FLIP_KEYPOINT_PERMUTATION = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256


def resolve_image_path(img_dir: str, image_file_rel: str) -> str:
    """
    Resolve image path while tolerating png/jpg extension mismatches between
    packaged NPZ metadata and the local extracted dataset.
    """
    image_file = os.path.join(img_dir, image_file_rel)
    if os.path.exists(image_file):
        return image_file

    root, ext = os.path.splitext(image_file)
    alt_exts = []
    if ext.lower() == '.png':
        alt_exts = ['.jpg', '.jpeg']
    elif ext.lower() in ('.jpg', '.jpeg'):
        alt_exts = ['.png']

    for alt_ext in alt_exts:
        alt_path = root + alt_ext
        if os.path.exists(alt_path):
            return alt_path

    return image_file

class ImageDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 rescale_factor = 2,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(ImageDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.skip_missing_images = kwargs.get('skip_missing_images', not train)

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)
        self.rescale_factor = rescale_factor

        self.img_dir = img_dir
        loaded = np.load(dataset_file, allow_pickle=True)
        self.data = {key: loaded[key] for key in loaded.files}
        loaded.close()

        self.imgname = self.data['imgname']
        if 'personid' in self.data:
            self.personid = self.data['personid'].astype(np.int32).reshape(-1)
        else:
            self.personid = np.zeros(len(self.imgname), dtype=np.int32)
        self.extra_info = self.data.get('extra_info', [{} for _ in range(len(self.imgname))])

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1) / 200.0
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))
        assert self.scale.shape == (len(self.center), 2)

        try:
            self.right = self.data['right']
        except KeyError:
            self.right = np.ones(len(self.imgname), dtype=np.float32)

        # Get gt MANO parameters, if available
        try:
            self.hand_pose = self.data['hand_pose'].astype(np.float32)
            self.has_hand_pose = self.data['has_hand_pose'].astype(np.float32)
        except KeyError:
            self.hand_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_hand_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            hand_keypoints_2d = self.data['hand_keypoints_2d']
        except KeyError:
            hand_keypoints_2d = np.zeros((len(self.center), 21, 3))

        self.keypoints_2d = hand_keypoints_2d

        # Try to get 3d keypoints, if available
        try:
            hand_keypoints_3d = self.data['hand_keypoints_3d'].astype(np.float32)
        except KeyError:
            hand_keypoints_3d = np.zeros((len(self.center), 21, 4), dtype=np.float32)

        self.keypoints_3d = hand_keypoints_3d
        self._apply_ho3d_official_subset_filter(dataset_file)
        self._filter_missing_images()

    def _apply_ho3d_official_subset_filter(self, dataset_file: str) -> None:
        """
        For HO3D evaluation-style NPZ files, keep only frames listed in the official
        evaluation.txt whitelist so the prediction count matches the public scorer.
        """
        if self.train:
            return

        dataset_file_l = str(dataset_file).lower()
        img_dir_l = str(self.img_dir).lower()
        if 'ho3d' not in dataset_file_l and 'ho-3d' not in dataset_file_l and 'ho3d' not in img_dir_l and 'ho-3d' not in img_dir_l:
            return

        candidate_subset_files = [
            os.path.join(self.img_dir, 'evaluation.txt'),
            os.path.join(os.path.dirname(self.img_dir), 'evaluation.txt'),
        ]
        subset_file = next((p for p in candidate_subset_files if os.path.exists(p)), None)
        if subset_file is None:
            return

        with open(subset_file, 'r') as f:
            whitelist = {line.strip() for line in f.readlines() if line.strip()}
        if not whitelist:
            return

        keep_mask = np.ones(len(self.imgname), dtype=np.bool_)
        matched = 0
        for idx, raw_name in enumerate(self.imgname):
            name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)
            norm_name = name.replace('\\', '/')
            stem = os.path.splitext(norm_name)[0]
            parts = stem.split('/')

            short_name = None
            # examples:
            # evaluation/SM1/rgb/0000.jpg -> SM1/0000
            # SM1/rgb/0000.png            -> SM1/0000
            if len(parts) >= 3 and parts[-2] == 'rgb':
                short_name = f"{parts[-3]}/{parts[-1]}"

            if short_name is None or short_name not in whitelist:
                keep_mask[idx] = False
            else:
                matched += 1

        if matched == len(self.imgname):
            return

        print(f"Applying HO3D official whitelist from {subset_file}: keeping {matched}/{len(self.imgname)}")
        self._apply_sample_mask(keep_mask)

    def _filter_missing_images(self) -> None:
        if not self.skip_missing_images:
            return

        num_samples = len(self.imgname)
        keep_mask = np.ones(num_samples, dtype=np.bool_)
        missing_examples = []

        for idx in range(num_samples):
            raw_name = self.imgname[idx]
            image_file_rel = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)
            image_file = resolve_image_path(self.img_dir, image_file_rel)
            if not os.path.isfile(image_file):
                keep_mask[idx] = False
                if len(missing_examples) < 5:
                    missing_examples.append(image_file)

        missing_count = int((~keep_mask).sum())
        if missing_count == 0:
            return

        self._apply_sample_mask(keep_mask)
        print(f"Skipping {missing_count} samples with missing images from {self.img_dir}")
        for image_file in missing_examples:
            print(f"  missing: {image_file}")

    def _apply_sample_mask(self, keep_mask: np.ndarray) -> None:
        original_len = keep_mask.shape[0]

        def maybe_mask(value):
            if isinstance(value, np.ndarray):
                if value.ndim > 0 and value.shape[0] == original_len:
                    return value[keep_mask]
                return value
            if isinstance(value, list) and len(value) == original_len:
                return [v for v, keep in zip(value, keep_mask.tolist()) if keep]
            return value

        for attr in [
            'imgname',
            'personid',
            'extra_info',
            'center',
            'scale',
            'right',
            'hand_pose',
            'has_hand_pose',
            'betas',
            'has_betas',
            'keypoints_2d',
            'keypoints_3d',
        ]:
            setattr(self, attr, maybe_mask(getattr(self, attr)))

        self.data = {key: maybe_mask(value) for key, value in self.data.items()}

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file_rel = self.imgname[idx]
        image_file = resolve_image_path(self.img_dir, image_file_rel)
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        right = self.right[idx].copy()
        if self.rescale_factor == -1:
            BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
            bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
            bbox_expand_factor = bbox_size / ((scale*200).max())
        else:
            bbox_expand_factor = self.rescale_factor
            bbox_size = bbox_expand_factor*scale.max()*200
        hand_pose = self.hand_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_hand_pose = self.has_hand_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        mano_params = {'global_orient': hand_pose[:3],
                       'hand_pose': hand_pose[3:],
                       'betas': betas
                      }

        has_mano_params = {'global_orient': has_hand_pose,
                           'hand_pose': has_hand_pose,
                           'betas': has_betas
                           }

        mano_params_is_axis_angle = {'global_orient': True,
                                     'hand_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    mano_params, has_mano_params,
                                                                                                    self.flip_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.train, right, augm_config)
        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['mano_params'] = mano_params
        item['has_mano_params'] = has_mano_params
        item['mano_params_is_axis_angle'] = mano_params_is_axis_angle
        item['imgname'] = image_file
        item['imgname_rel'] = image_file_rel
        item['personid'] = int(self.personid[idx])
        item['extra_info'] = copy.deepcopy(self.extra_info[idx])
        item['idx'] = idx
        item['_scale'] = scale
        item['right'] = self.right[idx].copy()
        return item

    @staticmethod
    def load_tars_as_webdataset(cfg: CfgNode, urls: Union[str, List[str]], train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        STD = 255. * np.array(cfg.MODEL.IMAGE_STD)

        def split_data(source):
            for item in source:
                datas = item['data.pyd']
                for data in datas:
                    if 'detection.npz' in item:
                        det_idx = data['extra_info']['detection_npz_idx']
                        mask = item['detection.npz']['masks'][det_idx]
                    else:
                        mask = np.ones_like(item['jpg'][:,:,0], dtype=bool)
                    yield {
                        '__key__': item['__key__'],
                        'jpg': item['jpg'],
                        'data.pyd': data,
                        'mask': mask,
                    }

        def suppress_bad_kps(item, thresh=0.0):
            if thresh > 0:
                kp2d = item['data.pyd']['keypoints_2d']
                kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
                item['data.pyd']['keypoints_2d'] = np.concatenate([kp2d[:,:2], kp2d_conf[:,None]], axis=1)
            return item

        def filter_numkp(item, numkp=4, thresh=0.0):
            kp_conf = item['data.pyd']['keypoints_2d'][:, 2]
            return (kp_conf > thresh).sum() > numkp

        def filter_reproj_error(item, thresh=10**4.5):
            losses = item['data.pyd'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
            reproj_loss = losses.get('reprojection_loss', None)
            return reproj_loss is None or reproj_loss < thresh

        def filter_bbox_size(item, thresh=1):
            bbox_size_min = item['data.pyd']['scale'].min().item() * 200.
            return bbox_size_min > thresh

        def filter_no_poses(item):
            return (item['data.pyd']['has_hand_pose'] > 0)

        def supress_bad_betas(item, thresh=3):
            has_betas = item['data.pyd']['has_betas']
            if thresh > 0 and has_betas:
                betas_abs = np.abs(item['data.pyd']['betas'])
                if (betas_abs > thresh).any():
                    item['data.pyd']['has_betas'] = False
            return item

        def supress_bad_poses(item):
            has_hand_pose = item['data.pyd']['has_hand_pose']
            if has_hand_pose:
                hand_pose = item['data.pyd']['hand_pose']
                pose_is_probable = poses_check_probable(torch.from_numpy(hand_pose)[None, 3:], amass_poses_hist100_smooth).item()
                if not pose_is_probable:
                    item['data.pyd']['has_hand_pose'] = False
            return item

        def poses_betas_simultaneous(item):
            # We either have both hand_pose and betas, or neither
            has_betas = item['data.pyd']['has_betas']
            has_hand_pose = item['data.pyd']['has_hand_pose']
            item['data.pyd']['has_betas'] = item['data.pyd']['has_hand_pose'] = np.array(float((has_hand_pose>0) and (has_betas>0)))
            return item

        def set_betas_for_reg(item):
            # Always have betas set to true
            has_betas = item['data.pyd']['has_betas']
            betas = item['data.pyd']['betas']

            if not (has_betas>0):
                item['data.pyd']['has_betas'] = np.array(float((True)))
                item['data.pyd']['betas'] = betas * 0
            return item

        # Load the dataset
        if epoch_size is not None:
            resampled = True
        #corrupt_filter = lambda sample: (sample['__key__'] not in CORRUPT_KEYS)
        import webdataset as wds
        dataset = wds.WebDataset(expand_urls(urls),
                                nodesplitter=wds.split_by_node,
                                shardshuffle=True,
                                resampled=resampled,
                                cache_dir=cache_dir,
                              ) #.select(corrupt_filter)
        if train:
            dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')

        # Process the dataset
        dataset = dataset.compose(split_data)

        # Filter/clean the dataset
        SUPPRESS_KP_CONF_THRESH = cfg.DATASETS.get('SUPPRESS_KP_CONF_THRESH', 0.0)
        SUPPRESS_BETAS_THRESH = cfg.DATASETS.get('SUPPRESS_BETAS_THRESH', 0.0)
        SUPPRESS_BAD_POSES = cfg.DATASETS.get('SUPPRESS_BAD_POSES', False)
        POSES_BETAS_SIMULTANEOUS = cfg.DATASETS.get('POSES_BETAS_SIMULTANEOUS', False)
        BETAS_REG = cfg.DATASETS.get('BETAS_REG', False)
        FILTER_NO_POSES = cfg.DATASETS.get('FILTER_NO_POSES', False)
        FILTER_NUM_KP = cfg.DATASETS.get('FILTER_NUM_KP', 4)
        FILTER_NUM_KP_THRESH = cfg.DATASETS.get('FILTER_NUM_KP_THRESH', 0.0)
        FILTER_REPROJ_THRESH = cfg.DATASETS.get('FILTER_REPROJ_THRESH', 0.0)
        FILTER_MIN_BBOX_SIZE = cfg.DATASETS.get('FILTER_MIN_BBOX_SIZE', 0.0)
        if SUPPRESS_KP_CONF_THRESH > 0:
            dataset = dataset.map(lambda x: suppress_bad_kps(x, thresh=SUPPRESS_KP_CONF_THRESH))
        if SUPPRESS_BETAS_THRESH > 0:
            dataset = dataset.map(lambda x: supress_bad_betas(x, thresh=SUPPRESS_BETAS_THRESH))
        if SUPPRESS_BAD_POSES:
            dataset = dataset.map(lambda x: supress_bad_poses(x))
        if POSES_BETAS_SIMULTANEOUS:
            dataset = dataset.map(lambda x: poses_betas_simultaneous(x))
        if FILTER_NO_POSES:
            dataset = dataset.select(lambda x: filter_no_poses(x))
        if FILTER_NUM_KP > 0:
            dataset = dataset.select(lambda x: filter_numkp(x, numkp=FILTER_NUM_KP, thresh=FILTER_NUM_KP_THRESH))
        if FILTER_REPROJ_THRESH > 0:
            dataset = dataset.select(lambda x: filter_reproj_error(x, thresh=FILTER_REPROJ_THRESH))
        if FILTER_MIN_BBOX_SIZE > 0:
            dataset = dataset.select(lambda x: filter_bbox_size(x, thresh=FILTER_MIN_BBOX_SIZE))
        if BETAS_REG:
            dataset = dataset.map(lambda x: set_betas_for_reg(x))       # NOTE: Must be at the end

        use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        # Process the dataset further
        dataset = dataset.map(lambda x: ImageDataset.process_webdataset_tar_item(x, train,
                                                        augm_config=cfg.DATASETS.CONFIG,
                                                        MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                        BBOX_SHAPE=BBOX_SHAPE,
                                                        use_skimage_antialias=use_skimage_antialias,
                                                        border_mode=border_mode,
                                                        ))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset

    @staticmethod
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    ):
        # Read data from item
        key = item['__key__']
        image = item['jpg']
        data = item['data.pyd']
        mask = item['mask']

        keypoints_2d = data['keypoints_2d']
        keypoints_3d = data['keypoints_3d']
        center = data['center']
        scale = data['scale']
        hand_pose = data['hand_pose']
        betas = data['betas']
        right = data['right']
        has_hand_pose = data['has_hand_pose']
        has_betas = data['has_betas']
        # image_file = data['image_file']

        # Process data
        orig_keypoints_2d = keypoints_2d.copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        if bbox_size < 1:
            breakpoint()


        mano_params = {'global_orient': hand_pose[:3],
                    'hand_pose': hand_pose[3:],
                    'betas': betas
                    }

        has_mano_params = {'global_orient': has_hand_pose,
                        'hand_pose': has_hand_pose,
                        'betas': has_betas
                        }

        mano_params_is_axis_angle = {'global_orient': True,
                                    'hand_pose': True,
                                    'betas': False
                                    }

        augm_config = copy.deepcopy(augm_config)
        # Crop image and (possibly) perform data augmentation
        img_rgba = np.concatenate([image, mask.astype(np.uint8)[:,:,None]*255], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size, trans = get_example(img_rgba,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    mano_params, has_mano_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
                                                                                                    IMG_SIZE, IMG_SIZE,
                                                                                                    MEAN, STD, train, right, augm_config,
                                                                                                    is_bgr=False, return_trans=True,
                                                                                                    use_skimage_antialias=use_skimage_antialias,
                                                                                                    border_mode=border_mode,
                                                                                                    )
        img_patch = img_patch_rgba[:3,:,:]
        mask_patch = (img_patch_rgba[3,:,:] / 255.0).clip(0,1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {}

        item['img'] = img_patch
        item['mask'] = mask_patch
        # item['img_og'] = image
        # item['mask_og'] = mask
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = center.copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['mano_params'] = mano_params
        item['has_mano_params'] = has_mano_params
        item['mano_params_is_axis_angle'] = mano_params_is_axis_angle
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = key
        # item['idx'] = idx
        return item
