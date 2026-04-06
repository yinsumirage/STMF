"""
Inspect packed NPZ hand GT samples by rendering 2D keypoints on source images.

This is meant as a quick sanity-check after exporting datasets such as:
- HO3D: `ho3d_train.npz`, `ho3d_evaluation.npz`
- FreiHAND-style packed NPZ files

Typical usage:

1. Inspect a few evenly spaced HO3D train samples:
python tools/data_prep/inspect_packed_gt.py \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3 \
  --out_dir /data/hand_data/HO-3D_v3/inspect_train \
  --dataset_type ho3d \
  --num_samples 12

2. Compare HO3D keypoints rendered in both model/OpenPose order and official order:
python tools/data_prep/inspect_packed_gt.py \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_evaluation.npz \
  --img_dir /data/hand_data/HO-3D_v3 \
  --out_dir /data/hand_data/HO-3D_v3/inspect_eval \
  --dataset_type ho3d \
  --packed_order openpose \
  --order both \
  --num_samples 8

3. Inspect specific sample indices:
python tools/data_prep/inspect_packed_gt.py \
  --dataset_file /data/hand_data/HO-3D_v3/ho3d_train.npz \
  --img_dir /data/hand_data/HO-3D_v3 \
  --out_dir /data/hand_data/HO-3D_v3/inspect_debug \
  --dataset_type ho3d \
  --indices 0,1,2,100,200

Notes:
- For HO3D, `order=openpose` means the packed keypoints are interpreted in the
  model-native HaMeR/OpenPose order.
- `order=official` converts the same packed keypoints back to official MANO order.
  This view intentionally renders points + indices only, not OpenPose-style
  finger connections, because the official order does not match the drawing
  topology used by `render_openpose()`.
- `packed_order` describes what order is currently stored in the NPZ.
  Old HO3D exports were usually `official`; newly fixed exports should be `openpose`.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from hamer.utils.render_openpose import render_openpose
from tools.data_prep.ho3d_process import HO3D_OFFICIAL_TO_OPENPOSE


HO3D_OPENPOSE_TO_OFFICIAL = np.argsort(HO3D_OFFICIAL_TO_OPENPOSE)


def parse_indices(indices: str) -> List[int]:
    return [int(x.strip()) for x in indices.split(",") if x.strip()]


def resolve_image_path(img_dir: str, rel_path: str) -> str:
    path = os.path.join(img_dir, rel_path)
    if os.path.exists(path):
        return path

    stem = os.path.splitext(path)[0]
    for ext in (".jpg", ".png", ".jpeg"):
        alt = stem + ext
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"Image not found for packed sample: {rel_path}")


def draw_bbox(img: np.ndarray, center: np.ndarray, scale: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        w = h = float(scale[0])
    else:
        w, h = float(scale[0]), float(scale[1])
    cx, cy = float(center[0]), float(center[1])
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 3, color, -1)
    return out


def add_caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_panel(img_bgr: np.ndarray, kps: np.ndarray, center: np.ndarray, scale: np.ndarray, title: str) -> np.ndarray:
    with_bbox = draw_bbox(img_bgr, center, scale)
    rendered = render_openpose(with_bbox, kps.copy())
    rendered = np.ascontiguousarray(rendered.astype(np.uint8))
    return add_caption(rendered, title)


def render_points_with_indices(img_bgr: np.ndarray, kps: np.ndarray, center: np.ndarray, scale: np.ndarray, title: str) -> np.ndarray:
    out = draw_bbox(img_bgr, center, scale)
    for i, kp in enumerate(kps):
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf <= 0.1:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, 3, (0, 255, 0), -1)
        cv2.putText(out, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return add_caption(out, title)


def choose_indices(total: int, num_samples: int) -> List[int]:
    if total <= 0:
        return []
    if num_samples >= total:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, num=num_samples, dtype=int).tolist()))


def build_contact_sheet(panels: List[np.ndarray], ncols: int = 1, pad: int = 8, bg=(30, 30, 30)) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to render")
    h, w = panels[0].shape[:2]
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    sheet = np.full((nrows * h + (nrows + 1) * pad, ncols * w + (ncols + 1) * pad, 3), bg, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // ncols
        c = idx % ncols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = panel
    return sheet


def iter_selected_indices(total: int, args) -> Iterable[int]:
    if args.indices:
        return parse_indices(args.indices)
    return choose_indices(total, args.num_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True, help="Packed NPZ file to inspect")
    parser.add_argument("--img_dir", type=str, required=True, help="Image root directory used by imgname inside NPZ")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save rendered inspection images")
    parser.add_argument("--dataset_type", type=str, default="ho3d", choices=["ho3d", "generic"], help="Whether to enable HO3D order conversions")
    parser.add_argument("--packed_order", type=str, default="openpose", choices=["openpose", "official"], help="Joint order currently stored inside the packed NPZ")
    parser.add_argument("--order", type=str, default="both", choices=["openpose", "official", "both"], help="How to interpret packed keypoint order")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of evenly spaced samples to render when --indices is not given")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated sample indices to inspect")
    parser.add_argument("--ncols", type=int, default=1, help="Columns in the contact sheet")
    args = parser.parse_args()

    data = np.load(args.dataset_file, allow_pickle=True)
    imgnames = data["imgname"]
    centers = data["center"]
    scales = data["scale"]
    keypoints_2d = data["hand_keypoints_2d"]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    panels = []
    selected = list(iter_selected_indices(len(imgnames), args))
    for idx in selected:
        rel_path = imgnames[idx]
        if isinstance(rel_path, bytes):
            rel_path = rel_path.decode("utf-8")

        image_path = resolve_image_path(args.img_dir, rel_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[skip] failed to read image: {image_path}")
            continue

        packed_kps = keypoints_2d[idx].copy()
        center = centers[idx].copy()
        scale = scales[idx].copy()

        if args.dataset_type == "ho3d":
            if args.packed_order == "openpose":
                kps_openpose = packed_kps
                kps_official = packed_kps[HO3D_OPENPOSE_TO_OFFICIAL]
            else:
                kps_official = packed_kps
                kps_openpose = packed_kps[HO3D_OFFICIAL_TO_OPENPOSE]
        else:
            kps_openpose = packed_kps
            kps_official = packed_kps

        sample_panels = [
            add_caption(draw_bbox(img_bgr, center, scale), f"idx={idx} raw+bbox"),
        ]

        if args.order in ("openpose", "both"):
            sample_panels.append(
                render_panel(img_bgr, kps_openpose, center, scale, "GT skeleton (openpose/model order)")
            )

        if args.dataset_type == "ho3d" and args.order in ("official", "both"):
            sample_panels.append(
                render_points_with_indices(img_bgr, kps_official, center, scale, "GT points+indices (official MANO order)")
            )

        row = build_contact_sheet(sample_panels, ncols=len(sample_panels), pad=6, bg=(20, 20, 20))
        panels.append(row)

        out_file = os.path.join(args.out_dir, f"{idx:06d}_{Path(rel_path).stem}.jpg")
        cv2.imwrite(out_file, row)

    if panels:
        summary = build_contact_sheet(panels, ncols=args.ncols, pad=10, bg=(15, 15, 15))
        summary_path = os.path.join(args.out_dir, "summary.jpg")
        cv2.imwrite(summary_path, summary)
        print(f"Saved {len(panels)} sample renders to {args.out_dir}")
        print(f"Saved summary sheet to {summary_path}")
    else:
        print("No panels were rendered.")


if __name__ == "__main__":
    main()
