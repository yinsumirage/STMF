#!/usr/bin/env bash
set -euo pipefail

# Full cached STMF-v2 HO3D pipeline for the dual4090 remote machine.
#
# Typical usage on the remote host:
#
#   cd /home/user/code/STMF
#   . /home/user/miniconda3/etc/profile.d/conda.sh
#   conda activate STMF
#   bash scripts/run_stmf_v2_full_ho3d.sh
#
# This script intentionally keeps all data/log/result paths explicit so local
# WSL path differences do not leak into remote experiments.

RUN_DATE="${RUN_DATE:-20260603}"
DATA_ROOT="${DATA_ROOT:-/data/hand_data/HO-3D_v3}"
CHECKPOINT="${CHECKPOINT:-./_DATA/hamer_ckpts/checkpoints/hamer.ckpt}"
TRAIN_NPZ="${TRAIN_NPZ:-${DATA_ROOT}/ho3d_train.npz}"
EVAL_NPZ="${EVAL_NPZ:-${DATA_ROOT}/ho3d_evaluation.npz}"
TRAIN_CACHE="${TRAIN_CACHE:-${DATA_ROOT}/ho3d_train_hamer_base_cache.npz}"
EVAL_CACHE="${EVAL_CACHE:-${DATA_ROOT}/ho3d_evaluation_hamer_base_cache.npz}"
RESULT_ROOT="${RESULT_ROOT:-results/sensor_refiner/ho3d_v3_${RUN_DATE}}"
LOG_ROOT="${LOG_ROOT:-logs_remote/ho3d_v3_${RUN_DATE}}"

BATCH_CACHE="${BATCH_CACHE:-1024}"
BATCH_TRAIN="${BATCH_TRAIN:-8192}"
BATCH_METRICS="${BATCH_METRICS:-4096}"
EPOCHS="${EPOCHS:-20}"
WINDOW_SIZE="${WINDOW_SIZE:-5}"
NUM_WORKERS_CACHE="${NUM_WORKERS_CACHE:-2}"
NUM_WORKERS_TRAIN="${NUM_WORKERS_TRAIN:-4}"
GPU_ZERO="${GPU_ZERO:-0}"
GPU_SENSOR="${GPU_SENSOR:-1}"
HISTORY_SOURCE="${HISTORY_SOURCE:-base}"
MIXED_GT_PROB="${MIXED_GT_PROB:-0.5}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-2}"
SMOOTHNESS_WEIGHT="${SMOOTHNESS_WEIGHT:-0.0}"
GLOBAL_ORIENT_WEIGHT="${GLOBAL_ORIENT_WEIGHT:-0.0}"
TRAIN_BASE_POSE_NOISE_STD="${TRAIN_BASE_POSE_NOISE_STD:-0.0}"
BLACKOUT_STRATEGY="${BLACKOUT_STRATEGY:-hold}"
BLACKOUT_1_LEN="${BLACKOUT_1_LEN:-1}"
BLACKOUT_3_LEN="${BLACKOUT_3_LEN:-3}"
BASE_POSE_NOISE_STD="${BASE_POSE_NOISE_STD:-0.0}"
SENSOR_DROPOUT="${SENSOR_DROPOUT:-0.0}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

echo "RUN_DATE=${RUN_DATE}"
echo "HEAD=$(git rev-parse --short HEAD)"
echo "GPU_ZERO=${GPU_ZERO}"
echo "GPU_SENSOR=${GPU_SENSOR}"
echo "BATCH_CACHE=${BATCH_CACHE}"
echo "BATCH_TRAIN=${BATCH_TRAIN}"
echo "BATCH_METRICS=${BATCH_METRICS}"
echo "EPOCHS=${EPOCHS}"
echo "WINDOW_SIZE=${WINDOW_SIZE}"
echo "HISTORY_SOURCE=${HISTORY_SOURCE}"
echo "MIXED_GT_PROB=${MIXED_GT_PROB}"
echo "LR=${LR}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "HIDDEN_DIM=${HIDDEN_DIM}"
echo "NUM_LAYERS=${NUM_LAYERS}"
echo "SMOOTHNESS_WEIGHT=${SMOOTHNESS_WEIGHT}"
echo "GLOBAL_ORIENT_WEIGHT=${GLOBAL_ORIENT_WEIGHT}"
echo "TRAIN_BASE_POSE_NOISE_STD=${TRAIN_BASE_POSE_NOISE_STD}"
echo "BLACKOUT_STRATEGY=${BLACKOUT_STRATEGY}"
echo "BASE_POSE_NOISE_STD=${BASE_POSE_NOISE_STD}"
echo "SENSOR_DROPOUT=${SENSOR_DROPOUT}"

cache_split() {
  local gpu="$1"
  local dataset_file="$2"
  local output_file="$3"
  local log_file="$4"
  if [[ -f "${output_file}" ]]; then
    echo "Skip existing cache: ${output_file}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/cache_base_hamer_predictions.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset_file "${dataset_file}" \
    --img_dir "${DATA_ROOT}" \
    --output_file "${output_file}" \
    --split train \
    --batch_size "${BATCH_CACHE}" \
    --num_workers "${NUM_WORKERS_CACHE}" \
    2>&1 | tee "${log_file}"
}

cache_split "${GPU_ZERO}" "${TRAIN_NPZ}" "${TRAIN_CACHE}" "${LOG_ROOT}/cache_train.log" &
cache_train_pid=$!
cache_split "${GPU_SENSOR}" "${EVAL_NPZ}" "${EVAL_CACHE}" "${LOG_ROOT}/cache_eval.log" &
cache_eval_pid=$!
wait "${cache_train_pid}"
wait "${cache_eval_pid}"

train_refiner() {
  local mode="$1"
  local gpu="$2"
  local output_dir="${LOG_ROOT}/${mode}_w${WINDOW_SIZE}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_sensor_refiner.py \
    --dataset_file "${TRAIN_NPZ}" \
    --base_pred_file "${TRAIN_CACHE}" \
    --output_dir "${output_dir}" \
    --history_source "${HISTORY_SOURCE}" \
    --sensor_mode "${mode}" \
    --mixed_gt_prob "${MIXED_GT_PROB}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --num_layers "${NUM_LAYERS}" \
    --window_size "${WINDOW_SIZE}" \
    --batch_size "${BATCH_TRAIN}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_workers "${NUM_WORKERS_TRAIN}" \
    --smoothness_weight "${SMOOTHNESS_WEIGHT}" \
    --global_orient_weight "${GLOBAL_ORIENT_WEIGHT}" \
    --base_pose_noise_std "${TRAIN_BASE_POSE_NOISE_STD}" \
    --device cuda \
    --log_every 20 \
    2>&1 | tee "${LOG_ROOT}/train_${mode}.log"
}

eval_refiner() {
  local mode="$1"
  local gpu="$2"
  local stress_name="$3"
  shift 3
  local checkpoint_path="${LOG_ROOT}/${mode}_w${WINDOW_SIZE}/last.pt"
  local pred_file="${RESULT_ROOT}/${mode}_${stress_name}_stateful.npz"
  local metrics_json="${RESULT_ROOT}/${mode}_${stress_name}_metrics.json"
  local metrics_csv="${RESULT_ROOT}/${mode}_${stress_name}_metrics.csv"

  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/eval_sensor_refiner.py \
    --checkpoint "${checkpoint_path}" \
    --dataset_file "${EVAL_NPZ}" \
    --base_pred_file "${EVAL_CACHE}" \
    --output_file "${pred_file}" \
    --window_size "${WINDOW_SIZE}" \
    --stateful \
    --base_pose_noise_std "${BASE_POSE_NOISE_STD}" \
    --sensor_dropout "${SENSOR_DROPOUT}" \
    --device cuda \
    "$@" \
    2>&1 | tee "${LOG_ROOT}/eval_${mode}_${stress_name}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/eval_sensor_refiner_metrics.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset_file "${EVAL_NPZ}" \
    --prediction_file "${pred_file}" \
    --output_json "${metrics_json}" \
    --output_csv "${metrics_csv}" \
    --batch_size "${BATCH_METRICS}" \
    --device cuda \
    2>&1 | tee "${LOG_ROOT}/metrics_${mode}_${stress_name}.log"
}

run_mode() {
  local mode="$1"
  local gpu="$2"
  train_refiner "${mode}" "${gpu}"
  eval_refiner "${mode}" "${gpu}" clean
  eval_refiner "${mode}" "${gpu}" blackout1 --blackout_len "${BLACKOUT_1_LEN}" --blackout_strategy "${BLACKOUT_STRATEGY}"
  eval_refiner "${mode}" "${gpu}" blackout3 --blackout_len "${BLACKOUT_3_LEN}" --blackout_strategy "${BLACKOUT_STRATEGY}"
}

run_mode zero "${GPU_ZERO}" &
zero_pid=$!
run_mode sensor "${GPU_SENSOR}" &
sensor_pid=$!
wait "${zero_pid}"
wait "${sensor_pid}"

export RESULT_ROOT
python - <<'PY'
import csv
import os
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
rows = []
for csv_path in sorted(result_root.glob("*_metrics.csv")):
    stem = csv_path.stem.replace("_metrics", "")
    parts = stem.split("_")
    mode = parts[0]
    stress = "_".join(parts[1:])
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prediction") != "refined":
                continue
            row["mode"] = mode
            row["stress"] = stress
            rows.append(row)

summary_path = result_root / "summary_refined_metrics.csv"
if rows:
    fieldnames = ["mode", "stress"] + sorted(k for k in rows[0].keys() if k not in {"mode", "stress"})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {summary_path}")
else:
    print("No refined metrics rows found")
PY
