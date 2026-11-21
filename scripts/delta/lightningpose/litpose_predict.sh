#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH -t 0-2:59:00
#SBATCH -J litpose
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose/litpose_predict_ibl_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate lp

export PATH="${PATH}:/sw/external/python/anaconda3/bin"
export HF_HUB_CACHE=/work/hdd/bfsr/xdai3/hf_cache
export HF_HUB_OFFLINE=1

REPO=/u/xdai3/project3d/pose2visual/pose2visual
ROOT=/work/hdd/bfsr/xdai3/IBL-2view
DRAW=${1:?usage: sbatch litpose_predict.sh <draw index 0-2>}
MODEL_ROOT=/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/mm-anchored-lora_ibl_v14/mm-anchored-lora
LIGHTNING_POSE_MODEL_DIR=${MODEL_ROOT}/tfall-draw${DRAW}
SCRIPT=${REPO}/src/pose2visual/lightningpose/predict_ibl.py
CONFIG=${REPO}/configs/lightningpose/extraction_keypoints.yaml

SESSION_IDS=(
  51e53aff-1d5d-4182-a684-aba783d50ae5
  b22f694e-4a34-4142-ab9d-2556c3487086
)

if [ "${#SESSION_IDS[@]}" -eq 0 ]; then
  echo "SESSION_IDS is empty; add at least one session id" >&2
  exit 1
fi

# to skip labeled overlay videos, add: --skip_viz

PYTHONPATH="${REPO}/src:${PYTHONPATH}" python -u "${SCRIPT}" \
  --root "${ROOT}" \
  --model-dir "${LIGHTNING_POSE_MODEL_DIR}" \
  --config "${CONFIG}" \
  --sessionids "${SESSION_IDS[@]}" \
  --skip-existing \
  -- --overrides "data.data_dir=${LIGHTNING_POSE_MODEL_DIR}"

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."