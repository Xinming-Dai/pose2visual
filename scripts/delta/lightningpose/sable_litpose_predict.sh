#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH -t 0-4:59:00
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
LIGHTNING_POSE_MODEL_DIR=/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E3a-headmlp256/face+ibl+cheese+caz+kondo+hmv+c3d+kauf_train/supervised/sampling-T2/tf1/vits_dinov3/seed0
SCRIPT=${REPO}/src/pose2visual/lightningpose/predict_ibl.py
CONFIG=${REPO}/configs/lightningpose/extraction_keypoints.yaml

SESSION_IDS=(
  3e6a97d3-3991-49e2-b346-6948cb4580fb
  5dcee0eb-b34d-4652-acc3-d10afc6eae68
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