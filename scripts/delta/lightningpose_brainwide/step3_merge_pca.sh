#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -t 0-03:00:00
#SBATCH -J bw_merge_pca
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/step3_merge_pca_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast
source /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/common.sh

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${REPO}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.encoding.list_litpose_keypoints \
  --config "${ENCODING_CONFIG}" \
  --video-timestamps-dir "${OUT_ROOT}/inputs/timestamps" \
  --eids "${SESSION_IDS[@]}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
