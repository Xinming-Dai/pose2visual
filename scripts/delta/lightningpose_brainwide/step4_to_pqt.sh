#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 40G
#SBATCH -t 0-01:30:00
#SBATCH -J bw_pqt
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/step4_to_pqt_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast
source /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/common.sh

RUN="${RUN:-brainwide}"
EXTRA_ARGS=()
if [ -n "${OVERWRITE:-}" ]; then
  EXTRA_ARGS+=(--overwrite)
fi
cd "${REPO}"

PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -m pose2visual.lightningpose.litpose_to_pqt \
  --out-root "${OUT_ROOT}" \
  --run "${RUN}" \
  --config "${ENCODING_CONFIG}" \
  --sessionids "${SESSION_IDS[@]}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  "${EXTRA_ARGS[@]}"
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
