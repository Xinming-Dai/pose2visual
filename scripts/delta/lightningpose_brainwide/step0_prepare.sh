#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 4G
#SBATCH -t 0-00:10:00
#SBATCH -J bw_prepare
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/step0_prepare_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate lp
source /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/common.sh

# symlinks only: videos/timestamps under the IBL-2view names, shadow dirs of the 2 models x 3 draws.
# add --dry-run to only print them
PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.lightningpose.predict_brainwide prepare \
  --root "${ROOT}" \
  --out-root "${OUT_ROOT}" \
  --draws "${DRAWS[@]}"

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
