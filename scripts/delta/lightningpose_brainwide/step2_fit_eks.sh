#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 40G
#SBATCH -t 0-03:00:00
#SBATCH -J bw_eks
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/step2_fit_eks_%j.log
#SBATCH --export=ALL

# check sacct -j <id> -o MaxRSS,Elapsed after the first run of each and adjust.
# One job = one model x one session: Pose-EKS over the 3 draws for the left and right camera, run at
# the same time (--workers = the 2 cpus). 
#   sbatch --export=ALL,MODEL=digits,EID=<eid> -J bw_eks_digits step2_fit_eks.sh

exec 2>&1
source ~/.bashrc
conda activate beast
source /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/common.sh

if [ "${MODEL:-}" != "digits" ] && [ "${MODEL:-}" != "rest" ]; then
  echo "MODEL must be digits or rest, got '${MODEL:-}'" >&2
  exit 1
fi
if [ -z "${EID:-}" ]; then
  echo "EID must be set, e.g. sbatch --export=ALL,MODEL=digits,EID=<eid> step2_fit_eks.sh" >&2
  exit 1
fi

EKS_RUN="${OUT_ROOT}/models/${MODEL}/eks_run"
if [ -f "${EKS_RUN}/eks/_iblrig_leftCamera.downsampled.${EID}.csv" ] \
    && [ -f "${EKS_RUN}/eks/_iblrig_rightCamera.downsampled.${EID}.csv" ]; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] model=${MODEL} session=${EID} already fitted, skipping"
  exit 0
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Fitting Pose-EKS: model=${MODEL} session=${EID}"

PYTHONPATH="${REPO}/src" python -m pose2visual.lightningpose.fit_eks_latent_model \
  --model-dir "${OUT_ROOT}/models/${MODEL}" \
  --seed-dirs "${DRAWS[@]}" \
  --sessions "${EID}" \
  --out-dir "${EKS_RUN}/${EID}" \
  --csv-dir "${EKS_RUN}/eks" \
  --workers "${SLURM_CPUS_PER_TASK:-2}"
STATUS=$?

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
