#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH -t 0-02:00:00
#SBATCH -J bw_predict
#SBATCH -o /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/step1_predict_%j.log
#SBATCH --export=ALL

# One job = one model, the draws in RUN_DRAWS x the sessions in RUN_SESSION_IDS, run one after the
# other on a single GPU (like scripts/delta/lightningpose/litpose_predict.sh). Submit as many jobs
# as you want to spend, e.g. one model and a few sessions at a time:
#   sbatch --export=ALL,MODEL=digits scripts/delta/lightningpose_brainwide/step1_predict.sh
#   sbatch --export=ALL,MODEL=rest,CAMERAS=right,RUN_DRAWS_OVERRIDE="tfall-draw0",EIDS="<eid1> <eid2>" -t 0-24:00:00 \
#     scripts/delta/lightningpose_brainwide/step1_predict.sh # --overrides dataloader.test_batch_size=72

exec 2>&1
source ~/.bashrc
conda activate lp
source /u/xdai3/project3d/pose2visual/pose2visual/scripts/delta/lightningpose_brainwide/common.sh

export PATH="${PATH}:/sw/external/python/anaconda3/bin"
export HF_HUB_CACHE=/work/hdd/bfsr/xdai3/hf_cache
export HF_HUB_OFFLINE=1

# digits | rest; sbatch --export=ALL,MODEL=rest overrides this
MODEL="${MODEL:-digits}"
RUN_DRAWS=(
  # tfall-draw0
  tfall-draw1
  # tfall-draw2
)
RUN_SESSION_IDS=(
  # 0802ced5-33a3-405e-8336-b65ebc5cb07c  # mainenlab/ZFM-02373
  # 0841d188-8ef2-4f20-9828-76a94d5343a4  # churchlandlab_ucla/UCLA049
  239dd3c9-35f3-4462-95ee-91b822a22e6b  # mainenlab/ZFM-04308
  # 2bdf206a-820f-402f-920a-9e86cd5388a4  # churchlandlab_ucla/UCLA037
  # 3e6a97d3-3991-49e2-b346-6948cb4580fb  # steinmetzlab/NR_0020
  # 4a45c8ba-db6f-4f11-9403-56e06a33dfa4  # danlab/DY_020
)
if [ -n "${RUN_DRAWS_OVERRIDE:-}" ]; then
  read -r -a RUN_DRAWS <<< "${RUN_DRAWS_OVERRIDE}"
fi
if [ -n "${EIDS:-}" ]; then
  read -r -a RUN_SESSION_IDS <<< "${EIDS}"
fi

CAMERA_ARGS=()
if [ -n "${CAMERAS:-}" ]; then
  CAMERA_ARGS=(--cameras ${CAMERAS})
fi

if [ "${MODEL}" != "digits" ] && [ "${MODEL}" != "rest" ]; then
  echo "MODEL must be digits or rest, got '${MODEL}'" >&2
  exit 1
fi
if [ "${#RUN_SESSION_IDS[@]}" -eq 0 ] || [ "${#RUN_DRAWS[@]}" -eq 0 ]; then
  echo "RUN_DRAWS and RUN_SESSION_IDS must not be empty" >&2
  exit 1
fi

STATUS=0
for DRAW in "${RUN_DRAWS[@]}"; do
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] model=${MODEL} draw=${DRAW} sessions=${RUN_SESSION_IDS[*]}"
  PYTHONPATH="${REPO}/src:${PYTHONPATH:-}" python -u -m pose2visual.lightningpose.predict_brainwide predict \
    --out-root "${OUT_ROOT}" \
    --model "${MODEL}" \
    --draw "${DRAW}" \
    "${CAMERA_ARGS[@]}" \
    --sessionids "${RUN_SESSION_IDS[@]}"
  # keep going with the other draws if one fails; the job still exits non-zero
  if [ "$?" -ne 0 ]; then
    echo "draw ${DRAW} failed" >&2
    STATUS=1
  fi
done

conda deactivate
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done (exit status ${STATUS})."
exit "${STATUS}"
