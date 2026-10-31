# shared settings of the Brainwide Lightning Pose scripts; sourced by step0..step4 (not run alone).
# Pipeline (see pose2visual/lightningpose/predict_brainwide.py for the layout under OUT_ROOT):
#   step0_prepare.sh   stage videos + timestamps, build shadow model dirs
#   step1_predict.sh   litpose predict (GPU), one job per model; draws and sessions are set in the
#                      script or by RUN_DRAWS_OVERRIDE / EIDS, so you decide how many jobs to run
#   step2_fit_eks.sh   EKS over the 3 draws, one job per model x session (MODEL=, EID= via --export)
#   step3_merge_pca.sh merge digits + rest, list keypoints, PCA (encoding step 0)
#   step4_to_pqt.sh    eks.digits / eks.rest / merged / pca csv -> .pqt
# step3 and step4 run once for all sessions, after every step2 job has finished.

REPO=/u/xdai3/project3d/pose2visual/pose2visual
ROOT=/work/hdd/bfsx/yzhang39/brainwide/video/raw
OUT_ROOT=${OUT_ROOT:-/work/hdd/bezq/xdai3/brainwide-litpose}
ENCODING_CONFIG=${REPO}/configs/encoding/litpose_encoding_brainwide.yaml

MODELS=(digits rest)
DRAWS=(tfall-draw0 tfall-draw1 tfall-draw2)
SESSION_IDS=(
  0802ced5-33a3-405e-8336-b65ebc5cb07c  # mainenlab/ZFM-02373
  0841d188-8ef2-4f20-9828-76a94d5343a4  # churchlandlab_ucla/UCLA049
  239dd3c9-35f3-4462-95ee-91b822a22e6b  # mainenlab/ZFM-04308
  2bdf206a-820f-402f-920a-9e86cd5388a4  # churchlandlab_ucla/UCLA037
  3e6a97d3-3991-49e2-b346-6948cb4580fb  # steinmetzlab/NR_0020
  4a45c8ba-db6f-4f11-9403-56e06a33dfa4  # danlab/DY_020
)
