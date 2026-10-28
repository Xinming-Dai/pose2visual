"""Helpers for running Lightning Pose inference on pose2visual data.

Modules:
    predict_ibl: run ``litpose predict`` on every IBL-2view session.
    fit_eks_latent_model: fit per-group, per-camera Pose-EKS rank-reduced latent smoothers.
    predict_brainwide: stage the Brainwide (ONE cache layout) videos under the IBL-2view names and
        run ``litpose predict`` per model draw on shadow model directories.
    litpose_to_pqt: convert the Brainwide EKS / merged / PCA csvs to ``.pqt`` tables.
"""
