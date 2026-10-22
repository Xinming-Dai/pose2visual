"""Helpers for running Lightning Pose inference on pose2visual data.

Modules:
    predict_ibl: run ``litpose predict`` on every IBL-2view session.
    fit_eks_latent_model: fit per-group, per-camera Pose-EKS rank-reduced latent smoothers.
    pca_eks: per-group PCA reprojection of the Pose-EKS keypoints, saved with its config.
"""
