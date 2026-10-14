"""Quality-control tools for pose predictions.

Subpackages:
    rainbow: rainbow scatter plots of per-seed predictions, the ensemble median and Pose-EKS
        output, by body-part group (`python -m pose2visual.qc.rainbow`).
    overlay: keypoint overlay video from a DLC-style predictions csv
        (`python -m pose2visual.qc.overlay`).
    prune_run: delete a run folder and its registry entry (`python -m pose2visual.qc.prune_run`).
    video: ensemble median vs Pose-EKS comparison video with SD crosses and coordinate traces
        (`python -m pose2visual.qc.video`).

Importing this package is cheap; the plotting dependencies (matplotlib, opencv, PIL) are
only imported by the submodules that need them. Install them with `pip install -e .[qc]`.
"""
