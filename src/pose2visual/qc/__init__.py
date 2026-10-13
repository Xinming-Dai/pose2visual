"""Quality-control tools for pose predictions.

Subpackages:
    rainbow: rainbow scatter plots of per-seed predictions, the ensemble median and Pose-EKS
        output, by body-part group (`python -m pose2visual.qc.rainbow`).
    video: quality-control video renderers (prototype scripts, run directly as scripts).

Importing this package is cheap; the plotting dependencies (matplotlib, opencv, PIL) are
only imported by the submodules that need them. Install them with `pip install -e .[qc]`.
"""
