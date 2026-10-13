"""Rainbow scatter QC of pose predictions across network seeds.

Modules:
    cli: argument parsing and `main`; run with `python -m pose2visual.qc.rainbow`.
    runner: resolves sessions, seeds and videos, fans tasks out to workers, writes the manifest.
    camera_task: `CameraTask` and `process_camera`, the work for one (session, camera) pair.
    loading: finds and loads seed csvs, background frames and Pose-EKS output.
    render: matplotlib rows and the combined figure.
    canvas: palette and the PIL title/legend banners stacked around a figure.
    stats: per-keypoint summary statistics.
"""
