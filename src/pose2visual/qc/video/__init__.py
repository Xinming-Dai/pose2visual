"""QC video of the ensemble median vs the single-camera Pose-EKS output.

Run with `python -m pose2visual.qc.video` (see `cli` for the flags); needs
`pip install -e .[qc]`.

Modules:
    cli: argument parsing and `main`.
    runner: resolves sessions, groups, csvs and videos and renders one video per session.
    loading: reads the requested frames and keypoints of a Pose-EKS csv into `CameraData`.
    render_compare: `VideoJob` and `render_video`, the per-frame layout and the ffmpeg encode.
    trace_panels: the scrolling x/y coordinate traces under the video.
    style: keypoint palette and text helpers.

Importing this package is cheap; opencv and imageio-ffmpeg are only imported by the submodules
that need them.
"""
