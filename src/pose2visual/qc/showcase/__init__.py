"""Showcase figures of the Pose-EKS keypoints: labeled frames and keypoint traces.

Run with `python -m pose2visual.qc.showcase` (see `cli` for the flags); needs
`pip install -e .[qc]`. For one anchor frame of a session it writes

- `kp_frame.<session>.frame<anchor>`: the left and right camera frames side by side with every
  keypoint above the likelihood gate drawn in its own colour (like a LightningPose labeled video);
- `kp_traces.<session>.frame<anchor>.pm<seconds>s`: the same frames plus the x/y traces of those
  keypoints from `seconds` before to `seconds` after the anchor, the anchor marked by a large dot.

The digit tips of the paw on the far side of each camera (occluded, the model predicts the nearer
paw) are dropped by default; `--fingers none` drops every digit tip. The left frame is resized to
the size of the right one unless `--native-aspect` is given. Each figure is saved as png and pdf.
A SLURM launcher lives in `scripts/delta/qc/render_showcase.sh`.

Modules:
    cli: argument parsing and `main`.
    runner: resolves csvs and videos per session, loads the data and saves the figures.
    loading: reads one video frame and the Pose-EKS window around the anchor.
    figures: the matplotlib figures.

Importing this package is cheap; matplotlib and opencv are only imported by the submodules that
need them.
"""
