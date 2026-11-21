"""Command-line entry point of the QC video: `python -m pose2visual.qc.video`.

Renders, for each session, a video comparing the ensemble median with the single-camera Pose-EKS
output (written by `pose2visual.lightningpose.fit_eks_latent_model`, see
`scripts/delta/lightningpose/fit_eks.sh`) for the keypoints of a body-part group. Every
group of `--group-set` gets its own video unless `--groups` names some. Both cameras are shown side
by side by default; the source videos are auto-discovered from the csv names (IBL-2view
convention, rooted at `--video-root` or `$IBL_2VIEW_ROOT`).

Default output, one folder per group, in the EKS run folder (next to its `manifest.json`):
`<eks-dir>/../qc/video_qc/<group>/qc_video.<session>.<group>.<cameras>.frames<start>_<end>.mp4`
(`<eks-dir>/..` is the run folder). With `--save-diagnostics` also a `.diagnostics.csv` with the
same stem. A SLURM launcher lives in `scripts/delta/qc/render_video.sh`.

PCA: when encoding step 0 (`scripts/delta/lightningpose_encoding/step0_list_keypoints.sh`) has
saved the PCA of the `--config` yaml for every camera, the video gets a third image row under
Pose-EKS with the PCA position of every keypoint it covers, and the coordinate traces a thin red
line (`--no-pca` hides both).

Zoom: a group is zoomed to its per-camera box in the crop boxes file (`--crop-boxes-file`, default
`paths.crop_boxes_json` of `--config`, `scripts/delta/data/qc_crop_boxes.json`). The file has a
`default` entry and, for sessions whose boxes differ, a `sessions.<eid>` entry; both hold
`[group_set][group][camera] = [x0, y0, x1, y1]`, and per camera the session's box wins over the
default. Draw boxes with the mouse using `scripts/delta/qc/extract_crop_frames.py` and
`scripts/delta/qc/select_crops.py --session <eid>`. Precedence: `--crop-left/--crop-right`, then
the crop boxes file, then `--auto-crop-pad`, then the full frame.
"""

import argparse
import logging
import sys

from pose2visual.encoding.litpose_keypoints import load_encoding_yaml
from pose2visual.qc.video.runner import run

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; defaults to `sys.argv[1:]`.

    Returns:
        the parsed namespace; `groups_file`, `sessions_file` and `crop_boxes_file` are filled from
        the `paths` section of `--config` when not given explicitly.
    """
    parser = argparse.ArgumentParser(
        description='Ensemble median vs Pose-EKS comparison video, by body-part group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m pose2visual.qc.video \\
    --eks-dir /path/to/models/eks/<run>/eks --config configs/encoding/litpose_encoding.yaml \\
    --sessions 9b528ad0-4599-4a55-9148-96cc1d93fb24 --groups ear_related

  python -m pose2visual.qc.video \\
    --eks-dir /path/to/models/eks/<run>/eks --groups-file scripts/delta/data/body_parts.json \\
    --sessions <uuid> --groups ear_related eye_related --cameras left \\
    --start-frame 3000 --n-frames 600 --auto-crop-pad 12
        ''',
    )
    parser.add_argument(
        '--eks-dir',
        required=True,
        help='Directory with _iblrig_<camera>Camera.downsampled.<session>.csv Pose-EKS csvs, as '
        'written by fit_eks_latent_model.py',
    )
    session_arg = parser.add_mutually_exclusive_group(required=True)
    session_arg.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='Session IDs (UUIDs); one video each',
    )
    session_arg.add_argument(
        '--session-group',
        default=None,
        help='Name of a session group to load from --sessions-file (e.g. "sable", "all")',
    )
    parser.add_argument(
        '--sessions-file',
        default=None,
        help='Path to an ibl_session_ids.json-style file, required with --session-group '
        '(default: paths.eid_groups_json of --config)',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        choices=['left', 'right'],
        default=['left', 'right'],
        help='Cameras to show, side by side in this order (default: left right)',
    )
    parser.add_argument(
        '--far-side',
        action='store_true',
        help='Draw the bilateral keypoints of the side far from each camera (right hand in the '
        'left camera, left hand in the right camera) instead of the near side; the PCA trace is '
        'not drawn',
    )
    parser.add_argument(
        '--groups-file',
        default=None,
        help='Path to a body_parts.json-style file (default: paths.body_parts_json of --config)',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Encoding yaml (e.g. configs/encoding/litpose_encoding.yaml) whose '
        'paths.body_parts_json and paths.eid_groups_json fill --groups-file and '
        '--sessions-file when those are not given',
    )
    parser.add_argument(
        '--group-set',
        default='nonoverlapping_analysis_groups',
        help='Top-level key to load from --groups-file (default: nonoverlapping_analysis_groups)',
    )
    parser.add_argument(
        '--groups',
        nargs='+',
        default=None,
        help='Body-part groups to render, one video (in its own folder) per group '
        '(default: every group in --group-set)',
    )
    parser.add_argument(
        '--video-left',
        default=None,
        help='Override video path for left camera (auto-discovered by default)',
    )
    parser.add_argument(
        '--video-right',
        default=None,
        help='Override video path for right camera (auto-discovered by default)',
    )
    parser.add_argument(
        '--video-root',
        default=None,
        help='Root of the IBL-2view video folders used for video auto-discovery '
        '(default: $IBL_2VIEW_ROOT)',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='First frame of the clip (default: 0)',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=2000,
        help='Number of frames of the clip (default: 2000)',
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.7,
        help='A keypoint is drawn on frames where its mean ensemble likelihood exceeds this '
        '(default: 0.7)',
    )
    parser.add_argument(
        '--bar-scale',
        type=float,
        default=5.0,
        help='Magnification of the +-1 SD cross arms; the strips show true pixel SDs '
        '(default: 5)',
    )
    parser.add_argument(
        '--trace-window',
        type=int,
        default=300,
        help='Number of trailing frames in the coordinate traces (default: 300)',
    )
    parser.add_argument(
        '--crop-left',
        type=int,
        nargs=4,
        metavar=('X0', 'Y0', 'X1', 'Y1'),
        default=None,
        help='Zoom the left camera panels to this source-pixel box (expanded to the frame '
        'aspect ratio)',
    )
    parser.add_argument(
        '--crop-right',
        type=int,
        nargs=4,
        metavar=('X0', 'Y0', 'X1', 'Y1'),
        default=None,
        help='Zoom the right camera panels to this source-pixel box (expanded to the frame '
        'aspect ratio)',
    )
    parser.add_argument(
        '--auto-crop-pad',
        type=float,
        default=None,
        help='For cameras without an explicit crop, zoom to the drawn keypoints plus this many '
        'source pixels of margin (default: no zoom)',
    )
    parser.add_argument(
        '--crop-boxes-file',
        default=None,
        help='json file with the zoom boxes: "default" and per-session "sessions.<eid>" entries '
        '(default: paths.crop_boxes_json of --config; none = no zoom from a file)',
    )
    parser.add_argument(
        '--no-json-crops',
        action='store_true',
        help='Ignore the crop boxes file (by default each group is zoomed to its box there, per '
        'camera and session, unless --crop-left/--crop-right are given)',
    )
    parser.add_argument(
        '--pca-dir',
        default=None,
        help='Directory with the _iblrig_<camera>Camera.downsampled.<session>.csv files saved by '
        'encoding step 0, drawn as a thin red trace (default: <paths.video_preds_dir of --config>/'
        'encoding/pca/<pca.group_set of --config>; a missing file only skips the red trace)',
    )
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='Do not draw the PCA trace',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: <eks-dir>/../qc/video_qc)',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional extra part of the output file name',
    )
    parser.add_argument(
        '--save-diagnostics',
        action='store_true',
        help='Also write <video name>.diagnostics.csv: per keypoint, axis and method, the '
        'roughness of the trace and its difference from the ensemble median (default: off)',
    )

    args = parser.parse_args(argv)
    if args.n_frames < 1:
        parser.error('--n-frames must be >= 1')
    if args.start_frame < 0:
        parser.error('--start-frame must be >= 0')
    if args.trace_window < 2:
        parser.error('--trace-window must be >= 2')
    if args.bar_scale <= 0:
        parser.error('--bar-scale must be > 0')
    if args.config is not None:
        config = load_encoding_yaml(args.config)
        if args.groups_file is None:
            args.groups_file = config.get('body_parts_json')
        if args.sessions_file is None:
            args.sessions_file = config.get('eid_groups_json')
        if args.crop_boxes_file is None:
            args.crop_boxes_file = config.get('crop_boxes_json')
    if args.groups_file is None:
        parser.error('--groups-file is required (or set paths.body_parts_json in --config)')
    if args.session_group is not None and args.sessions_file is None:
        parser.error(
            '--session-group requires --sessions-file (or paths.eid_groups_json in --config)',
        )
    return args


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        run(parse_args())
    except Exception as error:
        logger.error(f'Error: {error}')
        sys.exit(1)


if __name__ == '__main__':
    main()
