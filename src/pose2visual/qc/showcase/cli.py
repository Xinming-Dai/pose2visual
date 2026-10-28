"""Command-line entry point of the showcase figures: `python -m pose2visual.qc.showcase`.

For each session, saves (png and pdf) the left and right camera frames at the anchor frame with
every Pose-EKS keypoint above the likelihood gate drawn in its own colour (without the digit
tips of the paw far from each camera, see `--fingers`), and the same frames with the
x/y traces of those keypoints from `--half-window-sec` before to the same time after the
anchor. The source videos are auto-discovered from the csv names (IBL-2view convention, rooted at
`--video-root` or `$IBL_2VIEW_ROOT`).

Default output, in the folder of the csvs:
`<eks-dir>/qc/showcase/{png/*.png,pdf/*.pdf}`, named `kp_frame.<session>.frame<anchor>` and
`kp_traces.<session>.frame<anchor>.pm<seconds>s`.
A SLURM launcher lives in `scripts/delta/qc/render_showcase.sh`.
"""

import argparse
import logging
import sys

from pose2visual.qc.showcase.runner import run

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; defaults to `sys.argv[1:]`.

    Returns:
        the parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description='Labeled frames and keypoint traces of the Pose-EKS output at an anchor frame',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m pose2visual.qc.showcase \\
    --eks-dir /path/to/mm_ibl_ear/<run> --sessions 51e53aff-1d5d-4182-a684-aba783d50ae5

  python -m pose2visual.qc.showcase --eks-dir /path/to/<run> --sessions <uuid> \\
    --anchor-frame 3000 --half-window-sec 1 --figures traces

  python -m pose2visual.qc.showcase --eks-dir /path/to/<run> --sessions <uuid> --fingers none
        ''',
    )
    parser.add_argument(
        '--eks-dir',
        required=True,
        help='Directory with _iblrig_<camera>Camera.downsampled.<session>.csv Pose-EKS csvs',
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        required=True,
        help='Session IDs (UUIDs); one set of figures each',
    )
    parser.add_argument(
        '--anchor-frame',
        type=int,
        default=200,
        help='0-based video frame shown, and the centre of the traces (default: 200)',
    )
    parser.add_argument(
        '--half-window-sec',
        type=float,
        default=2.5,
        help='Seconds of trace before and after the anchor (default: 2.5)',
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.7,
        help='A keypoint is drawn when its likelihood exceeds this at the anchor frame; in the '
        'traces, frames at or below it are left blank (default: 0.7)',
    )
    parser.add_argument(
        '--colormap',
        default='cool',
        help='Matplotlib colormap whose evenly spaced colours label the keypoints, as in '
        'LightningPose labeled videos (default: cool)',
    )
    parser.add_argument(
        '--dot-size',
        type=float,
        default=10.0,
        help='Diameter in points of the keypoint dots on the frames (default: 10)',
    )
    parser.add_argument(
        '--fingers',
        choices=['near', 'all', 'none'],
        default='near',
        help="Digit tips (the keypoints d<n>_tip_*) to show: 'near' drops those of the paw on the "
        'far side of each camera (right-side tips on the left camera and vice versa; that paw is '
        'occluded and the model predicts the nearer one), all keeps every one, and none drops all '
        "of them. 'none' and 'all' add .nofingers / .allfingers to the file names "
        '(default: near)',
    )
    parser.add_argument(
        '--native-aspect',
        action='store_true',
        help='Keep every frame at its own size and aspect ratio. By default the left frame is '
        'resized to the size of the right one (its keypoints and traces are scaled to match)',
    )
    parser.add_argument(
        '--figures',
        nargs='+',
        choices=['frames', 'traces'],
        default=['frames', 'traces'],
        help='Figures to save (default: frames traces)',
    )
    parser.add_argument(
        '--video-root',
        default=None,
        help='Root of the IBL-2view video folders used for video auto-discovery '
        '(default: $IBL_2VIEW_ROOT)',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: <eks-dir>/qc/showcase)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Resolution of the png files (default: 200)',
    )

    args = parser.parse_args(argv)
    if args.anchor_frame < 0:
        parser.error('--anchor-frame must be >= 0')
    if args.half_window_sec <= 0:
        parser.error('--half-window-sec must be > 0')
    if args.dot_size <= 0:
        parser.error('--dot-size must be > 0')
    if args.dpi < 1:
        parser.error('--dpi must be >= 1')
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
