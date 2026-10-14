"""Command-line entry point of the rainbow QC: `python -m pose2visual.qc.rainbow`.

Rainbow scatter plots of pose predictions across network seeds, by body-part group.
Each figure scatters all frames' predictions (every frame available by default) on a single
static background frame. Each dot is one frame's (x, y) for one landmark from one network.
Compares single-network seeds, their per-frame ensemble median, and optionally Pose-EKS
smoothed output. Renders both ungated (all frames) and gated (high-confidence frames per
landmark).

Body parts are grouped per a `body_parts.json`-style file (`--groups-file`, `--group-set`;
both json paths can instead come from the `paths` section of an encoding yaml via `--config`);
every group is rendered as its own row of panels (one column per method), stacked vertically
into one combined figure per camera per gated/ungated mode. Real keypoint column names (and
whether each is bilateral or midline) are discovered from the prediction csv header at
runtime, so any body part works without a hardcoded sub-part list.

The video is auto-discovered from the csv filename (IBL-2view convention, rooted at
`--video-root` or `$IBL_2VIEW_ROOT`); the output directory is timestamped.

`--session-group <name>` runs every session in that named group from `--sessions-file`
(an `ibl_session_ids.json`-style file, e.g. its `"sable"`/`"all"` groups under
`"neural_sessions"`) in one invocation, sharing one output run folder. Use
`--sessions <uuid> [<uuid> ...]` instead to name sessions explicitly.

Default output: `<model_dir>/rainbow_qc/<timestamp>[__<tag>]/{rainbow_qc,ensemble_median,
qc_manifest.json}`. The folder name is `YYYYMMDD_HHMMSS`, suffixed with `__<tag>` when `--tag`
is given. A SLURM launcher lives in `scripts/delta/qc/plot_rainbow.sh`.
"""

import argparse
import logging
import sys

from pose2visual.encoding.litpose_keypoints import load_encoding_yaml
from pose2visual.qc.rainbow.runner import run

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; defaults to `sys.argv[1:]`.

    Returns:
        the parsed namespace; `groups_file` and `sessions_file` are filled from the `paths`
        section of `--config` when not given explicitly.
    """
    parser = argparse.ArgumentParser(
        description='Rainbow scatter plots of pose predictions across seeds, by body-part group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m pose2visual.qc.rainbow \\
    --model-dir /path/to/models --seed-dirs seed0 seed1 seed2 \\
    --groups-file scripts/delta/data/body_parts.json \\
    --sessions-file scripts/delta/data/ibl_session_ids.json --session-group sable

  python -m pose2visual.qc.rainbow \\
    --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
    --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
    --groups-file scripts/delta/data/body_parts.json \\
    --sessions 9b528ad0-4599-4a55-9148-96cc1d93fb24

  python -m pose2visual.qc.rainbow \\
    --model-dir /path/to/models \\
    --seed-dirs seed0 seed1 seed2 \\
    --sessions <uuid1> <uuid2> \\
    --groups-file scripts/delta/data/body_parts.json \\
    --n-frames 5000 --conf-threshold 0.7 \\
    --eks-dir /path/to/models/eks/<run>/eks \\
    --groups eye_related ear_related
        ''',
    )

    parser.add_argument(
        '--model-dir',
        required=True,
        help='Root directory holding seed folders',
    )
    parser.add_argument(
        '--seed-dirs',
        nargs='+',
        required=True,
        help='Seed folder names (relative to model-dir) or absolute paths',
    )
    session_group_arg = parser.add_mutually_exclusive_group(required=True)
    session_group_arg.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='Session IDs (UUIDs) to process together, sharing one output run folder',
    )
    session_group_arg.add_argument(
        '--session-group',
        default=None,
        help='Name of a session group to load from --sessions-file (e.g. "sable", "all")',
    )
    parser.add_argument(
        '--sessions-file',
        default=None,
        help=(
            'Path to an ibl_session_ids.json-style file ({"neural_sessions": {<group>: '
            '[<uuid>, ...]}}), required with --session-group (default: paths.eid_groups_json '
            'of --config)'
        ),
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=['left', 'right'],
        help='Camera names to process (default: left right)',
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
        '--out-dir',
        default=None,
        help='Output directory (default: <model-dir>/rainbow_qc/<timestamp>[__<tag>])',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional tag for output filenames',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to process (default: every frame available, from start-frame)',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame index (default: 0)',
    )
    parser.add_argument(
        '--background-frame',
        type=int,
        default=None,
        help='Background frame index (default: start_frame + n_frames//2)',
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for gating (default: 0.7)',
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
        '--sessions-file when those are not given; relative paths are relative to the '
        'working directory',
    )
    parser.add_argument(
        '--group-set',
        default='nonoverlapping_analysis_groups',
        help='Top-level key to load from --groups-file, e.g. nonoverlapping_analysis_groups, '
        'overlapping_analysis_groups or nonoverlapping_analysis_groups_ibl_ears '
        '(default: nonoverlapping_analysis_groups)',
    )
    parser.add_argument(
        '--groups',
        nargs='+',
        default=None,
        help='Subset of group names to render (default: every group in --group-set), one row '
        'per group, stacked vertically',
    )
    parser.add_argument(
        '--save-ensemble-median',
        action='store_true',
        help='Also write the ensemble median (every discovered keypoint, not just the plotted '
        'groups) to ensemble_median/<source-csv-name>.csv, i.e. the same filename as the raw '
        'seed csv (default: off)',
    )
    parser.add_argument(
        '--ensemble-median-dir',
        default=None,
        help='Directory of already-saved median csvs, named <source-csv-name>.csv (as written '
        'by --save-ensemble-median), to reuse instead of recomputing from the seeds, for any '
        '(session, camera) found there',
    )
    parser.add_argument(
        '--eks-dir',
        default=None,
        help='Directory containing _iblrig_<camera>Camera.downsampled.<session>.csv Pose-EKS '
        'csvs, as written by fit_eks_latent_model.py (optional)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Worker processes; each handles one (session, camera) pair at a time '
        '(default: $SLURM_CPUS_PER_TASK if set, else 1, i.e. serial)',
    )
    parser.add_argument(
        '--roi-left',
        type=int,
        nargs=4,
        default=None,
        help='Left camera ROI (x0 y0 x1 y1) for statistics',
    )
    parser.add_argument(
        '--roi-right',
        type=int,
        nargs=4,
        default=None,
        help='Right camera ROI (x0 y0 x1 y1) for statistics',
    )

    args = parser.parse_args(argv)
    if args.num_workers is not None and args.num_workers < 1:
        parser.error('--num-workers must be >= 1')
    if args.config is not None:
        config = load_encoding_yaml(args.config)
        if args.groups_file is None:
            args.groups_file = config.get('body_parts_json')
        if args.sessions_file is None:
            args.sessions_file = config.get('eid_groups_json')
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
        args = parse_args()
        run(args)
    except Exception as e:
        logger.error(f'Error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
