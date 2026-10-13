#!/usr/bin/env python3
"""Extract and save the ensemble median across network seeds, in DLC csv format.

For one session, loads every seed's full prediction csv, computes the per-frame
per-landmark median across seeds for every keypoint found in the csv header (no
keypoint subsetting — this produces a drop-in median csv with the same columns as the
raw seed csvs), and writes one median csv per camera.

The median is a plain `np.median` across seeds (see
`pose2visual.lightningpose.ensemble_median.compute_ensemble_median`): it does not
threshold on likelihood or filter NaNs, matching `pose2visual.qc.rainbow`'s own ensemble
median panel exactly, so csv output and QC plot stay consistent.

Usage:
    python export_ensemble_median.py \\
        --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
        --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
        --session 9b528ad0-4599-4a55-9148-96cc1d93fb24

Default output: <model_dir>/ensemble_median/<timestamp>[__<tag>]/csv/ensemble_median_<camera>.csv
Folder name is `YYYYMMDD_HHMMSS`, suffixed with `__<tag>` when `--tag` is given. Every
run's manifest is also appended to `<model_dir>/ensemble_median/registry.json` (keyed by
folder name). Use `prune_run.py` to delete a run and its registry entry together.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path for development/skill usage (project-agnostic import)
_pose2visual_src = Path(__file__).parent.parent.parent.parent / 'src'
if _pose2visual_src.exists() and str(_pose2visual_src) not in sys.path:
    sys.path.insert(0, str(_pose2visual_src))

from pose2visual.lightningpose.ensemble_median import (
    compute_ensemble_median,
    write_ensemble_median_csv,
)
from pose2visual.lightningpose.fit_eks_latent_model import (
    discover_keypoint_names,
    find_seed_csv,
    find_video_preds_dir,
)
from pose2visual.run_registry import append_entry


def _count_csv_rows(csv_path: Path) -> int:
    """Number of data rows in a 3-level-header DLC-style csv.

    Args:
        csv_path: path to the csv.

    Returns:
        number of data rows (excludes the 3 header rows).
    """
    with open(csv_path) as f:
        return sum(1 for _ in f) - 3


def load_all_keypoints(
    csv_path: Path,
    keypoints: list[str],
    start_frame: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (x, y, likelihood) for every keypoint from one seed's prediction csv.

    Args:
        csv_path: path to predictions csv.
        keypoints: keypoint names to load (as they appear in the csv header).
        start_frame: starting frame index.
        n_frames: number of frames to load.

    Returns:
        Tuple of (xy, likelihood), shapes (n_frames, n_keypoints, 2) and
        (n_frames, n_keypoints).
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)

    xy_list = []
    conf_list = []
    for key in keypoints:
        xy_list.append(df[key][['x', 'y']].values)
        conf_list.append(df[key]['likelihood'].values)

    xy = np.stack(xy_list, axis=1)  # (n_total, n_keypoints, 2)
    conf = np.stack(conf_list, axis=1)  # (n_total, n_keypoints)

    end_frame = min(start_frame + n_frames, len(xy))
    xy = xy[start_frame:end_frame]
    conf = conf[start_frame:end_frame]

    if len(xy) < n_frames:
        xy = np.pad(xy, ((0, n_frames - len(xy)), (0, 0), (0, 0)), constant_values=np.nan)
        conf = np.pad(conf, ((0, n_frames - len(conf)), (0, 0)), constant_values=np.nan)

    return xy, conf


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract and save the ensemble median across network seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python export_ensemble_median.py \\
    --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
    --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
    --session 9b528ad0-4599-4a55-9148-96cc1d93fb24
        ''',
    )

    parser.add_argument('--model-dir', required=True, help='Root directory holding seed folders')
    parser.add_argument(
        '--seed-dirs',
        nargs='+',
        required=True,
        help='Seed folder names (relative to model-dir) or absolute paths',
    )
    parser.add_argument('--session', required=True, help='Session ID (UUID)')
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=['left', 'right'],
        help='Camera names to process (default: left right)',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame index (default: 0)',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to process (default: every frame available, from start-frame)',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: <model-dir>/ensemble_median/<timestamp>[__<tag>])',
    )
    parser.add_argument('--tag', default=None, help='Optional tag for output filenames')

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Main execution logic."""
    model_dir = Path(args.model_dir).resolve()
    session_id = args.session
    start_frame = args.start_frame
    tag = args.tag

    seed_dirs = {}
    for seed_name in args.seed_dirs:
        seed_path = Path(seed_name)
        if not seed_path.is_absolute():
            seed_path = model_dir / seed_name
        seed_path = seed_path.resolve()
        if not seed_path.exists():
            print(f'Error: seed directory not found: {seed_path}', file=sys.stderr)
            sys.exit(1)
        seed_dirs[seed_name] = seed_path

    print('Resolving seed predictions...')
    csv_paths = {}  # {camera: {seed_name: path}}
    for camera in args.cameras:
        csv_paths[camera] = {}
        for seed_name, seed_path in seed_dirs.items():
            preds_dir = find_video_preds_dir(seed_path)
            csv_path = find_seed_csv(preds_dir, session_id, camera)
            csv_paths[camera][seed_name] = csv_path
            print(f'  {camera}/{seed_name}: {csv_path}')

    if args.n_frames is not None:
        n_frames = args.n_frames
    else:
        available = min(
            _count_csv_rows(path)
            for paths in csv_paths.values()
            for path in paths.values()
        )
        n_frames = available - start_frame
        print(f'--n-frames not given; using all available frames: {n_frames}')

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    if tag:
        run_name = f'{run_name}__{tag}'

    out_dir = Path(args.out_dir).resolve() if args.out_dir else model_dir / 'ensemble_median' / run_name
    csv_dir = out_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output: {out_dir}')

    outputs = {}
    for camera in args.cameras:
        print(f'Processing {camera} camera...')
        seed_names_sorted = sorted(csv_paths[camera].keys())

        keypoints = discover_keypoint_names(csv_paths[camera][seed_names_sorted[0]])
        print(f'  discovered {len(keypoints)} keypoints')

        xy_seeds = []
        conf_seeds = []
        for seed_name in seed_names_sorted:
            xy, conf = load_all_keypoints(
                csv_paths[camera][seed_name], keypoints, start_frame, n_frames,
            )
            xy_seeds.append(xy)
            conf_seeds.append(conf)

        xy_median, conf_median = compute_ensemble_median(
            np.stack(xy_seeds, axis=0), np.stack(conf_seeds, axis=0),
        )

        out_path = csv_dir / f'ensemble_median_{camera}.csv'
        write_ensemble_median_csv(xy_median, conf_median, keypoints, out_path)
        print(f'  Saved: {out_path}')

        outputs[camera] = {
            'path': str(out_path),
            'n_frames': n_frames,
            'n_keypoints': len(keypoints),
            'keypoints': keypoints,
        }

    manifest = {
        'run_name': run_name,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'model_dir': str(model_dir),
        'seed_folders': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'session': session_id,
        'cameras': args.cameras,
        'csv_paths': {
            cam: {name: str(path) for name, path in paths.items()}
            for cam, paths in csv_paths.items()
        },
        'start_frame': start_frame,
        'n_frames': n_frames,
        'tag': tag,
        'outputs': outputs,
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved: {manifest_path}')

    registry_path = model_dir / 'ensemble_median' / 'registry.json'
    append_entry(registry_path, run_name, manifest)
    print(f'Updated registry: {registry_path}')

    print(f'Done. Output in: {out_dir}')


def main() -> None:
    """CLI entry point."""
    try:
        args = parse_args()
        run(args)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
