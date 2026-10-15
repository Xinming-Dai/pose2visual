"""Fit single-camera Pose-EKS smoothers, one csv per session and camera, for qc.rainbow.

For each session, runs the `eks <https://github.com/paninski-lab/eks>`_ single-camera Ensemble
Kalman Smoother (the same model as ``eks.fit_eks_singlecam``) on the ensemble of seed models'
prediction csvs, independently for each camera. The (session, camera) fits are independent of
each other, so ``--workers`` runs them in parallel inside one run. Every keypoint in the csv is
smoothed.

This calls ``eks.singlecam_smoother.ensemble_kalman_smoother_singlecam`` rather than
``fit_eks_singlecam`` itself, since the latter always smooths every frame of the input csvs (its
``s_frames`` only chooses the frames used to fit ``s``), and we want ``--start-frame`` and
``--n-frames``.

Writes one csv per session and camera in the same ``(scorer, bodyparts, coords)`` 3-level-header
format as the raw prediction csvs, named ``_iblrig_<camera>Camera.downsampled.<session>.csv`` so
it can be used directly as a ``video_preds_dir`` (e.g. by the encoding pipeline, whose
``position.columns`` picks ``x, y`` or ``x_ens_median, y_ens_median``).
Each keypoint has the coords ``x``, ``y`` (smoothed), ``likelihood`` (mean ensemble confidence),
``x_ens_median``, ``y_ens_median``, ``x_ens_var``, ``y_ens_var``, ``x_posterior_var`` and
``y_posterior_var``. The fitted ``s`` per keypoint is recorded in the run's manifest.

Needs only ``pandas``, ``numpy``, ``jax`` and ``eks`` (``pip install
git+https://github.com/paninski-lab/eks.git`` — it is not on PyPI under this name); no
``litpose``/``lightning_pose`` import, so any conda env with those four is fine. ``eks`` does
not require a GPU.

Usage::

    PYTHONPATH=src python -m pose2visual.lightningpose.fit_eks_latent_model \\
      --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
      --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
      --sessions 9b528ad0-4599-4a55-9148-96cc1d93fb24 <more session ids ...> \\
      --workers 4

Default output, for every session and camera:
``<model-dir>/eks/<timestamp>[__<tag>]/eks/_iblrig_<camera>Camera.downsampled.<session>.csv``,
plus one ``<model-dir>/eks/<timestamp>[__<tag>]/manifest.json`` for the whole run. 
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from pose2visual.run_registry import append_entry

logger = logging.getLogger(__name__)

# run-folder timestamps use this timezone, not the cluster's local time (as in qc.rainbow)
RUN_TIMEZONE = ZoneInfo('America/New_York')


def discover_keypoint_names(csv_path: str | Path) -> list[str]:
    """Read the deduplicated keypoint names from a DLC-style 3-level csv header.

    Args:
        csv_path: path to a predictions csv with a (scorer, bodyparts, coords) header.

    Returns:
        keypoint names in header order, each appearing once.
    """
    header = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0, nrows=0)
    seen: dict[str, None] = {}
    for _, keypoint, _ in header.columns:
        seen.setdefault(keypoint, None)
    return list(seen)


def resolve_group_bases(keypoint_names: list[str], tokens: list[str]) -> list[tuple[str, bool]]:
    """Resolve a body-part group's real landmarks and whether each is bilateral.

    A keypoint is bilateral when it ends in ``_left``/``_right`` and its sibling (the same name
    with the other suffix) also exists — the suffix names which anatomical side (e.g. which ear)
    it belongs to, not a camera. A keypoint without such a sibling is midline (e.g. ``mouth``)
    and has a single column shared by both cameras' csvs.

    A landmark is in the group if the first ``_``-separated segment of its base name is one of
    ``tokens`` (e.g. ``ear_top`` matches tokens ``['ear']`` via its ``ear`` segment).

    Args:
        keypoint_names: real keypoint names from a prediction csv header, in header order.
        tokens: `body_parts.json` group tokens, e.g. ``['mouth', 'upperlip', 'lowerlip',
            'tongue']``.

    Returns:
        ``(base, is_bilateral)`` pairs, in header order, with duplicates removed.

    Raises:
        ValueError: if no keypoint matches any of `tokens`.
    """
    names = set(keypoint_names)
    bases: dict[str, bool] = {}
    for name in keypoint_names:
        if name.endswith('_left') or name.endswith('_right'):
            base = name.rsplit('_', 1)[0]
            sibling_suffix = '_right' if name.endswith('_left') else '_left'
            if f'{base}{sibling_suffix}' in names:
                bases.setdefault(base, True)
                continue
        bases.setdefault(name, False)

    selected = [
        (base, is_bilateral) for base, is_bilateral in bases.items()
        if base.split('_')[0] in tokens
    ]
    if not selected:
        raise ValueError(f'no landmarks matched tokens {tokens} in {list(bases)}')
    return selected


def group_columns_for_camera(bases: list[tuple[str, bool]], camera: str) -> list[str]:
    """Per-camera column names for a group's resolved landmarks.

    Args:
        bases: `(base, is_bilateral)` pairs from `resolve_group_bases`.
        camera: camera name (`'left'` or `'right'`); selects the matching anatomical side for
            bilateral landmarks.

    Returns:
        column names to read from that camera's prediction csvs, in `bases` order.
    """
    return [f'{base}_{camera}' if is_bilateral else base for base, is_bilateral in bases]


def load_groups(
    groups_file: str | Path,
    group_set: str = 'nonoverlapping_analysis_groups',
) -> dict[str, list[str]]:
    """Load a group-set from a `body_parts.json`-style file.

    Args:
        groups_file: path to the json file.
        group_set: top-level key to load, e.g. `'nonoverlapping_analysis_groups'` or
            `'overlapping_analysis_groups'`.

    Returns:
        mapping of group name to its token list.

    Raises:
        ValueError: if `group_set` is missing or empty in `groups_file`.
    """
    raw = json.loads(Path(groups_file).read_text())
    groups = raw.get(group_set)
    if not groups:
        raise ValueError(f"'{group_set}' missing or empty in {groups_file}")
    return {name: list(tokens) for name, tokens in groups.items()}


def find_video_preds_dir(seed_path: str | Path) -> Path:
    """Find the single `video_preds` directory under a seed folder.

    Mirrors the helper of the same name in `pose2visual.qc.rainbow.loading` (kept local so this
    module does not depend on the qc package).

    Args:
        seed_path: path to a seed folder, or directly to a `video_preds` directory.

    Returns:
        absolute path to the `video_preds` directory.

    Raises:
        FileNotFoundError: if no `video_preds` directory is found.
        ValueError: if more than one `video_preds` directory is found.
    """
    seed_path = Path(seed_path).resolve()
    if seed_path.name == 'video_preds':
        return seed_path

    candidates = [c for c in seed_path.rglob('video_preds') if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(f'no video_preds directory found under {seed_path}')
    if len(candidates) > 1:
        raise ValueError(
            f'multiple video_preds directories found under {seed_path}: '
            f'{[str(c) for c in candidates]}',
        )
    return candidates[0]


def find_seed_csv(preds_dir: str | Path, session_id: str, camera: str) -> Path:
    """Find the prediction csv for one seed and camera.

    Args:
        preds_dir: path to a `video_preds` directory.
        session_id: session UUID.
        camera: camera name (`'left'` or `'right'`).

    Returns:
        path to the csv.

    Raises:
        FileNotFoundError: if the csv does not exist.
    """
    csv_path = Path(preds_dir) / f'_iblrig_{camera}Camera.downsampled.{session_id}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'csv not found: {csv_path}')
    return csv_path



def parse_s_frames(spec: list[str] | None) -> list[tuple[int | None, int | None]] | None:
    """Parse ``--s-frames`` ranges into the ``(start, end)`` tuples eks expects.

    Args:
        spec: strings of the form ``'start:end'`` (half-open, 0-based); either side may be empty
            for an open end, e.g. ``':100'`` or ``'50:'``. ``None`` or an empty list means every
            frame.

    Returns:
        list of ``(start, end)`` tuples, or ``None`` if `spec` is empty.

    Raises:
        ValueError: if an entry is not of the form ``'start:end'``.
    """
    if not spec:
        return None
    ranges: list[tuple[int | None, int | None]] = []
    for item in spec:
        parts = item.split(':')
        if len(parts) != 2:
            raise ValueError(f"s-frames range {item!r} must look like 'start:end'")
        start, end = (int(part) if part else None for part in parts)
        ranges.append((start, end))
    return ranges


def fit_eks_singlecam_camera(
    seed_csvs: list[Path],
    start_frame: int,
    n_frames: int | None,
    smooth_param: float | None,
    s_frames: list[tuple[int | None, int | None]] | None,
) -> tuple[pd.DataFrame, list[float], list[str]]:
    """Run the single-camera Ensemble Kalman Smoother on one camera's seed ensemble.

    Mirrors ``eks.fit_eks_singlecam`` (ensemble median + confidence-weighted variance, independent
    random-walk state per keypoint, one fitted smoothing scale per keypoint), except that it can
    restrict the fit to a range of frames.

    Args:
        seed_csvs: the ensemble's prediction csvs, all for the same camera.
        start_frame: first frame to fit.
        n_frames: number of frames to fit, or `None` for every frame available (the minimum row
            count across `seed_csvs`, past `start_frame`).
        smooth_param: fixed smoothing scale shared by every keypoint, or `None` to fit one per
            keypoint.
        s_frames: frame ranges (relative to `start_frame`) used to fit the smoothing scale, or
            `None` for all frames; the final smoothing always covers every frame.

    Returns:
        `(df_smoothed, s_finals, keypoint_names)`: the eks output frame (9 coords per keypoint, see
        the module docstring), the smoothing scale used for each keypoint, and the keypoint names
        in the same order as `s_finals`.
    """
    # lazy: avoids requiring the eks package (and its jax dependency) for callers that only use
    # the helpers above, and for cpu-only test/dev environments without eks.
    from eks.marker_array import input_dfs_to_markerArray
    from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
    from eks.utils import format_data

    logger.info(f'  loading {len(seed_csvs)} seed csvs')
    dfs, keypoint_names = format_data([str(p) for p in seed_csvs])
    end_frame = start_frame + n_frames if n_frames is not None else min(len(df) for df in dfs)
    dfs = [df.iloc[start_frame:end_frame].reset_index(drop=True) for df in dfs]
    logger.info(
        f'  fitting {len(keypoint_names)} keypoints over frames [{start_frame}, {end_frame}) '
        f'({end_frame - start_frame} total)',
    )

    marker_array = input_dfs_to_markerArray([dfs], keypoint_names, [''])
    df_smoothed, s_finals = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=[],
        avg_mode='median',
        var_mode='confidence_weighted_var',
    )
    logger.info('  kalman smoother done')
    return df_smoothed, [float(s) for s in s_finals], list(keypoint_names)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Fit single-camera Pose-EKS smoothers (one csv per session and camera) for '
            'pose2visual.qc.rainbow.'
        ),
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
    parser.add_argument(
        '--sessions',
        nargs='+',
        required=True,
        help='Session IDs (UUIDs); all are fit in one run',
    )
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
        help='First frame to fit (default: 0)',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to fit (default: every frame available)',
    )
    parser.add_argument(
        '--smooth-param',
        type=float,
        default=None,
        help='Fixed smoothing scale s shared by every keypoint; smaller values smooth more '
        '(default: fit one s per keypoint)',
    )
    parser.add_argument(
        '--s-frames',
        nargs='+',
        default=None,
        help="Frame ranges 'start:end' (half-open, relative to --start-frame; either side may be "
        "empty) used only to fit s, to speed up the fit, e.g. ':5000'. Ignored with "
        '--smooth-param. The final smoothing always covers every frame (default: all frames)',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Run folder; writes csvs under <out-dir>/eks and the manifest at '
        '<out-dir>/manifest.json (default: <model-dir>/eks/<timestamp>[__<tag>])',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional tag appended to the default run folder name',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Worker processes to fit (session, camera) tasks in parallel (default: 1, '
        'sequential). Every task is an independent fit, so up to len(--sessions) * '
        'len(--cameras) workers are useful; memory use grows with the number of workers',
    )
    return parser


def _resolve_seed_dirs(model_dir: Path, seed_dirs: list[str]) -> dict[str, Path]:
    """Resolve seed folder names (or absolute paths) against a model root.

    Args:
        model_dir: root directory holding seed folders.
        seed_dirs: seed folder names (relative to `model_dir`) or absolute paths.

    Returns:
        mapping of seed folder name to its resolved, existing path.

    Raises:
        FileNotFoundError: if a resolved seed path does not exist.
    """
    resolved = {}
    for seed_name in seed_dirs:
        seed_path = Path(seed_name)
        if not seed_path.is_absolute():
            seed_path = model_dir / seed_name
        seed_path = seed_path.resolve()
        if not seed_path.exists():
            raise FileNotFoundError(f'seed directory not found: {seed_path}')
        resolved[seed_name] = seed_path
    return resolved


def main(argv: list[str] | None = None) -> None:
    """Entry point: fit and write one smoothed csv per (session, camera).

    All sessions of one invocation share a single run folder, manifest and registry entry.
    A failing fit is logged and recorded in the manifest's `failed` list without stopping the
    other fits.

    Args:
        argv: command line arguments; defaults to `sys.argv[1:]`.

    Raises:
        ValueError: if `--sessions` contains duplicates.
        FileNotFoundError: if a seed folder or a seed prediction csv is missing (raised before
            any fit starts).
        RuntimeError: if any fit failed, after the other fits' outputs and the manifest are
            written.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)

    if len(set(args.sessions)) != len(args.sessions):
        raise ValueError(f'duplicate session ids in --sessions: {args.sessions}')

    model_dir = Path(args.model_dir).resolve()
    seed_dirs = _resolve_seed_dirs(model_dir, args.seed_dirs)
    s_frames = parse_s_frames(args.s_frames)

    # resolve every task's seed csvs up front so a missing file aborts before any fit starts
    preds_dirs = {name: find_video_preds_dir(path) for name, path in seed_dirs.items()}
    seed_csvs: dict[tuple[str, str], list[Path]] = {}
    for session in args.sessions:
        for camera in args.cameras:
            seed_csvs[session, camera] = []
            for seed_name, preds_dir in preds_dirs.items():
                csv_path = find_seed_csv(preds_dir, session, camera)
                seed_csvs[session, camera].append(csv_path)
                logger.info(f'resolved {session[:8]}/{seed_name}/{camera}: {csv_path}')
    tasks = list(seed_csvs)
    n_tasks = len(tasks)

    run_name = datetime.now(RUN_TIMEZONE).strftime('%Y%m%d_%H%M%S')
    if args.tag:
        run_name = f'{run_name}__{args.tag}'

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = model_dir / 'eks' / run_name
    csv_dir = out_dir / 'eks'
    csv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'output: {out_dir}')

    outputs: dict[str, dict[str, dict]] = {}
    failed: list[dict[str, str]] = []

    def _write_output(
        session: str,
        camera: str,
        result: tuple[pd.DataFrame, list[float], list[str]],
    ) -> None:
        """Save one task's csv and record it in `outputs` — always runs in the main process."""
        df_smoothed, s_finals, keypoints = result
        out_path = csv_dir / f'_iblrig_{camera}Camera.downsampled.{session}.csv'
        df_smoothed.to_csv(out_path)
        outputs.setdefault(session, {})[camera] = {
            'path': str(out_path),
            'n_frames': len(df_smoothed),
            'keypoints': keypoints,
            's_finals': dict(zip(keypoints, s_finals)),
        }
        logger.info(f'wrote {out_path} ({len(df_smoothed)} frames, {len(keypoints)} keypoints)')

    def _record_failure(session: str, camera: str, error: Exception) -> None:
        """Log one failed task and remember it for the manifest."""
        logger.error(f'{session[:8]}/{camera} failed: {error!r}')
        failed.append({'session': session, 'camera': camera, 'error': repr(error)})

    fit_args = (args.start_frame, args.n_frames, args.smooth_param, s_frames)
    if args.workers <= 1:
        for task_idx, (session, camera) in enumerate(tasks, start=1):
            logger.info(f'[{task_idx}/{n_tasks}] fitting {session[:8]}/{camera}...')
            try:
                result = fit_eks_singlecam_camera(seed_csvs[session, camera], *fit_args)
            except Exception as error:
                _record_failure(session, camera, error)
                continue
            _write_output(session, camera, result)
    else:
        # each worker's own jax/blas calls shouldn't also try to grab many threads — without
        # this, `workers` processes each using all cores oversubscribes the allocation.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        n_workers = min(args.workers, n_tasks)
        logger.info(f'fitting {n_tasks} tasks across {n_workers} worker processes...')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=multiprocessing.get_context('fork'),
        ) as pool:
            futures = {
                pool.submit(fit_eks_singlecam_camera, seed_csvs[task], *fit_args): task
                for task in tasks
            }
            for task_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                session, camera = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    _record_failure(session, camera, error)
                    continue
                logger.info(f'[{task_idx}/{n_tasks}] {session[:8]}/{camera} done')
                _write_output(session, camera, result)

    # completion order varies with --workers; keep the manifest in --sessions/--cameras order
    outputs = {
        session: {
            camera: outputs[session][camera]
            for camera in args.cameras if camera in outputs.get(session, {})
        }
        for session in args.sessions if session in outputs
    }
    failed.sort(key=lambda item: (args.sessions.index(item['session']), item['camera']))

    manifest = {
        'run_name': run_name,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_dir': str(model_dir),
        'seed_dirs': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'sessions': args.sessions,
        'cameras': args.cameras,
        'start_frame': args.start_frame,
        'n_frames_requested': args.n_frames,
        'smooth_param': args.smooth_param,
        's_frames': args.s_frames,
        'tag': args.tag,
        'outputs': outputs,
        'failed': failed,
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    logger.info(f'wrote {manifest_path}')

    registry_path = model_dir / 'eks' / 'registry.json'
    append_entry(registry_path, run_name, manifest)
    logger.info(f'updated registry: {registry_path}')

    if failed:
        raise RuntimeError(
            f'{len(failed)} of {n_tasks} fits failed: '
            + ', '.join(f"{item['session'][:8]}/{item['camera']}" for item in failed),
        )
    logger.info(f'done: wrote {n_tasks} smoothed csvs under {csv_dir}')


if __name__ == '__main__':
    main()
