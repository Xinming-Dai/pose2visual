"""Fit Pose-EKS rank-reduced latent smoothers for ``pose2visual.qc.rainbow``.

For one session, fits one rank-reduced (probabilistic-PCA-style) Ensemble Kalman Smoother per
body-part group (from a ``body_parts.json``-style file's ``nonoverlapping_analysis_groups``) per
camera, using the `eks <https://github.com/paninski-lab/eks>`_ package. Each group's landmarks
are fit jointly, independently per camera and independently from every other group, using
mixture ("ensemble") variance across the seeds as the observation uncertainty.

``eks`` has no public API that exposes the raw latent means and loading matrix for this case
(its multi-camera smoother assumes real camera views sharing one keypoint, not multiple
landmarks sharing one camera). This module calls ``eks``'s lower-level functions directly with a
"pseudo-camera" trick: it places each group's landmarks on the *camera* axis of the
``MarkerArray`` eks expects, so its per-keypoint PCA ends up fit jointly across the whole group's
(x, y) instead of once per individual landmark.

For each group and camera, writes a ``<group>_<camera>_latent-model.npz`` with the four arrays
``pose2visual.qc.rainbow`` reads (``mean``, ``smoothed_means``, ``W``, ``observation_medians``,
such that ``mean + smoothed_means @ W.T`` reconstructs the smoothed (x, y) for every landmark in
the group) plus a ``columns`` array labelling each observed dimension (e.g. ``'ear_top_left_x'``),
so the npz is self-describing for anyone loading it later without the run's log output.

Needs only ``pandas``, ``numpy``, ``jax`` and ``eks`` (``pip install
git+https://github.com/paninski-lab/eks.git`` — it is not on PyPI under this name); no
``litpose``/``lightning_pose`` import, so any conda env with those four is fine. ``eks`` does
not require a GPU.

Usage::

    PYTHONPATH=src python -m pose2visual.lightningpose.fit_eks_latent_model \\
      --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
      --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
      --session 9b528ad0-4599-4a55-9148-96cc1d93fb24

Default output:
``<model-dir>/eks/<timestamp>[__<tag>]/<session>/<group>/<group>_<camera>_latent-model.npz``
(folder name is ``YYYYMMDD_HHMMSS``, suffixed with ``__<tag>`` when ``--tag`` is given,
mirroring ``pose2visual.qc.rainbow``'s own ``<model-dir>/rainbow_qc/<timestamp>[__<tag>]``
convention). Every run's manifest is also appended to ``<model-dir>/eks/registry.json``
(keyed by folder name). Pass ``--out-root`` to override the output location, or
``prune_run.py`` to delete a run and its registry entry together.
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.run_registry import append_entry

logger = logging.getLogger(__name__)


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


def fit_latent_model(
    seed_csvs: list[Path],
    columns: list[str],
    n_latent: int,
    start_frame: int,
    n_frames: int | None,
) -> dict[str, np.ndarray]:
    """Fit a rank-`n_latent` latent smoother jointly over `columns`, for one camera.

    Builds an `eks` `MarkerArray` with each of `columns` placed on the *camera* axis (a trick to
    make `eks`'s per-keypoint PCA span all of `columns`' (x, y) jointly, rather than fitting one
    PCA per column independently), so the Kalman smoother fits one shared rank-`n_latent` latent
    space across the whole group.

    This call sequence was derived by reading `eks` source on GitHub, not by running it (it was
    not installed anywhere reachable at the time this was written). Before trusting it on real
    data, smoke-test it against a tiny synthetic ensemble and confirm every intermediate shape
    matches the comments below, on a compute node — not on an HPC login node, and not by this
    function's docstring alone.

    Args:
        seed_csvs: the ensemble's prediction csvs, all for the same camera.
        columns: resolved per-camera column names to fit jointly (see `resolve_group_bases` and
            `group_columns_for_camera`).
        n_latent: latent rank; must be strictly less than `2 * len(columns)`.
        start_frame: first frame to fit.
        n_frames: number of frames to fit, or `None` for every frame available (the minimum row
            count across `seed_csvs`, past `start_frame`).

    Returns:
        dict with `mean` (`2 * len(columns)`,), `smoothed_means` (`n_frames`, `n_latent`),
        `W` (`2 * len(columns)`, `n_latent`), `observation_medians` (`n_frames`,
        `2 * len(columns)`), and `columns` (`2 * len(columns)`,) — the per-observed-dimension
        label (e.g. `'ear_top_left_x'`), in the same order as the rows of `mean`/`W` and the
        columns of `observation_medians`, so the npz is self-describing without the caller
        needing to separately track column order or the x/y interleaving convention.

    Raises:
        ValueError: if `n_latent` is not in `[1, 2 * len(columns))`.
    """
    if not 1 <= n_latent < 2 * len(columns):
        raise ValueError(
            f'n_latent={n_latent} must be in [1, {2 * len(columns)}) for {len(columns)} columns',
        )

    # lazy: avoids requiring the eks package (and its jax dependency) for callers that only use
    # the group-resolution helpers above, and for cpu-only test/dev environments without eks.
    import jax.numpy as jnp
    from eks.core import ensemble, run_kalman_smoother
    from eks.marker_array import MarkerArray, input_dfs_to_markerArray, mA_to_stacked_array
    from eks.multicam_smoother import initialize_kalman_filter_pca
    from eks.stats import compute_pca
    from eks.utils import center_predictions, format_data

    logger.info(f'  loading {len(seed_csvs)} seed csvs for columns {columns}')
    dfs, _ = format_data([str(p) for p in seed_csvs])
    end_frame = start_frame + n_frames if n_frames is not None else min(len(df) for df in dfs)
    dfs = [df.iloc[start_frame:end_frame].reset_index(drop=True) for df in dfs]
    logger.info(f'  fitting frames [{start_frame}, {end_frame}) ({end_frame - start_frame} total)')

    marker_array = input_dfs_to_markerArray([dfs], columns, ['view'])
    # pseudo-camera trick: move the per-landmark axis onto the camera axis so eks's per-keypoint
    # pca spans all of `columns` jointly, instead of fitting one pca per landmark.
    marker_array = MarkerArray(
        np.transpose(marker_array.get_array(), (0, 3, 2, 1, 4)),
        data_fields=['x', 'y', 'likelihood'],
    )

    logger.info('  computing ensemble mean/variance across seeds...')
    ens = ensemble(marker_array, avg_mode='median', var_mode='confidence_weighted_var')
    observation_medians = np.asarray(mA_to_stacked_array(ens.slice_fields('x', 'y'), 0))

    logger.info(f'  fitting rank-{n_latent} pca basis...')
    valid_mask, emA_centered, emA_good_centered, emA_means = center_predictions(
        ens, quantile_keep_pca=50.0,
    )
    pcas, good_pcs = compute_pca(
        valid_mask, emA_centered, emA_good_centered, n_components=n_latent,
    )
    m0, s0, a_mats, q_mats, c_mats = initialize_kalman_filter_pca(
        good_pcs, pcas, n_latent=n_latent,
    )

    logger.info('  running kalman smoother...')
    ys = np.stack([mA_to_stacked_array(emA_centered, 0)])
    ensemble_vars = np.stack([mA_to_stacked_array(ens.slice_fields('var_x', 'var_y'), 0)])
    _, smoothed_means_per_group, _ = run_kalman_smoother(
        ys=jnp.asarray(ys),
        m0s=m0,
        S0s=s0,
        As=a_mats,
        Qs=q_mats,
        Cs=c_mats,
        ensemble_vars=np.swapaxes(ensemble_vars, 0, 1),
        s_frames=None,
        smooth_param=None,
    )
    logger.info('  kalman smoother done')

    # one label per observed dimension (x, y interleaved per landmark, matching eks's own
    # stacking order), so mean/W/observation_medians are self-describing without the caller
    # needing to separately remember `columns` or the x/y interleaving convention.
    obs_columns = np.array([f'{column}_{axis}' for column in columns for axis in ('x', 'y')])

    return {
        'mean': np.asarray(mA_to_stacked_array(emA_means, 0)[0]),
        'smoothed_means': np.asarray(smoothed_means_per_group[0]),
        'W': np.asarray(c_mats[0]),
        'observation_medians': observation_medians,
        'columns': obs_columns,
    }


def load_latent_model(npz_path: str | Path) -> dict:
    """Load a fitted latent model npz and reconstruct smoothed (x, y) per landmark.

    `n_landmarks` is derived from the npz itself (`len(mean) // 2`), never hardcoded, so this
    works for any group size — unlike the reshape this replaces in `pose2visual.qc.rainbow`.

    Args:
        npz_path: path to a `<group>_<camera>_latent-model.npz` written by `fit_latent_model`.

    Returns:
        dict with:
            `xy`: `(n_frames, n_landmarks, 2)`, i.e. `mean + smoothed_means @ W.T` reshaped.
            `observation_medians_xy`: `(n_frames, n_landmarks, 2)`, `observation_medians`
                reshaped the same way.
            `landmarks`: list of `n_landmarks` base column names (e.g. `'ear_top_left'`, no
                `_x`/`_y` suffix), in `xy`'s landmark order; `None` if the npz predates the
                `columns` array (reconstruction still works — only the names are unavailable).
            the raw arrays `mean`, `W`, `smoothed_means`, `observation_medians`, and `columns`
                (if present), for callers that want them directly.
    """
    data = np.load(npz_path)
    n_landmarks = len(data['mean']) // 2
    n_frames = data['smoothed_means'].shape[0]
    recon = data['mean'] + data['smoothed_means'] @ data['W'].T

    landmarks = None
    if 'columns' in data.files:
        seen: dict[str, None] = {}
        for column in data['columns']:
            seen.setdefault(str(column).rsplit('_', 1)[0], None)
        landmarks = list(seen)

    result = {
        'xy': recon.reshape(n_frames, n_landmarks, 2),
        'observation_medians_xy': data['observation_medians'].reshape(n_frames, n_landmarks, 2),
        'landmarks': landmarks,
        'mean': data['mean'],
        'W': data['W'],
        'smoothed_means': data['smoothed_means'],
        'observation_medians': data['observation_medians'],
    }
    if 'columns' in data.files:
        result['columns'] = data['columns']
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Fit per-group, per-camera Pose-EKS rank-reduced latent smoothers for '
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
    parser.add_argument('--session', required=True, help='Session ID (UUID)')
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=['left', 'right'],
        help='Camera names to process (default: left right)',
    )
    parser.add_argument(
        '--groups-file',
        default='scripts/delta/data/body_parts.json',
        help="Path to a body_parts.json-style file with 'nonoverlapping_analysis_groups'",
    )
    parser.add_argument(
        '--group-set',
        choices=['nonoverlapping_analysis_groups', 'overlapping_analysis_groups'],
        default='nonoverlapping_analysis_groups',
        help="Top-level key to load from --groups-file (default: nonoverlapping_analysis_groups)",
    )
    parser.add_argument(
        '--groups',
        nargs='+',
        default=None,
        help='Subset of group names to fit (default: every group in --groups-file/--group-set)',
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
        '--out-root',
        default=None,
        help='Output root; writes <out-root>/<session>/<group>/<group>_<camera>_'
        'latent-model.npz (default: <model-dir>/eks/<timestamp>[__<tag>], mirroring '
        'pose2visual.qc.rainbow\'s <model-dir>/rainbow_qc/<timestamp>[__<tag>])',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional tag appended to the default output folder name',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Worker processes to fit groups/cameras in parallel (default: 1, sequential). '
        'The per-(group, camera) fits are fully independent, so this parallelizes well; set '
        'to the number of cpus requested (e.g. --cpus-per-task in sbatch)',
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
    """Entry point: fit every selected group's latent model for every camera.

    Args:
        argv: command line arguments; defaults to `sys.argv[1:]`.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)

    model_dir = Path(args.model_dir).resolve()
    seed_dirs = _resolve_seed_dirs(model_dir, args.seed_dirs)
    groups = load_groups(args.groups_file, group_set=args.group_set)
    if args.groups is not None:
        missing = sorted(set(args.groups) - set(groups))
        if missing:
            raise ValueError(f'groups not found in {args.groups_file}: {missing}')
        groups = {name: groups[name] for name in args.groups}

    seed_csvs: dict[str, list[Path]] = {camera: [] for camera in args.cameras}
    for seed_name, seed_path in seed_dirs.items():
        preds_dir = find_video_preds_dir(seed_path)
        for camera in args.cameras:
            csv_path = find_seed_csv(preds_dir, args.session, camera)
            seed_csvs[camera].append(csv_path)
            logger.info(f'resolved {seed_name}/{camera}: {csv_path}')

    keypoint_names = discover_keypoint_names(seed_csvs[args.cameras[0]][0])
    logger.info(f'discovered {len(keypoint_names)} keypoints; fitting {len(groups)} groups')

    run_name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    if args.tag:
        run_name = f'{run_name}__{args.tag}'

    if args.out_root is not None:
        out_root_base = Path(args.out_root).resolve()
    else:
        out_root_base = model_dir / 'eks' / run_name
    out_root = out_root_base / args.session
    logger.info(f'output root: {out_root}')

    # resolve every (group, camera) task's columns up front; cheap, no eks involved.
    tasks = []  # (group_name, camera, columns, n_latent, landmarks)
    outputs: dict[str, dict[str, dict]] = {name: {} for name in groups}
    for group_idx, (group_name, tokens) in enumerate(groups.items(), start=1):
        bases = resolve_group_bases(keypoint_names, tokens)
        n_latent = len(bases)
        landmarks = [base for base, _ in bases]
        logger.info(
            f'[group {group_idx}/{len(groups)}] {group_name}: '
            f'{len(bases)} landmarks ({landmarks}), n_latent={n_latent}',
        )
        for camera in args.cameras:
            columns = group_columns_for_camera(bases, camera)
            tasks.append((group_name, camera, columns, n_latent, landmarks))
    n_tasks = len(tasks)

    def _write_output(
        group_name: str, camera: str, n_latent: int, landmarks: list[str], result: dict,
    ) -> None:
        """Save one task's npz and record it in `outputs` — always runs in the main process."""
        out_path = out_root / group_name / f'{group_name}_{camera}_latent-model.npz'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **result)
        n_frames_fit = result['smoothed_means'].shape[0]
        outputs[group_name][camera] = {
            'path': str(out_path),
            'n_frames': n_frames_fit,
            'n_latent': n_latent,
            'landmarks': landmarks,
        }
        logger.info(f'wrote {out_path} ({n_frames_fit} frames, start_frame={args.start_frame})')

    if args.workers <= 1:
        for task_idx, (group_name, camera, columns, n_latent, landmarks) in enumerate(
            tasks, start=1,
        ):
            logger.info(f'[{task_idx}/{n_tasks}] fitting {group_name}/{camera}...')
            result = fit_latent_model(
                seed_csvs[camera], columns, n_latent, args.start_frame, args.n_frames,
            )
            _write_output(group_name, camera, n_latent, landmarks, result)
    else:
        # each worker's own jax/blas calls shouldn't also try to grab many threads — without
        # this, `workers` processes each using all cores oversubscribes the allocation.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        logger.info(f'fitting {n_tasks} tasks across {args.workers} worker processes...')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context('fork'),
        ) as pool:
            futures = {
                pool.submit(
                    fit_latent_model, seed_csvs[camera], columns, n_latent,
                    args.start_frame, args.n_frames,
                ): (group_name, camera, n_latent, landmarks)
                for group_name, camera, columns, n_latent, landmarks in tasks
            }
            for task_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                group_name, camera, n_latent, landmarks = futures[future]
                logger.info(f'[{task_idx}/{n_tasks}] {group_name}/{camera} done')
                _write_output(group_name, camera, n_latent, landmarks, future.result())

    manifest = {
        'run_name': run_name,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_dir': str(model_dir),
        'seed_dirs': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'session': args.session,
        'cameras': args.cameras,
        'groups_file': str(args.groups_file),
        'group_set': args.group_set,
        'groups': groups,
        'start_frame': args.start_frame,
        'n_frames_requested': args.n_frames,
        'tag': args.tag,
        'outputs': outputs,
    }
    manifest_path = out_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    logger.info(f'wrote {manifest_path}')

    registry_path = model_dir / 'eks' / 'registry.json'
    append_entry(registry_path, run_name, manifest)
    logger.info(f'updated registry: {registry_path}')

    logger.info(f'done: wrote {n_tasks} latent models under {out_root}')


if __name__ == '__main__':
    main()
