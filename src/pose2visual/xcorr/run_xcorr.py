"""Step 1: neural-behavior cross-correlation of one session (Figure 20 of the progress report).

Reads the continuous spikes of step 0 (``extract_neural_continuous``) and, for every camera
(``--cameras``, default left and right), builds the behavioral traces on the same bin grid with
the encoding pipeline's functions (``behavior``). The neural activity is smoothed like the RRR
encoder's targets, every neuron is cross-correlated with every behavior over
``+-xcorr.max_lag_sec`` (``r(tau) = Corr(neural(t), behavior(t + tau))``, positive lag = neural
activity first). Optionally (``--n-permutations``, off by default) the peak of each curve is tested
against a circular-shift null (``permutation``); all cameras share the same shifts and one
multiple-comparison family. The
two time axes are never assumed to match: the behavior grid must equal the spike bin grid, the csv
must have one row per frame timestamp, and the frame times must be increasing; the video must lie
inside the spike recording (step 0).

The yaml is copied once at the start (``settings.freeze_config``), so editing it while the job
runs changes nothing; the copy is saved as ``xcorr_config.yaml``. ``session.camera`` is not
used. ``--exclude`` drops keypoints by glob pattern (``--exclude 'd*'`` = the fingers), and
``--n-jobs 2`` processes the two cameras in parallel.

This is Experiment A (baseline). Writes under
``<video_preds_dir>/<xcorr.output_name>/<eid>/experiment_A_baseline/``:

* ``xcorr_summary.<eid>.csv`` — one row per camera-neuron-behavior pair: ids, peak signed
  correlation, peak absolute correlation, lag of the peak and, when the permutation test was
  run, permutation p, adjusted q and significance.
* ``xcorr_arrays.<eid>.npz`` — correlations ``r [C, N, B, L]`` (NaN where a camera lacks a
  behavior), lags, names, cameras, neuron order, p values and the null 95th percentile.
* ``xcorr_params.json`` and ``xcorr_config.yaml`` — settings and alignment diagnostics.
* ``<YYYYmmdd_HHMMSS>/`` (New York time of the run; ``--plots-subdir`` names it instead) —
  ``heatmap.all.<eid>.{png,pdf}``: every behavior on one figure,
  rows = cameras (Figure 20), each row labelled with its own camera's keypoint names; and
  ``example_curves.<eid>.{png,pdf}`` (only with ``--n-examples N``, off by default).
  ``replot`` redraws them from the files above without recomputing anything.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.run_xcorr \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --video-timestamps-dir <dir> --cameras left right --exclude 'd*' --n-jobs 2
"""

import argparse
import json
import logging
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    ANATOMICAL_GROUP_ALL,
    CAMERAS,
    DEFAULT_PCA_GROUP_SET,
    load_encoding_yaml,
)
from pose2visual.lightningpose.fit_eks_latent_model import load_groups
from pose2visual.xcorr.behavior import build_continuous_behavior
from pose2visual.xcorr.crosscorr import (
    constant_columns,
    cross_correlation,
    smooth_neural,
    summarize_peaks,
)
from pose2visual.xcorr.ordering import ORDERS, neuron_order
from pose2visual.xcorr.permutation import (
    correct_pvalues,
    draw_shifts,
    observed_and_null,
    permutation_pvalues,
)
from pose2visual.xcorr.peth import load_peth
from pose2visual.xcorr.plots import DEFAULT_N_EXAMPLES, timestamp_subdir, write_plots
from pose2visual.xcorr.settings import (
    baseline_dir,
    freeze_config,
    load_xcorr_settings,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

OTHER_GROUP = 'other'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Cross-correlate neural activity with keypoint-derived behavior.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument(
        '--video-timestamps-dir', type=str, required=True,
        help='directory containing _ibl_{left,right}Camera.times.<eid>.npy.',
    )
    parser.add_argument(
        '--cameras', type=str, nargs='+', choices=CAMERAS, default=list(CAMERAS),
        help='cameras to analyse and draw as rows of one figure (default: left right).',
    )
    parser.add_argument(
        '--exclude', type=str, nargs='*', default=[], metavar='PATTERN',
        help="glob patterns of keypoints / features to leave out, e.g. 'd*' for the fingers "
        '(d1_tip_left ... d4_tip_right); quote them so the shell does not expand them.',
    )
    parser.add_argument(
        '--order', type=str, choices=ORDERS, default='rastermap',
        help='neuron order of every heatmap strip (default: rastermap).',
    )
    parser.add_argument(
        '--n-permutations', type=int, default=None,
        help='override xcorr.n_permutations; 0 skips the significance test.',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='cameras processed in parallel (default: 1; 2 = one process per camera).',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='fix the colour scale to +-VMAX (Figure 20 uses 0.6); default: from the data.',
    )
    parser.add_argument(
        '--n-examples', type=int, default=DEFAULT_N_EXAMPLES,
        help='number of example curves to draw (default 0 = none; 12 was the former default).',
    )
    parser.add_argument(
        '--no-peth', action='store_true',
        help='do not draw the reward / movement-onset event strips (xcorr and PETH) even if '
        'peth.<eid>.npz exists.',
    )
    parser.add_argument(
        '--peth-vmax', type=float, default=1.0,
        help='half-range of the PETH colour scale (default: 1.0 = a doubling of the rate).',
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='folder of the figures inside experiment_A_baseline (default: a new folder named by '
        'the New York time of the run, YYYYmmdd_HHMMSS, so no earlier figure is overwritten).',
    )
    return parser.parse_args(argv)


def behavior_groups(names: list[str], groups: dict[str, list[str]]) -> dict[str, list[int]]:
    """Assign each behavior to its body-part group by the first `_`-separated name segment.

    Args:
        names: behavior names (keypoint names, `<group>_visible` features).
        groups: group name -> tokens, from `body_parts.json`.

    Returns:
        Group name -> indices into `names`, in `groups` order; behaviors without a group go to
        `'other'`; empty groups are left out.
    """
    out: dict[str, list[int]] = {}
    for group, tokens in groups.items():
        idx = [i for i, name in enumerate(names) if name.split('_')[0] in tokens]
        if idx:
            out[group] = idx
    claimed = {i for idx in out.values() for i in idx}
    rest = [i for i in range(len(names)) if i not in claimed]
    if rest:
        out[OTHER_GROUP] = rest
    return out


def grouped_behaviors(
    cfg: dict[str, Any],
    camera_names: list[list[str]],
) -> tuple[list[str], list[str]]:
    """Union of the behaviors of every camera, ordered and labelled by anatomical group.

    Args:
        cfg: encoding yaml (`anatomical_group`, `pca_group_set`, `body_parts_json`).
        camera_names: behavior names of each camera.

    Returns:
        `(names, groups)`: the union of the names, grouped by body part, and the group of each.
    """
    group_set = cfg.get('anatomical_group')
    if group_set in (None, ANATOMICAL_GROUP_ALL):
        group_set = cfg.get('pca_group_set') or DEFAULT_PCA_GROUP_SET
    body_groups = load_groups(cfg['body_parts_json'], str(group_set))
    union = list(dict.fromkeys(name for names in camera_names for name in names))
    column_groups = behavior_groups(union, body_groups)
    names = [union[idx] for idxs in column_groups.values() for idx in idxs]
    groups = [group for group, idxs in column_groups.items() for _ in idxs]
    return names, groups


def _camera_job(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    intervals: np.ndarray,
    bin_right_edges: np.ndarray,
    binsize: float,
    exclude: list[str],
    neural: np.ndarray,
    max_lag_bins: int,
    shifts: np.ndarray | None,
) -> dict[str, Any]:
    """Behavior traces and cross-correlation of one camera (runs in a worker process).

    Args:
        config_path: frozen encoding yaml.
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with the camera timestamp files.
        intervals: contiguous video intervals of the neural bundle.
        bin_right_edges: spike bin grid of the neural bundle.
        binsize: bin size in seconds.
        exclude: glob patterns of behaviors to leave out.
        neural: smoothed neural activity ``[T, N]``.
        max_lag_bins: largest lag in bins.
        shifts: circular shifts of the permutation null, or `None` for no test.

    Returns:
        Dict with `camera`, `traces`, `names`, `r` ``[N, B, L]``, `null_max` (``[N, B, P]`` or
        `None`) and the diagnostics of `build_continuous_behavior`.

    Raises:
        RuntimeError: naming the camera, if its behavior cannot be built or its bin grid differs
            from the spike bin grid or its length from the spikes.
    """
    try:
        behavior = build_continuous_behavior(
            config_path, eid, camera, video_timestamps_dir, intervals, binsize, exclude,
        )
        # the two series must sit on the same bins; nothing is resampled or shifted to fit
        if behavior['grid'].shape != bin_right_edges.shape or not np.allclose(
            behavior['grid'], bin_right_edges, atol=1e-9,
        ):
            raise ValueError('behavior bin grid differs from the spike bin grid')
        traces = behavior['traces']
        if neural.shape[0] != traces.shape[0]:
            raise ValueError(f'{neural.shape[0]} spike bins but {traces.shape[0]} behavior bins')
        if traces.shape[1] == 0:
            raise ValueError('no behavior left to correlate')
        if shifts is not None:
            r, null_max = observed_and_null(neural, traces, max_lag_bins, shifts)
        else:
            r, null_max = cross_correlation(neural, traces, max_lag_bins), None
    except Exception as error:
        raise RuntimeError(f'{camera} camera: {error}') from error
    behavior.update({'camera': camera, 'r': r, 'null_max': null_max})
    return behavior


def run_cameras(
    cameras: list[str],
    n_jobs: int,
    **job_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run `_camera_job` for every camera, in parallel when `n_jobs > 1`.

    Args:
        cameras: camera names; results come back in this order.
        n_jobs: number of processes; 1 runs inline.
        **job_kwargs: the remaining arguments of `_camera_job`.

    Returns:
        One result dict per camera.

    Raises:
        RuntimeError: from the first failing camera (no camera is silently dropped).
    """
    workers = max(1, min(n_jobs, len(cameras)))
    if workers == 1:
        return [_camera_job(camera=camera, **job_kwargs) for camera in cameras]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_camera_job, camera=camera, **job_kwargs) for camera in cameras]
        return [future.result() for future in futures]


def run_session(
    config_path: str | Path,
    eid: str,
    video_timestamps_dir: str | Path,
    cameras: list[str] | None = None,
    order: str = 'rastermap',
    n_permutations: int | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    exclude: list[str] | None = None,
    n_jobs: int = 1,
    vmax: float | None = None,
    plots_subdir: str | None = None,
    show_peth: bool = True,
    peth_vmax: float = 1.0,
) -> dict[str, Path]:
    """Cross-correlate the neurons of one session with the behavior of every camera.

    Args:
        config_path: path of the encoding yaml (use `freeze_config` to protect it from edits).
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `CAMERAS`.
        order: neuron order of the heatmaps, one of `ORDERS`.
        n_permutations: overrides `xcorr.n_permutations`; 0 skips the significance test.
        n_examples: number of example curves; 0 (default) draws none.
        exclude: glob patterns of keypoints / features to leave out (e.g. `['d*']`).
        n_jobs: cameras processed in parallel.
        vmax: half-range of the heatmap colour scale; `None` takes it from the data.
        plots_subdir: folder of the figures inside `experiment_A_baseline`; `None` makes a new
            folder named by the current New York time (`plots.timestamp_subdir`).
        show_peth: draw the PETH strips when `peth.<eid>.npz` exists (see `peth`).
        peth_vmax: half-range of the PETH colour scale.

    Returns:
        Dict of the main outputs: `summary`, `arrays`, `params` and `heatmap` (the png).

    Raises:
        FileNotFoundError: if step 0 has not been run for the session.
        RuntimeError: if a camera's behavior cannot be built (the message names the camera).
        ValueError: if no neuron or camera is left, or `cameras` has duplicates or unknown names.
    """
    cameras = list(dict.fromkeys(cameras or CAMERAS))
    unknown = sorted(set(cameras) - set(CAMERAS))
    if unknown:
        raise ValueError(f'unknown cameras {unknown}; use {list(CAMERAS)}')
    exclude = list(exclude or [])
    cfg = load_encoding_yaml(config_path)
    settings = load_xcorr_settings(config_path)
    if n_permutations is not None:
        settings['n_permutations'] = n_permutations
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    out_dir = baseline_dir(config_path, eid, settings['output_name'])

    npz_path = session_dir / 'neural_continuous.npz'
    if not npz_path.is_file():
        raise FileNotFoundError(f'{npz_path} not found; run extract_neural_continuous first')
    neural_file = np.load(npz_path)
    spikes = neural_file['spikes']
    intervals = neural_file['intervals']
    bin_right_edges = neural_file['bin_right_edges']
    binsize = float(neural_file['binsize'])
    with (session_dir / f'{eid}_meta.pkl').open('rb') as f:
        meta = pickle.load(f)

    neural = smooth_neural(spikes, settings['smooth_sigma_bins'])
    keep = ~constant_columns(neural)
    if not keep.all():
        logger.warning(f'dropping {int((~keep).sum())} neurons without variance')
    neural = neural[:, keep]
    if neural.shape[1] == 0:
        raise ValueError('no neuron left to correlate')
    neuron_info = pd.DataFrame({
        'neuron_id': np.asarray(meta['uuids'], dtype=object)[keep],
        'region': np.asarray(meta['cluster_regions'], dtype=object)[keep],
        'depth': np.asarray(meta['cluster_depths'], dtype=np.float64)[keep],
        'channel': np.asarray(meta['cluster_channels'])[keep],
    })

    n_time = neural.shape[0]
    max_lag_bins = int(round(settings['max_lag_sec'] / binsize))
    lags_sec = np.arange(-max_lag_bins, max_lag_bins + 1) * binsize
    n_perm = int(settings['n_permutations'])
    shifts = None
    if n_perm > 0:
        rng = np.random.default_rng(settings['seed'])
        min_shift_bins = max(int(np.ceil(settings['min_shift_sec'] / binsize)), max_lag_bins + 1)
        shifts = draw_shifts(n_time, n_perm, min_shift_bins, rng)

    results = run_cameras(
        cameras, n_jobs,
        config_path=config_path, eid=eid, video_timestamps_dir=video_timestamps_dir,
        intervals=intervals, bin_right_edges=bin_right_edges, binsize=binsize,
        exclude=exclude, neural=neural, max_lag_bins=max_lag_bins, shifts=shifts,
    )
    logger.info(
        f'{neural.shape[1]} neurons, {n_time} bins ({n_time * binsize:.0f} s), '
        f'{lags_sec.size} lags; behaviors per camera: '
        f'{ {res["camera"]: len(res["names"]) for res in results} }',
    )

    # p values of every camera form one family for the multiple-comparison correction
    tables = []
    p_by_camera: list[np.ndarray | None] = [None] * len(results)
    q_by_camera: list[np.ndarray | None] = [None] * len(results)
    if shifts is not None:
        p_by_camera = [permutation_pvalues(res['r'], res['null_max']) for res in results]
        q_flat, sig_flat = correct_pvalues(
            np.concatenate([p.ravel() for p in p_by_camera]),
            settings['correction'], settings['alpha'],
        )
        logger.info(
            f'{int(sig_flat.sum())} / {sig_flat.size} pairs significant '
            f'({settings["correction"]}, alpha={settings["alpha"]})',
        )
        start = 0
        for idx, p in enumerate(p_by_camera):
            q_by_camera[idx] = q_flat[start:start + p.size].reshape(p.shape)
            start += p.size
    for res, p, q in zip(results, p_by_camera, q_by_camera):
        table = summarize_peaks(
            res['r'], lags_sec, neuron_info, res['names'], eid, res['camera'],
        )
        if p is not None:
            table['p_perm'] = p.ravel()
            table['q_value'] = q.ravel()
            table['significant'] = q.ravel() <= settings['alpha']
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)

    idx_order = neuron_order(order, neural, neuron_info['depth'].to_numpy())
    table['neuron_rank'] = np.argsort(idx_order)[table['neuron_idx'].to_numpy()]

    # union of the behaviors, grouped by body part; a camera without one gets NaN
    names, groups = grouped_behaviors(cfg, [res['names'] for res in results])
    r_all = np.full((len(results), neural.shape[1], len(names), lags_sec.size), np.nan)
    p_all = np.full((len(results), neural.shape[1], len(names)), np.nan)
    q_all = p_all.copy()
    for idx_cam, res in enumerate(results):
        for idx_b, name in enumerate(res['names']):
            idx_col = names.index(name)
            r_all[idx_cam, :, idx_col, :] = res['r'][:, idx_b, :]
            if p_by_camera[idx_cam] is not None:
                p_all[idx_cam, :, idx_col] = p_by_camera[idx_cam][:, idx_b]
                q_all[idx_cam, :, idx_col] = q_by_camera[idx_cam][:, idx_b]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f'xcorr_summary.{eid}.csv'
    table.to_csv(summary_path, index=False)
    arrays_path = out_dir / f'xcorr_arrays.{eid}.npz'
    arrays: dict[str, Any] = {
        'r': r_all.astype(np.float32),
        'lags_sec': lags_sec,
        'cameras': np.asarray(cameras),
        'behavior_names': np.asarray(names),
        'behavior_groups': np.asarray(groups),
        'valid': ~np.isnan(r_all).all(axis=(1, 3)),
        'neuron_ids': neuron_info['neuron_id'].to_numpy().astype(str),
        'neuron_order': idx_order,
    }
    if shifts is not None:
        arrays['p_perm'] = p_all
        arrays['q_value'] = q_all
        # 95th percentile of the null peak |r|: the threshold line of the example curves
        null_p95 = np.full_like(p_all, np.nan)
        for idx_cam, res in enumerate(results):
            for idx_b, name in enumerate(res['names']):
                null_p95[idx_cam, :, names.index(name)] = np.percentile(
                    res['null_max'][:, idx_b], 95, axis=1,
                )
        arrays['null_p95'] = null_p95
    np.savez_compressed(arrays_path, **arrays)
    shutil.copyfile(config_path, out_dir / 'xcorr_config.yaml')

    # figures: one overview with every behavior, and example curves, in their own subfolder
    peth = None
    if show_peth:
        peth = load_peth(out_dir, eid, neuron_info['neuron_id'].to_numpy())
        if peth is None:
            logger.info(
                'no PETH strips: run pose2visual.xcorr.extract_trial_events and '
                'pose2visual.xcorr.peth to add them',
            )
    plots_subdir = plots_subdir or timestamp_subdir()
    plots = write_plots(
        out_dir / plots_subdir, eid, r_all, lags_sec, names, groups, cameras,
        arrays['valid'], idx_order, table, order, vmax, n_examples, arrays.get('null_p95'),
        peth, peth_vmax,
    )

    # sync diagnostic only: where does the population rate peak against each behavior
    pop = neural.sum(axis=1, keepdims=True)
    per_camera: dict[str, Any] = {}
    for res in results:
        pop_r = cross_correlation(pop, res['traces'], max_lag_bins)[0]
        pop_peak_lag = lags_sec[np.argmax(np.abs(pop_r), axis=1)]
        logger.info(
            f'{res["camera"]}: population rate vs behavior, lag of peak |r|: median '
            f'{np.median(pop_peak_lag):+.3f} s, range [{pop_peak_lag.min():+.3f}, '
            f'{pop_peak_lag.max():+.3f}] s',
        )
        per_camera[res['camera']] = {
            'behaviors': res['names'],
            'dropped_constant_behaviors': res['dropped_constant'],
            'excluded': res['excluded'],
            'csv_path': res['csv_path'],
            'timestamps_path': res['timestamps_path'],
            'frame_timing': res['timing'],
            'masked_fraction': res['masked_fraction'],
            'population_rate_peak_lag_sec': {
                'median': float(np.median(pop_peak_lag)),
                'min': float(pop_peak_lag.min()),
                'max': float(pop_peak_lag.max()),
            },
        }

    params_path = out_dir / 'xcorr_params.json'
    with params_path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'eid': eid,
                'cameras': cameras,
                'settings': settings,
                'order': order,
                'exclude_patterns': exclude,
                'n_jobs': n_jobs,
                'binsize': binsize,
                'n_time_bins': n_time,
                'n_neurons': int(neural.shape[1]),
                'n_behaviors_union': len(names),
                'per_camera': per_camera,
                'n_significant': (
                    None if shifts is None else int(table['significant'].sum())
                ),
                'heatmap_vmax': plots['vmax'],
                'plots_subdir': plots_subdir,
                'peth_drawn': peth is not None,
            },
            f,
            indent=2,
        )
    return {
        'summary': summary_path, 'arrays': arrays_path, 'params': params_path,
        'heatmap': plots['heatmap'][0],
    }


def main(argv: list[str] | None = None) -> None:
    """Run step 1 for every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    # a private copy of the yaml: editing the original while the job runs changes nothing
    # one folder name for the whole job, so every session's figures carry the same run time
    plots_subdir = args.plots_subdir or timestamp_subdir()
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                run_session(
                    config, eid, args.video_timestamps_dir, args.cameras, args.order,
                    args.n_permutations, args.n_examples, args.exclude, args.n_jobs, args.vmax,
                    plots_subdir, not args.no_peth, args.peth_vmax,
                )
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
