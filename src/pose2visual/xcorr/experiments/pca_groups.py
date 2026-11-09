"""Experiment D: PCA of each anatomical group, first components of every group.

Question: does a couple of principal components per body part (eye, mouth, nose/pad, ear, forelimb)
carry the neuron relationship the baseline sees in the individual keypoints? Experiment B answers
it for one pooled face group and leaves the paws alone; this experiment fits one PCA per
anatomical group of ``body_parts.json`` (the group set of Experiment C, ``pca.group_set`` of the
yaml), on the group's keypoint coordinates of one camera. A camera only has the keypoints of its
own side (``same_side_filter``), so the left and right forelimb are never fitted together: the left
camera's forelimb group is the left wrist and left digits only.

For every group of a camera with two or more usable keypoints, the ``[n_frames, 2 * n_keypoints]``
position matrix is projected onto its top ``n_components`` principal components (the same
eigendecomposition as ``pose2visual.encoding.litpose_pca``, fit once per camera and group on every
frame of the session). Two kinds of cross-correlation follow from each fit:

* **PCA on positions**: the component scores, ``<group>_pc1_position``,
  ``<group>_pc2_position``, ..., are new behaviors cross-correlated with the neurons directly.
* **PCA on speed**: the positions are reprojected onto the same components (denoised, back in
  pixels), each keypoint's speed is recomputed from its denoised position with the baseline's
  speed function, and the result is cross-correlated under the keypoint's own name -- comparable,
  keypoint for keypoint, to the baseline.

Groups with a single keypoint (e.g. the visibility feature of the tongue is not a keypoint at all)
and keypoints in no group keep their baseline speed unchanged. The original speed
cross-correlations are recomputed with the baseline's own code (``common.compute_original``), so
this does not wait for Experiment A. Everything is written to
``<session>/results/experiment_D_pca_groups/<run>/`` (arrays, csv, json) and
``<session>/plot_video/experiment_D_pca_groups/<run>_<order>/`` (figures and their guide).

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.pca_groups \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --video-timestamps-dir <dir>
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import VISIBILITY_SUFFIX, speed_from_positions
from pose2visual.encoding.litpose_pca import project_group, reproject_group
from pose2visual.xcorr.experiments.common import (
    BASELINE_COLORBAR_LABEL,
    compute_original,
    relabel_overview,
    resolve_order,
    save_signed_comparison,
    write_figure_guide,
)
from pose2visual.xcorr.experiments.masked_crosscorr import masked_cross_correlation
from pose2visual.xcorr.experiments.pca import DEFAULT_N_COMPONENTS, anatomical_groups
from pose2visual.xcorr.experiments.relative_motion import (
    DEFAULT_MIN_VALID_FRACTION,
    MIN_RELIABLE_KEYPOINTS,
    bin_with_gaps,
    load_camera_positions,
    plan_groups,
    resolve_body_groups,
    screen_traces,
)
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.settings import (
    EXPERIMENT_PCA_GROUPS,
    freeze_config,
    new_run_dirs,
    resolve_eids,
)

logger = logging.getLogger(__name__)

SIDES = ('left', 'right')
# (stem, meaning) rows of the figure guide (`common.write_figure_guide`)
HEATMAP_GUIDE = (
    (
        'heatmap.original_speed',
        "baseline xcorr of each keypoint's absolute speed (the same quantity as Experiment A; "
        'same colour scale, neuron order, keypoints and lags as `heatmap.pca_speed`)',
    ),
    (
        'heatmap.pca_speed',
        "PCA on speed: each keypoint's position is reprojected onto the top principal components "
        "of its own group (denoised, back in pixels), its speed recomputed and cross-correlated "
        "under the keypoint's own name; single-keypoint groups and ungrouped keypoints keep the "
        'original xcorr',
    ),
)
POSITION_PCA_GUIDE = (
    'heatmap.position_pca',
    'PCA on positions: xcorr of the component scores themselves (`<group>_pc1_position`, '
    '`<group>_pc2_position`, ...), one column per group and component, with its own colour scale '
    '(see the figure title)',
)
SCATTER_GUIDE = (
    'scatter.peak_abs_r',
    'one panel per camera (row) and reduced anatomical group (column), same axes in every panel; '
    'one dot per neuron-keypoint pair, coloured per keypoint (same colour for both sides and in '
    'every camera and session): peak \\|r\\| over lags of the original speed (x) against the PCA '
    'speed (y). The sign is ignored and the two peaks may be at different lags (see '
    '`scatter.signed_r_at_original_peak`); below the dashed diagonal = the denoising lost '
    'correlation, above = gained; unreduced keypoints (on the diagonal by construction) are not '
    'drawn',
)
VARIANCE_GUIDE = (
    'bar.variance',
    'per keypoint, variance of position (top) and speed (bottom), original against '
    'PCA-denoised, log axis; `*` = PCA applied',
)
GROUPS_TABLE_GUIDE = (
    'groups_table',
    'which groups got PCA and which kept their original speed, and the fraction of frames defined '
    '(the table is shared with Experiment C: read "centroid subtracted" as "PCA applied")',
)
EXPLAINED_VARIANCE_GUIDE = (
    'bar.explained_variance',
    '% of the position variance of each group explained by each kept component, one panel per '
    'camera and group',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Experiment D: PCA of each anatomical group, first components of every group.',
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
        '--cameras', type=str, nargs='+', default=None,
        help='cameras to analyse (default: left right).',
    )
    parser.add_argument(
        '--exclude', type=str, nargs='*', default=[], metavar='PATTERN',
        help="glob patterns of keypoints to leave out, e.g. 'd*'; use the same patterns as the "
        'baseline run to compare like with like. Quote them.',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='cameras processed in parallel for the original xcorr (default: 1).',
    )
    parser.add_argument(
        '--n-components', type=int, default=DEFAULT_N_COMPONENTS,
        help=f'number of principal components to keep for every group (default '
        f'{DEFAULT_N_COMPONENTS}); a group with fewer x/y features keeps all of them.',
    )
    parser.add_argument(
        '--min-valid-fraction', type=float, default=DEFAULT_MIN_VALID_FRACTION,
        help='a keypoint whose denoised speed is defined in a smaller fraction of the bins '
        f'is left out and reported (default: {DEFAULT_MIN_VALID_FRACTION}).',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='shared colour scale +-VMAX of the original and pca-speed heatmaps; default: the '
        "baseline's rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='run name: results/experiment_D_pca_groups/<name> and the figure folders '
        'plot_video/experiment_D_pca_groups/<name>_<order>, e.g. <name>_rastermap (default: New '
        'York time).',
    )
    parser.add_argument(
        '--diff-vmax', type=float, default=None,
        help='half-range of the one colour scale of every difference strip of '
        'heatmap.signed_difference; fix it to compare sessions (default: from the data, once per '
        'figure).',
    )
    parser.add_argument(
        '--no-depth-plot', action='store_true',
        help='do not also draw the figures in probe-depth order into <plots-subdir>_depth.',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='compute and save the arrays/csv/json only, draw no figures; redraw them later '
        'with pose2visual.xcorr.experiments.replot_pca_groups.',
    )
    return parser.parse_args(argv)


def check_same_side(members: list[str], camera: str) -> None:
    """Check that no keypoint of the other camera's side is in a group.

    Args:
        members: keypoint names of one group.
        camera: `'left'` or `'right'`.

    Raises:
        ValueError: if a member ends in the other side's suffix (`_right` for the left camera),
            i.e. step 0 was run without the `same_side_filter` the yaml now asks for.
    """
    other = next(side for side in SIDES if side != camera)
    wrong = [name for name in members if name.endswith(f'_{other}')]
    if wrong:
        raise ValueError(
            f'{camera} camera: {wrong} belong to the {other} side; a group would mix both sides. '
            'Rerun step 0 (list_litpose_keypoints) with the yaml\'s same_side_filter',
        )


def fit_group_pca(
    positions: dict[str, tuple[np.ndarray, np.ndarray]],
    reliable: dict[str, np.ndarray],
    members: list[str],
    n_components: int,
) -> dict[str, Any]:
    """PCA of the x/y positions of one group.

    Args:
        positions: keypoint -> finite imputed `(x, y)` arrays ``[F]``.
        reliable: keypoint -> boolean ``[F]``, frames that are tracked (not imputed).
        members: the group's keypoints (two or more).
        n_components: requested number of components.

    Returns:
        Dict with `scores` ``[F, K]``, `xy_hat` ``[F, 2 * members]`` (x then y of each member),
        `ratio` (explained variance of the kept components), `n_components` (effective),
        `valid_position` (at least `MIN_RELIABLE_KEYPOINTS` members tracked) and `valid_speed`
        (`valid_position` at the frame and both neighbours).
    """
    n_reliable = np.sum([np.asarray(reliable[k], dtype=bool) for k in members], axis=0)
    valid_position = n_reliable >= min(MIN_RELIABLE_KEYPOINTS, len(members))
    left = np.concatenate([[True], valid_position[:-1]])
    right = np.concatenate([valid_position[1:], [True]])
    xy = np.stack([v for name in members for v in positions[name]], axis=1)
    effective = min(n_components, xy.shape[1])
    scores, ratio = project_group(xy, effective)
    xy_hat, _ = reproject_group(xy, effective)
    return {
        'scores': scores, 'xy_hat': xy_hat, 'ratio': ratio, 'n_components': effective,
        'valid_position': valid_position, 'valid_speed': valid_position & left & right,
    }


def camera_pca_groups(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    intervals: np.ndarray,
    binsize: float,
    exclude: list[str],
    body_groups: dict[str, list[str]],
    n_components: int,
    baseline_names: list[str],
) -> dict[str, Any]:
    """PCA of every group and per-keypoint statistics of one camera, on the spike grid.

    Args:
        config_path: frozen encoding yaml.
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with the camera timestamp files.
        intervals: contiguous intervals of the neural bundle.
        binsize: bin size in seconds.
        exclude: glob patterns of keypoints to leave out.
        body_groups: group name -> tokens (`relative_motion.resolve_body_groups`).
        n_components: requested number of components per group.
        baseline_names: behaviors the baseline kept for this camera; keypoints outside this list
            (dropped as constant, or hidden) are not analysed.

    Returns:
        Dict with `names` (analysed keypoints, in `baseline_names` order), `pca_applied` (bool,
        one per name), `speed_traces` (name -> binned denoised speed, reduced keypoints only),
        `position_traces` (`<group>_pcK_position` -> binned score, `K` from 1), `position_groups`
        (the group of each position trace), `pca_info` (group -> `members`,
        `n_components_requested`, `n_components`, `explained_variance_ratio`, `valid_fraction`),
        `plan` (one dict per non-empty group: `group`, `members`, `corrected`, `reason`,
        `valid_fraction`) and `stats` (name -> variance and coverage statistics).

    Raises:
        ValueError: if a group mixes both sides (`check_same_side`).
    """
    data = load_camera_positions(config_path, eid, camera, video_timestamps_dir, exclude)
    times, positions, reliable = data['times'], data['positions'], data['reliable']
    lengths = intervals[:, 1] - intervals[:, 0]
    n_bins = int(np.ceil(lengths[0] / binsize))

    plan = plan_groups(data['keypoints'], body_groups, [])
    speed_traces: dict[str, np.ndarray] = {}
    position_traces: dict[str, np.ndarray] = {}
    position_groups: dict[str, str] = {}
    pca_info: dict[str, dict[str, Any]] = {}
    xy_hat: dict[str, np.ndarray] = {}
    frames_of: dict[str, np.ndarray] = {}
    for entry in plan:
        members = entry['members']
        check_same_side(members, camera)
        entry['valid_fraction'] = None
        if not entry['corrected']:
            if len(members) == 1:
                entry['reason'] = 'single keypoint: no PCA, original speed kept'
            else:
                entry['reason'] = 'no anatomical group: original speed kept'
            continue
        fit = fit_group_pca(positions, reliable, members, n_components)
        effective = fit['n_components']
        if effective < n_components:
            logger.warning(
                f'{camera} {entry["group"]}: {len(members)} keypoints ({2 * len(members)} x/y '
                f'features) but {n_components} components were requested; using {effective}',
            )
        for idx_comp in range(effective):
            name = f'{entry["group"]}_pc{idx_comp + 1}_position'
            position_traces[name] = bin_with_gaps(
                fit['scores'][:, idx_comp], fit['valid_position'], times, intervals, binsize,
                n_bins,
            )
            position_groups[name] = entry['group']
        for idx_kp, name in enumerate(members):
            xy_hat[name] = fit['xy_hat'][:, 2 * idx_kp:2 * idx_kp + 2]
            frames_of[name] = fit['valid_speed']
            speed = speed_from_positions(
                xy_hat[name][:, 0], xy_hat[name][:, 1], times,
            )
            speed_traces[name] = bin_with_gaps(
                speed, fit['valid_speed'], times, intervals, binsize, n_bins,
            )
        entry['valid_fraction'] = float(fit['valid_speed'].mean())
        entry['reason'] = f'{effective} of {n_components} requested components kept'
        pca_info[entry['group']] = {
            'members': members, 'n_components_requested': n_components,
            'n_components': effective,
            'explained_variance_ratio': [float(v) for v in fit['ratio']],
            'valid_fraction': entry['valid_fraction'],
        }
        logger.info(
            f'{camera}: {entry["group"]} PCA, {len(members)} keypoints -> {effective} '
            f'components, explained variance {float(np.sum(fit["ratio"])):.4f}',
        )

    names = [name for name in baseline_names if name in data['keypoints']]
    stats: dict[str, dict[str, Any]] = {}
    for name in names:
        x, y = positions[name]
        absolute_speed = speed_from_positions(x, y, times)
        if name in speed_traces:
            x_hat, y_hat = xy_hat[name][:, 0], xy_hat[name][:, 1]
            frames = frames_of[name]
            pca_speed = speed_from_positions(x_hat, y_hat, times)
        else:
            x_hat, y_hat, pca_speed = x, y, absolute_speed
            frames = np.ones(times.size, dtype=bool)
        stats[name] = {
            'corrected': name in speed_traces,
            'var_position_original': float(np.var(x[frames]) + np.var(y[frames])),
            'var_position_relative': float(np.var(x_hat[frames]) + np.var(y_hat[frames])),
            'var_speed_original': float(np.var(absolute_speed[frames])),
            'var_speed_relative': float(np.var(pca_speed[frames])),
            'frame_fraction_defined': float(frames.mean()),
        }
    return {
        'names': names,
        'pca_applied': np.asarray([name in speed_traces for name in names]),
        'speed_traces': speed_traces,
        'position_traces': position_traces,
        'position_groups': position_groups,
        'pca_info': pca_info,
        'plan': plan,
        'stats': stats,
    }


def explanation_text(report: dict[str, Any]) -> str:
    """Short markdown explanation plus the group decisions of one run.

    Args:
        report: the group report built by `run_session`.

    Returns:
        Markdown text.
    """
    lines = [
        f'# Experiment D: PCA of each group ({report["eid"]})',
        '',
        'Every anatomical group with two or more keypoints of a camera is fitted on its own '
        '(x/y positions, a camera only has its own side, so the left and right forelimb are '
        'never mixed). The component scores are new behaviors (`<group>_pcK_position`); the '
        'positions reprojected onto the same components give a denoised speed per keypoint, kept '
        "under the keypoint's own name. Single-keypoint groups and ungrouped keypoints keep their "
        'original speed.',
        '',
    ]
    for camera, info in report['cameras'].items():
        lines += [
            f'## {camera} camera', '',
            '| group | keypoints | treatment | reason | frames defined |',
            '|---|---|---|---|---|',
        ]
        for row in info['plan']:
            valid = row['valid_fraction']
            fraction = '-' if valid is None else f'{100 * valid:.1f} %'
            treatment = 'PCA applied' if row['corrected'] else 'original speed kept'
            lines.append(
                f'| {row["group"]} | {", ".join(row["members"])} | {treatment} | '
                f'{row["reason"]} | {fraction} |',
            )
        if info['dropped']:
            lines += ['', '**Left out (speed or component unusable):** ' + '; '.join(
                f'{name} ({why})' for name, why in info['dropped'].items())]
        for group, group_info in info['pca_info'].items():
            lines += [
                '',
                f"**{group} explained variance** ({group_info['n_components']} components): "
                + ', '.join(f'{100 * v:.1f} %' for v in group_info['explained_variance_ratio']),
            ]
        lines.append('')
    return '\n'.join(lines)


def run_session(
    config_path: str | Path,
    eid: str,
    video_timestamps_dir: str | Path,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    vmax: float | None = None,
    plots_subdir: str | None = None,
    skip_plot: bool = False,
    n_jobs: int = 1,
    depth_plot: bool = True,
    diff_vmax: float | None = None,
) -> dict[str, Path | None]:
    """Run Experiment D on one session.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `run_xcorr.CAMERAS`.
        exclude: glob patterns of keypoints to leave out (use the baseline's patterns).
        n_components: number of principal components to keep for every group.
        min_valid_fraction: smallest fraction of defined bins of a denoised keypoint.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule to the
            original xcorrs.
        plots_subdir: run name of the figure folder, to which the neuron order is
            appended; `None` = New York time of the run.
        skip_plot: compute and save only, draw no figures; redraw them later with
            `pose2visual.xcorr.experiments.replot_pca_groups`.
        n_jobs: cameras processed in parallel for the original xcorr.
        depth_plot: also draw the figures in probe-depth order into `<plots_subdir>_depth`
            (skipped when the first order is already depth, or when `skip_plot` is `True`).
        diff_vmax: fixed half-range of the difference strips of `heatmap.signed_difference`
            (to compare sessions); `None` = data driven.

    Returns:
        Dict of the written paths: `speed_arrays`, `position_arrays`, `summary`, `groups`,
        `groups_md`, `plots` and `plots_depth` (each `None` when not drawn).

    Raises:
        FileNotFoundError: if step 0 is missing.
        ValueError: if a group mixes both sides of the animal.
    """
    base = compute_original(config_path, eid, video_timestamps_dir, cameras, exclude, n_jobs)
    exclude = base['exclude_patterns']
    neural = base['neural']
    cams, lags_sec = base['cameras'], base['lags_sec']
    body_groups = resolve_body_groups(config_path)
    plots_stem = plots_subdir or timestamp_subdir()
    out_dir, plots_root = new_run_dirs(base['session_dir'], EXPERIMENT_PCA_GROUPS, plots_stem)

    idx_motion = [i for i, n in enumerate(base['names']) if not n.endswith(VISIBILITY_SUFFIX)]
    names = [base['names'][i] for i in idx_motion]
    valid_base = base['valid'][:, idx_motion]
    r_original_all = base['r'][:, :, idx_motion, :]
    groups_out = anatomical_groups(config_path, names)

    n_cam, n_neuron, n_beh, n_lag = r_original_all.shape
    r_speed_pca = np.full(r_original_all.shape, np.nan)
    pca_applied = np.zeros((n_cam, n_beh), dtype=bool)
    valid = np.zeros((n_cam, n_beh), dtype=bool)
    camera_report: dict[str, Any] = {}
    stat_rows: list[dict[str, Any]] = []
    position_fits: dict[str, dict[str, np.ndarray]] = {}
    position_group_of: dict[str, str] = {}

    for idx_c, camera in enumerate(cams):
        baseline_names = [n for n, ok in zip(names, valid_base[idx_c]) if ok]
        result = camera_pca_groups(
            config_path, eid, camera, video_timestamps_dir, neural['intervals'],
            neural['binsize'], exclude, body_groups, n_components, baseline_names,
        )
        cam_names = result['names']
        trace_matrix = np.full((neural['neural'].shape[0], len(cam_names)), np.nan)
        for idx_n, name in enumerate(cam_names):
            if name in result['speed_traces']:
                trace_matrix[:, idx_n] = result['speed_traces'][name]
        usable, dropped = screen_traces(
            trace_matrix, cam_names, result['pca_applied'], min_valid_fraction,
        )
        for name, why in dropped.items():
            logger.warning(f'{camera} {name}: left out, {why}')
        idx_fit = [i for i in range(len(cam_names)) if usable[i] and result['pca_applied'][i]]
        if idx_fit:
            fitted = masked_cross_correlation(
                neural['neural'], trace_matrix[:, idx_fit], neural['max_lag_bins'],
            )
        for idx_k, name in enumerate(cam_names):
            if not usable[idx_k]:
                continue
            idx_b = names.index(name)
            valid[idx_c, idx_b] = True
            pca_applied[idx_c, idx_b] = bool(result['pca_applied'][idx_k])
            if result['pca_applied'][idx_k]:
                r_speed_pca[idx_c, :, idx_b, :] = fitted[:, idx_fit.index(idx_k), :]
            else:
                # not reduced: the baseline's own xcorr, not a recomputation
                r_speed_pca[idx_c, :, idx_b, :] = r_original_all[idx_c, :, idx_b, :]
            stat_rows.append({
                'camera': camera, 'behavior': name, 'group': groups_out[idx_b],
                **result['stats'][name],
            })

        if result['position_traces']:
            pc_names = list(result['position_traces'])
            position_matrix = np.stack([result['position_traces'][n] for n in pc_names], axis=1)
            # a component of a group with too little variance (e.g. one tracked keypoint that
            # hardly moves) is constant or sparse, and has no correlation
            usable_pc, dropped_pc = screen_traces(
                position_matrix, pc_names, np.ones(len(pc_names), dtype=bool), min_valid_fraction,
            )
            for name, why in dropped_pc.items():
                logger.warning(f'{camera} {name}: left out, {why}')
            dropped.update(dropped_pc)
            idx_pc = np.flatnonzero(usable_pc)
            if idx_pc.size:
                fitted_pos = masked_cross_correlation(
                    neural['neural'], position_matrix[:, idx_pc], neural['max_lag_bins'],
                )
                position_fits[camera] = {
                    pc_names[i]: fitted_pos[:, j, :] for j, i in enumerate(idx_pc)
                }
                position_group_of.update({pc_names[i]: result['position_groups'][pc_names[i]]
                                          for i in idx_pc})

        camera_report[camera] = {
            'plan': result['plan'], 'dropped': dropped, 'pca_info': result['pca_info'],
        }
        logger.info(f'{camera}: groups {[entry["group"] for entry in result["plan"]]}')

    # position behaviors of every camera, grouped like the keypoints (first-seen order)
    group_order = list(dict.fromkeys(groups_out))
    position_names = sorted(
        position_group_of,
        key=lambda n: (group_order.index(position_group_of[n]), n),
    )
    r_position_pca = np.full((n_cam, n_neuron, len(position_names), n_lag), np.nan)
    valid_position = np.zeros((n_cam, len(position_names)), dtype=bool)
    for idx_c, camera in enumerate(cams):
        for name, fit in position_fits.get(camera, {}).items():
            idx_p = position_names.index(name)
            r_position_pca[idx_c, :, idx_p, :] = fit
            valid_position[idx_c, idx_p] = True
    position_groups = [position_group_of[n] for n in position_names]

    r_original = np.where(valid[:, None, :, None], r_original_all, np.nan)
    r_speed_pca_masked = np.where(valid[:, None, :, None], r_speed_pca, np.nan)
    summary = pd.DataFrame(stat_rows)

    speed_arrays_path = out_dir / f'pca_groups_speed_arrays.{eid}.npz'
    np.savez_compressed(
        speed_arrays_path, r_pca=r_speed_pca_masked.astype(np.float32),
        r_original=r_original.astype(np.float32), pca_applied=pca_applied, valid=valid,
        lags_sec=lags_sec, cameras=np.asarray(cams), behavior_names=np.asarray(names),
        behavior_groups=np.asarray(groups_out), neuron_ids=base['neuron_ids'],
        neuron_depths=base['neuron_depths'],
    )
    position_arrays_path = out_dir / f'pca_groups_position_arrays.{eid}.npz'
    np.savez_compressed(
        position_arrays_path, r_position=r_position_pca.astype(np.float32), valid=valid_position,
        lags_sec=lags_sec, cameras=np.asarray(cams), pc_names=np.asarray(position_names),
        pc_groups=np.asarray(position_groups), neuron_ids=base['neuron_ids'],
        neuron_depths=base['neuron_depths'],
    )
    summary_path = out_dir / f'pca_groups_summary.{eid}.csv'
    summary.to_csv(summary_path, index=False)
    report = {'eid': eid, 'cameras': camera_report, 'n_components': n_components}
    groups_path = out_dir / f'pca_groups_report.{eid}.json'
    groups_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    groups_md_path = out_dir / f'pca_groups_report.{eid}.md'
    groups_md_path.write_text(explanation_text(report), encoding='utf-8')

    plots_dir = None
    plots_dir_depth = None
    if not skip_plot:
        orders = ['saved', 'depth'] if depth_plot else ['saved']
        drawn: set[str] = set()
        for order in orders:
            idx_order, order_label = resolve_order(
                config_path, eid, order, base['neuron_ids'], base['neuron_depths'],
            )
            if order_label in drawn:
                continue  # the saved order already was depth
            drawn.add(order_label)
            this_dir = plots_root / order_subdir(plots_stem, order_label)
            plot_results(
                this_dir, eid, lags_sec, cams, names, groups_out, valid, r_original,
                r_speed_pca_masked, idx_order, order_label, vmax, position_names,
                position_groups, r_position_pca, valid_position, pca_applied, summary,
                camera_report, diff_vmax,
            )
            if order == 'saved':
                plots_dir = this_dir
            else:
                plots_dir_depth = this_dir
    return {
        'speed_arrays': speed_arrays_path, 'position_arrays': position_arrays_path,
        'summary': summary_path, 'groups': groups_path, 'groups_md': groups_md_path,
        'plots': plots_dir, 'plots_depth': plots_dir_depth,
    }


def plot_results(
    plots_dir: str | Path,
    eid: str,
    lags_sec: np.ndarray,
    cams: list[str],
    names: list[str],
    groups_out: list[str],
    valid: np.ndarray,
    r_original: np.ndarray,
    r_speed_pca_masked: np.ndarray,
    idx_order: np.ndarray,
    order_label: str,
    vmax: float | None,
    position_names: list[str],
    position_groups: list[str],
    r_position_pca: np.ndarray,
    valid_position: np.ndarray,
    pca_applied: np.ndarray,
    summary: pd.DataFrame,
    camera_report: dict[str, Any],
    diff_vmax: float | None = None,
) -> list[Path]:
    """Draw every Experiment D figure from already-computed (or reloaded) arrays.

    Shared by `run_session` (freshly computed arrays) and
    `pose2visual.xcorr.experiments.replot_pca_groups` (arrays reloaded from the saved
    npz/csv/json), so a figure always looks the same whether it is drawn right after the compute
    step or redrawn later with different `vmax`/`cameras`.

    Args:
        plots_dir: folder the figures are written into (created if missing).
        eid: session id.
        lags_sec: lag axis.
        cams: cameras, in plotting order.
        names: behavior (keypoint) names, baseline order.
        groups_out: anatomical group of each behavior (`names` order).
        valid: boolean `[C, B]`.
        r_original: baseline xcorr `[C, N, B, L]`, NaN where not `valid`.
        r_speed_pca_masked: PCA-denoised speed xcorr `[C, N, B, L]`, NaN where not `valid`.
        idx_order: neuron order to draw the heatmaps in.
        order_label: name of the neuron order, for the y-axis label.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule.
        position_names: `<group>_pcK_position` names.
        position_groups: anatomical group of each position behavior.
        r_position_pca: position-PCA xcorr `[C, N, P, L]`.
        valid_position: boolean `[C, P]`.
        pca_applied: boolean `[C, B]`, whether PCA was applied to that behavior.
        summary: per-keypoint statistics table (`run_session`'s `summary`).
        camera_report: camera -> `plan`/`dropped`/`pca_info` (`run_session`'s `camera_report`, or
            the `cameras` entry of a saved `pca_groups_report.<eid>.json`).
        diff_vmax: half-range of the one colour scale of every difference strip of
            `heatmap.signed_difference`; `None` takes it from the data, once per figure.

    Returns:
        The written png files, then the figure guide `figures.<eid>.md`
        (`common.write_figure_guide`).
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.experiments import diagnostics
    from pose2visual.xcorr.figures import heatmap_vmax, plot_xcorr_overview, save_figure

    plots_dir = Path(plots_dir)
    paths: list[Path] = []
    guide = list(HEATMAP_GUIDE)
    scale = vmax if vmax is not None else heatmap_vmax(r_original)
    ylabel = f'neurons ({order_label} order)'
    panels = [
        ('original_speed', r_original, 'original speed (absolute)'),
        (
            'pca_speed', r_speed_pca_masked,
            'denoised speed (each group reprojected onto its top components)',
        ),
    ]
    for stem, data, label in panels:
        fig = plot_xcorr_overview(
            data[:, idx_order], lags_sec, names, cams, groups_out, valid, eid, scale,
            ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: {label}; colour scale +-{scale:.2f}; each strip: neurons x lag, '
            'number = max |value|', BASELINE_COLORBAR_LABEL,
        )
        paths += save_figure(fig, plots_dir / f'heatmap.{stem}.{eid}')
    if valid_position.any():
        pos_scale = heatmap_vmax(r_position_pca)
        fig = plot_xcorr_overview(
            r_position_pca[:, idx_order], lags_sec, position_names, cams, position_groups,
            valid_position, eid, pos_scale, ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: PCA on positions (component scores of each group); colour scale '
            f'+-{pos_scale:.2f}', BASELINE_COLORBAR_LABEL,
        )
        paths += save_figure(fig, plots_dir / f'heatmap.position_pca.{eid}')
        guide.append(POSITION_PCA_GUIDE)
    with warnings.catch_warnings():
        # a camera without a keypoint is all NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        peak_original = np.nanmax(np.abs(r_original), axis=3)
        peak_pca = np.nanmax(np.abs(r_speed_pca_masked), axis=3)
    scatter_panels = []
    for idx_c, camera in enumerate(cams):
        idx_b = np.flatnonzero(valid[idx_c])
        scatter_panels.append({
            'camera': camera, 'original': peak_original[idx_c][:, idx_b],
            'relative': peak_pca[idx_c][:, idx_b],
            'groups': [groups_out[i] for i in idx_b], 'corrected': pca_applied[idx_c, idx_b],
            'names': [names[i] for i in idx_b],
        })
    paths += save_figure(
        diagnostics.plot_peak_scatter(scatter_panels), plots_dir / f'scatter.peak_abs_r.{eid}',
    )
    guide.append(SCATTER_GUIDE)
    signed_paths, signed_guide = save_signed_comparison(
        plots_dir, eid, lags_sec, cams, names, groups_out, valid, r_original,
        r_speed_pca_masked, idx_order, scale, ylabel, 'pca', pca_applied, groups_out,
        diff_vmax,
    )
    paths += signed_paths
    guide += signed_guide
    if not summary.empty:
        paths += save_figure(
            diagnostics.plot_variance_bars(summary), plots_dir / f'bar.variance.{eid}',
        )
        guide.append(VARIANCE_GUIDE)
    group_rows = [
        {'camera': camera, **entry}
        for camera, info in camera_report.items() for entry in info['plan']
    ]
    paths += save_figure(
        diagnostics.plot_group_table(group_rows), plots_dir / f'groups_table.{eid}',
    )
    guide.append(GROUPS_TABLE_GUIDE)
    variance_entries = [
        {
            'camera': camera, 'explained_variance_ratio': group_info['explained_variance_ratio'],
            'title': f'{camera}: {group}', 'label': f'% of {group} variance explained',
        }
        for camera, info in camera_report.items() for group, group_info in info['pca_info'].items()
    ]
    if variance_entries:
        paths += save_figure(
            diagnostics.plot_pca_explained_variance(variance_entries),
            plots_dir / f'bar.explained_variance.{eid}',
        )
        guide.append(EXPLAINED_VARIANCE_GUIDE)
    paths.append(write_figure_guide(
        plots_dir, eid, 'Experiment D: PCA of each group', order_label, scale, 'pca_speed', guide,
    ))
    return paths


def main(argv: list[str] | None = None) -> None:
    """Run Experiment D for every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    plots_subdir = args.plots_subdir or timestamp_subdir()
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                paths = run_session(
                    config, eid, args.video_timestamps_dir, args.cameras, args.exclude,
                    args.n_components, args.min_valid_fraction, args.vmax, plots_subdir,
                    args.no_plot, args.n_jobs, not args.no_depth_plot,
                    args.diff_vmax,
                )
                logger.info(
                    f'wrote {[str(p) for p in paths.values() if isinstance(p, Path)]}',
                )
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
