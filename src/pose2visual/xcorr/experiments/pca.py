"""Experiment B: PCA of the face group; left/right paw keypoints pass through unchanged.

Question: can the many near-correlated face keypoints (eye, pupil, mouth, upperlip, lowerlip,
nose, pad, ear -- both sides pooled together) be summarized by a couple of principal
components, and do those components already carry the neuron relationship the baseline sees in
the individual keypoints? The tongue is not included: its keypoints are replaced by the binary
``tongue_visible`` feature (yaml ``visibility`` section), so no tongue position enters the fit.

Starting from the keypoint coordinates (not from the derived speeds), the face keypoints of a
camera are stacked into one ``[n_frames, 2 * n_face_keypoints]`` matrix and projected onto their
top ``n_components`` principal components (the same eigendecomposition as
``pose2visual.encoding.litpose_pca``, fit once per camera on every frame of the session). Two
kinds of cross-correlation follow from that one fit:

* **PCA on positions**: the component scores themselves, ``face_pc1_position``,
  ``face_pc2_position``, ..., are new behaviors in their own right, cross-correlated with the
  neurons directly (``pose2visual.encoding.litpose_pca.project_group``).
* **PCA on speed**: the face positions are reprojected onto the same components (denoised, mapped
  back to pixel space, ``litpose_pca.reproject_group``), each keypoint's own speed is computed
  from its denoised position with the baseline's speed function, and the result is
  cross-correlated under the keypoint's own name -- comparable, keypoint for keypoint, to the
  baseline.

Left and right paw keypoints (``wrist``, ``d1``-``d4``, grouped by their ``_left`` / ``_right``
suffix into ``left_paw`` / ``right_paw``) are not reduced: a paw group needs no PCA (in practice it
keeps only ``wrist_left`` / ``wrist_right`` once the digits are excluded, as this lab's xcorr runs
normally do), so its baseline speed is kept unchanged. Keypoints matching neither a face nor a paw
token are likewise left unchanged and reported as ``unassigned``.

The original speed cross-correlations it is compared against are recomputed here with the
baseline's own code (``common.compute_original``), so this does not wait for Experiment A; only the
neuron order of the figures is taken from A when A has run (``common.resolve_order``). Everything
is written to ``<session>/experiment_B_pca/``.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.pca \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --video-timestamps-dir <dir> --exclude 'd*'
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    VISIBILITY_SUFFIX,
    load_encoding_yaml,
    speed_from_positions,
)
from pose2visual.encoding.litpose_pca import project_group, reproject_group
from pose2visual.xcorr.experiments.common import (
    BASELINE_COLORBAR_LABEL,
    compute_original,
    experiment_dir,
    relabel_overview,
    resolve_order,
    write_figure_guide,
)
from pose2visual.xcorr.experiments.masked_crosscorr import masked_cross_correlation
from pose2visual.xcorr.experiments.relative_motion import (
    DEFAULT_MIN_VALID_FRACTION,
    MIN_RELIABLE_KEYPOINTS,
    bin_with_gaps,
    load_camera_positions,
    screen_traces,
)
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.settings import EXPERIMENT_PCA, freeze_config, resolve_eids

logger = logging.getLogger(__name__)

PAW_TOKENS = frozenset({'wrist', 'd1', 'd2', 'd3', 'd4'})
UNASSIGNED_GROUP = 'unassigned'
DEFAULT_N_COMPONENTS = 2
DIFFERENCE_COLORBAR = 'pca minus original correlation'
# (stem, meaning) rows of the figure guide (`common.write_figure_guide`)
HEATMAP_GUIDE = (
    (
        'heatmap.original_speed',
        "baseline xcorr of each keypoint's absolute speed (the same quantity as Experiment A)",
    ),
    (
        'heatmap.pca_speed',
        "PCA on speed: each face keypoint's position is reprojected onto the face group's top "
        'principal components (denoised, back in pixels), its speed recomputed and '
        "cross-correlated under the keypoint's own name; paw and unassigned keypoints keep the "
        'original xcorr',
    ),
    (
        'heatmap.difference',
        '`pca_speed - original_speed` per keypoint, neuron and lag: positive = the denoised speed '
        'correlates more positively than the raw speed; non-face columns are exactly 0',
    ),
)
POSITION_PCA_GUIDE = (
    'heatmap.position_pca',
    'PCA on positions: xcorr of the face component scores themselves (`face_pc1_position`, '
    '`face_pc2_position`, ...), one column per component, with its own colour scale (see the '
    'figure title)',
)
SCATTER_GUIDE = (
    'scatter.peak_abs_r',
    'one dot per neuron-keypoint pair: peak \\|r\\| over lags of the original speed (x) against '
    'the PCA speed (y); grey = PCA not applied (on the diagonal by construction); below the '
    'diagonal = the denoising lost correlation',
)
VARIANCE_GUIDE = (
    'bar.variance',
    'per keypoint, variance of position (top) and speed (bottom), original against '
    'PCA-denoised, log axis; `*` = PCA applied',
)
GROUPS_TABLE_GUIDE = (
    'groups_table',
    'which group got PCA and which kept its original speed, and the fraction of frames defined '
    '(the table is shared with Experiment C: read "centroid subtracted" as "PCA applied")',
)
EXPLAINED_VARIANCE_GUIDE = (
    'bar.explained_variance',
    '% of the face position variance explained by each kept component, one panel per camera',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Experiment B: PCA of the face group; left/right paw keypoints pass through.',
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
        help=f'number of principal components to keep for the face group (default '
        f'{DEFAULT_N_COMPONENTS}).',
    )
    parser.add_argument(
        '--min-valid-fraction', type=float, default=DEFAULT_MIN_VALID_FRACTION,
        help='a face keypoint whose denoised speed is defined in a smaller fraction of the bins '
        f'is left out and reported (default: {DEFAULT_MIN_VALID_FRACTION}).',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='shared colour scale +-VMAX of the original and pca-speed heatmaps; default: the '
        "baseline's rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='figure folder inside experiment_B_pca (default: New York time); the neuron order is '
        'appended, e.g. <name>_rastermap.',
    )
    parser.add_argument(
        '--no-depth-plot', action='store_true',
        help='do not also draw the figures in probe-depth order into <plots-subdir>_depth.',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='compute and save the arrays/csv/json only, draw no figures; redraw them later '
        'with pose2visual.xcorr.experiments.replot_pca.',
    )
    return parser.parse_args(argv)


def load_face_tokens(body_parts_json: str | Path) -> frozenset[str]:
    """Face-related keypoint tokens: `body_parts.json`'s `'all'` list minus the paw tokens.

    Args:
        body_parts_json: path of a `body_parts.json`-style file.

    Returns:
        The face tokens, e.g. `{'eye', 'pupil', 'mouth', ...}`.

    Raises:
        ValueError: if the file has no non-empty `'all'` list.
    """
    raw = json.loads(Path(body_parts_json).read_text(encoding='utf-8'))
    all_tokens = raw.get('all')
    if not isinstance(all_tokens, list) or not all_tokens:
        raise ValueError(f"{body_parts_json} has no non-empty 'all' list of keypoint tokens")
    return frozenset(all_tokens) - PAW_TOKENS


def assign_face_paw_groups(
    keypoints: list[str],
    face_tokens: frozenset[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Assign each keypoint to `face`, `left_paw` or `right_paw` by its token and side.

    A keypoint belongs to a paw group when the first `_`-separated segment of its name is a paw
    token (`wrist`, `d1`-`d4`) and its last segment is `left` or `right` (`wrist_left` ->
    `left_paw`); to `face` when that first segment is a face token, regardless of side (so
    `eye_left` and `eye_right` both join the one `face` group); otherwise it is unassigned (a paw
    token with no side, or a token in neither set).

    Args:
        keypoints: keypoint names, e.g. the usable keypoints of a camera.
        face_tokens: face tokens, from `load_face_tokens`.

    Returns:
        Tuple `(groups, unassigned)`: `groups` has keys `'face'`, `'left_paw'`, `'right_paw'`,
        each mapping to the matching keypoints in input order (possibly empty), and `unassigned`
        lists the keypoints matching neither.
    """
    groups: dict[str, list[str]] = {'face': [], 'left_paw': [], 'right_paw': []}
    unassigned: list[str] = []
    for name in keypoints:
        segments = name.split('_')
        token = segments[0]
        side = segments[-1] if segments[-1] in ('left', 'right') else None
        if token in PAW_TOKENS:
            if side is None:
                unassigned.append(name)
            else:
                groups[f'{side}_paw'].append(name)
        elif token in face_tokens:
            groups['face'].append(name)
        else:
            unassigned.append(name)
    return groups, unassigned


def camera_pca(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    intervals: np.ndarray,
    binsize: float,
    exclude: list[str],
    face_tokens: frozenset[str],
    n_components: int,
    baseline_names: list[str],
) -> dict[str, Any]:
    """PCA of the face group and per-keypoint statistics of one camera, on the spike grid.

    Args:
        config_path: frozen encoding yaml.
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with the camera timestamp files.
        intervals: contiguous intervals of the neural bundle.
        binsize: bin size in seconds.
        exclude: glob patterns of keypoints to leave out.
        face_tokens: face tokens (`load_face_tokens`).
        n_components: requested number of components for the face group.
        baseline_names: behaviors the baseline kept for this camera; keypoints outside this list
            (dropped as constant, or hidden) are not analysed.

    Returns:
        Dict with `names` (analysed keypoints, in `baseline_names` order), `pca_applied` (bool,
        one per name: `True` for a face keypoint with a usable PCA fit), `speed_traces` (name ->
        binned denoised speed, face keypoints only), `position_traces` (`face_pcK_position` ->
        binned score, `K` from 1), `pca_info` (`None` if no usable face keypoint, else `members`,
        `n_components_requested`, `n_components` (effective), `explained_variance_ratio` and
        `valid_fraction`), `plan` (one dict per non-empty group, for the report: `group`,
        `members`, `corrected`, `reason`, `valid_fraction`) and `stats` (name -> variance and
        coverage statistics, for the summary table).
    """
    data = load_camera_positions(config_path, eid, camera, video_timestamps_dir, exclude)
    times, positions, reliable = data['times'], data['positions'], data['reliable']
    lengths = intervals[:, 1] - intervals[:, 0]
    n_bins = int(np.ceil(lengths[0] / binsize))

    groups_map, unassigned = assign_face_paw_groups(data['keypoints'], face_tokens)
    face_members = groups_map['face']

    speed_traces: dict[str, np.ndarray] = {}
    position_traces: dict[str, np.ndarray] = {}
    pca_info: dict[str, Any] | None = None
    xy_hat = None
    valid_speed_face = None
    if face_members:
        n_reliable = np.sum([np.asarray(reliable[k], dtype=bool) for k in face_members], axis=0)
        threshold = min(MIN_RELIABLE_KEYPOINTS, len(face_members))
        valid_position = n_reliable >= threshold
        left = np.concatenate([[True], valid_position[:-1]])
        right = np.concatenate([valid_position[1:], [True]])
        valid_speed_face = valid_position & left & right

        xy = np.stack([v for name in face_members for v in positions[name]], axis=1)
        n_features = xy.shape[1]
        effective = min(n_components, n_features)
        if effective < n_components:
            logger.warning(
                f'{camera}: face group has {len(face_members)} usable keypoints '
                f'({n_features} x/y features) but {n_components} components were requested; '
                f'using {effective}',
            )
        scores, ratio = project_group(xy, effective)
        xy_hat, _ = reproject_group(xy, effective)
        for idx_comp in range(effective):
            position_traces[f'face_pc{idx_comp + 1}_position'] = bin_with_gaps(
                scores[:, idx_comp], valid_position, times, intervals, binsize, n_bins,
            )
        for idx_kp, name in enumerate(face_members):
            speed = speed_from_positions(xy_hat[:, 2 * idx_kp], xy_hat[:, 2 * idx_kp + 1], times)
            speed_traces[name] = bin_with_gaps(
                speed, valid_speed_face, times, intervals, binsize, n_bins,
            )
        pca_info = {
            'members': face_members, 'n_components_requested': n_components,
            'n_components': effective, 'explained_variance_ratio': [float(v) for v in ratio],
            'valid_fraction': float(valid_speed_face.mean()),
        }
        logger.info(
            f'{camera}: face PCA, {len(face_members)} keypoints -> {effective} components, '
            f'explained variance {float(np.sum(ratio)):.4f}',
        )

    plan = []
    for group in ('face', 'left_paw', 'right_paw'):
        members = groups_map[group]
        if not members:
            continue
        if group == 'face' and pca_info is not None:
            corrected, valid_fraction = True, pca_info['valid_fraction']
            reason = (
                f"{pca_info['n_components']} of {pca_info['n_components_requested']} requested "
                'components kept'
            )
        else:
            corrected, valid_fraction = False, None
            reason = 'no PCA: original speed kept'
        plan.append({
            'group': group, 'members': members, 'corrected': corrected, 'reason': reason,
            'valid_fraction': valid_fraction,
        })
    if unassigned:
        plan.append({
            'group': UNASSIGNED_GROUP, 'members': unassigned, 'corrected': False,
            'reason': 'no face or paw token matched', 'valid_fraction': None,
        })

    names = [name for name in baseline_names if name in data['keypoints']]
    stats: dict[str, dict[str, Any]] = {}
    for name in names:
        x, y = positions[name]
        absolute_speed = speed_from_positions(x, y, times)
        if name in speed_traces:
            idx_kp = face_members.index(name)
            x_hat = xy_hat[:, 2 * idx_kp]
            y_hat = xy_hat[:, 2 * idx_kp + 1]
            frames = valid_speed_face
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
        f'# Experiment B: PCA ({report["eid"]})',
        '',
        '**Face** (eye, pupil, mouth, upperlip, lowerlip, nose, pad, ear; both sides '
        'pooled into one fit): PCA on x/y positions. The component scores are new behaviors '
        '(`face_pcK_position`); the positions reprojected onto the same components give a '
        "denoised speed per keypoint, kept under the keypoint's own name. **Left/right paw** "
        '(`wrist`, `d1`-`d4` by side) and any keypoint matching neither a face nor a paw token '
        'keep their original speed unchanged.',
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
            lines += ['', '**Left out (denoised speed unusable):** ' + '; '.join(
                f'{name} ({why})' for name, why in info['dropped'].items())]
        if info['explained_variance_ratio'] is not None:
            lines += [
                '',
                f"**Face explained variance** ({info['n_components']} components): "
                + ', '.join(f'{100 * v:.1f} %' for v in info['explained_variance_ratio']),
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
) -> dict[str, Path | None]:
    """Run Experiment B on one session.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `run_xcorr.CAMERAS`.
        exclude: glob patterns of keypoints to leave out (use the baseline's patterns).
        n_components: number of principal components to keep for the face group.
        min_valid_fraction: smallest fraction of defined bins of a denoised face keypoint.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule to the
            original xcorrs.
        plots_subdir: run name of the figure folder, to which the neuron order is
            appended; `None` = New York time of the run.
        skip_plot: compute and save only, draw no figures; redraw them later with
            `pose2visual.xcorr.experiments.replot_pca`.
        n_jobs: cameras processed in parallel for the original xcorr.
        depth_plot: also draw the figures in probe-depth order into `<plots_subdir>_depth`
            (skipped when the first order is already depth, or when `skip_plot` is `True`).

    Returns:
        Dict of the written paths: `speed_arrays`, `position_arrays`, `summary`, `groups`,
        `groups_md`, `plots` and `plots_depth` (each `None` when not drawn).

    Raises:
        FileNotFoundError: if step 0 is missing.
    """
    base = compute_original(config_path, eid, video_timestamps_dir, cameras, exclude, n_jobs)
    exclude = base['exclude_patterns']
    neural = base['neural']
    cams, lags_sec = base['cameras'], base['lags_sec']
    cfg = load_encoding_yaml(config_path)
    face_tokens = load_face_tokens(cfg['body_parts_json'])
    out_dir = experiment_dir(base['session_dir'], EXPERIMENT_PCA)

    idx_motion = [i for i, n in enumerate(base['names']) if not n.endswith(VISIBILITY_SUFFIX)]
    names = [base['names'][i] for i in idx_motion]
    valid_base = base['valid'][:, idx_motion]
    r_original_all = base['r'][:, :, idx_motion, :]
    groups_map_all, unassigned_all = assign_face_paw_groups(names, face_tokens)
    group_of_all = {name: UNASSIGNED_GROUP for name in unassigned_all}
    for group, members in groups_map_all.items():
        for name in members:
            group_of_all[name] = group
    groups_out = [group_of_all[name] for name in names]

    n_cam, n_neuron, n_beh, n_lag = r_original_all.shape
    r_speed_pca = np.full(r_original_all.shape, np.nan)
    pca_applied = np.zeros((n_cam, n_beh), dtype=bool)
    valid = np.zeros((n_cam, n_beh), dtype=bool)
    position_names = [f'face_pc{idx + 1}_position' for idx in range(n_components)]
    r_position_pca = np.full((n_cam, n_neuron, n_components, n_lag), np.nan)
    valid_position = np.zeros((n_cam, n_components), dtype=bool)
    camera_report: dict[str, Any] = {}
    stat_rows: list[dict[str, Any]] = []

    for idx_c, camera in enumerate(cams):
        baseline_names = [n for n, ok in zip(names, valid_base[idx_c]) if ok]
        result = camera_pca(
            config_path, eid, camera, video_timestamps_dir, neural['intervals'],
            neural['binsize'], exclude, face_tokens, n_components, baseline_names,
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
                r_speed_pca[idx_c, :, idx_b, :] = r_original_all[idx_c, :, idx_b, :]
            stat_rows.append({
                'camera': camera, 'behavior': name, 'group': group_of_all[name],
                **result['stats'][name],
            })

        if result['position_traces']:
            pc_names = list(result['position_traces'])
            position_matrix = np.stack([result['position_traces'][n] for n in pc_names], axis=1)
            fitted_pos = masked_cross_correlation(
                neural['neural'], position_matrix, neural['max_lag_bins'],
            )
            for idx_pc, pc_name in enumerate(pc_names):
                idx_target = position_names.index(pc_name)
                r_position_pca[idx_c, :, idx_target, :] = fitted_pos[:, idx_pc, :]
                valid_position[idx_c, idx_target] = True

        camera_report[camera] = {
            'plan': result['plan'], 'dropped': dropped,
            'explained_variance_ratio': (
                result['pca_info']['explained_variance_ratio'] if result['pca_info'] else None
            ),
            'n_components': result['pca_info']['n_components'] if result['pca_info'] else None,
        }
        logger.info(f'{camera}: groups {[entry["group"] for entry in result["plan"]]}')

    r_original = np.where(valid[:, None, :, None], r_original_all, np.nan)
    r_speed_pca_masked = np.where(valid[:, None, :, None], r_speed_pca, np.nan)
    summary = pd.DataFrame(stat_rows)

    speed_arrays_path = out_dir / f'pca_speed_arrays.{eid}.npz'
    np.savez_compressed(
        speed_arrays_path, r_pca=r_speed_pca_masked.astype(np.float32),
        r_original=r_original.astype(np.float32), pca_applied=pca_applied, valid=valid,
        lags_sec=lags_sec, cameras=np.asarray(cams), behavior_names=np.asarray(names),
        behavior_groups=np.asarray(groups_out), neuron_ids=base['neuron_ids'],
        neuron_depths=base['neuron_depths'],
    )
    position_arrays_path = out_dir / f'pca_position_arrays.{eid}.npz'
    np.savez_compressed(
        position_arrays_path, r_position=r_position_pca.astype(np.float32), valid=valid_position,
        lags_sec=lags_sec, cameras=np.asarray(cams), pc_names=np.asarray(position_names),
        neuron_ids=base['neuron_ids'], neuron_depths=base['neuron_depths'],
    )
    summary_path = out_dir / f'pca_summary.{eid}.csv'
    summary.to_csv(summary_path, index=False)
    report = {'eid': eid, 'cameras': camera_report, 'n_components': n_components}
    groups_path = out_dir / f'pca_groups.{eid}.json'
    groups_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    groups_md_path = out_dir / f'pca_groups.{eid}.md'
    groups_md_path.write_text(explanation_text(report), encoding='utf-8')

    plots_dir = None
    plots_dir_depth = None
    if not skip_plot:
        plots_stem = plots_subdir or timestamp_subdir()
        orders = ['saved', 'depth'] if depth_plot else ['saved']
        drawn: set[str] = set()
        for order in orders:
            idx_order, order_label = resolve_order(
                config_path, eid, order, base['neuron_ids'], base['neuron_depths'],
            )
            if order_label in drawn:
                continue  # the saved order already was depth
            drawn.add(order_label)
            this_dir = out_dir / order_subdir(plots_stem, order_label)
            plot_results(
                this_dir, eid, lags_sec, cams, names, groups_out, valid, r_original,
                r_speed_pca_masked, idx_order, order_label, vmax, position_names,
                r_position_pca, valid_position, pca_applied, summary, camera_report,
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
    r_position_pca: np.ndarray,
    valid_position: np.ndarray,
    pca_applied: np.ndarray,
    summary: pd.DataFrame,
    camera_report: dict[str, Any],
) -> list[Path]:
    """Draw every Experiment B figure from already-computed (or reloaded) arrays.

    Shared by `run_session` (freshly computed arrays) and
    `pose2visual.xcorr.experiments.replot_pca` (arrays reloaded from the saved npz/csv/json), so
    a figure always looks the same whether it is drawn right after the compute step or redrawn
    later with different `vmax`/`cameras`.

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
        position_names: `face_pcK_position` names.
        r_position_pca: position-PCA xcorr `[C, N, n_components, L]`.
        valid_position: boolean `[C, n_components]`.
        pca_applied: boolean `[C, B]`, whether PCA was applied to that behavior.
        summary: per-keypoint statistics table (`run_session`'s `summary`).
        camera_report: camera -> `plan`/`explained_variance_ratio`/`n_components` (`run_session`'s
            `camera_report`, or the `cameras` entry of a saved `pca_groups.<eid>.json`).

    Returns:
        The written figure files, png and pdf of each figure, then the figure guide
        `figures.<eid>.md` (`common.write_figure_guide`).
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.experiments import diagnostics
    from pose2visual.xcorr.figures import heatmap_vmax, plot_xcorr_overview, save_figure

    paths: list[Path] = []
    guide = list(HEATMAP_GUIDE)
    scale = vmax if vmax is not None else heatmap_vmax(r_original)
    ylabel = f'neurons ({order_label} order)'
    difference = r_speed_pca_masked - r_original
    difference_scale = heatmap_vmax(difference)
    panels = [
        (
            'original_speed', r_original, scale, 'original speed (absolute)',
            BASELINE_COLORBAR_LABEL,
        ),
        (
            'pca_speed', r_speed_pca_masked, scale,
            'denoised speed (face group reprojected onto its top components)',
            BASELINE_COLORBAR_LABEL,
        ),
        ('difference', difference, difference_scale, 'pca minus original speed xcorr',
         DIFFERENCE_COLORBAR),
    ]
    for stem, data, panel_scale, label, colorbar in panels:
        fig = plot_xcorr_overview(
            data[:, idx_order], lags_sec, names, cams, groups_out, valid, eid, panel_scale,
            ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: {label}; colour scale +-{panel_scale:.2f}; each strip: neurons x lag, '
            'number = max |value|', colorbar,
        )
        paths += save_figure(fig, plots_dir / f'heatmap.{stem}.{eid}')
    if valid_position.any():
        pos_scale = heatmap_vmax(r_position_pca)
        pos_groups = ['face_position'] * len(position_names)
        fig = plot_xcorr_overview(
            r_position_pca[:, idx_order], lags_sec, position_names, cams, pos_groups,
            valid_position, eid, pos_scale, ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: PCA on positions (face group component scores); colour scale '
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
        })
    paths += save_figure(
        diagnostics.plot_peak_scatter(scatter_panels), plots_dir / f'scatter.peak_abs_r.{eid}',
    )
    guide.append(SCATTER_GUIDE)
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
        {'camera': camera, 'explained_variance_ratio': info['explained_variance_ratio']}
        for camera, info in camera_report.items() if info['explained_variance_ratio'] is not None
    ]
    if variance_entries:
        paths += save_figure(
            diagnostics.plot_pca_explained_variance(variance_entries),
            plots_dir / f'bar.explained_variance.{eid}',
        )
        guide.append(EXPLAINED_VARIANCE_GUIDE)
    paths.append(write_figure_guide(
        plots_dir, eid, 'Experiment B: PCA', order_label, scale, difference_scale, 'pca_speed',
        guide,
    ))
    return paths


def main(argv: list[str] | None = None) -> None:
    """Run Experiment B for every selected session; failures are listed at the end.

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
                )
                logger.info(f'wrote {[str(p) for p in paths.values()]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
