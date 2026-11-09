"""Experiment C: remove the shared translation of an anatomical group before the xcorr.

Question: do the eye, ear, nose and mouth keypoints correlate with the same neurons in the same way
because they move together (head or face translation), or does a neuron-specific relationship
remain in their *relative* movement?

Starting from the keypoint coordinates (not from the derived speeds), the keypoints are organized
into the anatomical groups of ``body_parts.json`` (the baseline's group set). For a group with at
least two distinct, usable keypoints the centroid of the members is subtracted at every frame,
``relative_position[k, t] = position[k, t] - group_mean_position[t]``, and the relative speed is
computed with the baseline's speed function (``litpose_keypoints.speed_from_positions``, a central
difference on the frame times) from the relative positions. Then the relative speed is
cross-correlated with the same smoothed full-session neural activity on the same 60 Hz grid over
the same lags (``masked_crosscorr``, which equals the baseline estimator when no sample is
missing).

Which groups are corrected (documented per camera in ``relative_groups.<eid>.{json,md}`` and in a
figure):

* **corrected**: groups with two or more usable keypoints (after ``--exclude``), except
  ``--no-centroid-groups`` (default none).
* **skipped, original speed kept**: groups with a single keypoint (the centroid would be the
  keypoint itself and the relative motion identically zero), groups named in
  ``--no-centroid-groups``, and keypoints that belong to no anatomical group (unrelated keypoints
  are never pooled to make a centroid). Their traces are the baseline's, unchanged, and their
  xcorrs are the baseline's saved arrays. The report lists every skipped keypoint and why.

Missing data: the centroid of a group always averages the same members (the group's usable
keypoints in that camera), using the baseline's imputed positions, so a keypoint dropping out
cannot make the centroid jump. A frame counts as defined for the group only if at least two members
are reliably tracked (not masked by the likelihood filter or NaN); a speed sample is defined only
if the frame and both neighbours it is differentiated from are. Undefined samples are NaN, stay NaN
after binning onto the spike grid (a bin touching an undefined frame is undefined) and are excluded
from the correlation; nothing is filled in. The fraction of defined frames is reported per group.

Differences from the baseline, by design: visibility features (``tongue_visible``) are not
movement and are left out; no permutation test is run (the masked estimator would need a
per-shift normalization); the spec's backward difference ``|dp| / dt`` is replaced by the
baseline's central difference so that the only change is the centroid subtraction.

Figures (``png/`` and vector ``pdf/`` subfolders; see ``HEATMAP_GUIDE`` and the other
``*_GUIDE`` entries, written to
``figures.<eid>.md``): ``heatmap.original_speed`` and ``heatmap.relative_speed`` (same neurons,
keypoints, lags and colour scale), ``scatter.peak_abs_r`` (peak |r| before against after the
centroid subtraction, coloured by anatomical group), ``bar.variance``, ``centroid_trajectories``
and ``groups_table``.

The original speed cross-correlations it is compared against are recomputed here with the
baseline's own code (``common.compute_original``), so this does not wait for Experiment A; only the
neuron order of the figures is taken from A when A has run (``common.resolve_order``). Everything
is written to ``<session>/results/experiment_C_relative_motion/<run>/`` (arrays, csv, json, pickle)
and ``<session>/plot_video/experiment_C_relative_motion/<run>_<order>/`` (figures and their guide).

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.relative_motion \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --video-timestamps-dir <dir> --exclude 'd*'
"""

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    ANATOMICAL_GROUP_ALL,
    DEFAULT_PCA_GROUP_SET,
    VISIBILITY_SUFFIX,
    bin_to_intervals,
    check_filter_settings_match,
    csv_filename,
    get_keypoint_coords,
    impute_keypoint_positions,
    keypoint_names_json_path,
    load_encoding_yaml,
    load_filter_settings,
    load_frame_times,
    load_keypoint_csv,
    load_position_columns,
    load_same_side_excluded,
    load_unfiltered_keypoints,
    load_video_preds_dir,
    load_visibility_features,
    masked_frames,
    speed_from_positions,
)
from pose2visual.lightningpose.fit_eks_latent_model import load_groups
from pose2visual.xcorr.behavior import (
    build_continuous_behavior,
    check_frame_times,
    excluded_names,
)
from pose2visual.xcorr.crosscorr import constant_columns
from pose2visual.xcorr.experiments.common import (
    BASELINE_COLORBAR_LABEL,
    compute_original,
    relabel_overview,
    resolve_order,
    save_signed_comparison,
    write_figure_guide,
)
from pose2visual.xcorr.experiments.masked_crosscorr import masked_cross_correlation
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.run_xcorr import OTHER_GROUP, behavior_groups
from pose2visual.xcorr.settings import (
    EXPERIMENT_RELMOTION,
    freeze_config,
    new_run_dirs,
    resolve_eids,
)

logger = logging.getLogger(__name__)

DEFAULT_NO_CENTROID_GROUPS: tuple[str, ...] = ()
DEFAULT_MIN_VALID_FRACTION = 0.5
# a group needs this many reliably tracked keypoints at a frame for its relative motion to exist
MIN_RELIABLE_KEYPOINTS = 2
# tolerance of the self-check between this module's absolute speed and the baseline's trace
# (the baseline stores its binned traces in float32)
SELF_CHECK_RTOL = 1e-4
SELF_CHECK_ATOL = 1e-3
# a group's centroid speed is named `<group>_centroid` in the centroid xcorr
CENTROID_SUFFIX = '_centroid'
# (stem, meaning) rows of the figure guide (`common.write_figure_guide`)
HEATMAP_GUIDE = (
    (
        'heatmap.original_speed',
        "baseline xcorr of each keypoint's absolute speed (same colour scale, neuron order, "
        'keypoints and lags as `heatmap.relative_speed`)',
    ),
    (
        'heatmap.relative_speed',
        "xcorr of each keypoint's speed after its group centroid (the shared translation of the "
        "group's keypoints) is subtracted at every frame; uncorrected keypoints keep the "
        'original xcorr. Compare with `heatmap.original_speed` strip by strip: a correlation '
        'that fades here came from the shared (head/face) translation',
    ),
)
SCATTER_GUIDE = (
    'scatter.peak_abs_r',
    'one panel per camera (row) and corrected anatomical group (column), same axes in every '
    'panel; one dot per neuron-keypoint pair, coloured per keypoint (same colour for both sides '
    'and in every camera and session): peak \\|r\\| over lags of the original speed (x) against '
    'the relative speed (y). The sign is ignored and the two peaks may be at different lags (see '
    '`scatter.signed_r_at_original_peak`); below the dashed diagonal = the group lost correlation '
    'when its centroid was removed, above = gained; uncorrected keypoints (on the diagonal by '
    'construction) are not drawn',
)
VARIANCE_GUIDE = (
    'bar.variance',
    'per keypoint, variance of position (top) and speed (bottom), original against centroid '
    'removed, log axis; `*` = corrected',
)
CENTROID_GUIDE = (
    'centroid_trajectories',
    'one row per corrected camera/group: centroid x and y over a middle excerpt (undefined '
    'frames shaded), centroid speed against the mean member speed (how much of the motion is '
    'shared), and x and y over the whole session',
)
CENTROID_HEATMAP_GUIDE = (
    'heatmap.centroid_speed',
    "xcorr of each corrected group's centroid SPEED MAGNITUDE (px/s of the centroid trajectory; "
    'the direction of the movement is not kept), one strip per group and camera, same colour '
    'scale and neuron order. It relates neural activity to the group\'s translational speed, '
    'while `heatmap.relative_speed` is the speed of each keypoint relative to the centroid: '
    'different behavioral measures, read side by side rather than as one quantity. A tight group '
    'looks like its keypoints in `heatmap.original_speed`; it differs when the members deform. '
    'Drawn unless --no-centroid-xcorr',
)
GROUPS_TABLE_GUIDE = (
    'groups_table',
    'which groups were centroid-corrected and which kept their original speed, why, and the '
    'fraction of frames defined',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Experiment C: xcorr of keypoint speed after removing the group centroid.',
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
        'baseline run to compare like with like. An excluded keypoint is neither compared nor '
        'part of a centroid. Quote them.',
    )
    parser.add_argument(
        '--no-centroid-groups', type=str, nargs='*', default=list(DEFAULT_NO_CENTROID_GROUPS),
        metavar='GROUP',
        help='groups that keep their original speed although they have two or more keypoints '
        '(default: none, every group with two or more keypoints is corrected).',
    )
    parser.add_argument(
        '--min-valid-fraction', type=float, default=DEFAULT_MIN_VALID_FRACTION,
        help='a corrected keypoint whose relative speed is defined in a smaller fraction of the '
        f'bins is left out and reported (default: {DEFAULT_MIN_VALID_FRACTION}).',
    )
    parser.add_argument(
        '--no-centroid-xcorr', action='store_true',
        help="do not correlate the neurons with each corrected group's centroid speed magnitude "
        '(the extra heatmap.centroid_speed and r_centroid in the npz are written by default).',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='shared colour scale +-VMAX of the original and relative heatmaps; default: the '
        "baseline's rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='run name: results/experiment_C_relative_motion/<name> and the figure folders '
        'plot_video/experiment_C_relative_motion/<name>_<order>, e.g. <name>_rastermap '
        '(default: New York time).',
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
        'with pose2visual.xcorr.experiments.replot_relative_motion.',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='cameras processed in parallel for the original xcorr (default: 1).',
    )
    return parser.parse_args(argv)


def resolve_body_groups(config_path: str | Path) -> dict[str, list[str]]:
    """The group -> tokens mapping the baseline used to group its behaviors.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        Group name -> keypoint name tokens, resolved as `run_xcorr.run_session` does
        (``anatomical_group``, else ``pca_group_set``, else the default set).
    """
    cfg = load_encoding_yaml(config_path)
    group_set = cfg.get('anatomical_group')
    if group_set in (None, ANATOMICAL_GROUP_ALL):
        group_set = cfg.get('pca_group_set') or DEFAULT_PCA_GROUP_SET
    return load_groups(cfg['body_parts_json'], str(group_set))


def plan_groups(
    keypoints: list[str],
    body_groups: dict[str, list[str]],
    no_centroid_groups: list[str],
) -> list[dict[str, Any]]:
    """Decide which anatomical groups get a centroid subtraction.

    Args:
        keypoints: usable keypoints of one camera (after `--exclude`).
        body_groups: group name -> tokens (`resolve_body_groups`).
        no_centroid_groups: groups that keep their original speed regardless of their size.

    Returns:
        One dict per non-empty group, in `body_groups` order (the group of keypoints without
        an anatomical group last): `group`, `members` (keypoint names), `corrected` (bool) and
        `reason`.
    """
    plan = []
    for group, idx in behavior_groups(keypoints, body_groups).items():
        members = [keypoints[i] for i in idx]
        if group == OTHER_GROUP:
            corrected, reason = False, 'no anatomical group: unrelated keypoints are not pooled'
        elif group in no_centroid_groups:
            corrected, reason = False, 'listed in --no-centroid-groups (separate body parts)'
        elif len(members) < MIN_RELIABLE_KEYPOINTS:
            corrected = False
            reason = 'single keypoint: the centroid is the keypoint, relative motion would be 0'
        else:
            corrected, reason = True, f'{len(members)} keypoints: centroid subtracted'
        plan.append({'group': group, 'members': members, 'corrected': corrected, 'reason': reason})
    return plan


def relative_group_motion(
    positions: dict[str, tuple[np.ndarray, np.ndarray]],
    reliable: dict[str, np.ndarray],
    members: list[str],
    times: np.ndarray,
) -> dict[str, Any]:
    """Centroid, relative positions and relative speeds of one group at the frame rate.

    The centroid always averages all `members` (constant membership), over their imputed
    positions. Frames with fewer than `MIN_RELIABLE_KEYPOINTS` reliably tracked members are
    flagged undefined.

    Args:
        positions: keypoint -> finite imputed `(x, y)` arrays ``[F]``.
        reliable: keypoint -> boolean ``[F]``, True where the keypoint was tracked (not masked).
        members: the group's keypoints.
        times: frame timestamps ``[F]`` in seconds, strictly increasing.

    Returns:
        Dict with `centroid` (`(x, y)`), `relative_positions` (keypoint -> `(x, y)`),
        `relative_speed` and `absolute_speed` (keypoint -> ``[F]``, finite everywhere; the caller
        applies `valid_speed`), `valid_position` (boolean ``[F]``, enough reliable members) and
        `valid_speed` (boolean ``[F]``, additionally both frames of the central difference
        defined).

    Raises:
        ValueError: if the group has fewer than `MIN_RELIABLE_KEYPOINTS` members.
    """
    if len(members) < MIN_RELIABLE_KEYPOINTS:
        raise ValueError(
            f'a centroid needs at least {MIN_RELIABLE_KEYPOINTS} keypoints, got {members}',
        )
    xs = np.stack([positions[k][0] for k in members])
    ys = np.stack([positions[k][1] for k in members])
    centroid = (xs.mean(axis=0), ys.mean(axis=0))
    n_reliable = np.sum([np.asarray(reliable[k], dtype=bool) for k in members], axis=0)
    valid_position = n_reliable >= MIN_RELIABLE_KEYPOINTS
    left = np.concatenate([[True], valid_position[:-1]])
    right = np.concatenate([valid_position[1:], [True]])
    relative_positions = {
        k: (positions[k][0] - centroid[0], positions[k][1] - centroid[1]) for k in members
    }
    return {
        'centroid': centroid,
        'relative_positions': relative_positions,
        'relative_speed': {
            k: speed_from_positions(*relative_positions[k], times) for k in members
        },
        'absolute_speed': {k: speed_from_positions(*positions[k], times) for k in members},
        'valid_position': valid_position,
        'valid_speed': valid_position & left & right,
    }


def bin_with_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    times: np.ndarray,
    intervals: np.ndarray,
    binsize: float,
    n_bins: int,
) -> np.ndarray:
    """Bin a frame-rate signal onto the spike grid, keeping undefined stretches undefined.

    Same interpolation as the baseline (`bin_to_intervals`); a bin is defined only when both
    frames it is interpolated from are.

    Args:
        values: signal at the frame times ``[F]``; ignored where `valid` is False.
        valid: boolean ``[F]``, where the signal is defined.
        times: frame timestamps ``[F]``.
        intervals: contiguous intervals ``[n_intervals, 2]``.
        binsize: bin size in seconds.
        n_bins: bins per interval.

    Returns:
        Continuous series ``[n_intervals * n_bins]`` with NaN in undefined bins.
    """
    filled = np.where(valid, values, 0.0)
    binned = bin_to_intervals(filled, times, intervals, binsize, n_bins).ravel()
    coverage = bin_to_intervals(valid.astype(np.float64), times, intervals, binsize, n_bins)
    return np.where(coverage.ravel() >= 1.0 - 1e-9, binned, np.nan)


def load_camera_positions(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Imputed positions, reliability masks and frame times of one camera.

    The setup is that of `behavior.build_continuous_behavior` (same step A json, filter settings,
    position columns, unfiltered keypoints and `exclude`), up to the speed computation.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with the camera timestamp files.
        exclude: `fnmatch` patterns of keypoints to leave out.

    Returns:
        Dict with `keypoints` (usable keypoints after `exclude`), `positions` (keypoint -> finite
        imputed `(x, y)` ``[F]``), `reliable` (keypoint -> boolean ``[F]``), `masked_fraction`,
        `times`, `timing` and `csv_path`.

    Raises:
        FileNotFoundError: if the step A json is missing.
        ValueError: if the settings, csv and timestamps disagree (see the baseline), or `exclude`
            removes every keypoint.
    """
    video_preds_dir = load_video_preds_dir(config_path)
    if video_preds_dir is None:
        raise ValueError(f'{config_path}: paths.video_preds_dir is required')
    likelihood_threshold, max_masked_frac = load_filter_settings(config_path)
    json_path = keypoint_names_json_path(video_preds_dir, camera, eid)
    if not json_path.is_file():
        raise FileNotFoundError(
            f'{json_path} not found; run pose2visual.encoding.list_litpose_keypoints first',
        )
    with json_path.open(encoding='utf-8') as f:
        keypoint_info = json.load(f)
    exempt = load_unfiltered_keypoints(config_path, keypoint_info['keypoints'])
    position_columns = load_position_columns(config_path)
    check_filter_settings_match(
        keypoint_info, likelihood_threshold, max_masked_frac, json_path, exempt,
        load_visibility_features(config_path, keypoint_info['keypoints']), position_columns,
        load_same_side_excluded(config_path, keypoint_info['keypoints'], camera),
    )
    usable = list(keypoint_info['usable'])
    left_out = set(excluded_names(usable, list(exclude or [])))
    keypoints = [name for name in usable if name not in left_out]
    if not keypoints:
        raise ValueError(f'--exclude {exclude} removes every keypoint of the {camera} camera')
    csv_path = Path(video_preds_dir) / csv_filename(camera, eid)
    df = load_keypoint_csv(csv_path)
    times, _ = load_frame_times(video_timestamps_dir, camera, eid)
    timing = check_frame_times(times, len(df), csv_path.name)
    positions, masked_fraction = impute_keypoint_positions(
        df, keypoints, times, likelihood_threshold, position_columns, exempt,
    )
    reliable = {}
    for name in keypoints:
        x, y, likelihood = get_keypoint_coords(df, name, position_columns)
        threshold = None if name in exempt else likelihood_threshold
        reliable[name] = ~masked_frames(x, y, likelihood, threshold)
    return {
        'keypoints': keypoints,
        'positions': positions,
        'reliable': reliable,
        'masked_fraction': masked_fraction,
        'times': times,
        'timing': timing,
        'csv_path': str(csv_path),
    }


def _variances(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    frames: np.ndarray,
) -> tuple[float, float]:
    """Position variance (x plus y) and speed variance over a set of frames.

    Args:
        x: x position ``[F]``.
        y: y position ``[F]``.
        speed: speed ``[F]``.
        frames: boolean ``[F]``, the frames to use.

    Returns:
        `(var_position, var_speed)` in px^2 and px^2 / s^2.
    """
    return float(np.var(x[frames]) + np.var(y[frames])), float(np.var(speed[frames]))


def camera_motion(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    intervals: np.ndarray,
    bin_right_edges: np.ndarray,
    binsize: float,
    exclude: list[str],
    body_groups: dict[str, list[str]],
    no_centroid_groups: list[str],
    baseline_names: list[str],
) -> dict[str, Any]:
    """Relative-motion traces of one camera on the spike grid, plus all diagnostics.

    Args:
        config_path: frozen encoding yaml.
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with the camera timestamp files.
        intervals: contiguous intervals of the neural bundle.
        bin_right_edges: spike bin grid of the neural bundle.
        binsize: bin size in seconds.
        exclude: glob patterns of keypoints to leave out.
        body_groups: group name -> tokens.
        no_centroid_groups: groups that keep their original speed.
        baseline_names: behaviors the baseline kept for this camera; keypoints outside this list
            (dropped as constant, or hidden) are not analysed.

    Returns:
        Dict with `names` (analysed keypoints), `traces` ``[T, B]`` (relative speed for corrected
        keypoints with NaN where undefined, the baseline trace otherwise), `corrected` (bool
        ``[B]``), `plan` (group decisions with `valid_fraction`), `stats` (one dict per
        keypoint: variances and valid fraction), `centroids` (plot data), `timing`,
        `csv_path` and `masked_fraction`.

    Raises:
        ValueError: if the baseline bin grid or this module's absolute speeds disagree with the
            baseline traces (so the comparison would not be like for like).
    """
    data = load_camera_positions(config_path, eid, camera, video_timestamps_dir, exclude)
    times, positions, reliable = data['times'], data['positions'], data['reliable']
    behavior = build_continuous_behavior(
        config_path, eid, camera, video_timestamps_dir, intervals, binsize, exclude,
    )
    if behavior['grid'].shape != bin_right_edges.shape or not np.allclose(
        behavior['grid'], bin_right_edges, atol=1e-9,
    ):
        raise ValueError('behavior bin grid differs from the spike bin grid')
    lengths = intervals[:, 1] - intervals[:, 0]
    n_bins = int(np.ceil(lengths[0] / binsize))
    baseline_trace = {
        name: behavior['traces'][:, idx] for idx, name in enumerate(behavior['names'])
        if not name.endswith(VISIBILITY_SUFFIX)
    }
    keep = [name for name in baseline_names if name in baseline_trace]

    plan = plan_groups(data['keypoints'], body_groups, no_centroid_groups)
    traces: dict[str, np.ndarray] = {}
    corrected: dict[str, bool] = {}
    stats: dict[str, dict[str, Any]] = {}
    centroids: list[dict[str, Any]] = []
    centroid_traces: dict[str, np.ndarray] = {}
    centroid_groups: dict[str, str] = {}
    group_of: dict[str, str] = {}
    for entry in plan:
        members = entry['members']
        all_frames = np.ones(times.size, dtype=bool)
        entry['valid_fraction'] = None
        motion = None
        if entry['corrected']:
            motion = relative_group_motion(positions, reliable, members, times)
            entry['valid_fraction'] = float(motion['valid_speed'].mean())
            entry['interpolated_fraction'] = float(
                np.mean(~np.all([reliable[k] for k in members], axis=0)),
            )
            centroid_speed = speed_from_positions(*motion['centroid'], times)
            centroid_name = f'{entry["group"]}{CENTROID_SUFFIX}'
            centroid_traces[centroid_name] = bin_with_gaps(
                centroid_speed, motion['valid_speed'], times, intervals, binsize, n_bins,
            )
            centroid_groups[centroid_name] = entry['group']
            centroids.append({
                'camera': camera, 'group': entry['group'], 'times': times,
                'x': motion['centroid'][0], 'y': motion['centroid'][1],
                'speed': centroid_speed,
                'member_speed': np.mean([motion['absolute_speed'][k] for k in members], axis=0),
                'valid': motion['valid_speed'],
            })
        for name in members:
            group_of[name] = entry['group']
            absolute = speed_from_positions(*positions[name], times)
            if name in keep:
                mine = bin_to_intervals(absolute, times, intervals, binsize, n_bins).ravel()
                if not np.allclose(mine, baseline_trace[name], SELF_CHECK_RTOL, SELF_CHECK_ATOL):
                    raise ValueError(
                        f'{camera} {name}: absolute speed recomputed from positions differs '
                        'from the baseline trace; the experiment would not compare like with like',
                    )
            x, y = positions[name]
            if motion is None:
                traces[name] = baseline_trace.get(name)
                corrected[name] = False
                frames = all_frames
                rel_x, rel_y, relative = x, y, absolute
            else:
                rel_x, rel_y = motion['relative_positions'][name]
                relative = motion['relative_speed'][name]
                frames = motion['valid_speed']
                traces[name] = bin_with_gaps(
                    relative, frames, times, intervals, binsize, n_bins,
                )
                corrected[name] = True
            var_pos_o, var_speed_o = _variances(x, y, absolute, frames)
            var_pos_r, var_speed_r = _variances(rel_x, rel_y, relative, frames)
            stats[name] = {
                'group': entry['group'],
                'corrected': corrected[name],
                'reason': entry['reason'],
                'var_position_original': var_pos_o,
                'var_position_relative': var_pos_r,
                'var_speed_original': var_speed_o,
                'var_speed_relative': var_speed_r,
                'frame_fraction_defined': float(frames.mean()),
                'masked_fraction': float(data['masked_fraction'][name]),
            }
    names = [name for name in keep if traces.get(name) is not None]
    return {
        'names': names,
        'groups': [group_of[name] for name in names],
        'traces': np.stack([traces[name] for name in names], axis=1),
        'corrected': np.asarray([corrected[name] for name in names]),
        'centroid_names': list(centroid_traces),
        'centroid_groups': [centroid_groups[name] for name in centroid_traces],
        'centroid_traces': np.stack(
            list(centroid_traces.values()), axis=1,
        ) if centroid_traces else np.zeros((len(intervals) * n_bins, 0)),
        'plan': plan,
        'stats': {name: stats[name] for name in names},
        'centroids': centroids,
        'timing': data['timing'],
        'csv_path': data['csv_path'],
        'masked_fraction': data['masked_fraction'],
    }


def screen_traces(
    traces: np.ndarray,
    names: list[str],
    corrected: np.ndarray,
    min_valid_fraction: float,
) -> tuple[np.ndarray, dict[str, str]]:
    """Mark corrected keypoints whose relative speed is too sparse or constant.

    Args:
        traces: ``[T, B]`` with NaN where undefined.
        names: the ``B`` keypoint names.
        corrected: boolean ``[B]``.
        min_valid_fraction: smallest fraction of defined bins.

    Returns:
        `(usable, dropped)`: boolean ``[B]`` (True = keep) and keypoint -> reason for the dropped
        ones. Uncorrected keypoints are always kept (they are the baseline's traces).
    """
    usable = np.ones(len(names), dtype=bool)
    dropped: dict[str, str] = {}
    for idx, name in enumerate(names):
        if not corrected[idx]:
            continue
        defined = np.isfinite(traces[:, idx])
        fraction = float(defined.mean())
        if fraction < min_valid_fraction:
            dropped[name] = f'relative speed defined in only {100 * fraction:.1f} % of the bins'
        elif constant_columns(traces[defined, idx][:, None])[0]:
            dropped[name] = 'relative speed is constant'
        else:
            continue
        usable[idx] = False
    return usable, dropped


def explanation_text(report: dict[str, Any]) -> str:
    """Short markdown explanation plus the group decisions of one run.

    Args:
        report: the group report built by `run_session`.

    Returns:
        Markdown text.
    """
    lines = [
        f'# Experiment C: relative motion ({report["eid"]})',
        '',
        '**Removed**: the group centroid (the shared translation of the members) at every frame, '
        'for groups with two or more tracked keypoints. **Remains**: the movement of each '
        'keypoint relative to its group (rotation, deformation, opening/closing). A keypoint of '
        'a corrected group can still correlate with a neuron through movement the group does '
        'not share. Groups that were not corrected keep their original absolute speed.',
        '',
        'The relative speed is computed with the baseline speed function (central difference) '
        'from the relative positions. Frames with fewer than two reliably tracked members are '
        'undefined (NaN) and left out of the correlation.',
        '',
    ]
    for camera, info in report['cameras'].items():
        lines += [
            f'## {camera} camera', '',
            '| group | keypoints | treatment | reason | frames defined |',
            '|---|---|---|---|---|',
        ]
        for row in info['groups']:
            valid = row['valid_fraction']
            fraction = '-' if valid is None else f'{100 * valid:.1f} %'
            treatment = 'centroid subtracted' if row['corrected'] else 'original speed kept'
            lines.append(
                f'| {row["group"]} | {", ".join(row["members"])} | {treatment} | '
                f'{row["reason"]} | {fraction} |',
            )
        skipped = [
            f'{row["group"]}: {", ".join(row["members"])}'
            for row in info['groups'] if not row['corrected']
        ]
        lines += ['', '**Centroid subtraction intentionally skipped for:** '
                  + ('; '.join(skipped) if skipped else 'none')]
        if info['dropped']:
            lines += ['', '**Left out (relative speed unusable):** ' + '; '.join(
                f'{name} ({why})' for name, why in info['dropped'].items())]
        lines.append('')
    return '\n'.join(lines)


def run_session(
    config_path: str | Path,
    eid: str,
    video_timestamps_dir: str | Path,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    no_centroid_groups: list[str] | None = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    vmax: float | None = None,
    plots_subdir: str | None = None,
    skip_plot: bool = False,
    n_jobs: int = 1,
    depth_plot: bool = True,
    centroid_xcorr: bool = True,
    diff_vmax: float | None = None,
) -> dict[str, Path | None]:
    """Run Experiment C on one session.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `run_xcorr.CAMERAS`.
        exclude: glob patterns of keypoints to leave out (use the baseline's patterns).
        no_centroid_groups: groups that keep their original speed; `None` = the default
            (`DEFAULT_NO_CENTROID_GROUPS`), `[]` = correct every group with two or more keypoints.
        min_valid_fraction: smallest fraction of defined bins of a corrected keypoint.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule to the
            original xcorrs.
        plots_subdir: run name of the figure folder, to which the neuron order is
            appended; `None` = New York time of the run.
        skip_plot: compute and save only, draw no figures; redraw them later with
            `pose2visual.xcorr.experiments.replot_relative_motion`.
        n_jobs: cameras processed in parallel for the original xcorr.
        depth_plot: also draw the figures in probe-depth order into `<plots_subdir>_depth`
            (skipped when the first order is already depth, or when `skip_plot` is `True`).
        centroid_xcorr: also correlate the neurons with the centroid speed magnitude (px/s, no
            direction) of every corrected group (`r_centroid` in the arrays npz and
            `heatmap.centroid_speed`).
        diff_vmax: fixed half-range of the difference strips of `heatmap.signed_difference`
            (to compare sessions); `None` = data driven.

    Returns:
        Dict of the written paths: `arrays`, `summary`, `groups`, `groups_md` and `plots`
        (`None` when `skip_plot` is `True`).

    Raises:
        FileNotFoundError: if step 0 is missing.
        ValueError: if the absolute speeds recomputed from positions disagree with the
            baseline traces, or no keypoint is left to compare.
    """
    no_centroid = list(
        DEFAULT_NO_CENTROID_GROUPS if no_centroid_groups is None else no_centroid_groups,
    )
    base = compute_original(config_path, eid, video_timestamps_dir, cameras, exclude, n_jobs)
    exclude = base['exclude_patterns']
    neural = base['neural']
    cams, lags_sec = base['cameras'], base['lags_sec']
    body_groups = resolve_body_groups(config_path)
    plots_stem = plots_subdir or timestamp_subdir()
    out_dir, plots_root = new_run_dirs(base['session_dir'], EXPERIMENT_RELMOTION, plots_stem)

    # keypoints (not visibility features) of the baseline, in its grouped order
    idx_motion = [i for i, n in enumerate(base['names']) if not n.endswith(VISIBILITY_SUFFIX)]
    names = [base['names'][i] for i in idx_motion]
    groups = [base['groups'][i] for i in idx_motion]
    valid_base = base['valid'][:, idx_motion]
    r_original_all = base['r'][:, :, idx_motion, :]

    n_cam, n_neuron, n_beh, n_lag = r_original_all.shape
    r_relative = np.full(r_original_all.shape, np.nan)
    corrected = np.zeros((n_cam, n_beh), dtype=bool)
    valid = np.zeros((n_cam, n_beh), dtype=bool)
    group_rows: list[dict[str, Any]] = []
    camera_report: dict[str, Any] = {}
    stat_rows: list[dict[str, Any]] = []
    centroids: list[dict[str, Any]] = []
    # camera -> {centroid trace name: (group, xcorr [N, L])}, only with centroid_xcorr
    centroid_fits: dict[str, dict[str, tuple[str, np.ndarray]]] = {}
    for idx_c, camera in enumerate(cams):
        baseline_names = [n for n, ok in zip(names, valid_base[idx_c]) if ok]
        motion = camera_motion(
            config_path, eid, camera, video_timestamps_dir, neural['intervals'],
            neural['bin_right_edges'], neural['binsize'], exclude, body_groups,
            no_centroid, baseline_names,
        )
        usable, dropped = screen_traces(
            motion['traces'], motion['names'], motion['corrected'], min_valid_fraction,
        )
        for name, why in dropped.items():
            logger.warning(f'{camera} {name}: left out, {why}')
        idx_fit = [i for i in range(len(motion['names'])) if usable[i] and motion['corrected'][i]]
        if idx_fit:
            fitted = masked_cross_correlation(
                neural['neural'], motion['traces'][:, idx_fit], neural['max_lag_bins'],
            )
        for idx_k, name in enumerate(motion['names']):
            if not usable[idx_k]:
                continue
            idx_b = names.index(name)
            valid[idx_c, idx_b] = True
            corrected[idx_c, idx_b] = bool(motion['corrected'][idx_k])
            if motion['corrected'][idx_k]:
                r_relative[idx_c, :, idx_b, :] = fitted[:, idx_fit.index(idx_k), :]
            else:
                # not corrected: the baseline's own xcorr, not a recomputation
                r_relative[idx_c, :, idx_b, :] = r_original_all[idx_c, :, idx_b, :]
            if groups[idx_b] != motion['groups'][idx_k]:
                logger.warning(
                    f'{camera} {name}: baseline group {groups[idx_b]!r} but this run grouped it '
                    f'under {motion["groups"][idx_k]!r}',
                )
        if centroid_xcorr and motion['centroid_names']:
            usable_c, dropped_c = screen_traces(
                motion['centroid_traces'], motion['centroid_names'],
                np.ones(len(motion['centroid_names']), dtype=bool), min_valid_fraction,
            )
            for name, why in dropped_c.items():
                logger.warning(f'{camera} {name}: left out, {why}')
            idx_keep = np.flatnonzero(usable_c)
            if idx_keep.size:
                fitted_c = masked_cross_correlation(
                    neural['neural'], motion['centroid_traces'][:, idx_keep],
                    neural['max_lag_bins'],
                )
                centroid_fits[camera] = {
                    motion['centroid_names'][i]: (motion['centroid_groups'][i], fitted_c[:, j, :])
                    for j, i in enumerate(idx_keep)
                }
        for entry in motion['plan']:
            group_rows.append({'camera': camera, **entry})
        for name in motion['names']:
            if usable[motion['names'].index(name)]:
                stat_rows.append({'camera': camera, 'behavior': name, **motion['stats'][name]})
        centroids += motion['centroids']
        camera_report[camera] = {
            'groups': motion['plan'], 'dropped': dropped, 'frame_timing': motion['timing'],
            'csv_path': motion['csv_path'],
        }
        logger.info(
            f'{camera}: corrected groups '
            f'{[e["group"] for e in motion["plan"] if e["corrected"]]}, kept original speed '
            f'{[e["group"] for e in motion["plan"] if not e["corrected"]]}',
        )

    if not valid.any():
        left_out = {c: info['dropped'] for c, info in camera_report.items()}
        raise ValueError(f'no keypoint left to compare; left out: {left_out}')
    r_original = np.where(valid[:, None, :, None], r_original_all, np.nan)
    r_relative = np.where(valid[:, None, :, None], r_relative, np.nan)
    summary = pd.DataFrame(stat_rows)
    with warnings.catch_warnings():
        # keypoints a camera does not have are all NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        peak_original = np.nanmax(np.abs(r_original), axis=3)
        peak_relative = np.nanmax(np.abs(r_relative), axis=3)
    for idx, row in summary.iterrows():
        idx_c, idx_b = cams.index(row['camera']), names.index(row['behavior'])
        summary.loc[idx, 'frac_position_variance_removed'] = 1 - (
            row['var_position_relative'] / row['var_position_original']
        )
        summary.loc[idx, 'median_peak_original'] = np.median(peak_original[idx_c, :, idx_b])
        summary.loc[idx, 'median_peak_relative'] = np.median(peak_relative[idx_c, :, idx_b])
        summary.loc[idx, 'map_correlation'] = np.corrcoef(
            r_original[idx_c, :, idx_b, :].ravel(), r_relative[idx_c, :, idx_b, :].ravel(),
        )[0, 1]

    extra_arrays: dict[str, np.ndarray] = {}
    centroid_plot: dict[str, Any] = {}
    if centroid_fits:
        # union of the centroid traces over cameras, in first-seen order
        centroid_names = list(dict.fromkeys(n for fits in centroid_fits.values() for n in fits))
        centroid_groups = [
            next(fits[n][0] for fits in centroid_fits.values() if n in fits)
            for n in centroid_names
        ]
        r_centroid = np.full((n_cam, n_neuron, len(centroid_names), n_lag), np.nan)
        valid_centroid = np.zeros((n_cam, len(centroid_names)), dtype=bool)
        for idx_c, camera in enumerate(cams):
            for name, (_, fit) in centroid_fits.get(camera, {}).items():
                idx_g = centroid_names.index(name)
                r_centroid[idx_c, :, idx_g, :] = fit
                valid_centroid[idx_c, idx_g] = True
        extra_arrays = {
            'r_centroid': r_centroid.astype(np.float32), 'valid_centroid': valid_centroid,
            'centroid_names': np.asarray(centroid_names),
            'centroid_groups': np.asarray(centroid_groups),
        }
        centroid_plot = {
            'r_centroid': r_centroid, 'centroid_names': centroid_names,
            'centroid_groups': centroid_groups, 'valid_centroid': valid_centroid,
        }
    arrays_path = out_dir / f'relative_arrays.{eid}.npz'
    np.savez_compressed(
        arrays_path, r_relative=r_relative.astype(np.float32),
        r_original=r_original.astype(np.float32), corrected=corrected, valid=valid,
        lags_sec=lags_sec, cameras=np.asarray(cams), behavior_names=np.asarray(names),
        behavior_groups=np.asarray(groups), neuron_ids=base['neuron_ids'],
        neuron_depths=base['neuron_depths'], **extra_arrays,
    )
    summary_path = out_dir / f'relative_summary.{eid}.csv'
    summary.to_csv(summary_path, index=False)
    report = {
        'eid': eid, 'cameras': camera_report, 'no_centroid_groups': no_centroid,
        'min_valid_fraction': min_valid_fraction,
    }
    groups_path = out_dir / f'relative_groups.{eid}.json'
    groups_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    groups_md_path = out_dir / f'relative_groups.{eid}.md'
    groups_md_path.write_text(explanation_text(report), encoding='utf-8')
    # centroid trajectories are frame-rate position traces, not part of the spike-grid arrays
    # above and not reconstructible from them; pickled here so replot_relative_motion can redraw
    # the trajectory figure with no recomputation (positions would otherwise need reloading from
    # the keypoint csvs). group_rows is not pickled: it is exactly the flattening of
    # report['cameras'][camera]['groups'] over cameras, already in relative_groups.<eid>.json.
    centroids_path = out_dir / f'relative_centroids.{eid}.pkl'
    with centroids_path.open('wb') as f:
        pickle.dump(centroids, f)

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
                this_dir, eid, lags_sec, cams, names, groups, valid, r_original, r_relative,
                idx_order, order_label, vmax, corrected, summary, group_rows, centroids,
                diff_vmax=diff_vmax, **centroid_plot,
            )
            if order == 'saved':
                plots_dir = this_dir
            else:
                plots_dir_depth = this_dir
    return {
        'arrays': arrays_path, 'summary': summary_path, 'groups': groups_path,
        'groups_md': groups_md_path, 'plots': plots_dir, 'plots_depth': plots_dir_depth,
    }


def plot_results(
    plots_dir: str | Path,
    eid: str,
    lags_sec: np.ndarray,
    cams: list[str],
    names: list[str],
    groups: list[str],
    valid: np.ndarray,
    r_original: np.ndarray,
    r_relative: np.ndarray,
    idx_order: np.ndarray,
    order_label: str,
    vmax: float | None,
    corrected: np.ndarray,
    summary: pd.DataFrame,
    group_rows: list[dict[str, Any]],
    centroids: list[dict[str, Any]],
    r_centroid: np.ndarray | None = None,
    centroid_names: list[str] | None = None,
    centroid_groups: list[str] | None = None,
    valid_centroid: np.ndarray | None = None,
    diff_vmax: float | None = None,
) -> list[Path]:
    """Draw every Experiment C figure from already-computed (or reloaded) arrays.

    Shared by `run_session` (freshly computed arrays) and
    `pose2visual.xcorr.experiments.replot_relative_motion` (arrays reloaded from the saved
    npz/csv/json), so a figure always looks the same whether it is drawn right after the compute
    step or redrawn later with different `vmax`/`cameras`.

    Args:
        plots_dir: folder the figures are written into (created if missing).
        eid: session id.
        lags_sec: lag axis.
        cams: cameras, in plotting order.
        names: behavior (keypoint) names, baseline order.
        groups: anatomical group of each behavior (`names` order).
        valid: boolean `[C, B]`.
        r_original: baseline xcorr `[C, N, B, L]`, NaN where not `valid`.
        r_relative: relative-motion xcorr `[C, N, B, L]`, NaN where not `valid`.
        idx_order: neuron order to draw the heatmaps in.
        order_label: name of the neuron order, for the y-axis label.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule.
        corrected: boolean `[C, B]`, whether the centroid was subtracted for that behavior.
        summary: per-keypoint statistics table (`run_session`'s `summary`).
        group_rows: one dict per camera/group entry, for the group-decision table
            (`run_session`'s `group_rows`, or the concatenation of a saved
            `relative_groups.<eid>.json`'s per-camera `groups` plans).
        centroids: centroid trajectory data, one dict per corrected camera/group
            (`run_session`'s `centroids`); empty skips the trajectory figure.
        r_centroid: xcorr of the centroid speeds `[C, N, G, L]`; `None` skips their heatmap.
        centroid_names: the `G` centroid trace names (`<group>_centroid`).
        centroid_groups: anatomical group of each centroid trace.
        valid_centroid: boolean `[C, G]`.
        diff_vmax: half-range of the one colour scale of every difference strip of
            `heatmap.signed_difference`; `None` takes it from the data, once per figure.

    Returns:
        The written png files, then the figure guide `figures.<eid>.md`
        (`common.write_figure_guide`).
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.experiments import diagnostics
    from pose2visual.xcorr.figures import heatmap_vmax, plot_xcorr_overview, save_figure

    paths: list[Path] = []
    guide = [*HEATMAP_GUIDE, SCATTER_GUIDE, VARIANCE_GUIDE]
    with warnings.catch_warnings():
        # keypoints a camera does not have are all NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        peak_original = np.nanmax(np.abs(r_original), axis=3)
        peak_relative = np.nanmax(np.abs(r_relative), axis=3)
    # one colour scale for both heatmaps, so they are comparable without a difference figure
    scale = vmax if vmax is not None else heatmap_vmax(r_original)
    ylabel = f'neurons ({order_label} order)'
    panels = [
        ('original_speed', r_original, 'original speed (absolute)'),
        (
            'relative_speed', r_relative,
            'relative speed (group centroid removed where corrected)',
        ),
    ]
    for stem, data, label in panels:
        fig = plot_xcorr_overview(
            data[:, idx_order], lags_sec, names, cams, groups, valid, eid, scale, ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: {label}; colour scale +-{scale:.2f}; each strip: neurons x lag, '
            'number = max |value|', BASELINE_COLORBAR_LABEL,
        )
        paths += save_figure(fig, plots_dir / f'heatmap.{stem}.{eid}')
    if r_centroid is not None:
        fig = plot_xcorr_overview(
            r_centroid[:, idx_order], lags_sec, centroid_names, cams, centroid_groups,
            valid_centroid, eid, scale, ylabel=ylabel,
        )
        relabel_overview(
            fig, f'{eid}: group centroid speed; colour scale +-{scale:.2f}; each strip: '
            'neurons x lag, number = max |value|', BASELINE_COLORBAR_LABEL,
        )
        paths += save_figure(fig, plots_dir / f'heatmap.centroid_speed.{eid}')
        guide.insert(2, CENTROID_HEATMAP_GUIDE)
    scatter_panels = []
    for idx_c, camera in enumerate(cams):
        idx_b = np.flatnonzero(valid[idx_c])
        scatter_panels.append({
            'camera': camera, 'original': peak_original[idx_c][:, idx_b],
            'relative': peak_relative[idx_c][:, idx_b],
            'groups': [groups[i] for i in idx_b], 'corrected': corrected[idx_c, idx_b],
            'names': [names[i] for i in idx_b],
        })
    paths += save_figure(
        diagnostics.plot_peak_scatter(scatter_panels), plots_dir / f'scatter.peak_abs_r.{eid}',
    )
    signed_paths, signed_guide = save_signed_comparison(
        plots_dir, eid, lags_sec, cams, names, groups, valid, r_original, r_relative, idx_order,
        scale, ylabel, 'relative', corrected, groups, diff_vmax,
    )
    paths += signed_paths
    guide += signed_guide
    paths += save_figure(
        diagnostics.plot_variance_bars(summary), plots_dir / f'bar.variance.{eid}',
    )
    if centroids:
        paths += save_figure(
            diagnostics.plot_centroid_trajectories(centroids),
            plots_dir / f'centroid_trajectories.{eid}',
        )
        guide.append(CENTROID_GUIDE)
    paths += save_figure(
        diagnostics.plot_group_table(group_rows), plots_dir / f'groups_table.{eid}',
    )
    guide.append(GROUPS_TABLE_GUIDE)
    paths.append(write_figure_guide(
        plots_dir, eid, 'Experiment C: relative motion', order_label, scale, 'relative_speed',
        guide,
    ))
    return paths


def main(argv: list[str] | None = None) -> None:
    """Run Experiment C for every selected session; failures are listed at the end.

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
                    args.no_centroid_groups, args.min_valid_fraction, args.vmax, plots_subdir,
                    args.no_plot, args.n_jobs, not args.no_depth_plot,
                    not args.no_centroid_xcorr, args.diff_vmax,
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
