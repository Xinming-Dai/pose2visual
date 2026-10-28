"""Continuous behavioral traces from Lightning Pose keypoints, on the spike bin grid.

Uses exactly the encoding pipeline's functions: likelihood masking and linear interpolation
(``litpose_keypoints.impute_keypoint_positions``), keypoint speed, the ``<group>_visible``
features, and binning onto the right edge of each spike bin (``build_dynamic_vars``). The
only difference is that the intervals are all the contiguous 1 s intervals of the video, so the
binned values concatenate into one continuous series per feature. The features are those of the
encoding without PCA: the speed of each usable keypoint and the binary visibility features.
"""

import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.encoding.extract_litpose_dynamic_vars_trials import build_dynamic_vars
from pose2visual.encoding.litpose_keypoints import (
    check_filter_settings_match,
    csv_filename,
    keypoint_names_json_path,
    load_filter_settings,
    load_frame_times,
    load_keypoint_csv,
    load_position_columns,
    load_same_side_excluded,
    load_unfiltered_keypoints,
    load_video_preds_dir,
    load_visibility_features,
)
from pose2visual.xcorr.crosscorr import constant_columns
from pose2visual.xcorr.neural import interval_bin_grid

logger = logging.getLogger(__name__)

# a frame interval longer than this many median intervals counts as a gap in the video
GAP_FACTOR = 2.0


def check_frame_times(times: np.ndarray, n_rows: int, csv_name: str) -> dict[str, Any]:
    """Check that the frame timestamps fit the prediction csv and are regular.

    Args:
        times: frame timestamps in seconds.
        n_rows: number of rows of the prediction csv.
        csv_name: csv file name, used in error messages.

    Returns:
        Diagnostics for `params.json`: `n_frames`, `t_start`, `t_end`, `median_dt`, `fps`,
        `n_gaps` (frame intervals above `GAP_FACTOR` median intervals) and `max_gap_sec`.

    Raises:
        ValueError: if the number of timestamps differs from `n_rows`, a timestamp is not
            finite, or the timestamps are not strictly increasing.
    """
    if len(times) != n_rows:
        raise ValueError(f'{csv_name} has {n_rows} rows but there are {len(times)} timestamps')
    if not np.all(np.isfinite(times)):
        raise ValueError('frame timestamps contain NaN or inf')
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError('frame timestamps are not strictly increasing')
    median_dt = float(np.median(dt))
    gaps = dt > GAP_FACTOR * median_dt
    info = {
        'n_frames': int(len(times)),
        't_start': float(times[0]),
        't_end': float(times[-1]),
        'median_dt': median_dt,
        'fps': 1.0 / median_dt,
        'n_gaps': int(gaps.sum()),
        'max_gap_sec': float(dt.max()),
    }
    if info['n_gaps']:
        logger.warning(
            f'{info["n_gaps"]} frame gaps > {GAP_FACTOR} x the median interval '
            f'(largest {info["max_gap_sec"]:.3f} s); speed is interpolated across them',
        )
    return info


def excluded_names(names: list[str], patterns: list[str]) -> list[str]:
    """Names matching any of the glob patterns.

    Args:
        names: keypoint or feature names.
        patterns: `fnmatch` patterns, e.g. `'d*'` for the finger keypoints `d1_tip_left` ...

    Returns:
        The matching names, in `names` order.
    """
    return [n for n in names if any(fnmatch.fnmatchcase(n, pattern) for pattern in patterns)]


def build_continuous_behavior(
    config_path: str | Path,
    eid: str,
    camera: str,
    video_timestamps_dir: str | Path,
    intervals: np.ndarray,
    binsize: float,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Behavioral features of one session on the continuous spike bin grid.

    Args:
        config_path: path of the encoding yaml (filter settings, position columns, visibility,
            `paths.video_preds_dir`).
        eid: session id.
        camera: `'left'` or `'right'`.
        video_timestamps_dir: directory with `_ibl_<camera>Camera.times.<eid>.npy`.
        intervals: contiguous video intervals ``[n_intervals, 2]`` of the neural bundle.
        binsize: bin size in seconds.
        exclude: `fnmatch` patterns of keypoints / features to leave out (e.g. `['d*']` for the
            fingers). They are removed before any trace is computed. `None` = keep everything.

    Returns:
        Dict with `traces` (float array ``[T, B]``, time-major), `names` (the `B` feature names,
        speeds then visibility features), `grid` (bin right edges ``[T]``), `masked_fraction`
        (keypoint -> masked frame fraction), `timing` (see `check_frame_times`),
        `dropped_constant` (features without variance, left out), `excluded` (names removed by
        `exclude`), `csv_path` and `timestamps_path`.

    Raises:
        FileNotFoundError: if the step A json is missing.
        ValueError: if the step A json was written with other settings than the yaml, the csv and
            timestamps disagree, the video bins are not covered by the frame times, or `exclude`
            removes every feature.
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
    no_filter_keypoints = load_unfiltered_keypoints(config_path, keypoint_info['keypoints'])
    visibility = load_visibility_features(config_path, keypoint_info['keypoints'])
    position_columns = load_position_columns(config_path)
    check_filter_settings_match(
        keypoint_info, likelihood_threshold, max_masked_frac, json_path, no_filter_keypoints,
        visibility, position_columns,
        load_same_side_excluded(config_path, keypoint_info['keypoints'], camera),
    )
    keypoints = list(keypoint_info['usable'])
    feature_members = {} if visibility is None else dict(visibility['features'])
    patterns = list(exclude or [])
    for pattern in patterns:
        if not excluded_names(keypoints + list(feature_members), [pattern]):
            logger.warning(f'--exclude pattern {pattern!r} matches no keypoint or feature')
    excluded = excluded_names(keypoints + list(feature_members), patterns)
    keypoints = [name for name in keypoints if name not in excluded]
    feature_members = {k: v for k, v in feature_members.items() if k not in excluded}
    if visibility is not None:
        visibility = {**visibility, 'features': feature_members} if feature_members else None
    if not keypoints and visibility is None:
        raise ValueError(f'--exclude {patterns} removes every keypoint and feature')
    if excluded:
        logger.info(f'excluding {len(excluded)} keypoints/features: {excluded}')

    csv_path = Path(video_preds_dir) / csv_filename(camera, eid)
    logger.info(f'loading keypoints from {csv_path}')
    df = load_keypoint_csv(csv_path)
    times, times_path = load_frame_times(video_timestamps_dir, camera, eid)
    timing = check_frame_times(times, len(df), csv_path.name)

    intervals = np.asarray(intervals, dtype=np.float64)
    lengths = intervals[:, 1] - intervals[:, 0]
    if not np.allclose(lengths, lengths[0]) or not np.allclose(
        intervals[1:, 0], intervals[:-1, 1],
    ):
        raise ValueError('intervals must be contiguous and of equal length')
    n_bins = int(np.ceil(lengths[0] / binsize))
    binned, masked_fraction = build_dynamic_vars(
        df, keypoints, times, {'all': intervals}, likelihood_threshold, binsize, n_bins,
        no_filter_keypoints=no_filter_keypoints,
        visibility=visibility,
        position_columns=position_columns,
    )
    # [K, n_bins, 1, D] -> [K * n_bins, D]
    traces = binned['all'][:, :, 0, :].reshape(-1, binned['all'].shape[-1]).astype(np.float64)
    names = list(keypoints) + ([] if visibility is None else list(visibility['features']))

    constant = constant_columns(traces)
    dropped = [name for name, flag in zip(names, constant) if flag]
    if dropped:
        logger.warning(f'dropping features without variance: {dropped}')
        traces = traces[:, ~constant]
        names = [name for name, flag in zip(names, constant) if not flag]
    return {
        'traces': traces,
        'names': names,
        'grid': interval_bin_grid(intervals, binsize, n_bins),
        'masked_fraction': masked_fraction,
        'timing': timing,
        'dropped_constant': dropped,
        'excluded': excluded,
        'csv_path': str(csv_path),
        'timestamps_path': str(times_path),
    }
