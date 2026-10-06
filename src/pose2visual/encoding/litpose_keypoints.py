"""Shared helpers for encoding from Lightning Pose keypoint predictions.

Lightning Pose writes one csv per camera video, ``_iblrig_<camera>Camera.downsampled.<eid>.csv``,
with three header rows (``scorer`` / ``bodyparts`` / ``coords``) and ``x, y, likelihood``
columns per keypoint, one row per video frame. This module holds the file naming conventions,
csv reading, keypoint speed computation and trial binning shared by
``list_litpose_keypoints``, ``extract_litpose_dynamic_vars_trials`` and
``run_litpose_encoding``.

Directory layout under ``video_preds_dir`` (``<model_dir>/video_preds``)::

    encoding/keypoint_names/<camera>_<eid>.json           step A, one per csv
    <output_name>/<eid>/dynamic_vars.npz                  step B
    <output_name>/<eid>/base/encoding_result.npz          step C, base fit, reused by variants
    <output_name>/<eid>/<variant_name>/encoding_result.npz  step C, one per additional set (the
        default name is the sorted additional keypoints joined by ``_``, plus a hash suffix when
        that folder holds a run with other settings)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CAMERAS = ('left', 'right')

# <eid> never contains an underscore, which keeps ``*_temporal_norm.csv`` files out
_CSV_PATTERN = re.compile(r'^_iblrig_(left|right)Camera\.downsampled\.([^_]+)\.csv$')


def csv_filename(camera: str, eid: str) -> str:
    """Name of the Lightning Pose prediction csv for one camera and session.

    Args:
        camera: `'left'` or `'right'`.
        eid: session id.

    Returns:
        File name such as `_iblrig_leftCamera.downsampled.<eid>.csv`.
    """
    return f'_iblrig_{camera}Camera.downsampled.{eid}.csv'


def list_keypoint_csvs(video_preds_dir: str | Path) -> list[tuple[str, str, Path]]:
    """Find every prediction csv in a ``video_preds`` directory.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.

    Returns:
        Sorted list of `(camera, eid, csv_path)`; `*_temporal_norm.csv` files are skipped.
    """
    found = []
    for path in Path(video_preds_dir).iterdir():
        match = _CSV_PATTERN.match(path.name)
        if match:
            found.append((match.group(1), match.group(2), path))
    return sorted(found, key=lambda item: (item[1], item[0]))


def keypoint_names_json_path(video_preds_dir: str | Path, camera: str, eid: str) -> Path:
    """Path of the step A json listing the keypoints of one camera and session.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        camera: `'left'` or `'right'`.
        eid: session id.

    Returns:
        `<video_preds_dir>/encoding/keypoint_names/<camera>_<eid>.json`.
    """
    return Path(video_preds_dir) / 'encoding' / 'keypoint_names' / f'{camera}_{eid}.json'


def session_output_dir(video_preds_dir: str | Path, output_name: str, eid: str) -> Path:
    """Directory holding the step B / step C outputs of one session.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: user-chosen folder name.
        eid: session id.

    Returns:
        `<video_preds_dir>/<output_name>/<eid>`.
    """
    return Path(video_preds_dir) / output_name / eid


def base_dir(
    video_preds_dir: str | Path,
    output_name: str,
    eid: str,
    key_hash: str | None = None,
) -> Path:
    """Directory holding the base-keypoints fits that the variants of a session reuse.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: user-chosen folder name.
        eid: session id.
        key_hash: hash of the base cache key; each distinct base input (pca on / off, number of
            components, seed, ...) gets its own subfolder so one fit never replaces another.

    Returns:
        `<video_preds_dir>/<output_name>/<eid>/base`, plus `/<key_hash>` when given.
    """
    root = session_output_dir(video_preds_dir, output_name, eid) / 'base'
    return root if key_hash is None else root / key_hash


def variant_dir(
    video_preds_dir: str | Path,
    output_name: str,
    eid: str,
    variant_name: str,
) -> Path:
    """Directory holding the results of one `additional_keypoints` set.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: user-chosen folder name.
        eid: session id.
        variant_name: name of the additional-keypoints variant.

    Returns:
        `<video_preds_dir>/<output_name>/<eid>/<variant_name>`.
    """
    return session_output_dir(video_preds_dir, output_name, eid) / variant_name


def load_keypoint_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a Lightning Pose prediction csv with its three header rows.

    Args:
        csv_path: path of the csv.

    Returns:
        Dataframe with `(scorer, bodypart, coord)` column levels, one row per frame.
    """
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def get_keypoint_names(df: pd.DataFrame) -> list[str]:
    """Keypoint names (second header row) deduplicated in file order.

    Args:
        df: dataframe from `load_keypoint_csv`.

    Returns:
        Keypoint names in header order.
    """
    return list(dict.fromkeys(df.columns.get_level_values(1)))


def get_keypoint_coords(
    df: pd.DataFrame,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract `x`, `y` and `likelihood` of one keypoint.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.

    Returns:
        Tuple of float64 arrays `(x, y, likelihood)`, each of shape `[n_frames]`.
    """
    sub = df.xs(name, axis=1, level=1)
    # the remaining column level is the coordinate name
    sub.columns = pd.Index(sub.columns.get_level_values(-1))
    return (
        sub['x'].to_numpy(dtype=np.float64),
        sub['y'].to_numpy(dtype=np.float64),
        sub['likelihood'].to_numpy(dtype=np.float64),
    )


def masked_frames(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    likelihood_threshold: float,
) -> np.ndarray:
    """Boolean mask of unusable frames: NaN coordinate/likelihood or low likelihood.

    Args:
        x: x position per frame.
        y: y position per frame.
        likelihood: likelihood per frame.
        likelihood_threshold: frames below this likelihood are masked.

    Returns:
        Boolean array `[n_frames]`, `True` for masked frames.
    """
    # NaN likelihood compares False in the threshold test, so mask it explicitly
    return (likelihood < likelihood_threshold) | np.isnan(x) | np.isnan(y) | np.isnan(likelihood)


def frame_fractions(
    df: pd.DataFrame,
    name: str,
    likelihood_threshold: float,
) -> dict[str, float]:
    """Split the unusable frames of a keypoint by cause.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.
        likelihood_threshold: frames below this likelihood count as low-likelihood.

    Returns:
        Dict with `masked` (any cause, as `masked_frames`), `nan` (NaN x, y or likelihood) and
        `low_likelihood` (finite likelihood below the threshold), each a fraction in `[0, 1]`.
    """
    x, y, likelihood = get_keypoint_coords(df, name)
    nan = np.isnan(x) | np.isnan(y) | np.isnan(likelihood)
    low = ~np.isnan(likelihood) & (likelihood < likelihood_threshold)
    return {
        'masked': float(np.mean(masked_frames(x, y, likelihood, likelihood_threshold))),
        'nan': float(np.mean(nan)),
        'low_likelihood': float(np.mean(low)),
    }


def masked_fraction(df: pd.DataFrame, name: str, likelihood_threshold: float) -> float:
    """Fraction of frames of a keypoint that `masked_frames` marks unusable.

    Keypoints on the far side of the animal are typically predicted with likelihood near 0 and
    therefore have a masked fraction near 1.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.
        likelihood_threshold: frames below this likelihood are masked.

    Returns:
        Fraction in `[0, 1]`.
    """
    x, y, likelihood = get_keypoint_coords(df, name)
    return float(np.mean(masked_frames(x, y, likelihood, likelihood_threshold)))


def compute_keypoint_speed(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    times: np.ndarray,
    likelihood_threshold: float,
) -> tuple[np.ndarray, float]:
    """Speed of one keypoint in pixels per second.

    Frames flagged by `masked_frames` (low likelihood or NaN) are masked and the
    position is linearly interpolated over them before differentiating.

    Args:
        x: x position per frame.
        y: y position per frame.
        likelihood: likelihood per frame.
        times: frame timestamps in seconds, strictly increasing.
        likelihood_threshold: frames below this likelihood are masked.

    Returns:
        Tuple `(speed, masked_fraction)`: speed per frame `[n_frames]` and the fraction of
        masked frames.

    Raises:
        ValueError: if fewer than two frames remain unmasked.
    """
    masked = masked_frames(x, y, likelihood, likelihood_threshold)
    idx_valid = np.flatnonzero(~masked)
    if idx_valid.size < 2:
        raise ValueError(
            f'fewer than 2 frames with likelihood >= {likelihood_threshold}; '
            'cannot compute keypoint speed',
        )
    x_interp = np.interp(times, times[idx_valid], x[idx_valid])
    y_interp = np.interp(times, times[idx_valid], y[idx_valid])
    speed = np.hypot(np.gradient(x_interp, times), np.gradient(y_interp, times))
    return speed, float(masked.mean())


def bin_to_intervals(
    values: np.ndarray,
    times: np.ndarray,
    intervals: np.ndarray,
    binsize: float,
    n_bins: int,
) -> np.ndarray:
    """Linearly interpolate a time series onto the per-interval bin grid used for neural data.

    The grid of each interval is `linspace(beg + binsize, end, n_bins)`, identical to
    `ibl_data_utils.get_behavior_per_interval`.

    Args:
        values: signal sampled at `times`, shape `[n_frames]`.
        times: sample timestamps in seconds, strictly increasing.
        intervals: interval start/end times, shape `[n_intervals, 2]`.
        binsize: bin size in seconds.
        n_bins: number of bins per interval.

    Returns:
        Array of shape `[n_intervals, n_bins]`.

    Raises:
        ValueError: if an interval lies outside the range of `times`.
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size and (intervals.min() < times[0] or intervals.max() > times[-1]):
        raise ValueError(
            f'intervals [{intervals.min():.3f}, {intervals.max():.3f}] fall outside the '
            f'frame times [{times[0]:.3f}, {times[-1]:.3f}]',
        )
    grid = np.linspace(intervals[:, 0] + binsize, intervals[:, 1], n_bins, axis=1)
    return np.interp(grid.ravel(), times, values).reshape(grid.shape)
