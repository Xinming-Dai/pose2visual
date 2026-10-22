"""Helpers for continuous 60 Hz spike counts aligned to a video.

Mirrors the alignment of ``encoding.extract_neural_data`` (same camera-synchronization check, same
integer-second interval rule, same bin edges) but keeps every contiguous 1 s interval of the video
instead of sampling 400 of them, so the counts can be concatenated into one continuous time
series. These helpers do not import ``one``/``brainbox``, so they can be tested without them.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# the pooled spike train of a session has thousands of spikes per second; a video that starts
# or ends this long outside the first / last spike is not covered by the recording
COVERAGE_TOLERANCE_SEC = 1.0


def video_time_range(
    timestamps: list[np.ndarray],
) -> tuple[int, int]:
    """Integer-second range of the video, with the rule of ``extract_neural_data``.

    Args:
        timestamps: frame timestamps (seconds) of every available camera, one array each.

    Returns:
        Tuple `(min_timestamp, max_timestamp)`: `round(first frame) + 1` and `round(last frame)
        - 1`, so both lie inside the video.

    Raises:
        ValueError: if no timestamps are given.
        AssertionError: if two cameras' timestamps differ (they must be synchronized).
    """
    if not timestamps:
        raise ValueError('no camera timestamps given')
    if len(timestamps) == 2 and not np.array_equal(timestamps[0], timestamps[1]):
        raise AssertionError('left and right camera timestamps must be synchronized.')
    return (
        int(round(min(a.min() for a in timestamps))) + 1,
        int(round(max(a.max() for a in timestamps))) - 1,
    )


def contiguous_intervals(min_timestamp: float, max_timestamp: float) -> np.ndarray:
    """All contiguous 1 s intervals of the video, the pool step 0 samples its trials from.

    Args:
        min_timestamp: start of the first interval (seconds).
        max_timestamp: end bound, see `ibl_data_utils.create_intervals`.

    Returns:
        Array ``[n_intervals, 2]`` of start / end times; each start is the previous end.
    """
    # lazy: avoids importing the heavy ibl dependencies unless intervals are built
    from pose2visual.encoding.ibl_data_utils import create_intervals

    return create_intervals(min_timestamp, max_timestamp, 1)


def interval_bin_grid(intervals: np.ndarray, binsize: float, n_bins: int) -> np.ndarray:
    """Right edge of every bin of every interval, ``linspace(beg + binsize, end, n_bins)``.

    This is the grid the encoding pipeline samples behavior on.

    Args:
        intervals: ``[n_intervals, 2]`` start / end times.
        binsize: bin size in seconds.
        n_bins: bins per interval.

    Returns:
        Flat array ``[n_intervals * n_bins]`` in time order.
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    return np.linspace(intervals[:, 0] + binsize, intervals[:, 1], n_bins, axis=1).ravel()


def concat_intervals(binned: np.ndarray) -> np.ndarray:
    """Concatenate per-interval counts into one continuous time series.

    Args:
        binned: counts ``[n_intervals, n_bins, N]`` from `bin_spiking_data`.

    Returns:
        Array ``[n_intervals * n_bins, N]``.
    """
    n_intervals, n_bins, n_units = binned.shape
    return binned.reshape(n_intervals * n_bins, n_units)


def check_spike_coverage(spike_times: np.ndarray, intervals: np.ndarray) -> None:
    """Fail when the video extends beyond the recorded spikes.

    Bins outside the recording would be all zero and look like silence, so they must not be
    treated as data.

    Args:
        spike_times: spike times of the session in seconds (the camera clock).
        intervals: video intervals ``[n_intervals, 2]``.

    Raises:
        ValueError: if the first spike is later than the video start, or the last spike earlier
            than the video end, by more than `COVERAGE_TOLERANCE_SEC`.
    """
    first, last = float(np.min(spike_times)), float(np.max(spike_times))
    start, end = float(intervals[0, 0]), float(intervals[-1, 1])
    if first > start + COVERAGE_TOLERANCE_SEC or last < end - COVERAGE_TOLERANCE_SEC:
        raise ValueError(
            f'spikes span [{first:.2f}, {last:.2f}] s but the video bins span '
            f'[{start:.2f}, {end:.2f}] s; the recording does not cover the video, so the '
            'time axes cannot be assumed to match',
        )


def select_units(counts: np.ndarray, binsize: float, min_fr_hz: float) -> np.ndarray:
    """Indices of the units whose mean firing rate exceeds a threshold.

    Args:
        counts: spike counts ``[T, N]``.
        binsize: bin size in seconds.
        min_fr_hz: a unit is kept when its mean rate is strictly above this.

    Returns:
        Integer indices of the kept units.
    """
    mean_fr = counts.sum(axis=0) / (counts.shape[0] * binsize)
    return np.flatnonzero(mean_fr > min_fr_hz)
