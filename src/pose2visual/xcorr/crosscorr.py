"""Normalized cross-correlation between neural activity and behavioral traces.

Convention: ``r(tau) = Corr(neural(t), behavior(t + tau))``. A positive lag means the behavior
follows the neural activity, i.e. the neural activity precedes the behavior. Both signals are
z-scored over the whole recording and the correlation is the mean product at each lag, with the
signals treated as circular (the samples that wrap around affect at most ``max_lag / T`` of the
product, 0.06 % for 60 lags of a 100 000-sample recording). The circular estimator makes the
permutation null of ``pose2visual.xcorr.permutation`` exact: a circular shift of the behavior
only changes which lags of the same circular cross-correlation are read.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def constant_columns(x: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Boolean mask of the columns of a ``[T, K]`` array that do not vary over time.

    A column counts as constant when its peak-to-peak range is at most `rtol` times its largest
    absolute value (at least 1), so floating-point noise of a derived trace, e.g. the ~1e-14
    speed of a keypoint that never moves, is not mistaken for signal.

    Args:
        x: array of shape ``[T, K]``.
        rtol: relative tolerance.

    Returns:
        Boolean array ``[K]``, True where the column is constant.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.maximum(np.abs(x).max(axis=0), 1.0)
    return np.ptp(x, axis=0) <= rtol * scale


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score each column of a ``[T, K]`` array over time (population std, ``ddof=0``).

    Args:
        x: array of shape ``[T, K]``.

    Returns:
        Float64 array of the same shape with zero mean and unit variance per column.

    Raises:
        ValueError: if a column is constant (see `constant_columns`) or not finite, since its
            correlation is undefined.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('zscore: input contains NaN or inf')
    constant = constant_columns(x)
    if np.any(constant):
        raise ValueError(
            f'zscore: constant columns {np.flatnonzero(constant).tolist()}; drop them first',
        )
    return (x - x.mean(axis=0)) / x.std(axis=0)


def smooth_neural(spikes: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian-smooth spike counts along time, over the whole recording.

    Args:
        spikes: counts of shape ``[T, N]``.
        sigma_bins: gaussian sigma in bins; 0 returns the counts as floats.

    Returns:
        Float64 array ``[T, N]``.
    """
    spikes = np.asarray(spikes, dtype=np.float64)
    if sigma_bins == 0:
        return spikes
    return gaussian_filter1d(spikes, sigma_bins, axis=0, mode='nearest')


def lag_indices(max_lag_bins: int, n_time: int) -> np.ndarray:
    """Indices of the lags ``-max_lag_bins .. +max_lag_bins`` in a circular correlation.

    Args:
        max_lag_bins: largest lag in bins.
        n_time: number of time samples ``T``.

    Returns:
        Integer array ``[2 * max_lag_bins + 1]``: lag ``tau`` is stored at index ``tau mod T``.

    Raises:
        ValueError: if ``max_lag_bins`` is negative or the lag window does not fit in ``T``.
    """
    if max_lag_bins < 0:
        raise ValueError(f'max_lag_bins must be >= 0, got {max_lag_bins}')
    if n_time <= 2 * max_lag_bins + 1:
        raise ValueError(f'{n_time} samples are too few for {2 * max_lag_bins + 1} lags')
    return np.arange(-max_lag_bins, max_lag_bins + 1) % n_time


def full_circular_xcorr(
    fft_neural: np.ndarray,
    fft_behavior: np.ndarray,
    n_time: int,
) -> np.ndarray:
    """Circular cross-correlation at every shift, for one behavior and all neurons.

    Args:
        fft_neural: ``rfft`` over time of the z-scored neural activity, shape ``[F, N]``.
        fft_behavior: ``rfft`` over time of one z-scored behavior, shape ``[F]``.
        n_time: number of time samples ``T`` the transforms were taken over.

    Returns:
        Array ``[N, T]`` with ``C[n, k] = mean_t z_neural[t, n] * z_behavior[(t + k) mod T]``.
    """
    product = np.conj(fft_neural) * fft_behavior[:, None]
    return np.fft.irfft(product, n=n_time, axis=0).T / n_time


def cross_correlation(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag_bins: int,
) -> np.ndarray:
    """Normalized cross-correlation of every neuron with every behavior over a lag window.

    Args:
        neural: neural activity ``[T, N]`` (already smoothed, if wanted).
        behavior: behavioral traces ``[T, B]`` on the same time grid.
        max_lag_bins: largest lag in bins; the output has ``2 * max_lag_bins + 1`` lags.

    Returns:
        Array ``[N, B, 2 * max_lag_bins + 1]``, ``r[n, b, i] = Corr(neural_n(t),
        behavior_b(t + tau_i))`` with ``tau = -max_lag_bins .. max_lag_bins``.

    Raises:
        ValueError: if the time axes differ, a column is constant or not finite, or ``T`` is too
            short for the lag window.
    """
    neural = np.asarray(neural)
    behavior = np.asarray(behavior)
    if neural.shape[0] != behavior.shape[0]:
        raise ValueError(
            f'neural has {neural.shape[0]} time bins but behavior has {behavior.shape[0]}',
        )
    n_time = neural.shape[0]
    idx_lag = lag_indices(max_lag_bins, n_time)
    fft_neural = np.fft.rfft(zscore(neural), axis=0)
    fft_behavior = np.fft.rfft(zscore(behavior), axis=0)
    out = np.empty((neural.shape[1], behavior.shape[1], idx_lag.size))
    for idx_b in range(behavior.shape[1]):
        out[:, idx_b, :] = full_circular_xcorr(fft_neural, fft_behavior[:, idx_b], n_time)[
            :, idx_lag
        ]
    return out


def summarize_peaks(
    r: np.ndarray,
    lags_sec: np.ndarray,
    neuron_info: pd.DataFrame,
    behavior_names: list[str],
    eid: str,
    camera: str | None = None,
) -> pd.DataFrame:
    """One row per neuron-behavior pair with the peak of its cross-correlation curve.

    The peak is the lag of the largest absolute correlation, so ``peak_r`` keeps its sign.

    Args:
        r: correlations ``[N, B, L]`` from `cross_correlation`.
        lags_sec: lag in seconds of each of the ``L`` entries.
        neuron_info: one row per neuron (``N`` rows, same order as ``r``); every column is copied
            to the output (e.g. `neuron_id`, `region`, `depth`).
        behavior_names: the ``B`` behavior names.
        eid: session id, written in every row.
        camera: camera name, written in a `camera` column after `eid` when given.

    Returns:
        Dataframe of ``N * B`` rows, neuron-major, with `eid`, `camera` (when given),
        `neuron_idx`, the `neuron_info` columns, `behavior`, `peak_r`, `peak_abs_r`,
        `peak_lag_sec` and `peak_lag_bins`.

    Raises:
        ValueError: if the shapes of the inputs disagree.
    """
    n_neurons, n_behaviors, n_lags = r.shape
    if len(neuron_info) != n_neurons or len(behavior_names) != n_behaviors:
        raise ValueError('neuron_info / behavior_names do not match the shape of r')
    if len(lags_sec) != n_lags:
        raise ValueError(f'{len(lags_sec)} lags but r has {n_lags}')
    idx_peak = np.argmax(np.abs(r), axis=2)
    peak_r = np.take_along_axis(r, idx_peak[..., None], axis=2)[..., 0]
    lag_bins = np.arange(n_lags) - n_lags // 2
    info = neuron_info.reset_index(drop=True)
    table = pd.DataFrame(index=np.arange(n_neurons * n_behaviors))
    table['eid'] = eid
    if camera is not None:
        table['camera'] = camera
    table['neuron_idx'] = np.repeat(np.arange(n_neurons), n_behaviors)
    for column in info.columns:
        table[column] = np.repeat(info[column].to_numpy(), n_behaviors)
    table['behavior'] = np.tile(np.asarray(behavior_names), n_neurons)
    table['peak_r'] = peak_r.ravel()
    table['peak_abs_r'] = np.abs(peak_r).ravel()
    table['peak_lag_sec'] = np.asarray(lags_sec)[idx_peak].ravel()
    table['peak_lag_bins'] = lag_bins[idx_peak].ravel()
    return table
