"""Neural-behavior cross-correlation for behavior traces with missing samples.

``pose2visual.xcorr.crosscorr.cross_correlation`` needs finite behavior traces. A relative-motion
trace (``relative_motion``) is undefined where fewer than two keypoints of its group are tracked,
and a missing sample must stay missing, not be replaced by zero or a mean. This module computes the
same quantity over the valid samples only: for every lag ``tau`` the Pearson correlation of
``neural(t)`` and ``behavior(t + tau)`` over the pairs whose behavior sample is valid, with the
same circular convention and lag indexing as the baseline. The statistics of the neural activity
are re-evaluated on the pairs of each lag (they change with the lag when the valid set is not the
whole recording), so the result is exact, not a zero-filled approximation. With no missing sample
it equals ``cross_correlation``.
"""

import numpy as np

from pose2visual.xcorr.crosscorr import (
    constant_columns,
    full_circular_xcorr,
    lag_indices,
    zscore,
)

# below this many valid samples the correlation of a behavior is not meaningful
MIN_VALID_SAMPLES = 10


def masked_cross_correlation(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag_bins: int,
) -> np.ndarray:
    """Pearson cross-correlation of every neuron with every behavior over the valid samples.

    ``r[n, b, i] = Corr(neural_n(t), behavior_b(t + tau_i))`` over the times ``t`` for which
    ``behavior_b(t + tau_i)`` is finite, with ``tau = -max_lag_bins .. max_lag_bins`` and the
    signals treated as circular (see `pose2visual.xcorr.crosscorr`).

    Args:
        neural: neural activity ``[T, N]`` (already smoothed), finite.
        behavior: behavioral traces ``[T, B]`` on the same grid; a missing sample is NaN.
        max_lag_bins: largest lag in bins.

    Returns:
        Array ``[N, B, 2 * max_lag_bins + 1]``.

    Raises:
        ValueError: if the time axes differ, a neural column is constant, a behavior has fewer
            than `MIN_VALID_SAMPLES` valid samples or no variance over them, or ``T`` is too short
            for the lag window.
    """
    neural = np.asarray(neural, dtype=np.float64)
    behavior = np.asarray(behavior, dtype=np.float64)
    if neural.shape[0] != behavior.shape[0]:
        raise ValueError(
            f'neural has {neural.shape[0]} time bins but behavior has {behavior.shape[0]}',
        )
    n_time = neural.shape[0]
    idx_lag = lag_indices(max_lag_bins, n_time)
    # z-scoring over the whole recording does not change a Pearson correlation; it only keeps
    # the sums of squares below well conditioned
    neural_z = zscore(neural)
    fft_neural = np.fft.rfft(neural_z, axis=0)
    fft_neural_sq = np.fft.rfft(neural_z ** 2, axis=0)
    out = np.empty((neural.shape[1], behavior.shape[1], idx_lag.size))
    for idx_b in range(behavior.shape[1]):
        trace = behavior[:, idx_b]
        valid = np.isfinite(trace)
        n_valid = int(valid.sum())
        if n_valid < MIN_VALID_SAMPLES:
            raise ValueError(
                f'behavior column {idx_b} has {n_valid} valid samples '
                f'(need at least {MIN_VALID_SAMPLES})',
            )
        if constant_columns(trace[valid][:, None])[0]:
            raise ValueError(f'behavior column {idx_b} is constant over its valid samples')
        mask = valid.astype(np.float64)
        centered = np.where(valid, trace - trace[valid].mean(), 0.0)
        std_behavior = np.sqrt(np.sum(centered ** 2) / n_valid)
        # sums over t of neural(t) * g(t + k), for g = mask, centered behavior; the neural sum of
        # squares pairs with the mask. full_circular_xcorr divides by T, hence the factor T.
        fft_mask = np.fft.rfft(mask)
        s_n = full_circular_xcorr(fft_neural, fft_mask, n_time)[:, idx_lag] * n_time
        s_nn = full_circular_xcorr(fft_neural_sq, fft_mask, n_time)[:, idx_lag] * n_time
        s_nb = full_circular_xcorr(fft_neural, np.fft.rfft(centered), n_time)[:, idx_lag] * n_time
        mean_neural = s_n / n_valid
        var_neural = s_nn / n_valid - mean_neural ** 2
        cov = s_nb / n_valid
        with np.errstate(divide='ignore', invalid='ignore'):
            out[:, idx_b, :] = cov / (np.sqrt(var_neural) * std_behavior)
    return out
