"""Circular-shift permutation test for the peak of a neural-behavior cross-correlation.

The statistic of a neuron-behavior pair is ``max_tau |r(tau)|`` over the lag window, which
handles the multiplicity of the lags. The null shifts the behavior circularly against the
neural activity, which keeps the autocorrelation of both signals. A shift changes only which
lags of the circular cross-correlation (``crosscorr.full_circular_xcorr``) are read, so one FFT
product per behavior serves the observed correlation and all permutations. The same shift is
used for every neuron and behavior in a permutation, which keeps their dependence intact.
Correction for the number of neuron-behavior pairs is done on the p values
(``correct_pvalues``).

The shifted windows are strongly dependent when a signal is slow: the number of independent null
draws is about ``T / tau`` for an autocorrelation time ``tau``, however many permutations are run.
The p values are only as fine as that number, so the test needs recordings that span many
autocorrelation times (a few hundred seconds or more for behaviors that vary over seconds); for a
short or strongly periodic series the tail of the null is coarse and p values are liberal.
"""

import numpy as np

from pose2visual.xcorr.crosscorr import full_circular_xcorr, lag_indices, zscore


def draw_shifts(
    n_time: int,
    n_permutations: int,
    min_shift_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw circular shifts uniformly from ``[min_shift_bins, n_time - min_shift_bins]``.

    Args:
        n_time: number of time samples ``T``.
        n_permutations: number of shifts.
        min_shift_bins: smallest distance of a shift from zero (and from ``T``), in bins.
        rng: random generator.

    Returns:
        Integer array ``[n_permutations]``.

    Raises:
        ValueError: if the range of shifts is empty.
    """
    if n_time <= 2 * min_shift_bins:
        raise ValueError(f'{n_time} samples are too few for a minimum shift of {min_shift_bins}')
    return rng.integers(min_shift_bins, n_time - min_shift_bins + 1, size=n_permutations)


def observed_and_null(
    neural: np.ndarray,
    behavior: np.ndarray,
    max_lag_bins: int,
    shifts: np.ndarray,
    chunk: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation of every pair and the null distribution of its peak.

    Args:
        neural: neural activity ``[T, N]``.
        behavior: behavioral traces ``[T, B]`` on the same time grid.
        max_lag_bins: largest lag in bins.
        shifts: circular shifts in bins, from `draw_shifts`; each must exceed `max_lag_bins`
            and leave ``T - shift > max_lag_bins`` so the null window never overlaps the
            observed one.
        chunk: permutations handled at once, bounds memory to ``N * chunk * L`` floats.

    Returns:
        Tuple `(r, null_max)`: correlations ``[N, B, L]`` (identical to `cross_correlation`) and
        ``null_max[n, b, p] = max_tau |r_p(tau)|`` of permutation ``p``, shape
        ``[N, B, len(shifts)]``.

    Raises:
        ValueError: if the time axes differ, the lag window does not fit in ``T`` or a shift
            lies inside the lag window.
    """
    neural = np.asarray(neural)
    behavior = np.asarray(behavior)
    if neural.shape[0] != behavior.shape[0]:
        raise ValueError(
            f'neural has {neural.shape[0]} time bins but behavior has {behavior.shape[0]}',
        )
    n_time = neural.shape[0]
    idx_lag = lag_indices(max_lag_bins, n_time)
    shifts = np.asarray(shifts, dtype=np.int64)
    if shifts.size and (shifts.min() <= max_lag_bins or shifts.max() >= n_time - max_lag_bins):
        raise ValueError('every shift must lie more than max_lag_bins away from 0 and T')
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    fft_neural = np.fft.rfft(zscore(neural), axis=0)
    fft_behavior = np.fft.rfft(zscore(behavior), axis=0)
    r = np.empty((neural.shape[1], behavior.shape[1], idx_lag.size))
    null_max = np.empty((neural.shape[1], behavior.shape[1], shifts.size))
    for idx_b in range(behavior.shape[1]):
        full = full_circular_xcorr(fft_neural, fft_behavior[:, idx_b], n_time)
        r[:, idx_b, :] = full[:, idx_lag]
        for start in range(0, shifts.size, chunk):
            idx_window = (shifts[start:start + chunk, None] + lags[None, :]) % n_time
            window = full[:, idx_window]  # [N, chunk, L]
            null_max[:, idx_b, start:start + chunk] = np.abs(window).max(axis=2)
    return r, null_max


def permutation_pvalues(r: np.ndarray, null_max: np.ndarray) -> np.ndarray:
    """P value of the peak absolute correlation of every pair against its shift null.

    Args:
        r: correlations ``[N, B, L]``.
        null_max: null peaks ``[N, B, P]`` from `observed_and_null`.

    Returns:
        Array ``[N, B]`` of ``(1 + #(null >= observed)) / (1 + P)``, never zero.
    """
    observed = np.abs(r).max(axis=2)
    n_perm = null_max.shape[2]
    n_exceed = (null_max >= observed[..., None]).sum(axis=2)
    return (1 + n_exceed) / (1 + n_perm)


def correct_pvalues(
    p: np.ndarray,
    method: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct p values for multiple comparisons across all pairs.

    Args:
        p: p values of any shape; all entries form one family.
        method: `'fdr_bh'` (Benjamini-Hochberg) or `'bonferroni'`.
        alpha: significance level.

    Returns:
        Tuple `(q, significant)` of adjusted p values (BH q values or Bonferroni-adjusted p) and
        the boolean mask `q <= alpha`, both shaped like `p`.

    Raises:
        ValueError: if `method` is unknown.
    """
    p = np.asarray(p, dtype=np.float64)
    flat = p.ravel()
    n = flat.size
    if method == 'bonferroni':
        q = np.minimum(flat * n, 1.0)
    elif method == 'fdr_bh':
        order = np.argsort(flat)
        scaled = flat[order] * n / np.arange(1, n + 1)
        # step-up: each q is the smallest scaled p of all larger ranks
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
        q = np.empty(n)
        q[order] = np.minimum(adjusted, 1.0)
    else:
        raise ValueError(f"unknown correction {method!r}; use 'fdr_bh' or 'bonferroni'")
    return q.reshape(p.shape), (q <= alpha).reshape(p.shape)
