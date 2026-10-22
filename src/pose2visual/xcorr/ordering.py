"""Neuron order shared by every panel of the cross-correlation heatmaps."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

ORDERS = ('rastermap', 'depth', 'none')


def rastermap_order(neural: np.ndarray) -> np.ndarray:
    """Order neurons with rastermap, fit once on the neural activity.

    Args:
        neural: smoothed neural activity ``[T, N]``.

    Returns:
        Permutation ``[N]`` of the neurons (`model.isort`), similar neurons next to each other.

    Raises:
        ImportError: if rastermap is not installed.
    """
    try:
        # lazy: optional dependency, only needed for this order
        from rastermap import Rastermap
    except ImportError as exc:
        raise ImportError(
            "rastermap is not installed; run `pip install rastermap` in the environment, or "
            "use `--order depth`",
        ) from exc
    n_neurons = neural.shape[1]
    spikes = np.asarray(neural, dtype=np.float32).T
    # z-score per neuron so low-rate units are not ignored; constant units were dropped earlier
    spikes = (spikes - spikes.mean(axis=1, keepdims=True)) / spikes.std(axis=1, keepdims=True)
    model = Rastermap(
        n_PCs=min(64, n_neurons - 1),
        n_clusters=100 if n_neurons > 200 else None,
        locality=0.75,
        time_lag_window=5,
        grid_upsample=10,
    ).fit(spikes)
    return np.asarray(model.isort, dtype=np.int64)


def neuron_order(
    method: str,
    neural: np.ndarray,
    depths: np.ndarray | None = None,
) -> np.ndarray:
    """Pick the neuron order used in every heatmap panel.

    Args:
        method: `'rastermap'`, `'depth'` (ascending probe depth) or `'none'` (file order).
        neural: smoothed neural activity ``[T, N]``.
        depths: unit depths ``[N]``, required for `'depth'`.

    Returns:
        Permutation ``[N]``.

    Raises:
        ValueError: if `method` is unknown or `'depth'` is used without depths.
    """
    if method == 'rastermap':
        return rastermap_order(neural)
    if method == 'depth':
        if depths is None:
            raise ValueError("order 'depth' needs the unit depths")
        return np.argsort(np.asarray(depths, dtype=np.float64), kind='stable')
    if method == 'none':
        return np.arange(neural.shape[1])
    raise ValueError(f'unknown order {method!r}; use one of {list(ORDERS)}')
