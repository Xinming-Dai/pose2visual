"""Preprocessing of Lightning Pose keypoint speed features before the encoders.

``dynamic_vars.npz`` (``extract_litpose_dynamic_vars_trials``) already masks NaN and
low-likelihood frames and interpolates over them, so the binned features are finite; this module
only verifies that (`check_finite`) instead of imputing. It then z-scores each keypoint column and
rotates the columns onto principal components (`preprocess_splits`).

The z-score statistics and the PCA basis are fit once on the train, val and test features pooled
and applied to every split, so all splits live in the same coordinate system and the held-out test
metrics stay comparable. The PCA sees only keypoint speeds, never spikes, but the test features do
influence the basis through those unsupervised statistics.

Used by ``run_litpose_encoding`` (step C).
"""

import numpy as np
from sklearn.decomposition import PCA

_SPLITS = ('train', 'val', 'test')


def check_finite(z_trials_time: dict[str, np.ndarray], name: str) -> None:
    """Raise if any feature value is NaN or infinite.

    Args:
        z_trials_time: split name -> `[K, T, 1, D]` features.
        name: label of the feature set used in the error message (e.g. `'base'`).

    Raises:
        ValueError: if a split holds a non-finite value.
    """
    for split, z in z_trials_time.items():
        n_bad = int(np.count_nonzero(~np.isfinite(z)))
        if n_bad:
            raise ValueError(
                f'{name} {split} features contain {n_bad} non-finite values; '
                'step 1 should have interpolated over masked frames',
            )


def fit_zscore_pca(z_pooled: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    """Fit a z-score followed by PCA on pooled features.

    Args:
        z_pooled: array of shape `[K, T, 1, D]` holding the trials of all splits.
        n_components: number of principal components to keep.

    Returns:
        State dict with `mean` and `std` (`[D]`, zero std replaced by 1), `pca_mean` (`[D]`),
        `components` (`[n_components, D]`) and `explained_variance_ratio` (`[n_components]`).

    Raises:
        ValueError: if `n_components` is not in `[1, D]`.
    """
    n_features = z_pooled.shape[-1]
    if not 1 <= n_components <= n_features:
        raise ValueError(f'n_components must be in [1, {n_features}], got {n_components}')
    flat = z_pooled.reshape(-1, n_features).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # a constant keypoint would divide by zero; keep it at 0 after centering
    std[std == 0] = 1.0
    pca = PCA(n_components=n_components, svd_solver='full').fit((flat - mean) / std)
    return {
        'mean': mean,
        'std': std,
        'pca_mean': pca.mean_,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
    }


def apply_zscore_pca(z: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    """Project features onto the principal components of a fitted state.

    Args:
        z: array of shape `[K, T, 1, D]`; `K` may be 0.
        state: output of `fit_zscore_pca`.

    Returns:
        float32 array of shape `[K, T, 1, n_components]`.
    """
    n_features = z.shape[-1]
    flat = z.reshape(-1, n_features).astype(np.float64)
    scaled = (flat - state['mean']) / state['std'] - state['pca_mean']
    projected = scaled @ state['components'].T
    return projected.reshape(*z.shape[:-1], projected.shape[-1]).astype(np.float32)


def preprocess_splits(
    z_trials_time: dict[str, np.ndarray],
    n_components: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Z-score and PCA the features of all splits with one basis fit on the pooled splits.

    Args:
        z_trials_time: split name -> `[K, T, 1, D]` features, with `train`, `val` and `test`.
        n_components: number of principal components to keep.

    Returns:
        Tuple `(z_pca, info)`: split name -> `[K, T, 1, n_components]` features, and a
        json-serializable dict with `n_components`, `explained_variance_ratio`, `scaling` and
        `fit_split`.

    Raises:
        ValueError: if `n_components` is not in `[1, D]`.
    """
    pooled = np.concatenate([z_trials_time[split] for split in _SPLITS], axis=0)
    state = fit_zscore_pca(pooled, n_components)
    z_pca = {split: apply_zscore_pca(z_trials_time[split], state) for split in _SPLITS}
    info = {
        'n_components': int(n_components),
        'explained_variance_ratio': [float(v) for v in state['explained_variance_ratio']],
        'scaling': 'zscore_pooled',
        'fit_split': 'pooled',
    }
    return z_pca, info
