"""Checks on the Lightning Pose keypoint speed features before the encoders.

``dynamic_vars.npz`` (``extract_litpose_dynamic_vars_trials``) already masks NaN and
low-likelihood frames, interpolates over them and, with ``pca: true``, reprojects the keypoint
positions of each body-part group onto its top principal components (``litpose_pca``), so the
binned features are finite and nothing is imputed or rotated here; this module only verifies
that (`check_finite`).

Used by ``run_litpose_encoding`` (step C).
"""

import numpy as np


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
