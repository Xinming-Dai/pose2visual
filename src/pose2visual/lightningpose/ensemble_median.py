"""Ensemble median across network seeds, shared by ``pose2visual.qc.rainbow`` and
``export_ensemble_median.py``.

The median here is a plain per-frame, per-landmark ``np.median`` across seeds, with no
likelihood gating or NaN filtering: a NaN in any one seed's (x, y) or likelihood
propagates into the output for that frame/landmark, exactly like
``pose2visual.qc.rainbow``'s original inline computation. This module only factors that
computation and its DLC-style CSV writer out into one place so both scripts agree.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def compute_ensemble_median(
    xy_seeds: np.ndarray,
    conf_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean/median over the ensemble members, independently for the 𝑥 and 𝑦 coordinates.

    Args:
        xy_seeds: seed coordinates, shape (n_seeds, n_frames, n_keypoints, 2).
        conf_seeds: seed likelihoods, shape (n_seeds, n_frames, n_keypoints).

    Returns:
        (xy_median, conf_median): shapes (n_frames, n_keypoints, 2) and
        (n_frames, n_keypoints). A NaN in any seed propagates to the output for that
        frame/landmark, since this is a plain `np.median`, not `np.nanmedian`.
    """
    xy_median = np.median(xy_seeds, axis=0)
    conf_median = np.median(conf_seeds, axis=0)
    return xy_median, conf_median


def write_ensemble_median_csv(
    xy_median: np.ndarray,
    conf_median: np.ndarray,
    keypoints: list[str],
    out_path: Path,
    scorer: str = 'ensemble_median',
) -> None:
    """Write the ensemble median to a DLC-style 3-level-header csv.

    Same `(scorer, bodyparts, coords)` structure as the raw seed prediction csvs, so
    the output is a drop-in for tools that expect that format (e.g. `pose2visual.qc.overlay`).

    Args:
        xy_median: (n_frames, n_keypoints, 2) median coordinates.
        conf_median: (n_frames, n_keypoints) median likelihoods.
        keypoints: keypoint name per `xy_median`'s second axis.
        out_path: output csv path.
        scorer: top header level value (default: `'ensemble_median'`).
    """
    n_frames = xy_median.shape[0]
    columns = pd.MultiIndex.from_product(
        [[scorer], keypoints, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = np.empty((n_frames, len(keypoints) * 3))
    data[:, 0::3] = xy_median[:, :, 0]
    data[:, 1::3] = xy_median[:, :, 1]
    data[:, 2::3] = conf_median
    df = pd.DataFrame(data, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
