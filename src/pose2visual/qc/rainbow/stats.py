"""Per-keypoint summary statistics of a prediction cloud."""

from typing import Any

import numpy as np


def compute_statistics(
    xy: np.ndarray,
    keep: np.ndarray,
    landmarks: list[str],
    roi: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    """Compute per-keypoint statistics.

    Args:
        xy: prediction array (n_frames, n_landmarks, 2).
        keep: boolean mask (n_frames, n_landmarks) indicating valid frames.
        landmarks: landmark names.
        roi: optional ROI (x0, y0, x1, y1) for coverage analysis.

    Returns:
        Dictionary {landmark_name: stats_dict}.
    """
    stats_dict = {}

    for k, landmark in enumerate(landmarks):
        z = xy[:, k, :][keep[:, k]]  # (n_kept, 2)

        if len(z) == 0:
            continue

        center = np.median(z, axis=0)
        radius = np.linalg.norm(z - center, axis=1)

        stat_entry = {
            'n': len(z),
            'median_x': float(center[0]),
            'median_y': float(center[1]),
            'radius_p95_px': float(np.quantile(radius, 0.95)),
        }

        if roi is not None:
            x0, y0, x1, y1 = roi
            outside = (z[:, 0] < x0) | (z[:, 0] >= x1) | (z[:, 1] < y0) | (
                z[:, 1] >= y1
            )
            stat_entry['outside_roi_count'] = int(outside.sum())
            stat_entry['outside_roi_fraction'] = float(outside.mean())

        stats_dict[landmark] = stat_entry

    return stats_dict
