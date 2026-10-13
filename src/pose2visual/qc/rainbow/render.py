"""Matplotlib rendering of the rainbow scatter rows and the combined per-camera figure."""

from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pose2visual.qc.rainbow.stats import compute_statistics


def render_group_row(
    axes_row: list[plt.Axes],
    xy_list: list[np.ndarray],
    confidences_list: list[np.ndarray],
    background_img: np.ndarray,
    landmarks: list[str],
    colors: list[str],
    camera: str,
    group_name: str,
    gated: bool,
    conf_threshold: float,
    method_names: list[str],
    roi: tuple[int, int, int, int] | None,
) -> tuple[list[dict[str, dict[str, Any]]], np.ndarray]:
    """Render one body-part group's row of panels (one per method).

    Args:
        axes_row: pre-created axes to draw into, one per entry in `method_names`
            (callers with fewer methods than the figure's column count pass only
            that many axes; the rest are left blank by the caller).
        xy_list: list of (n_frames, n_landmarks, 2) arrays, one per method.
        confidences_list: list of (n_frames, n_landmarks) arrays, one per seed (used
            for gating; does not include the ensemble median or EKS columns).
        background_img: background frame (H, W, 3).
        landmarks: landmark display names, matching `xy_list`'s landmark axis.
        colors: one colour per landmark, same order as `landmarks`.
        camera: camera name ('left' or 'right').
        group_name: body-part group name, shown in each panel's title.
        gated: whether to apply confidence gating.
        conf_threshold: confidence threshold for gating.
        method_names: names for each method (e.g. ['Seed 0', ..., 'Ensemble median']).
        roi: optional ROI (x0, y0, x1, y1) for statistics.

    Returns:
        Tuple of (stats_list, counts): `stats_list` is one per-method stats dict
        (`{landmark: stats}`), same order as `xy_list`; `counts` is the kept-frame count
        per landmark (shape `(n_landmarks,)`), identical across every method in this row
        since gating doesn't depend on the method — the caller uses it to build a legend
        instead of drawing per-panel text on the image.
    """
    n_frames = xy_list[0].shape[0]
    n_landmarks = len(landmarks)

    seeds_conf = np.stack(confidences_list, axis=0)  # (n_seeds, n_frames, n_landmarks)
    if gated:
        keep = (seeds_conf > conf_threshold).all(axis=0)
    else:
        keep = np.ones((n_frames, n_landmarks), dtype=bool)

    h, w = background_img.shape[:2]
    stats_list = []

    for m, (xy, method_name, ax) in enumerate(zip(xy_list, method_names, axes_row)):
        ax.imshow(background_img)

        xy_flat = xy.reshape(-1, 2)
        keep_flat = keep.ravel()
        pts = xy_flat[keep_flat]

        colors_array = np.tile(colors, n_frames)[keep_flat]

        order = np.random.default_rng(19).permutation(len(pts))
        ax.scatter(
            pts[order, 0],
            pts[order, 1],
            c=colors_array[order],
            s=1.4,
            alpha=0.18,
            linewidths=0,
        )

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis('off')
        ax.set_title(
            f'{camera.capitalize()} | {group_name} | {method_name}',
            color='white',
            fontsize=11,
        )

        ax.plot([12, 32], [h - 13, h - 13], color='white', lw=2)
        ax.text(12, h - 7, '20 px', color='white', fontsize=8)

        stats_list.append(compute_statistics(xy, keep, landmarks, roi))

    counts = keep.sum(axis=0)
    return stats_list, counts


def render_combined_figure(
    rows_data: list[dict[str, Any]],
    background_img: np.ndarray,
    session_id: str,
    camera: str,
    gated: bool,
    conf_threshold: float,
    roi: tuple[int, int, int, int] | None,
    n_frames: int,
) -> tuple[plt.Figure, dict[str, list[dict[str, dict[str, Any]]]], str, list[dict[str, Any]]]:
    """Render every body-part group's row, stacked vertically into one figure.

    Args:
        rows_data: one entry per group, each a dict with `group_name`, `landmarks`,
            `colors`, `xy_list`, `confidences_list`, `method_names` (see
            `render_group_row`).
        background_img: background frame (H, W, 3).
        session_id: session UUID.
        camera: camera name ('left' or 'right').
        gated: whether to apply confidence gating.
        conf_threshold: confidence threshold for gating.
        roi: optional ROI (x0, y0, x1, y1) for statistics.
        n_frames: number of frames plotted (for the title).

    Returns:
        Tuple of (figure, {group_name: stats_list}, title_text, rows_legend).
        `rows_legend` is one entry per row, in the same order as `rows_data`:
        `{'group_name', 'landmarks', 'colors', 'counts', 'n_frames'}`, for the caller to
        render as a legend canvas below the figure instead of drawing it on the image.
    """
    n_rows = len(rows_data)
    n_cols = max(len(row['method_names']) for row in rows_data)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        facecolor='#141414',
        squeeze=False,
    )

    stats_by_group = {}
    rows_legend = []
    for row_idx, row in enumerate(rows_data):
        n_methods = len(row['method_names'])
        axes_row = [axes[row_idx, col_idx] for col_idx in range(n_methods)]
        for col_idx in range(n_methods, n_cols):
            axes[row_idx, col_idx].axis('off')

        stats_list, counts = render_group_row(
            axes_row,
            row['xy_list'],
            row['confidences_list'],
            background_img,
            row['landmarks'],
            row['colors'],
            camera,
            row['group_name'],
            gated,
            conf_threshold,
            row['method_names'],
            roi,
        )
        stats_by_group[row['group_name']] = stats_list
        rows_legend.append({
            'group_name': row['group_name'],
            'landmarks': row['landmarks'],
            'colors': row['colors'],
            'counts': counts,
            'n_frames': n_frames,
        })

    gate_mode = (
        f'Matched frames per landmark: ALL seeds likelihood >{conf_threshold}' if gated
        else f'All {n_frames:,} frames; no confidence filtering'
    )
    group_names = ', '.join(row['group_name'] for row in rows_data)
    title_text = (
        f'Session {session_id[:8]} | {group_names}, frames 0–{n_frames - 1}\n{gate_mode}'
    )

    fig.subplots_adjust(top=0.96, bottom=0.03, left=0.015, right=0.985, hspace=0.22, wspace=0.055)

    return fig, stats_by_group, title_text, rows_legend
