"""Diagnostic and summary figures of the optional xcorr experiments.

Experiment B (``pca``): the explained-variance bars of the face group's kept components, the
variance removed by the PCA denoising, original against pca-denoised peak correlations, and the
table of PCA-applied and passthrough groups. Experiment C (``relative_motion``): the variance
removed by the centroid subtraction, original against relative peak correlations, the centroid
trajectories and the table of corrected and skipped groups. Like ``figures``, they build
``matplotlib.figure.Figure`` objects directly (no pyplot backend) and the caller saves them.
"""

import zlib
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

TEXT_COLOR = '0.15'
ORIGINAL_COLOR = '0.35'
PRIMARY_COLOR = '#0072B2'
RESIDUAL_COLOR = '#D55E00'
# Okabe-Ito, colour-blind safe
GROUP_PALETTE = (
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000',
)
# one fixed colour per anatomical group (`body_parts.json`), so a group looks the same in every
# camera, session and experiment
GROUP_COLORS = {
    'eye_related': '#0072B2',
    'ear_related': '#D55E00',
    'mouth_related': '#009E73',
    'nose_pad_related': '#CC79A7',
    'forelimb': '#E69F00',
    'tongue': '#56B4E9',
    'digits': '#000000',
}
# one fixed colour per anatomical feature (a keypoint without its `_left` / `_right` suffix), so a
# keypoint looks the same in every camera, session, run and experiment
KEYPOINT_COLORS = {
    'eye_back': '#1f77b4', 'eye_bottom': '#aec7e8', 'eye_front': '#17becf',
    'eye_top': '#9edae5', 'pupil_center': '#08306b',
    'ear_top': '#d62728', 'ear_tip': '#ff9896', 'ear_bottom': '#8c1c13',
    'ear_base': '#e6550d',
    'mouth': '#2ca02c', 'upperlip': '#98df8a', 'lowerlip': '#006d2c',
    'tongue_visible': '#66c2a4',
    'nose_top': '#e377c2', 'nose_bottom': '#f7b6d2', 'pad_center': '#7b3294',
    'pad_top': '#c2a5cf', 'pad_side': '#9467bd',
    'wrist': '#ff7f0e', 'd1_tip': '#8c564b', 'd2_tip': '#bcbd22', 'd3_tip': '#7f7f7f',
    'd4_tip': '#000000',
}
KEYPOINT_FALLBACK_PALETTE = (
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666',
)
UNCORRECTED_COLOR = '0.8'
EXCERPT_SEC = 20.0


def group_color(group: str) -> str:
    """Colour of an anatomical group, the same in every figure.

    Args:
        group: group name.

    Returns:
        The fixed colour of a known group; any other name gets a palette colour chosen by a stable
        hash of the name (not by the order the groups appear in).
    """
    if group in GROUP_COLORS:
        return GROUP_COLORS[group]
    return GROUP_PALETTE[zlib.crc32(group.encode()) % len(GROUP_PALETTE)]


def plot_pca_explained_variance(entries: list[dict[str, Any]]) -> Figure:
    """Explained-variance ratio of the face group's kept components, one bar panel per camera.

    Args:
        entries: one dict per panel with `camera` and `explained_variance_ratio` (list of
            floats, one per kept component, in component order), and optionally `title` (default
            `'<camera> camera'`) and `label` (y-axis label, default the face group's).

    Returns:
        The figure: one panel per camera, a bar per component.
    """
    fig = Figure(figsize=(3.2 * len(entries), 3.0), layout='constrained')
    axes = fig.subplots(1, len(entries), squeeze=False)[0]
    for ax, entry in zip(axes, entries):
        ratio = 100 * np.asarray(entry['explained_variance_ratio'])
        pos = np.arange(1, ratio.size + 1)
        ax.bar(pos, ratio, color=ORIGINAL_COLOR)
        ax.set_xticks(pos)
        ax.set_xlabel('component', fontsize=8)
        ax.set_ylabel(entry.get('label', '% of face variance explained'), fontsize=8)
        title = entry.get('title', f'{entry["camera"]} camera')
        ax.set_title(title, fontsize=9, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
    return fig


def plot_variance_bars(summary: pd.DataFrame) -> Figure:
    """Variance of each original and relative movement signal.

    Args:
        summary: one row per camera-keypoint with `camera`, `behavior`, `group`, `corrected`,
            `var_position_original`, `var_position_relative`, `var_speed_original` and
            `var_speed_relative`.

    Returns:
        The figure: one column per camera, position variance (x plus y, px^2) on top and speed
        variance (px^2 / s^2) below, original against relative, logarithmic axis. A star marks
        a keypoint whose group was centroid-corrected; the others are identical by construction.
    """
    cams = list(dict.fromkeys(summary['camera']))
    heights = [max(3.0, 0.22 * int((summary['camera'] == c).sum())) for c in cams]
    fig = Figure(figsize=(7.5 * len(cams), 2 * max(heights) + 1.5), layout='constrained')
    axes = fig.subplots(2, len(cams), squeeze=False)
    for idx_c, camera in enumerate(cams):
        sub = summary[summary['camera'] == camera].reset_index(drop=True)
        pos = np.arange(len(sub))
        labels = [
            f'{name}{" *" if flag else ""}'
            for name, flag in zip(sub['behavior'], sub['corrected'])
        ]
        for idx_r, (kind, unit) in enumerate((('position', 'px^2'), ('speed', 'px^2 / s^2'))):
            ax = axes[idx_r, idx_c]
            ax.barh(pos - 0.2, sub[f'var_{kind}_original'], height=0.38, color=ORIGINAL_COLOR,
                    label='original (absolute)')
            ax.barh(pos + 0.2, sub[f'var_{kind}_relative'], height=0.38, color=RESIDUAL_COLOR,
                    label='corrected (centroid removed, or PCA-denoised)')
            ax.set_xscale('log')
            ax.set_yticks(pos, labels, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel(f'variance of {kind} ({unit})', fontsize=7)
            ax.tick_params(axis='x', labelsize=7)
            ax.set_title(f'{camera} camera: {kind}', fontsize=8, color=TEXT_COLOR)
        axes[0, idx_c].legend(fontsize=6, frameon=False)
    fig.suptitle(
        'variance of the movement signals (* = a correction was applied, e.g. a group centroid '
        'removed or a PCA denoising; unstarred keypoints keep their original signal)',
        fontsize=8, color=TEXT_COLOR,
    )
    return fig


def keypoint_feature(name: str) -> str:
    """Anatomical feature of a keypoint: its name without the side suffix.

    Args:
        name: keypoint name, e.g. `'eye_top_left'`.

    Returns:
        The name without a trailing `_left` / `_right`, so the two sides (and the two cameras)
        of one feature share a colour, e.g. `'eye_top'`.
    """
    for side in ('_left', '_right'):
        if name.endswith(side):
            return name[:-len(side)]
    return name


def keypoint_color(name: str) -> str:
    """Colour of a keypoint, the same in every figure, camera, session and run.

    Args:
        name: keypoint name.

    Returns:
        The `KEYPOINT_COLORS` entry of its feature (`keypoint_feature`); an unknown feature gets a
        palette colour chosen by a stable hash (`zlib.crc32`, never the per-process `hash()`).
    """
    feature = keypoint_feature(name)
    if feature in KEYPOINT_COLORS:
        return KEYPOINT_COLORS[feature]
    return KEYPOINT_FALLBACK_PALETTE[
        zlib.crc32(feature.encode()) % len(KEYPOINT_FALLBACK_PALETTE)
    ]


def _scatter_groups(panels: list[dict[str, Any]]) -> list[str]:
    """Anatomical groups with a corrected keypoint in any panel, `GROUP_COLORS` order first.

    Args:
        panels: scatter panels (`plot_peak_scatter`).

    Returns:
        The group names, one scatter column each.
    """
    present = {
        group for panel in panels
        for group, flag in zip(panel['groups'], panel['corrected']) if flag
    }
    known = [group for group in GROUP_COLORS if group in present]
    return known + sorted(present - set(known))


def _draw_keypoint_dots(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    group: str,
    names: list[str] | None,
    legend: dict[str, str],
) -> None:
    """Scatter the dots of one panel, one colour per keypoint (per group without names).

    Args:
        ax: axes of the panel.
        x: values on the x axis ``[N, K]`` (one column per keypoint of the panel).
        y: values on the y axis ``[N, K]``.
        group: anatomical group of the panel.
        names: the ``K`` keypoint names, or `None` to colour by `group_color(group)`.
        legend: feature -> colour of the figure legend; extended with this panel's keypoints.
    """
    for idx in range(x.shape[1]):
        if names is None:
            color = group_color(group)
        else:
            color = keypoint_color(names[idx])
            legend.setdefault(keypoint_feature(names[idx]), color)
        ax.scatter(x[:, idx], y[:, idx], s=5, alpha=0.6, linewidths=0, color=color)


def _add_keypoint_legend(fig: Figure, legend: dict[str, str]) -> None:
    """One legend for the whole figure, right of the panels.

    Args:
        fig: the scatter figure.
        legend: feature -> colour of the keypoints drawn.
    """
    if not legend:
        return
    handles = [
        Line2D([], [], marker='o', ls='', markersize=4, color=color, label=feature)
        for feature, color in legend.items()
    ]
    fig.legend(
        handles=handles, loc='outside right upper', fontsize=7, frameon=False,
        title='keypoint (both sides)', title_fontsize=7,
    )


def plot_peak_scatter(panels: list[dict[str, Any]]) -> Figure:
    """Peak absolute xcorr of every neuron-keypoint pair, original against corrected speed.

    Args:
        panels: one dict per camera with `camera`, `original` and `relative` (``[N, B]`` peak
            absolute correlations over lags), `groups` (``B`` anatomical group names) and
            `corrected` (bool ``[B]``), and optionally `names` (``B`` keypoint names; with them
            the dots are coloured per keypoint and a figure legend is added, without them per
            group). Keypoints without a correction (e.g. a centroid subtraction or a PCA
            denoising) lie on the diagonal by construction and are not drawn.

    Returns:
        One figure with a row per camera and a column per anatomical group that has a corrected
        keypoint in any camera (`GROUP_COLORS` order first). Every panel has the identity line
        and the same square axes; points are coloured with `keypoint_color` (or `group_color`
        without names), one figure legend; a camera without that group leaves its cell empty.
        The axes say that the peak is of |r| (the sign is ignored and the peaks of the two
        conditions may be at different lags).
    """
    groups = _scatter_groups(panels)
    n_rows, n_cols = len(panels), max(1, len(groups))
    fig = Figure(figsize=(3.1 * n_cols + 1.6, 3.2 * n_rows + 0.4), layout='constrained')
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    top = 0.0
    for panel in panels:
        for idx_b, flag in enumerate(panel['corrected']):
            if flag:
                top = max(
                    top, float(np.nanmax(panel['original'][:, idx_b])),
                    float(np.nanmax(panel['relative'][:, idx_b])),
                )
    top = top or 1.0
    legend: dict[str, str] = {}
    for idx_r, panel in enumerate(panels):
        for idx_c, ax in enumerate(axes[idx_r]):
            group = groups[idx_c] if idx_c < len(groups) else None
            idx_b = [
                i for i, (g, flag) in enumerate(zip(panel['groups'], panel['corrected']))
                if flag and g == group
            ]
            if not idx_b:
                ax.axis('off')
                continue
            names = panel.get('names')
            _draw_keypoint_dots(
                ax, panel['original'][:, idx_b], panel['relative'][:, idx_b], group,
                None if names is None else [names[i] for i in idx_b], legend,
            )
            ax.plot([0, top], [0, top], color='k', lw=0.6, ls='--')
            ax.set_xlim(0, top * 1.03)
            ax.set_ylim(0, top * 1.03)
            ax.set_aspect('equal')
            ax.set_title(
                f'{panel["camera"]} camera: {group} ({len(idx_b)} keypoints)', fontsize=8,
                color=TEXT_COLOR,
            )
            ax.tick_params(labelsize=7)
            if idx_r == n_rows - 1 or not axes[idx_r + 1, idx_c].axison:
                ax.set_xlabel('peak |r| over lags, original speed', fontsize=7)
            if idx_c == 0 or not axes[idx_r, idx_c - 1].axison:
                ax.set_ylabel('peak |r| over lags, corrected speed', fontsize=7)
    fig.suptitle(
        'peak |r|: the sign is ignored and the two peaks may be at different lags',
        fontsize=8, color=TEXT_COLOR,
    )
    _add_keypoint_legend(fig, legend)
    return fig


def signed_r_at_original_peak(original: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, ...]:
    """Signed correlations of both conditions at the lag of the original's peak |r|.

    Args:
        original: correlations of the original speed ``[N, K, L]`` (NaN where undefined).
        other: correlations of the corrected speed, same shape.

    Returns:
        `(x, y)`, each ``[N, K]``: `original` and `other` at, for every neuron-keypoint pair,
        the lag where `|original|` is largest (NaN for a pair without any finite value). The lag
        is selected from the original only and reused for the other condition.
    """
    magnitude = np.where(np.isfinite(original), np.abs(original), -1.0)
    idx_lag = np.argmax(magnitude, axis=-1)[..., None]
    x = np.take_along_axis(original, idx_lag, axis=-1)[..., 0]
    y = np.take_along_axis(other, idx_lag, axis=-1)[..., 0]
    undefined = magnitude.max(axis=-1) < 0
    return np.where(undefined, np.nan, x), np.where(undefined, np.nan, y)


def plot_signed_scatter(panels: list[dict[str, Any]]) -> Figure:
    """Signed r of original and corrected speed at the lag of the original's peak |r|.

    Exploratory: the lag is picked as an extreme of the original data (`signed_r_at_original_peak`),
    so the original value is biased away from zero and the plot is a descriptive comparison, not an
    independent test and not a significance test.

    Args:
        panels: one dict per camera with `camera`, `original` and `relative` (signed
            ``[N, B, L]`` correlations), `groups` (``B`` anatomical group names), `corrected`
            (bool ``[B]``) and optionally `names` (``B`` keypoint names, see
            `plot_peak_scatter`).

    Returns:
        The figure, laid out like `plot_peak_scatter` (row per camera, column per corrected
        group, one legend) with symmetric axes, the identity line and the zero lines. A dot above
        the diagonal gained correlation at the matched lag, a dot across the zero lines reversed
        its sign.
    """
    groups = _scatter_groups(panels)
    n_rows, n_cols = len(panels), max(1, len(groups))
    fig = Figure(figsize=(3.1 * n_cols + 1.6, 3.2 * n_rows + 0.6), layout='constrained')
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    paired = []
    top = 0.0
    for panel in panels:
        x, y = signed_r_at_original_peak(
            np.asarray(panel['original']), np.asarray(panel['relative']),
        )
        paired.append((x, y))
        for idx_b, flag in enumerate(panel['corrected']):
            if flag:
                top = max(
                    top, float(np.nanmax(np.abs(x[:, idx_b]), initial=0.0)),
                    float(np.nanmax(np.abs(y[:, idx_b]), initial=0.0)),
                )
    top = top or 1.0
    legend: dict[str, str] = {}
    for idx_r, panel in enumerate(panels):
        x, y = paired[idx_r]
        for idx_c, ax in enumerate(axes[idx_r]):
            group = groups[idx_c] if idx_c < len(groups) else None
            idx_b = [
                i for i, (g, flag) in enumerate(zip(panel['groups'], panel['corrected']))
                if flag and g == group
            ]
            if not idx_b:
                ax.axis('off')
                continue
            names = panel.get('names')
            _draw_keypoint_dots(
                ax, x[:, idx_b], y[:, idx_b], group,
                None if names is None else [names[i] for i in idx_b], legend,
            )
            ax.plot([-top, top], [-top, top], color='k', lw=0.6, ls='--')
            ax.axhline(0.0, color='0.6', lw=0.4)
            ax.axvline(0.0, color='0.6', lw=0.4)
            ax.set_xlim(-top * 1.03, top * 1.03)
            ax.set_ylim(-top * 1.03, top * 1.03)
            ax.set_aspect('equal')
            ax.set_title(
                f'{panel["camera"]} camera: {group} ({len(idx_b)} keypoints)', fontsize=8,
                color=TEXT_COLOR,
            )
            ax.tick_params(labelsize=7)
            if idx_r == n_rows - 1 or not axes[idx_r + 1, idx_c].axison:
                ax.set_xlabel('r, original speed, at the lag of its peak |r|', fontsize=7)
            if idx_c == 0 or not axes[idx_r, idx_c - 1].axison:
                ax.set_ylabel('r, corrected speed, same lag', fontsize=7)
    fig.suptitle(
        'exploratory: the lag is chosen as the extreme of the original |r| and reused for the '
        'corrected speed; descriptive only, not a significance test',
        fontsize=8, color=TEXT_COLOR,
    )
    _add_keypoint_legend(fig, legend)
    return fig


def plot_centroid_trajectories(centroids: list[dict[str, Any]]) -> Figure:
    """Centroid trajectories of the corrected groups.

    Args:
        centroids: one dict per camera-group with `camera`, `group`, `times` (``[F]`` seconds),
            `x`, `y` (centroid position, pixels), `speed` (centroid speed), `member_speed` (mean
            absolute speed of the members) and `valid` (bool ``[F]``, relative motion defined).

    Returns:
        The figure: one row per camera-group. Left: centroid x and y over an excerpt of
        `EXCERPT_SEC` seconds from the middle of the recording (invalid stretches shaded);
        middle: speed of the centroid against the mean member speed, i.e. how much of the
        motion is shared; right: x and y over the whole recording.
    """
    fig = Figure(figsize=(16, 2.4 * len(centroids) + 0.8), layout='constrained')
    axes = fig.subplots(len(centroids), 3, squeeze=False)
    for idx, info in enumerate(centroids):
        times = np.asarray(info['times'])
        start = times[0] + 0.5 * (times[-1] - times[0] - EXCERPT_SEC)
        sel = (times >= start) & (times <= start + EXCERPT_SEC)
        ax_pos, ax_speed, ax_all = axes[idx]
        label = f'{info["camera"]} {info["group"]}'
        for ax, mask in ((ax_pos, sel), (ax_all, np.ones_like(sel))):
            ax.plot(
                times[mask], np.asarray(info['x'])[mask], color=PRIMARY_COLOR, lw=0.8, label='x',
            )
            ax.plot(times[mask], np.asarray(info['y'])[mask], color=RESIDUAL_COLOR, lw=0.8,
                    label='y')
            ax.set_ylabel('centroid (px)', fontsize=7)
            ax.tick_params(labelsize=7)
        invalid = ~np.asarray(info['valid'])
        ax_pos.fill_between(
            times[sel], 0, 1, where=invalid[sel], color='0.8', alpha=0.7,
            transform=ax_pos.get_xaxis_transform(), label='relative motion undefined',
        )
        ax_pos.set_title(f'{label}: centroid position, {EXCERPT_SEC:.0f} s excerpt', fontsize=8)
        ax_all.set_title(f'{label}: whole recording', fontsize=8)
        ax_speed.plot(times[sel], np.asarray(info['member_speed'])[sel], color=ORIGINAL_COLOR,
                      lw=0.8, label='mean absolute speed of the members')
        ax_speed.plot(times[sel], np.asarray(info['speed'])[sel], color=RESIDUAL_COLOR, lw=0.8,
                      label='speed of the centroid')
        ax_speed.set_ylabel('speed (px/s)', fontsize=7)
        ax_speed.set_title(f'{label}: shared translation', fontsize=8)
        ax_speed.tick_params(labelsize=7)
        if idx == 0:
            ax_pos.legend(fontsize=6, frameon=False)
            ax_speed.legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('time (s)', fontsize=7)
    return fig


def plot_group_table(rows: list[dict[str, Any]]) -> Figure:
    """Which groups were centroid-corrected and which kept their original speed.

    Args:
        rows: one dict per camera-group with `camera`, `group`, `members` (list of names),
            `corrected` (bool), `reason` and `valid_fraction`.

    Returns:
        The figure: a table in the order of `rows`.
    """
    fig = Figure(figsize=(14, 0.45 * len(rows) + 1.2), layout='constrained')
    ax = fig.add_subplot()
    ax.axis('off')
    cells = [
        [
            row['camera'], row['group'], ', '.join(row['members']),
            'centroid subtracted' if row['corrected'] else 'original speed kept', row['reason'],
            '-' if row['valid_fraction'] is None else f'{100 * row["valid_fraction"]:.1f} %',
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=cells,
        colLabels=['camera', 'group', 'keypoints', 'treatment', 'reason', 'frames defined'],
        loc='upper center', cellLoc='left',
        colWidths=[0.06, 0.12, 0.38, 0.14, 0.22, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    return fig
