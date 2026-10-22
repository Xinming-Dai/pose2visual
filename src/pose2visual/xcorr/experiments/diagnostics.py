"""Diagnostic and summary figures of the optional xcorr experiments.

Experiment A (``rank1``): the singular-value spectrum, the loadings of the leading component, its
neuron x lag pattern and the example curves of the residual. Experiment B (``relative_motion``):
the variance removed by the centroid subtraction, original against relative peak correlations,
the centroid trajectories and the table of corrected and skipped groups. Like ``figures``, they
build ``matplotlib.figure.Figure`` objects directly (no pyplot backend) and the caller saves them.
"""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

TEXT_COLOR = '0.15'
ORIGINAL_COLOR = '0.35'
RANK1_COLOR = '#0072B2'
RESIDUAL_COLOR = '#D55E00'
# Okabe-Ito, colour-blind safe; groups beyond the eighth reuse colours
GROUP_PALETTE = (
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000',
)
EXCERPT_SEC = 20.0


def group_colors(groups: list[str]) -> dict[str, str]:
    """Assign one colour to every group, in order of first appearance.

    Args:
        groups: group names, possibly repeated.

    Returns:
        Group name -> colour.
    """
    unique = list(dict.fromkeys(groups))
    return {g: GROUP_PALETTE[idx % len(GROUP_PALETTE)] for idx, g in enumerate(unique)}


def plot_rank1_diagnostics(
    cameras: list[dict[str, Any]],
    lags_sec: np.ndarray,
    title: str,
) -> Figure:
    """Spectrum, loadings, shared pattern and per-behavior explained fraction of rank 1.

    One row per camera, four panels: the leading squared singular values as a fraction of the
    squared Frobenius norm; the loadings ``v_0`` of the behaviors (coloured by group); the shared
    neuron x lag pattern ``u_0`` (neurons in display order); and the fraction of each behavior's
    own squared norm that rank 1 explains.

    Args:
        cameras: one dict per camera with `camera`, `singular_fractions`, `names`, `groups`,
            `loadings`, `pattern` (``[N, L]``, neurons in display order) and `frac_behavior`.
        lags_sec: lag in seconds of each pattern column.
        title: figure title.

    Returns:
        The figure.
    """
    n_rows = len(cameras)
    height_rows = [max(3.2, 0.2 * len(c['names'])) for c in cameras]
    fig = Figure(figsize=(17, sum(height_rows) + 1.2), layout='constrained')
    grid = fig.add_gridspec(n_rows, 4, height_ratios=height_rows, width_ratios=[1, 1.3, 1.3, 1])
    for idx_row, cam in enumerate(cameras):
        colors = group_colors(cam['groups'])
        ax = fig.add_subplot(grid[idx_row, 0])
        fractions = np.asarray(cam['singular_fractions'])
        ax.bar(np.arange(1, fractions.size + 1), 100 * fractions, color=ORIGINAL_COLOR)
        ax.set_xlabel('component', fontsize=7)
        ax.set_ylabel(f'{cam["camera"]} camera\n% of squared norm', fontsize=7)
        ax.set_title('singular values', fontsize=8, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
        pos = np.arange(len(cam['names']))
        ax = fig.add_subplot(grid[idx_row, 1])
        ax.barh(pos, cam['loadings'], color=[colors[g] for g in cam['groups']])
        ax.axvline(0.0, color='k', lw=0.5)
        ax.set_yticks(pos, cam['names'], fontsize=6)
        ax.invert_yaxis()
        ax.set_title('loadings v0 (colour = group)', fontsize=8, color=TEXT_COLOR)
        ax.tick_params(axis='x', labelsize=7)
        ax = fig.add_subplot(grid[idx_row, 2])
        pattern = np.asarray(cam['pattern'])
        limit = float(np.max(np.abs(pattern))) or 1.0
        image = ax.imshow(
            pattern, aspect='auto', cmap='bwr', vmin=-limit, vmax=limit, interpolation='nearest',
            extent=(float(lags_sec[0]), float(lags_sec[-1]), pattern.shape[0], 0),
        )
        ax.axvline(0.0, color='k', lw=0.4, ls=':')
        ax.set_xlabel('lag (s)', fontsize=7)
        ax.set_ylabel('neurons (display order)', fontsize=7)
        ax.set_title('shared pattern u0 (unit norm)', fontsize=8, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.8).ax.tick_params(labelsize=6)
        ax = fig.add_subplot(grid[idx_row, 3])
        ax.barh(pos, 100 * np.asarray(cam['frac_behavior']),
                color=[colors[g] for g in cam['groups']])
        ax.set_yticks(pos, cam['names'], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of the behavior\'s squared norm', fontsize=7)
        ax.set_title('explained by rank 1', fontsize=8, color=TEXT_COLOR)
        ax.tick_params(axis='x', labelsize=7)
    fig.suptitle(title, fontsize=9, color=TEXT_COLOR)
    return fig


def plot_residual_examples(
    examples: list[dict[str, Any]],
    lags_sec: np.ndarray,
    title: str,
    n_cols: int = 4,
) -> Figure:
    """Original, rank-1 and residual curves of individual neuron-behavior pairs.

    Args:
        examples: one dict per panel with `title` and the curves `original`, `rank1` and
            `residual`, each ``[L]``.
        lags_sec: lag in seconds of each curve entry.
        title: figure title.
        n_cols: panels per row.

    Returns:
        The figure.
    """
    n_cols = max(1, min(n_cols, len(examples)))
    n_rows = int(np.ceil(len(examples) / n_cols))
    fig = Figure(figsize=(3.4 * n_cols, 2.6 * n_rows), layout='constrained')
    axes = fig.subplots(n_rows, n_cols, sharex=True, squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, example in zip(axes.ravel(), examples):
        ax.set_visible(True)
        ax.axhline(0.0, color='k', lw=0.4)
        ax.axvline(0.0, color='k', lw=0.4, ls=':')
        ax.plot(lags_sec, example['original'], color=ORIGINAL_COLOR, lw=1.0, label='original')
        ax.plot(lags_sec, example['rank1'], color=RANK1_COLOR, lw=1.0, ls='--', label='rank 1')
        ax.plot(lags_sec, example['residual'], color=RESIDUAL_COLOR, lw=1.3, label='residual')
        ax.set_title(example['title'], fontsize=7, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=6, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('lag (s)', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('xcorr / SVD residual', fontsize=7)
    fig.suptitle(title, fontsize=9, color=TEXT_COLOR)
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
                    label='relative (group centroid removed)')
            ax.set_xscale('log')
            ax.set_yticks(pos, labels, fontsize=6)
            ax.invert_yaxis()
            ax.set_xlabel(f'variance of {kind} ({unit})', fontsize=7)
            ax.tick_params(axis='x', labelsize=7)
            ax.set_title(f'{camera} camera: {kind}', fontsize=8, color=TEXT_COLOR)
        axes[0, idx_c].legend(fontsize=6, frameon=False)
    fig.suptitle(
        'variance of the movement signals (* = group centroid subtracted; unstarred keypoints '
        'keep their original signal)', fontsize=8, color=TEXT_COLOR,
    )
    return fig


def plot_peak_scatter(panels: list[dict[str, Any]]) -> Figure:
    """Peak absolute xcorr of every neuron-keypoint pair, original against relative speed.

    Args:
        panels: one dict per camera with `camera`, `original` and `relative` (``[N, B]`` peak
            absolute correlations over lags), `groups` (``B`` names) and `corrected` (bool
            ``[B]``). Keypoints without centroid correction lie on the diagonal by
            construction and are drawn in grey.

    Returns:
        The figure: one panel per camera with the identity line.
    """
    fig = Figure(figsize=(5.6 * len(panels), 5.4), layout='constrained')
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, panel in zip(axes, panels):
        colors = group_colors(list(panel['groups']))
        top = 0.0
        for idx_b, group in enumerate(panel['groups']):
            corrected = bool(panel['corrected'][idx_b])
            x, y = panel['original'][:, idx_b], panel['relative'][:, idx_b]
            top = max(top, float(np.nanmax(x)), float(np.nanmax(y)))
            ax.scatter(
                x, y, s=5, alpha=0.5 if corrected else 0.2, linewidths=0,
                color=colors[group] if corrected else '0.6',
            )
        ax.plot([0, top], [0, top], color='k', lw=0.6, ls='--')
        ax.set_xlim(0, top * 1.03)
        ax.set_ylim(0, top * 1.03)
        ax.set_aspect('equal')
        ax.set_xlabel('peak |r|, original speed', fontsize=8)
        ax.set_ylabel('peak |r|, relative speed', fontsize=8)
        ax.set_title(f'{panel["camera"]} camera', fontsize=9, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
        handles = [
            ax.scatter([], [], s=12, color=color, label=group) for group, color in colors.items()
        ]
        handles.append(ax.scatter([], [], s=12, color='0.6', label='not corrected'))
        ax.legend(handles=handles, fontsize=6, frameon=False, loc='lower right')
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
            ax.plot(times[mask], np.asarray(info['x'])[mask], color=RANK1_COLOR, lw=0.8, label='x')
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
