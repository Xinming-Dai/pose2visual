"""Figures of the cross-correlation: a Figure 20-style overview and example curves.

Built on ``matplotlib.figure.Figure`` directly (no pyplot backend), saved as png and pdf like the
showcase figures. A strip of the overview is one behavior of one camera: neurons (rows, one order
for every strip) by lag (columns), coloured by the signed correlation on a symmetric scale centred
at zero that is shared by all strips, so weak behaviors stay pale instead of being stretched.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

COLORMAP = 'bwr'
# fractional rate change of the PETH strips: orange = more than baseline, purple = less
PETH_COLORMAP = 'PuOr_r'
PETH_GROUP = 'events'
TEXT_COLOR = '0.15'
PANEL_WIDTH = 0.5
MIN_FIGURE_WIDTH = 6.0
GROUP_GAP = 0.45
COLORBAR_RATIO = 0.6
GROUP_ROW_HEIGHT = 0.3
NAME_ROW_HEIGHT = 1.1
ROW_HEIGHT = 4.5
MAX_FIGURE_WIDTH = 26.0


def heatmap_vmax(r: np.ndarray, percentile: float = 99.5, step: float = 0.05) -> float:
    """Half-range of the shared colour scale.

    Args:
        r: correlations of any shape.
        percentile: percentile of `|r|` the scale must cover.
        step: the scale is rounded up to a multiple of this.

    Returns:
        A positive symmetric limit, at least `step`.
    """
    limit = float(np.nanpercentile(np.abs(r), percentile))
    return max(step, float(np.ceil(limit / step) * step))


def save_figure(fig: Figure, stem: Path, dpi: int = 200) -> list[Path]:
    """Save a figure as png and pdf.

    Args:
        fig: the figure.
        stem: output path without suffix; its folder is created.
        dpi: resolution of the png.

    Returns:
        The written paths, `<stem>.png` then `<stem>.pdf`.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    # a dot in the stem (the session id is separated from other parts by dots) is not a suffix
    paths = [stem.parent / f'{stem.name}.png', stem.parent / f'{stem.name}.pdf']
    for path in paths:
        fig.savefig(path, dpi=dpi)
    return paths


@dataclass
class Slot:
    """One column of the overview: a body part, with the behavior each camera shows for it.

    Attributes:
        group: body-part group of the column.
        label: column label (the merged name of a mirrored pair, else the behavior name).
        members: camera index -> behavior index drawn in that camera's row; a camera without an
            entry leaves the cell blank.
    """

    group: str
    label: str
    members: dict[int, int]


def _mirror(name: str) -> tuple[str, str] | None:
    """Name of the other body side and the side-less base of a `_left` / `_right` name.

    Args:
        name: behavior name.

    Returns:
        `(mirrored_name, base)`, e.g. `('eye_back_right', 'eye_back')` for `'eye_back_left'`, or
        `None` when the name does not end in a side.
    """
    for side, other in (('_left', '_right'), ('_right', '_left')):
        if name.endswith(side):
            base = name[:-len(side)]
            return base + other, base
    return None


def align_behaviors(
    names: list[str],
    groups: list[str],
    valid: np.ndarray,
) -> list[Slot]:
    """Arrange the behaviors of several cameras into shared columns.

    A camera sees the keypoints of its own side, so `eye_back_left` is valid only in the left
    camera and `eye_back_right` only in the right one. Such a mirrored pair becomes one column
    (the left row shows the first, the right row the second) instead of two columns that are each
    empty in one camera. The rule is data driven: a `X_left` / `X_right` pair is merged only when
    both are in the same group and valid in disjoint sets of cameras, so keypoints seen by both
    cameras (`wrist_left`, `wrist_right`, side-less names such as `ear_tip`) keep their own
    columns.

    Args:
        names: the ``B`` behavior names, grouped (behaviors of a group adjacent).
        groups: body-part group of each behavior.
        valid: boolean ``[C, B]``, whether camera `c` has behavior `b`.

    Returns:
        The columns in order of first appearance; behaviors valid in no camera are left out.
    """
    valid = np.asarray(valid, dtype=bool)
    index = {name: idx for idx, name in enumerate(names)}
    slots: list[Slot] = []
    used: set[int] = set()
    for idx, name in enumerate(names):
        if idx in used:
            continue
        members = {int(cam): idx for cam in np.flatnonzero(valid[:, idx])}
        label = name
        mirror = _mirror(name)
        if mirror is not None and mirror[0] in index:
            other = index[mirror[0]]
            cams_other = {int(cam) for cam in np.flatnonzero(valid[:, other])}
            if (
                members and cams_other and groups[other] == groups[idx]
                and not cams_other & set(members)
            ):
                members.update({cam: other for cam in cams_other})
                used.add(other)
                label = mirror[1]
        used.add(idx)
        if members:
            slots.append(Slot(groups[idx], label, members))
    return slots


def _draw_corr_strip(
    ax,
    data: np.ndarray,
    lags_sec: np.ndarray,
    vmax: float,
    ylabel: str | None,
):
    """Draw one neuron x lag correlation strip.

    Args:
        ax: axes of the strip.
        data: correlations ``[N, L]``.
        lags_sec: lag in seconds of each of the ``L`` columns.
        vmax: half-range of the colour scale.
        ylabel: y label for the first strip of a row (it also gets integer ticks); `None` hides
            the tick labels of the neuron axis.

    Returns:
        The image, for the colorbar.
    """
    image = ax.imshow(
        data, aspect='auto', cmap=COLORMAP, vmin=-vmax, vmax=vmax,
        extent=(float(lags_sec[0]), float(lags_sec[-1]), data.shape[0], 0),
        interpolation='nearest',
    )
    ax.axvline(0.0, color='k', lw=0.4, ls=':')
    ax.set_title(f'{np.nanmax(np.abs(data)):.2f}', fontsize=4, pad=1)
    ax.set_xticks([lags_sec[0], 0.0, lags_sec[-1]])
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.set_xlabel('lag (s)', fontsize=5, labelpad=1)
    # the outer labels lean inwards, so neighbouring strips do not overlap
    labels = ax.get_xticklabels()
    labels[0].set_ha('left')
    labels[-1].set_ha('right')
    if ylabel is not None:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(ylabel, fontsize=7)
    else:
        ax.tick_params(labelleft=False)
    return image


def plot_xcorr_overview(
    r: np.ndarray,
    lags_sec: np.ndarray,
    names: list[str],
    cameras: list[str],
    groups: list[str],
    valid: np.ndarray,
    title: str,
    vmax: float,
    ylabel: str = 'neurons',
    peth: dict | None = None,
    peth_vmax: float = 1.0,
) -> Figure:
    """Every behavior on one figure, like Figure 20 of the progress report.

    One row per camera (Figure 20's rows are the three versions of the data) and one narrow
    neuron x lag strip per behavior. Each row is labelled with its own camera's behavior names, so
    the left row reads `eye_back_left` where the right row reads `eye_back_right`; the two share a
    column (see `align_behaviors`) and a cell where a camera has no such behavior is blank. Body
    parts of one group sit together, separated by a gap and labelled once at the top. All strips
    share the neuron order, one symmetric colour scale centred at zero and one colorbar, so weak
    behaviors stay pale. With `peth`, an `events` group (reward, movement onset) follows, like the
    last two columns of Figure 20, drawn in every camera row so a neuron lines up horizontally
    across all strips. When `peth` has event cross-correlations they come first, in exactly the
    style of the behavior strips (same colormap, lag convention and colour scale and colorbar);
    the event-aligned PETH strips follow with their own colour scale and colorbar (fractional
    rate change against the pre-event baseline).

    Args:
        r: correlations ``[C, N, B, L]`` with the neurons already in display order.
        lags_sec: lag in seconds of each of the ``L`` columns.
        names: the ``B`` behavior names, grouped.
        cameras: the ``C`` camera names, in row order.
        groups: body-part group of each behavior (``B`` entries).
        valid: boolean ``[C, B]``, whether camera `c` has behavior `b`.
        title: figure title prefix (session id).
        vmax: half-range of the colour scale.
        ylabel: label of the neuron axis.
        peth: optional event-aligned PETHs, a dict with `labels` (``E`` names), `values`
            ``[E, N, Lp]`` (fractional change, neurons in the same display order as `r`),
            `lags_sec` ``[Lp]`` and `n` (events used per type), and optionally `xcorr`, a dict
            with `values` ``[E, N, Lx]`` (event cross-correlations, same neuron order) and
            `lags_sec` ``[Lx]``; `None` draws no event strip.
        peth_vmax: half-range of the PETH colour scale (+1 = a doubling).

    Returns:
        The figure.
    """
    n_cameras, n_neurons = r.shape[:2]
    slots = align_behaviors(names, groups, valid)
    # grid columns: one per slot with a narrow spacer between groups, then spacer and colorbar
    columns: list[int] = []
    ratios: list[float] = []
    for idx_slot, slot in enumerate(slots):
        if idx_slot and slot.group != slots[idx_slot - 1].group:
            ratios.append(GROUP_GAP)
        columns.append(len(ratios))
        ratios.append(1.0)
    peth_columns: list[int] = []
    event_xcorr_columns: list[int] = []
    event_xcorr = None if peth is None else peth.get('xcorr')
    if peth is not None:
        ratios.append(GROUP_GAP)
        if event_xcorr is not None:
            for _ in peth['labels']:
                event_xcorr_columns.append(len(ratios))
                ratios.append(1.0)
            ratios.append(GROUP_GAP / 2)
        for _ in peth['labels']:
            peth_columns.append(len(ratios))
            ratios.append(1.0)
    ratios += [GROUP_GAP, COLORBAR_RATIO]
    if peth is not None:
        # room for the tick labels and title of the correlation colorbar
        ratios += [GROUP_GAP * 3.5, COLORBAR_RATIO]
    width = min(MAX_FIGURE_WIDTH, max(MIN_FIGURE_WIDTH, PANEL_WIDTH * sum(ratios) + 1.5))
    height_ratios = [GROUP_ROW_HEIGHT] + [NAME_ROW_HEIGHT, ROW_HEIGHT] * n_cameras
    fig = Figure(figsize=(width, sum(height_ratios) / 0.86))
    grid = fig.add_gridspec(
        len(height_ratios), len(ratios), width_ratios=ratios, height_ratios=height_ratios,
        left=0.07, right=0.91, top=0.93, bottom=0.07, wspace=0.12, hspace=0.12,
    )
    image = None
    for idx_cam, camera in enumerate(cameras):
        first_in_row = True
        for slot, col in zip(slots, columns):
            if idx_cam not in slot.members:
                continue
            idx_b = slot.members[idx_cam]
            ax = fig.add_subplot(grid[2 + 2 * idx_cam, col])
            image = _draw_corr_strip(
                ax, r[idx_cam, :, idx_b, :], lags_sec, vmax,
                f'{camera} camera\n{ylabel}' if first_in_row else None,
            )
            first_in_row = False
            # the label of this camera's own behavior, directly above its strip
            header = fig.add_subplot(grid[1 + 2 * idx_cam, col])
            header.axis('off')
            header.text(
                0.5, 0.0, names[idx_b], rotation=90, ha='center', va='bottom', fontsize=5,
                color=TEXT_COLOR, transform=header.transAxes,
            )
    peth_image = None
    if event_xcorr is not None:
        for idx_cam in range(n_cameras):
            for idx_e, col in enumerate(event_xcorr_columns):
                ax = fig.add_subplot(grid[2 + 2 * idx_cam, col])
                _draw_corr_strip(
                    ax, np.asarray(event_xcorr['values'])[idx_e], event_xcorr['lags_sec'], vmax,
                    None,
                )
                header = fig.add_subplot(grid[1 + 2 * idx_cam, col])
                header.axis('off')
                header.text(
                    0.5, 0.0, f'xcorr {peth["labels"][idx_e]} (n={peth["n"][idx_e]})',
                    rotation=90, ha='center', va='bottom', fontsize=5, color=TEXT_COLOR,
                    transform=header.transAxes,
                )
    if peth is not None:
        peth_extent = (float(peth['lags_sec'][0]), float(peth['lags_sec'][-1]), n_neurons, 0)
        for idx_cam in range(n_cameras):
            for idx_e, col in enumerate(peth_columns):
                ax = fig.add_subplot(grid[2 + 2 * idx_cam, col])
                peth_image = ax.imshow(
                    peth['values'][idx_e], aspect='auto', cmap=PETH_COLORMAP, vmin=-peth_vmax,
                    vmax=peth_vmax, extent=peth_extent, interpolation='nearest',
                )
                ax.axvline(0.0, color='k', lw=0.6, ls='--')
                ax.set_xticks([peth['lags_sec'][0], 0.0, peth['lags_sec'][-1]])
                ax.tick_params(labelsize=5, length=2, pad=1, labelleft=False)
                ax.set_xlabel('time (s)', fontsize=5, labelpad=1)
                labels = ax.get_xticklabels()
                labels[0].set_ha('left')
                labels[-1].set_ha('right')
                header = fig.add_subplot(grid[1 + 2 * idx_cam, col])
                header.axis('off')
                header.text(
                    0.5, 0.0, f'PETH {peth["labels"][idx_e]} (n={peth["n"][idx_e]})',
                    rotation=90, ha='center', va='bottom', fontsize=5, color=TEXT_COLOR,
                    transform=header.transAxes,
                )
        first_event_col = (event_xcorr_columns or peth_columns)[0]
        ax = fig.add_subplot(grid[0, first_event_col:peth_columns[-1] + 1])
        ax.axis('off')
        ax.plot([0.02, 0.98], [0.1, 0.1], color=TEXT_COLOR, lw=0.8, transform=ax.transAxes)
        ax.text(
            0.5, 0.3, PETH_GROUP, ha='center', va='bottom', fontsize=7, color=TEXT_COLOR,
            fontweight='bold', transform=ax.transAxes,
        )
    begin = 0
    for stop in range(1, len(slots) + 1):
        if stop == len(slots) or slots[stop].group != slots[begin].group:
            ax = fig.add_subplot(grid[0, columns[begin]:columns[stop - 1] + 1])
            ax.axis('off')
            ax.plot([0.02, 0.98], [0.1, 0.1], color=TEXT_COLOR, lw=0.8, transform=ax.transAxes)
            ax.text(
                0.5, 0.3, slots[begin].group, ha='center', va='bottom', fontsize=7,
                color=TEXT_COLOR, fontweight='bold', transform=ax.transAxes,
            )
            begin = stop
    if peth is not None:
        cax = fig.add_subplot(grid[1:, -3])
        fig.colorbar(image, cax=cax, label='correlation')
        cax.tick_params(labelsize=6)
        cax_peth = fig.add_subplot(grid[1:, -1])
        fig.colorbar(peth_image, cax=cax_peth, label='PETH: fractional rate change')
        cax_peth.tick_params(labelsize=6)
    else:
        cax = fig.add_subplot(grid[1:, -1])
        fig.colorbar(image, cax=cax, label='correlation')
        cax.tick_params(labelsize=6)
    fig.suptitle(
        f'{title}: r(tau) = Corr(neural(t), behavior(t+tau)), positive lag = neural leads; '
        'each strip: neurons x lag, number = max |r|',
        fontsize=8, color=TEXT_COLOR,
    )
    if event_xcorr is not None:
        fig.text(
            0.07, 0.015,
            'events: xcorr uses the same lag convention (positive lag = neural leads the event, '
            'so a response after the event sits at negative lag); PETH time is relative to the '
            'event (positive = after)',
            fontsize=6, color=TEXT_COLOR,
        )
    return fig


def plot_example_curves(
    examples: list[dict],
    lags_sec: np.ndarray,
    title: str,
    n_cols: int = 4,
) -> Figure:
    """Cross-correlation curves of individual neuron-behavior pairs.

    Args:
        examples: one dict per panel with `curve` (``[L]`` correlations), `threshold` (95th
            percentile of the shift-null peak absolute correlation, drawn as a dashed line at
            plus and minus that value, or `None`), `title` and `significant` (bool, or `None`
            when no test was run).
        lags_sec: lag in seconds of each of the ``L`` entries.
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
        curve = np.asarray(example['curve'])
        if example.get('threshold') is not None:
            for sign in (-1, 1):
                ax.axhline(sign * example['threshold'], color='0.5', lw=0.7, ls='--')
        ax.axhline(0.0, color='k', lw=0.4)
        ax.axvline(0.0, color='k', lw=0.4, ls=':')
        color = {True: '#D55E00', False: '#0072B2', None: '#0072B2'}[example.get('significant')]
        ax.plot(lags_sec, curve, color=color, lw=1.2)
        idx_peak = int(np.argmax(np.abs(curve)))
        ax.plot(lags_sec[idx_peak], curve[idx_peak], 'o', color=color, ms=3)
        ax.set_title(example['title'], fontsize=7, color=TEXT_COLOR)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('lag (s)', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('correlation', fontsize=7)
    fig.suptitle(title, fontsize=9, color=TEXT_COLOR)
    return fig
