"""PSTH and activity-raster panels of one session: base vs base + additional keypoints.

Top row: trial-averaged firing rate (PSTH) of the ground truth and the two predictions for the
neurons with the highest test BPS. Bottom row: trial-by-time activity raster (ground truth and
both predictions) of the first of those neurons. Ported from ``psth_activity_raster_figure3.py`` of
the SABLE repo.

The run is found from the encoding yaml exactly as in ``plot_encoding_base_vs_additional.py``
(``pose2visual.encoding.litpose_results``), and the figure goes to the same ``plots`` folder, so no
results directory is passed. It reads the test predictions ``encoding_predictions.npz`` and the
per-neuron BPS ``encoding_result.npz`` of the session's variant folder. By default:

* the session is ``session.eid`` of the config (``--eid`` overrides it),
* the two methods are ``base`` (grey) and ``base_plus_additional`` (orange),
* the plotted neurons are the ``--n-neurons`` (3) with the highest BPS of the second method, ranked
  and labelled with the BPS over the full test set (the numbers of the bar and scatter plots),
* the plotted trials are ``--trial-idx 0,19`` (trials 0..19 of the test set; all of them when the
  test set is smaller).

Usage::

    PYTHONPATH=src python scripts/delta/analysis/encoding_plotting/plot_psth_activity_raster.py \\
      --config configs/encoding/litpose_encoding.yaml --no-show

Outputs ``png/psth_activity_raster.<eid>.png``, ``pdf/psth_activity_raster.<eid>.pdf`` and
``psth_activity_raster.<eid>.json`` (neurons, BPS and the variant folder that was read) in the
``plots`` folder of the run, unless ``-o`` is given.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from pose2visual.encoding.litpose_keypoints import session_output_dir
from pose2visual.encoding.litpose_results import (
    CONDITIONS,
    LEGEND_CONDITION_LABELS,
    MODELS,
    find_variant_dir,
    load_session_bps,
    load_session_predictions,
    load_wanted_settings,
    plots_dir,
    resolve_eid,
)
from pose2visual.plot_io import save_figure, strip_figure_suffix

logger = logging.getLogger(__name__)

GROUND_TRUTH_COLOR = 'black'
# light grey for the reference and orange for the alternative, as in the old figure-3 plot
CONDITION_PSTH_COLORS = {'base': '0.75', 'base_plus_additional': '#F28E2B'}
DEFAULT_TRIAL_IDX = (0, 19)
DEFAULT_N_NEURONS = 3
# matplotlib defaults, as in the old plot helpers
AXIS_LABEL_FONT_KWARGS = {'fontfamily': 'sans-serif', 'fontsize': 'medium', 'fontstyle': 'normal'}
AXIS_TICK_LABEL_FONT_KWARGS = AXIS_LABEL_FONT_KWARGS
PANEL_LABEL_FONT_KWARGS = {'fontsize': 14, 'fontweight': 'bold', 'fontstyle': 'normal'}


def parse_trials_idx_arg(text: str) -> tuple[int, ...]:
    """Parse `--trial-idx` as one int (trial count) or two ints (first / last 0-based index).

    Args:
        text: `N` or `K,N`.

    Returns:
        Tuple of one or two ints.

    Raises:
        argparse.ArgumentTypeError: if the text is empty, not integers, or has more than 2 values.
    """
    parts = [part for part in text.replace(',', ' ').split() if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--trial-idx: expected N or K,N')
    try:
        nums = tuple(int(part) for part in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError('--trial-idx: values must be integers') from err
    if len(nums) > 2:
        raise argparse.ArgumentTypeError('--trial-idx: pass one number (N) or two (K,N)')
    return nums


def trials_idx_to_bounds(
    n_trials_full: int,
    trials_idx: tuple[int, ...] | None,
) -> tuple[int, int]:
    """Map a 0-based trial selection to `(start, stop)`.

    Args:
        n_trials_full: number of trials available.
        trials_idx: `None` for the default (`DEFAULT_TRIAL_IDX`, clipped to the available
            trials), `(N,)` for the first N trials, `(K, N)` for indices K..N inclusive.

    Returns:
        Tuple `(start, stop)` for slicing the trial axis.

    Raises:
        IndexError: if the selection is empty or outside the available trials.
    """
    if trials_idx is None:
        return DEFAULT_TRIAL_IDX[0], min(DEFAULT_TRIAL_IDX[1] + 1, n_trials_full)
    if len(trials_idx) == 1:
        count = int(trials_idx[0])
        if count < 1:
            raise IndexError('--trial-idx N: N must be >= 1 (count of trials from index 0)')
        if count > n_trials_full:
            raise IndexError(f'--trial-idx N={count} exceeds n_trials={n_trials_full}')
        return 0, count
    first, last = int(trials_idx[0]), int(trials_idx[1])
    if first < 0 or last < first:
        raise IndexError(f'--trial-idx K,N: need 0 <= K <= N, got K={first}, N={last}')
    if last >= n_trials_full:
        raise IndexError(f'--trial-idx upper index {last} >= n_trials={n_trials_full}')
    return first, last + 1


def top_neurons(bps: np.ndarray, n_neurons: int) -> list[int]:
    """Indices of the neurons with the highest finite BPS.

    Args:
        bps: per-neuron BPS of shape `[n_neurons]`.
        n_neurons: how many neurons to return at most.

    Returns:
        Neuron indices, best first; fewer than `n_neurons` if fewer have a finite BPS.
    """
    finite_idx = np.flatnonzero(np.isfinite(bps))
    order = finite_idx[np.argsort(-bps[finite_idx], kind='stable')]
    return [int(idx) for idx in order[:n_neurons]]


def _psth(trace: np.ndarray) -> np.ndarray:
    """Average over trials, ignoring NaNs.

    Args:
        trace: array of shape `[n_trials, n_bins]`.

    Returns:
        Array of shape `[n_bins]`.
    """
    return np.nanmean(np.asarray(trace, dtype=np.float64), axis=0)


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    """Set the tick label font of both axes."""
    ax.tick_params(axis='both', labelsize=AXIS_TICK_LABEL_FONT_KWARGS['fontsize'])
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), **AXIS_TICK_LABEL_FONT_KWARGS)


def _normalize_traces(traces: list[np.ndarray]) -> list[np.ndarray]:
    """Scale the traces jointly to [0, 1].

    Args:
        traces: PSTHs of one neuron (ground truth and both predictions).

    Returns:
        The traces scaled with the shared min / max of their finite values; all zeros if the
        shared range is empty.
    """
    finite_chunks = [trace[np.isfinite(trace)] for trace in traces if np.any(np.isfinite(trace))]
    if not finite_chunks:
        return traces
    finite_values = np.concatenate(finite_chunks)
    low = float(np.nanmin(finite_values))
    high = float(np.nanmax(finite_values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return [np.zeros_like(trace, dtype=np.float64) for trace in traces]
    return [(trace - low) / (high - low) for trace in traces]


def _add_bps_label(
    ax: plt.Axes,
    bps_values: tuple[float, float],
    colors: tuple[str, str],
) -> None:
    """Print `BPS: <method 1> vs. <method 2>` above the top right corner of a PSTH panel.

    Args:
        ax: panel axes.
        bps_values: BPS of the first and second method.
        colors: text colors of the two values.
    """
    text = HPacker(
        children=[
            TextArea('BPS: ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps_values[0]:.2f}', textprops={'color': colors[0], 'size': 10}),
            TextArea(' vs. ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps_values[1]:.2f}', textprops={'color': colors[1], 'size': 10}),
        ],
        align='center',
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc='upper right',
        child=text,
        pad=0,
        frameon=False,
        bbox_to_anchor=(1.0, 1.12),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(box)


def _add_single_bps_label(ax: plt.Axes, bps: float, value_color: str) -> None:
    """Print `BPS: <value>` centered above a raster panel.

    Args:
        ax: panel axes.
        bps: BPS value.
        value_color: text color of the value.
    """
    text = HPacker(
        children=[
            TextArea('BPS: ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps:.2f}', textprops={'color': value_color, 'size': 10}),
        ],
        align='center',
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc='lower center',
        child=text,
        pad=0,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(box)


def _normalized_activity_raster(activity: np.ndarray) -> np.ndarray:
    """Center a trial-by-time raster on its PSTH and scale it by its 95th percentile.

    Args:
        activity: array of shape `[n_trials, n_bins]`.

    Returns:
        Array of the same shape clipped to [-1, 1].
    """
    activity = np.asarray(activity, dtype=np.float64)
    centered = activity - np.nanmean(activity, axis=0, keepdims=True)
    scale = np.nanpercentile(np.abs(centered), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(centered / scale, -1.0, 1.0)


def _plot_activity_raster(
    ax: plt.Axes,
    activity: np.ndarray,
    method_name: str,
    time_s: np.ndarray,
    trial_start: int,
    bps: float | None = None,
    bps_color: str = 'black',
    show_ylabel: bool = False,
) -> AxesImage:
    """Draw one trial-by-time raster.

    Args:
        ax: panel axes.
        activity: array of shape `[n_trials, n_bins]`.
        method_name: title printed above the panel.
        time_s: time of each bin in seconds.
        trial_start: 0-based index of the first plotted trial.
        bps: BPS label value; no label when `None`.
        bps_color: color of the BPS value.
        show_ylabel: whether to label the trial axis.

    Returns:
        The image, for the shared colorbar.
    """
    raster = _normalized_activity_raster(activity)
    trial_stop = trial_start + activity.shape[0] - 1
    image = ax.imshow(
        raster,
        aspect='auto',
        cmap='bwr',
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest',
        extent=(float(time_s[0]), float(time_s[-1]), trial_stop + 0.5, trial_start - 0.5),
    )
    if bps is not None:
        _add_single_bps_label(ax, bps, bps_color)
    # the name sits above the BPS label: a long name does not fit next to it on one line
    ax.text(
        0.5,
        1.22,
        method_name,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        color='black',
        clip_on=False,
    )
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xticks([float(time_s[0]), float(time_s[-1])])
    ax.set_xticklabels(['0', f'{time_s[-1]:.1f}'])
    ax.set_xlabel('Time (s)', labelpad=0, **AXIS_LABEL_FONT_KWARGS)
    if show_ylabel:
        ax.set_ylabel('Trial Index', **AXIS_LABEL_FONT_KWARGS)
        ax.set_yticks([trial_start, trial_stop])
    else:
        ax.set_yticks([])
    ax.tick_params(axis='both', width=1.0, length=3, pad=1)
    _apply_axis_tick_font(ax)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(1.0)
    return image


def plot_psth_comparison(
    gt: np.ndarray,
    preds: tuple[np.ndarray, np.ndarray],
    bps: tuple[np.ndarray, np.ndarray],
    methods: tuple[str, str],
    neuron_indices: list[int],
    labels: tuple[str, str],
    duration_s: float = 1.0,
    panel_label: str | None = None,
    trials_idx: tuple[int, ...] | None = None,
    raster_neuron_index: int | None = None,
    normalize: bool = True,
    save_path: Path | None = None,
    show: bool = True,
    colors: tuple[str, str] | None = None,
) -> plt.Figure:
    """Draw the PSTH row and the activity-raster row.

    Args:
        gt: held-out spikes of shape `[n_trials, n_bins, n_neurons]`.
        preds: predicted rates of the two methods, same shape as `gt`.
        bps: per-neuron BPS of the two methods (printed on the panels).
        methods: condition names of the two methods, selecting their colors.
        neuron_indices: neurons of the PSTH panels, one panel each.
        labels: legend names of the two methods.
        duration_s: right end of the time axis in seconds.
        panel_label: panel letter; none when `None`.
        trials_idx: trial selection, see `trials_idx_to_bounds`.
        raster_neuron_index: neuron of the raster row; the first of `neuron_indices` when `None`.
        normalize: scale each PSTH panel to [0, 1] instead of raw rates.
        save_path: png path; the png goes to its `png/` folder, the pdf to `pdf/` (same name);
            the figure is not saved when `None`.
        show: show the figure, otherwise close it.
        colors: colors of the two methods; `None` takes those of `methods` from
            `CONDITION_PSTH_COLORS`.

    Returns:
        The figure.

    Raises:
        IndexError: if a neuron index or the trial selection is out of range.
    """
    if colors is None:
        colors = (CONDITION_PSTH_COLORS[methods[0]], CONDITION_PSTH_COLORS[methods[1]])
    start, stop = trials_idx_to_bounds(gt.shape[0], trials_idx)
    gt = gt[start:stop]
    pred1, pred2 = preds[0][start:stop], preds[1][start:stop]

    n_neurons = gt.shape[2]
    for neuron_idx in neuron_indices:
        if not 0 <= neuron_idx < n_neurons:
            raise IndexError(f'neuron index {neuron_idx} out of range for n_neurons={n_neurons}')
    raster_idx = neuron_indices[0] if raster_neuron_index is None else int(raster_neuron_index)
    if not 0 <= raster_idx < n_neurons:
        raise IndexError(
            f'raster neuron index {raster_idx} out of range for n_neurons={n_neurons}',
        )

    n_panels = len(neuron_indices)
    fig = plt.figure(figsize=(max(2.0 * n_panels, 3.0), 4.2), dpi=200)
    grid = fig.add_gridspec(2, 12, height_ratios=(1.0, 1.0), hspace=0.78, wspace=0.45)
    col_edges = np.linspace(0, 12, n_panels + 1).round().astype(int)
    axes_psth = [
        fig.add_subplot(grid[0, col_edges[idx] : col_edges[idx + 1]]) for idx in range(n_panels)
    ]
    time_s = np.linspace(0.0, float(duration_s), gt.shape[1])

    for ax, neuron_idx in zip(axes_psth, neuron_indices):
        gt_trace = _psth(gt[:, :, neuron_idx])
        pred1_trace = _psth(pred1[:, :, neuron_idx])
        pred2_trace = _psth(pred2[:, :, neuron_idx])
        if normalize:
            gt_trace, pred1_trace, pred2_trace = _normalize_traces(
                [gt_trace, pred1_trace, pred2_trace],
            )
        ax.plot(time_s, gt_trace, color=GROUND_TRUTH_COLOR, linewidth=1.1)
        ax.plot(time_s, pred1_trace, color=colors[0], linewidth=3.0, alpha=0.9)
        ax.plot(time_s, pred2_trace, color=colors[1], linewidth=2.5)
        _add_bps_label(ax, (float(bps[0][neuron_idx]), float(bps[1][neuron_idx])), colors)

        ax.set_xlim(0.0, float(duration_s))
        ax.set_xticks([0.0, float(duration_s)])
        ax.set_xticklabels(['0', f'{duration_s:.1f}'])
        ax.tick_params(axis='both', width=1.2, length=3, pad=1)
        _apply_axis_tick_font(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.set_yticks([0.0, 1.0] if normalize else [])

    axes_psth[0].set_ylabel(
        'Normalized\nFiring Rate' if normalize else 'Firing Rate', **AXIS_LABEL_FONT_KWARGS,
    )
    for ax in axes_psth[1:]:
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)

    raster_axes = [
        fig.add_subplot(grid[1, 0:4]),
        fig.add_subplot(grid[1, 4:8]),
        fig.add_subplot(grid[1, 8:12]),
    ]
    image = _plot_activity_raster(
        raster_axes[0],
        gt[:, :, raster_idx],
        method_name='Ground Truth',
        time_s=time_s,
        trial_start=start,
        show_ylabel=True,
    )
    for ax, pred, label, bps_values, color in zip(
        raster_axes[1:], (pred1, pred2), labels, bps, colors,
    ):
        _plot_activity_raster(
            ax,
            pred[:, :, raster_idx],
            method_name=label,
            time_s=time_s,
            trial_start=start,
            bps=float(bps_values[raster_idx]),
            bps_color=color,
        )
    cax = fig.add_axes((0.935, 0.17, 0.012, 0.26))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([-1.0, 0.0, 1.0])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)

    label_colors = (GROUND_TRUTH_COLOR, colors[0], colors[1])
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=0, label=text)
        for color, text in zip(label_colors, ('Ground Truth', labels[0], labels[1]))
    ]
    legend = fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        ncol=3,
        fontsize=10,
        handlelength=0,
        handletextpad=0,
        columnspacing=0.9,
    )
    for text, color in zip(legend.get_texts(), label_colors):
        text.set_color(color)
    if panel_label:
        fig.text(0.005, 0.98, panel_label, **PANEL_LABEL_FONT_KWARGS, va='top', ha='left')

    fig.subplots_adjust(left=0.08, right=0.90, top=0.82, bottom=0.14, wspace=0.45, hspace=0.95)
    if save_path is not None:
        save_figure(fig, strip_figure_suffix(save_path), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='PSTH and activity rasters of one session: base vs base + additional.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='yaml config the encoding was run with (litpose_encoding.yaml)',
    )
    parser.add_argument(
        '--eid',
        type=str,
        default=None,
        help='session to plot (default: session.eid of the config)',
    )
    parser.add_argument(
        '--methods',
        nargs=2,
        choices=CONDITIONS,
        default=list(CONDITIONS),
        metavar=('METHOD1', 'METHOD2'),
        help=f'two conditions of {list(CONDITIONS)}; METHOD1 is grey, METHOD2 is orange '
        '(default: base base_plus_additional)',
    )
    parser.add_argument(
        '--neuron-indices',
        nargs='+',
        type=int,
        default=None,
        help='neurons to plot, one panel each (default: the --n-neurons with the highest BPS of '
        'METHOD2)',
    )
    parser.add_argument(
        '--n-neurons',
        type=int,
        default=DEFAULT_N_NEURONS,
        help='number of neurons plotted when --neuron-indices is not given',
    )
    parser.add_argument(
        '--encoder', choices=MODELS, default='cnn', help='encoder to plot (default: cnn)',
    )
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based test trials: N alone = first N trials; two values = indices K..N '
        'inclusive (default: 0,19)',
    )
    parser.add_argument(
        '--raster-neuron-index',
        type=int,
        default=None,
        help='neuron of the raster row (default: the first plotted neuron)',
    )
    parser.add_argument(
        '--duration-s', type=float, default=1.0, help='right end of the time axis in seconds',
    )
    parser.add_argument('--method1-label', type=str, default=None, help='legend label of METHOD1')
    parser.add_argument('--method2-label', type=str, default=None, help='legend label of METHOD2')
    parser.add_argument(
        '--panel-label', type=str, default=None, help='panel letter (default: none)',
    )
    parser.add_argument(
        '--no-normalize', action='store_true', help='plot raw PSTH values instead of 0-1 traces',
    )
    parser.add_argument(
        '--base-keypoints', nargs='+', default=None, help='overrides base_keypoints of the config',
    )
    parser.add_argument(
        '--additional-keypoints',
        nargs='+',
        default=None,
        help='overrides additional_keypoints of the config',
    )
    parser.add_argument(
        '--anatomical-group',
        type=str,
        default=None,
        help='overrides anatomical_group of the config',
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='only match runs of this seed (default: any)',
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help='output png path; the files go to png/ and pdf/ next to it (default: see docstring)',
    )
    parser.add_argument('--no-show', action='store_true', help='save without showing the figure')
    return parser.parse_args(argv)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Save the plotted neurons and where they came from next to the figure.

    Args:
        path: json path.
        summary: json-serializable provenance.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    """Load one session's finished run and write the PSTH / raster figure.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if the session has no finished matching run or no saved predictions.
        ValueError: if several variants match, or no neuron has a finite BPS.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    eid = resolve_eid(args.config, args.eid)
    wanted = load_wanted_settings(
        args.config,
        base_keypoints=args.base_keypoints,
        additional_keypoints=args.additional_keypoints,
        seed=args.seed,
        anatomical_group=args.anatomical_group,
    )
    variant_path = find_variant_dir(
        session_output_dir(wanted['video_preds_dir'], wanted['output_folder'], eid), wanted,
    )
    methods = (args.methods[0], args.methods[1])
    gt_list, pred_list, bps_list = [], [], []
    for method in methods:
        gt, pred = load_session_predictions(variant_path, method, args.encoder)
        gt_list.append(gt)
        pred_list.append(pred)
        bps_list.append(load_session_bps(variant_path, method, args.encoder))
    if gt_list[0].shape != gt_list[1].shape:
        raise ValueError(f'method shapes differ: {gt_list[0].shape} vs {gt_list[1].shape}')
    if not np.allclose(gt_list[0], gt_list[1], equal_nan=True):
        logger.warning('ground truth differs between the methods; plotting it from METHOD1')

    neuron_indices = args.neuron_indices
    if neuron_indices is None:
        neuron_indices = top_neurons(bps_list[1], args.n_neurons)
        if not neuron_indices:
            raise ValueError(f'{variant_path}: no neuron has a finite {methods[1]} BPS')
    logger.info(f'{eid}: plotting neurons {neuron_indices} (variant {variant_path.name})')

    labels = (
        args.method1_label or LEGEND_CONDITION_LABELS[methods[0]],
        args.method2_label or LEGEND_CONDITION_LABELS[methods[1]],
    )
    save_path = args.output
    if save_path is None:
        save_path = plots_dir(wanted) / f'psth_activity_raster.{eid}.png'
    if args.no_show:
        plt.switch_backend('Agg')
    plot_psth_comparison(
        gt_list[0],
        (pred_list[0], pred_list[1]),
        (bps_list[0], bps_list[1]),
        methods,
        neuron_indices,
        labels,
        duration_s=args.duration_s,
        panel_label=args.panel_label or None,
        trials_idx=args.trial_idx,
        raster_neuron_index=args.raster_neuron_index,
        normalize=not args.no_normalize,
        save_path=save_path,
        show=not args.no_show,
    )
    start, stop = trials_idx_to_bounds(gt_list[0].shape[0], args.trial_idx)
    write_summary_json(
        save_path.with_suffix('.json'),
        {
            'eid': eid,
            'variant_dir': str(variant_path),
            'encoder': args.encoder,
            'methods': list(methods),
            'neuron_indices': neuron_indices,
            'raster_neuron_index': (
                neuron_indices[0] if args.raster_neuron_index is None
                else args.raster_neuron_index
            ),
            'trials': [start, stop - 1],
            'bps': {
                method: {str(idx): float(bps[idx]) for idx in neuron_indices}
                for method, bps in zip(methods, bps_list)
            },
            'wanted': wanted,
        },
    )
    logger.info(
        f'wrote {save_path.name} (png/ and pdf/ in {save_path.parent}) and '
        f'{save_path.with_suffix(".json")}',
    )


if __name__ == '__main__':
    main()
