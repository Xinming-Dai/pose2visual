"""Per-neuron test BPS scatter of one session: method Y (y axis) vs method X (x axis).

Ported from ``scatter_plot_figure3.py`` of the SABLE repo. The run is found from the encoding yaml
exactly as in ``plot_encoding_base_vs_additional.py`` (``pose2visual.encoding.litpose_results``)
and the figure goes to the same ``plots`` folder, so no results directory is passed. The per-neuron
BPS is read from ``encoding_result.npz`` of the session's variant folder. By default the session is
``session.eid`` of the config (``--eid`` overrides it) and the methods are ``base_plus_additional``
(y) and ``base`` (x), so points above the diagonal are neurons the additional keypoints improve.

Usage::

    PYTHONPATH=src python scripts/delta/analysis/encoding_plotting/plot_bps_scatter.py \\
      --config configs/encoding/litpose_encoding.yaml --no-show

Outputs ``scatter_plot.<eid>.png`` in the ``plots`` folder of the run, unless ``-o`` is given.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pose2visual.encoding.litpose_keypoints import session_output_dir
from pose2visual.encoding.litpose_results import (
    CONDITIONS,
    LEGEND_CONDITION_LABELS,
    MODELS,
    find_variant_dir,
    load_session_bps,
    load_wanted_settings,
    plots_dir,
    resolve_eid,
)

logger = logging.getLogger(__name__)

TEXT_FONT_SIZE = 12
FIG_SIZE = 2.2
# matplotlib defaults, as in the old plot helpers
AXIS_LABEL_FONT_KWARGS = {'fontfamily': 'sans-serif', 'fontsize': 'medium', 'fontstyle': 'normal'}
AXIS_TICK_LABEL_FONT_KWARGS = AXIS_LABEL_FONT_KWARGS
PANEL_LABEL_FONT_KWARGS = {'fontsize': 14, 'fontweight': 'bold', 'fontstyle': 'normal'}


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    """Set the tick label font of both axes."""
    ax.tick_params(axis='both', labelsize=AXIS_TICK_LABEL_FONT_KWARGS['fontsize'])
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), **AXIS_TICK_LABEL_FONT_KWARGS)


def plot_bps_scatter(
    x_bps: np.ndarray,
    y_bps: np.ndarray,
    x_label: str,
    y_label: str,
    panel_label: str | None = None,
    show_bps: bool = False,
    show_axis_labels: bool = True,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Scatter the per-neuron BPS of two methods.

    Args:
        x_bps: per-neuron BPS on the x axis, shape `[n_neurons]`.
        y_bps: per-neuron BPS on the y axis, same shape.
        x_label: x axis label.
        y_label: y axis label.
        panel_label: panel letter; none when `None`.
        show_bps: print the mean BPS of both methods in the lower right corner.
        show_axis_labels: draw the axis labels.
        save_path: png path; the figure is not saved when `None`.
        also_save_pdf: also save a pdf next to the png.
        show: show the figure, otherwise close it.

    Returns:
        The figure.
    """
    mask = np.isfinite(x_bps) & np.isfinite(y_bps)
    x = np.asarray(x_bps[mask], dtype=np.float64)
    y = np.asarray(y_bps[mask], dtype=np.float64)

    mean_x = float(np.mean(x)) if x.size else float('nan')
    mean_y = float(np.mean(y)) if y.size else float('nan')
    logger.info(
        f'mean BPS over {x.size} neurons: {x_label} = {mean_x:.2f}, {y_label} = {mean_y:.2f}',
    )

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE), dpi=150)
    low = -0.25
    high = 1.85
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal', adjustable='box')
    ticks = (-0.25, 0.45, 1.15, 1.85)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for grid_value in (0.45, 1.15):
        ax.axhline(grid_value, color='0.85', linewidth=0.8, zorder=0)
        ax.axvline(grid_value, color='0.85', linewidth=0.8, zorder=0)
    ax.plot([low, high], [low, high], linestyle='--', color='0.55', linewidth=1.0, zorder=1)
    ax.scatter(
        x, y, s=18, facecolors=(0.25, 0.25, 0.25), edgecolors='none', alpha=0.35, zorder=2,
    )

    if show_axis_labels:
        ax.set_xlabel(x_label, **AXIS_LABEL_FONT_KWARGS)
        ax.set_ylabel(y_label, **AXIS_LABEL_FONT_KWARGS)
    _apply_axis_tick_font(ax)
    if panel_label:
        ax.text(
            -0.06,
            1.02,
            panel_label,
            transform=ax.transAxes,
            **PANEL_LABEL_FONT_KWARGS,
            va='bottom',
            ha='left',
            clip_on=False,
            zorder=10,
        )
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(1.0)
    fig.tight_layout()

    if show_bps and np.isfinite(mean_x) and np.isfinite(mean_y):
        # right-aligned `<x> vs <y>`, laid out piece by piece from the right edge
        x_right = 0.98
        y_bottom = 0.02
        text_y = ax.text(
            x_right, y_bottom, f'{mean_y:.2f}', transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE, va='bottom', ha='right', color='#1b5e20',
        )
        fig.canvas.draw()
        width_y = text_y.get_window_extent(renderer=fig.canvas.get_renderer()).width
        width_y /= fig.bbox.width
        text_vs = ax.text(
            x_right - width_y, y_bottom, '  vs  ', transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE, va='bottom', ha='right', color='black',
        )
        fig.canvas.draw()
        width_vs = text_vs.get_window_extent(renderer=fig.canvas.get_renderer()).width
        width_vs /= fig.bbox.width
        ax.text(
            x_right - width_y - width_vs, y_bottom, f'{mean_x:.2f}', transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE, va='bottom', ha='right', color='0.45',
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        if also_save_pdf:
            fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
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
        description='Per-neuron BPS scatter of one session: base + additional vs base.',
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
        default=[CONDITIONS[1], CONDITIONS[0]],
        metavar=('METHOD_Y', 'METHOD_X'),
        help=f'two conditions of {list(CONDITIONS)}: first on the y axis, second on the x axis '
        '(default: base_plus_additional base)',
    )
    parser.add_argument(
        '--encoder', choices=MODELS, default='cnn', help='encoder to plot (default: cnn)',
    )
    parser.add_argument(
        '--panel-label', type=str, default=None, help='panel letter (default: none)',
    )
    parser.add_argument('--show-bps', action='store_true', help='print the mean BPS on the plot')
    parser.add_argument('--no-axis-labels', action='store_true', help='hide the axis labels')
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
        '-o', '--output', type=Path, default=None, help='output png path (default: see docstring)',
    )
    parser.add_argument('--save-pdf', action='store_true', help='also save a pdf next to the png')
    parser.add_argument('--no-show', action='store_true', help='save without showing the figure')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load one session's finished run and write the BPS scatter.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if the session has no finished matching run.
        ValueError: if several variants match or the neuron counts of the methods differ.
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
    method_y, method_x = args.methods
    bps_y = load_session_bps(variant_path, method_y, args.encoder)
    bps_x = load_session_bps(variant_path, method_x, args.encoder)
    if bps_x.shape != bps_y.shape:
        raise ValueError(
            f'neuron count mismatch: {method_x} has {bps_x.shape}, {method_y} has {bps_y.shape}',
        )

    save_path = args.output
    if save_path is None:
        save_path = plots_dir(wanted) / f'scatter_plot.{eid}.png'
    if args.no_show:
        plt.switch_backend('Agg')
    plot_bps_scatter(
        bps_x,
        bps_y,
        x_label=LEGEND_CONDITION_LABELS[method_x],
        y_label=LEGEND_CONDITION_LABELS[method_y],
        panel_label=args.panel_label or None,
        show_bps=args.show_bps,
        show_axis_labels=not args.no_axis_labels,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )
    logger.info(f'wrote {save_path} (variant {variant_path})')


if __name__ == '__main__':
    main()
