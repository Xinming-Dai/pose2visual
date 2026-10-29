"""Bar plot of neural encoding BPS: base keypoints vs base + additional keypoints.

Reads the ``encoding_result.npz`` files written by ``step2_run_encoding.sh``
(``pose2visual.encoding.run_litpose_encoding``) and plots the held-out test bits-per-spike (BPS)
of the RRR and CNN encoders for the two input sets. One bar is the mean over all neurons of all
sessions (non-finite BPS dropped), the error bar is the standard error over those neurons. The
figure has no text below the axes: the base and additional keypoint names, the camera and the
neuron counts are written to a txt file next to it.

With ``anatomical_group`` set to a group set of ``body_parts_json`` (e.g.
``nonoverlapping_analysis_groups``), the two keypoint lists of the config hold group names such as
``eye_related``; the txt file then lists the group names (the keypoints a group resolved to can
differ between sessions, they are listed per session in the json next to the figure).

Result layout (see ``pose2visual.encoding.litpose_keypoints``)::

    <video_preds_dir>/<camera>_<output_name>/<eid>/<variant>/encoding_result.npz
    <video_preds_dir>/<camera>_<output_name>/<eid>/<variant>/encoding_params.json

A session can hold several variants (e.g. a pca and a no-pca run of the same keypoints), and
folder names carry hash suffixes, so the variant is chosen by its ``run_signature`` in
``encoding_params.json``: camera, base / additional keypoints (or groups), ``anatomical_group``
and pca settings (flag and group set). These come from the same yaml the run used (``configs/encoding/litpose_encoding.yaml``);
flags override single entries.
Every session must hold exactly one finished matching variant, otherwise the script raises.
The sessions are ``--eids``, else ``--eid-group``, else the ``session.eid_group`` of the config
(file: ``paths.eid_groups_json``), else every session folder with a ``dynamic_vars.npz``. With
``--only-if-complete`` the script logs the sessions that are not finished and exits without
plotting; ``step2_run_encoding.sh`` calls it that way after every session, so the plot appears
when the last session of the group finishes.

Usage::

    PYTHONPATH=src python scripts/delta/analysis/encoding_plotting/plot_encoding_base_vs_additional.py \\
      --config configs/encoding/litpose_encoding.yaml --no-show

Outputs ``<video_preds_dir>/<camera>_<output_name>/plots/encoding_base_vs_additional_<...>/`` with
``png/encoding_barplot.png``, ``pdf/encoding_barplot.pdf``, ``svg/encoding_barplot.svg`` and
``encoding_barplot.json`` (numbers and
provenance) and ``encoding_barplot_keypoints.txt`` (camera, step order, sessions, neuron counts and
the keypoint lists, one name per line; the figure itself carries no text below the axes), unless
``-o`` is given. The per-session figures of
``plot_psth_activity_raster.py`` and ``plot_bps_scatter.py`` go to the same folder;
``step2_run_encoding.sh`` makes those right after each session finishes, without waiting for the
other sessions. The helpers shared by the three scripts are in
``pose2visual.encoding.litpose_results``.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pose2visual.encoding.litpose_keypoints import session_output_dir
from pose2visual.encoding.litpose_results import (
    CONDITION_COLORS,
    CONDITIONS,
    LEGEND_CONDITION_LABELS,
    MODELS,
    RESULT_NAME,
    find_variant_dir,
    load_wanted_settings,
    plots_dir,
)
from pose2visual.plot_io import save_figure, strip_figure_suffix

logger = logging.getLogger(__name__)

# `anatomical_group` value meaning "the keypoint lists hold keypoint names"
_ANATOMICAL_GROUP_ALL = 'all'


@dataclass
class SessionResult:
    """Per-neuron test BPS of one session and the keypoints it was fitted with.

    Attributes:
        eid: session id.
        variant_dir: folder holding the `encoding_result.npz` that was read.
        bps: `(condition, model)` -> per-neuron BPS array of shape `[n_neurons]`.
        base_keypoints: base keypoint names as stored in the result (the keypoints the base
            groups resolved to in group mode).
        additional_keypoints: additional keypoint names as stored in the result.
        anatomical_group: `all` or the group set the run was fitted with.
        base_groups: base group names; empty when `anatomical_group` is `all`.
        additional_groups: additional group names; empty when `anatomical_group` is `all`.
    """

    eid: str
    variant_dir: Path
    bps: dict[tuple[str, str], np.ndarray]
    base_keypoints: list[str]
    additional_keypoints: list[str]
    anatomical_group: str = _ANATOMICAL_GROUP_ALL
    base_groups: list[str] = field(default_factory=list)
    additional_groups: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Bar plot of encoding BPS: base keypoints vs base + additional keypoints.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='yaml config the encoding was run with (litpose_encoding.yaml)',
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
        help=(
            'overrides anatomical_group of the config: all (keypoint names) or a group set of '
            'body_parts_json (group names)'
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='only match runs of this seed (default: any)',
    )
    parser.add_argument(
        '--eids',
        nargs='+',
        default=None,
        help='sessions to plot (default: the eid group, else every session with a '
        'dynamic_vars.npz)',
    )
    parser.add_argument(
        '--eid-group',
        type=str,
        default=None,
        help='eid group to plot, a key of paths.eid_groups_json (default: session.eid_group of '
        'the config); ignored when --eids is given',
    )
    parser.add_argument(
        '--only-if-complete',
        action='store_true',
        help='plot only when every session has a finished matching result; otherwise log the '
        'unfinished ones and exit without error (for the end of each step 2 job)',
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=None, help='output png path (default: see docstring)',
    )
    parser.add_argument(
        '--no-show', action='store_true', help='save the figure without showing it',
    )
    return parser.parse_args(argv)




def unfinished_eids(
    video_preds_dir: str | Path,
    output_folder: str,
    eids: list[str],
    wanted: dict[str, Any],
) -> list[str]:
    """List the sessions that have no finished matching result yet.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eids: sessions to check.
        wanted: output of `load_wanted_settings`.

    Returns:
        The eids, in order, without a finished variant that matches `wanted`.

    Raises:
        ValueError: if a session has several matching variants.
    """
    unfinished = []
    for eid in eids:
        try:
            find_variant_dir(session_output_dir(video_preds_dir, output_folder, eid), wanted)
        except FileNotFoundError:
            unfinished.append(eid)
    return unfinished


def load_session_result(eid: str, variant_path: Path) -> SessionResult:
    """Load the per-neuron test BPS and the keypoint names of one finished variant.

    Args:
        eid: session id.
        variant_path: folder holding `encoding_result.npz`.

    Returns:
        The `SessionResult` of the session.
    """
    data = np.load(variant_path / RESULT_NAME, allow_pickle=False)
    bps = {
        (cond, model): np.asarray(data[f'{cond}_{model}_bps_per_neuron'], dtype=np.float64)
        for cond in CONDITIONS
        for model in MODELS
    }
    # results from before group mode have none of the group entries
    files = data.files
    return SessionResult(
        eid=eid,
        variant_dir=variant_path,
        bps=bps,
        base_keypoints=[str(name) for name in data['base_keypoints']],
        additional_keypoints=[str(name) for name in data['additional_keypoints']],
        anatomical_group=(
            str(data['anatomical_group']) if 'anatomical_group' in files
            else _ANATOMICAL_GROUP_ALL
        ),
        base_groups=[str(name) for name in data['base_groups']] if 'base_groups' in files else [],
        additional_groups=(
            [str(name) for name in data['additional_groups']]
            if 'additional_groups' in files else []
        ),
    )


def list_session_eids(output_root: Path) -> list[str]:
    """List the sessions of an output folder.

    Args:
        output_root: `<video_preds_dir>/<camera>_<output_name>`.

    Returns:
        Sorted names of the subfolders holding a `dynamic_vars.npz` (step B output).
    """
    return sorted(
        path.name
        for path in output_root.iterdir()
        if path.is_dir() and (path / 'dynamic_vars.npz').is_file()
    )


def load_all_sessions(
    video_preds_dir: str | Path,
    output_folder: str,
    eids: list[str],
    wanted: dict[str, Any],
) -> list[SessionResult]:
    """Load the matching finished result of every session; all of them must exist.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eids: sessions to load.
        wanted: output of `load_wanted_settings`.

    Returns:
        One `SessionResult` per session, in the order of `eids`.

    Raises:
        FileNotFoundError: if some sessions have no finished matching result; a partial set of
            sessions is never plotted.
        ValueError: if a session has several matching variants.
    """
    results = []
    missing = {}
    for eid in eids:
        session_dir = session_output_dir(video_preds_dir, output_folder, eid)
        try:
            variant_path = find_variant_dir(session_dir, wanted)
        except FileNotFoundError as err:
            missing[eid] = str(err)
            continue
        results.append(load_session_result(eid, variant_path))
    if missing:
        details = '\n'.join(f'  {eid}: {reason}' for eid, reason in missing.items())
        raise FileNotFoundError(
            f'{len(missing)} of {len(eids)} sessions have no finished matching result '
            f'(restrict the sessions with --eids):\n{details}',
        )
    return results


def check_same_keypoints(results: list[SessionResult]) -> None:
    """Warn if the sessions were not fitted with the same base and additional keypoints.

    In group mode the usable keypoints a group resolves to differ between sessions, so the group
    names are compared instead of the keypoints. Keypoints that are masked too often are dropped
    per session, so a difference is logged as a warning and the sessions are still pooled.

    Args:
        results: sessions to compare.

    Raises:
        ValueError: if `results` is empty.
    """
    if not results:
        raise ValueError('no sessions to plot')
    first = results[0]
    for result in results[1:]:
        if first.anatomical_group != _ANATOMICAL_GROUP_ALL:
            same = (
                result.anatomical_group == first.anatomical_group
                and sorted(result.base_groups) == sorted(first.base_groups)
                and sorted(result.additional_groups) == sorted(first.additional_groups)
            )
        else:
            same = (
                result.anatomical_group == first.anatomical_group
                and sorted(result.base_keypoints) == sorted(first.base_keypoints)
                and sorted(result.additional_keypoints) == sorted(first.additional_keypoints)
            )
        if not same:
            logger.warning(
                f'keypoints of {result.eid} differ from {first.eid} (the txt file lists '
                f'those of {first.eid} in full): '
                f'{result.base_groups or result.base_keypoints} + '
                f'{result.additional_groups or result.additional_keypoints} vs '
                f'{first.base_groups or first.base_keypoints} + '
                f'{first.additional_groups or first.additional_keypoints}',
            )


def collect_stats(
    results: list[SessionResult],
) -> dict[tuple[str, str], dict[str, float]]:
    """Pool the finite per-neuron BPS of all sessions and take mean and standard error.

    Args:
        results: sessions to pool (`check_same_keypoints` warns if their keypoints or groups
            differ).

    Returns:
        `(condition, model)` -> dict with `mean`, `se` (std with ddof=1 over sqrt of the number
        of neurons, 0 for a single neuron) and `n_neurons`; `mean` and `se` are NaN without any
        finite value.

    Raises:
        ValueError: if `results` is empty.
    """
    check_same_keypoints(results)
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for key in ((cond, model) for cond in CONDITIONS for model in MODELS):
        pooled = np.concatenate([result.bps[key] for result in results])
        values = pooled[np.isfinite(pooled)]
        if values.size == 0:
            stats[key] = {'mean': float('nan'), 'se': float('nan'), 'n_neurons': 0}
            continue
        se = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        stats[key] = {'mean': float(np.mean(values)), 'se': se, 'n_neurons': int(values.size)}
    return stats


def _label_position(value: float, se: float, label_offset: float) -> tuple[float, str]:
    """Place a value label above positive bars and below negative ones.

    The offset is measured from the tip of the error bar, so the label never overlaps it.

    Args:
        value: bar height.
        se: error bar half-length.
        label_offset: gap between the error bar tip and the label.

    Returns:
        Tuple `(y, vertical alignment)`.
    """
    if value >= 0:
        return value + se + label_offset, 'bottom'
    return value - se - label_offset, 'top'


def plot_encoding_bars(
    stats: dict[tuple[str, str], dict[str, float]],
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw BPS bars per encoder (RRR, CNN), base next to base + additional.

    Args:
        stats: output of `collect_stats`.
        save_path: png path; the png goes to its `png/` folder, the pdf to `pdf/` (same name);
            the figure is not saved when `None`.
        show: show the figure, otherwise close it.

    Returns:
        The figure.
    """
    width = 0.38
    bar_center_spacing = 1.0
    edge_padding = 0.7
    x_centers = np.arange(len(MODELS), dtype=np.float64) * bar_center_spacing
    fig, ax = plt.subplots(figsize=(3.4, 3.1), dpi=200)

    finite = [entry for entry in stats.values() if np.isfinite(entry['mean'])]
    y_max = max((entry['mean'] + entry['se'] for entry in finite), default=1.0)
    y_min_data = min((entry['mean'] - entry['se'] for entry in finite), default=0.0)
    y_lim = max(y_max, 1e-6) * 1.30
    label_offset = y_lim * 0.03
    # headroom below zero (and for a below-bar label) only when a bar dips negative
    y_min = min(0.0, y_min_data - 3 * label_offset)

    # one group per encoder with the two conditions side by side, so base and base + additional
    # of the same encoder are adjacent; the colour tells the condition
    legend_done = set()
    for idx_model, model in enumerate(MODELS):
        for cond, shift in zip(CONDITIONS, (-width / 2, width / 2)):
            entry = stats[(cond, model)]
            if not np.isfinite(entry['mean']):
                continue
            x_pos = x_centers[idx_model] + shift
            label = LEGEND_CONDITION_LABELS[cond] if cond not in legend_done else None
            legend_done.add(cond)
            ax.bar(
                x_pos,
                entry['mean'],
                width,
                yerr=entry['se'],
                color=CONDITION_COLORS[cond],
                edgecolor='none',
                capsize=2,
                error_kw={'elinewidth': 0.8},
                label=label,
            )
            y_text, va_text = _label_position(entry['mean'], entry['se'], label_offset)
            ax.text(x_pos, y_text, f'{entry["mean"]:.3f}', ha='center', va=va_text, fontsize=8)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([model.upper() for model in MODELS], fontsize=10, fontweight='medium')
    ax.set_ylabel('Test BPS (mean over neurons)', fontsize=11, fontweight='medium')
    ax.tick_params(axis='y', labelsize=9, width=1.5, length=5, direction='out')
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_ylim(y_min, y_lim)
    ax.set_xlim(x_centers.min() - edge_padding, x_centers.max() + edge_padding)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    # zero line in place of the bottom spine, since the axes can extend below zero
    ax.axhline(0, color='black', linewidth=1.5, zorder=0.5)
    # above the axes so it never covers a value label
    ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        ncol=2,
        fontsize=7,
        handlelength=1.2,
        columnspacing=1.0,
    )

    fig.tight_layout(pad=0.4)

    if save_path is not None:
        save_figure(
            fig, strip_figure_suffix(save_path), dpi=400, svg=True, bbox_inches='tight',
            pad_inches=0.05, facecolor='white',
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def default_output_path(wanted: dict[str, Any]) -> Path:
    """Default png path under the `output_folder` of the results.

    Args:
        wanted: output of `load_wanted_settings`.

    Returns:
        `encoding_barplot.png` in `plots_dir(wanted)`; it is written to the `png/` subfolder
        there, with the pdf in `pdf/`.
    """
    return plots_dir(wanted) / 'encoding_barplot.png'


def write_keypoints_txt(
    path: Path,
    results: list[SessionResult],
    wanted: dict[str, Any],
    stats: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Save the setup behind the figure and the keypoint lists next to it.

    The header holds the camera, step pipeline, sessions and neuron counts; the keypoint lists
    follow, one name per line.

    The lists of the first session are written in full. Every other session gets a line saying
    whether its keypoints are identical to those of the first session; if not, the keypoints it
    has in addition to (`+`) or lacks (`-`) compared with the first session are listed.

    Args:
        path: txt path.
        results: sessions behind the bars.
        wanted: output of `load_wanted_settings`.
        stats: output of `collect_stats`.
    """
    first = results[0]
    counts = [int(entry['n_neurons']) for entry in stats.values()]
    n_neurons = str(counts[0]) if min(counts) == max(counts) else f'{min(counts)}-{max(counts)}'
    lines = [
        f'camera: {wanted["camera"]}',
        f'step_order: {wanted["step_order"]}',
        f'anatomical_group: {first.anatomical_group}',
        f'neurons: {n_neurons} (bars: mean ± SE over neurons)',
        f'sessions ({len(results)}): {", ".join(result.eid for result in results)}',
        f'lists below are those of session {first.eid}',
        '',
    ]
    sections = [
        ('Base groups', first.base_groups),
        ('Additional groups', first.additional_groups),
        ('Base keypoints', first.base_keypoints),
        ('Additional keypoints', first.additional_keypoints),
    ]
    for title, names in sections:
        if names:
            lines += [f'{title} ({len(names)}):', *(f'  {name}' for name in names), '']
    for result in results[1:]:
        diff_lines = []
        for title, names, ref_names in (
            ('base keypoints', result.base_keypoints, first.base_keypoints),
            ('additional keypoints', result.additional_keypoints, first.additional_keypoints),
        ):
            extra = sorted(set(names) - set(ref_names))
            missing = sorted(set(ref_names) - set(names))
            if extra or missing:
                diff_lines.append(f'  {title}:')
                diff_lines += [f'    + {name}' for name in extra]
                diff_lines += [f'    - {name}' for name in missing]
        if diff_lines:
            lines += [f'session {result.eid} differs from {first.eid}:', *diff_lines, '']
        else:
            lines += [f'session {result.eid}: same keypoints as {first.eid}', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_summary_json(
    path: Path,
    stats: dict[tuple[str, str], dict[str, float]],
    results: list[SessionResult],
    wanted: dict[str, Any],
) -> None:
    """Save the plotted numbers and where they came from next to the figure.

    Args:
        path: json path.
        stats: output of `collect_stats`.
        results: sessions behind the bars.
        wanted: output of `load_wanted_settings`.
    """
    summary = {
        'wanted': wanted,
        'anatomical_group': results[0].anatomical_group,
        'base_groups': results[0].base_groups,
        'additional_groups': results[0].additional_groups,
        'base_keypoints': results[0].base_keypoints,
        'additional_keypoints': results[0].additional_keypoints,
        # in group mode the resolved keypoints can differ between sessions
        'session_keypoints': {
            result.eid: {
                'base_keypoints': result.base_keypoints,
                'additional_keypoints': result.additional_keypoints,
            }
            for result in results
        },
        'stats': {f'{cond}_{model}': entry for (cond, model), entry in stats.items()},
        'sessions': {result.eid: str(result.variant_dir) for result in results},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    """Load the matching results of every session and write the bar plot.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if a session has no finished matching result.
        ValueError: if a session has several matching variants.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    wanted = load_wanted_settings(
        args.config,
        base_keypoints=args.base_keypoints,
        additional_keypoints=args.additional_keypoints,
        seed=args.seed,
        anatomical_group=args.anatomical_group,
        eid_group=args.eid_group,
    )
    output_root = Path(wanted['video_preds_dir']) / wanted['output_folder']
    if args.eids is not None:
        eids = args.eids
    elif wanted['eids'] is not None:
        eids = wanted['eids']
    else:
        eids = list_session_eids(output_root)
    if args.only_if_complete:
        unfinished = unfinished_eids(
            wanted['video_preds_dir'], wanted['output_folder'], eids, wanted,
        )
        if unfinished:
            logger.info(
                f'plot skipped: {len(unfinished)} of {len(eids)} sessions have no finished '
                f'matching result yet: {unfinished}',
            )
            return
    results = load_all_sessions(wanted['video_preds_dir'], wanted['output_folder'], eids, wanted)
    stats = collect_stats(results)
    for (cond, model), entry in stats.items():
        logger.info(
            f'{cond} {model.upper()}: BPS {entry["mean"]:.4f} ± {entry["se"]:.4f} '
            f'({entry["n_neurons"]} neurons, {len(results)} sessions)',
        )
    save_path = args.output if args.output is not None else default_output_path(wanted)
    keypoints_path = save_path.with_name(f'{save_path.stem}_keypoints.txt')
    if args.no_show:
        plt.switch_backend('Agg')
    plot_encoding_bars(stats, save_path=save_path, show=not args.no_show)
    write_summary_json(save_path.with_suffix('.json'), stats, results, wanted)
    write_keypoints_txt(keypoints_path, results, wanted, stats)
    logger.info(
        f'wrote {save_path.name} (png/ and pdf/ in {save_path.parent}), '
        f'{save_path.with_suffix(".json")} and {keypoints_path}',
    )


if __name__ == '__main__':
    main()
