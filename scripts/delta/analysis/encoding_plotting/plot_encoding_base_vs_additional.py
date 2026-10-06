"""Bar plot of neural encoding BPS: base keypoints vs base + additional keypoints.

Reads the ``encoding_result.npz`` files written by ``step2_run_encoding.sh``
(``pose2visual.encoding.run_litpose_encoding``) and plots the held-out test bits-per-spike (BPS)
of the RRR and CNN encoders for the two input sets. One bar is the mean over all neurons of all
sessions (non-finite BPS dropped), the error bar is the standard error over those neurons. The
base and additional keypoint names are printed under the axes, so a saved figure says which
keypoints it compares.

Result layout (see ``pose2visual.encoding.litpose_keypoints``)::

    <video_preds_dir>/<output_name>/<eid>/<variant>/encoding_result.npz
    <video_preds_dir>/<output_name>/<eid>/<variant>/encoding_params.json

A session can hold several variants (e.g. a pca and a no-pca run of the same keypoints), and
folder names carry hash suffixes, so the variant is chosen by its ``run_signature`` in
``encoding_params.json``: camera, base / additional keypoints and pca settings. These come from the
same yaml the run used (``configs/encoding/litpose_encoding.yaml``); flags override single entries.
Every session must hold exactly one finished matching variant, otherwise the script raises; use
``--eids`` to restrict the sessions.

Usage::

    PYTHONPATH=src python scripts/delta/analysis/plotting/plot_encoding_base_vs_additional.py \\
      --config configs/encoding/litpose_encoding.yaml --no-pca --no-show

Outputs ``<video_preds_dir>/<output_name>/plots/encoding_base_vs_additional_<...>/`` with
``encoding_barplot.png`` and ``encoding_barplot.json`` (numbers and provenance), unless ``-o`` is
given.
"""

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pose2visual.encoding.litpose_keypoints import session_output_dir

logger = logging.getLogger(__name__)

CONDITIONS = ('base', 'base_plus_additional')
MODELS = ('rrr', 'cnn')
LEGEND_CONDITION_LABELS = {'base': 'Base', 'base_plus_additional': 'Base + additional'}
# base hues of the old figure-3 plot: grey for the reference, orange for the alternative
CONDITION_COLORS = {'base': '#616161', 'base_plus_additional': '#F28E2B'}
_REQUIRED_CONFIG_KEYS = (
    'video_preds_dir',
    'output_name',
    'camera',
    'base_keypoints',
    'additional_keypoints',
)
_PCA_N_COMPONENTS_KEYS = ('pca_n_components_base', 'pca_n_components_base_plus_additional')
# folder of `output_name` that holds the plots, never a session
_PLOTS_DIRNAME = 'plots'
_KEYPOINT_TEXT_WIDTH = 46


@dataclass
class SessionResult:
    """Per-neuron test BPS of one session and the keypoints it was fitted with.

    Attributes:
        eid: session id.
        variant_dir: folder holding the `encoding_result.npz` that was read.
        bps: `(condition, model)` -> per-neuron BPS array of shape `[n_neurons]`.
        base_keypoints: base keypoint names as stored in the result.
        additional_keypoints: additional keypoint names as stored in the result.
    """

    eid: str
    variant_dir: Path
    bps: dict[tuple[str, str], np.ndarray]
    base_keypoints: list[str]
    additional_keypoints: list[str]


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
        '--pca',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='select the pca / no-pca run (default: `pca` of the config, true if absent)',
    )
    parser.add_argument(
        '--seed', type=int, default=None, help='only match runs of this seed (default: any)',
    )
    parser.add_argument(
        '--eids', nargs='+', default=None, help='sessions to plot (default: all with results)',
    )
    parser.add_argument(
        '-o', '--output', type=Path, default=None, help='output png path (default: see docstring)',
    )
    parser.add_argument(
        '--no-show', action='store_true', help='save the figure without showing it',
    )
    return parser.parse_args(argv)


def load_wanted_settings(
    config_path: str | Path,
    base_keypoints: list[str] | None = None,
    additional_keypoints: list[str] | None = None,
    pca: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Read the settings that identify the wanted run from the encoding yaml.

    Args:
        config_path: path of the yaml the encoding was run with.
        base_keypoints: replaces `base_keypoints` of the file when given.
        additional_keypoints: replaces `additional_keypoints` of the file when given.
        pca: replaces `pca` of the file when given.
        seed: only runs of this seed match; `None` matches any seed.

    Returns:
        Dict with `video_preds_dir`, `output_name`, `camera`, `base_keypoints`,
        `additional_keypoints`, `pca`, `pca_n_components_base`,
        `pca_n_components_base_plus_additional` and `seed`.

    Raises:
        ValueError: if the yaml is not a mapping or a required key is missing.
    """
    with Path(config_path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'{config_path}: expected a yaml mapping')
    missing = [key for key in _REQUIRED_CONFIG_KEYS if cfg.get(key) is None]
    if missing:
        raise ValueError(f'{config_path}: missing config keys {missing}')
    wanted: dict[str, Any] = {
        'video_preds_dir': str(cfg['video_preds_dir']),
        'output_name': str(cfg['output_name']),
        'camera': str(cfg['camera']),
        'base_keypoints': [str(name) for name in (base_keypoints or cfg['base_keypoints'])],
        'additional_keypoints': [
            str(name) for name in (additional_keypoints or cfg['additional_keypoints'])
        ],
        'pca': cfg.get('pca', True) if pca is None else pca,
        'seed': seed,
    }
    for key in _PCA_N_COMPONENTS_KEYS:
        wanted[key] = cfg.get(key)
    return wanted


def signature_matches(signature: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Whether the `run_signature` of a finished run is the wanted run.

    Keypoint lists are compared as sets of names. The pca component counts only matter when pca
    is on, and the seed only when `wanted['seed']` is set.

    Args:
        signature: `run_signature` stored in `encoding_params.json`.
        wanted: output of `load_wanted_settings`.

    Returns:
        `True` if camera, keypoints, pca settings (and seed) agree.
    """
    for key in ('base_keypoints', 'additional_keypoints'):
        if sorted(signature.get(key, [])) != sorted(wanted[key]):
            return False
    if signature.get('camera') != wanted['camera'] or signature.get('pca') != wanted['pca']:
        return False
    if wanted['pca']:
        for key in _PCA_N_COMPONENTS_KEYS:
            if signature.get(key) != wanted[key]:
                return False
    return wanted['seed'] is None or signature.get('seed') == wanted['seed']


def find_variant_dir(session_dir: Path, wanted: dict[str, Any]) -> Path:
    """Find the finished variant folder of one session that matches the wanted run.

    Args:
        session_dir: `<video_preds_dir>/<output_name>/<eid>`.
        wanted: output of `load_wanted_settings`.

    Returns:
        The variant folder holding `encoding_result.npz`.

    Raises:
        FileNotFoundError: if no variant matches, or the matching one has no
            `encoding_result.npz` yet (the run is unfinished or failed).
        ValueError: if more than one variant matches.
    """
    matches = []
    for params_path in sorted(session_dir.glob('*/encoding_params.json')):
        try:
            with params_path.open(encoding='utf-8') as f:
                signature = json.load(f).get('run_signature')
        except (OSError, ValueError):
            continue
        # results from before run_signature existed cannot be identified
        if isinstance(signature, dict) and signature_matches(signature, wanted):
            matches.append(params_path.parent)
    if not matches:
        raise FileNotFoundError(f'{session_dir}: no variant matches the wanted settings')
    if len(matches) > 1:
        raise ValueError(
            f'{session_dir}: {len(matches)} variants match the wanted settings: '
            f'{[path.name for path in matches]}',
        )
    if not (matches[0] / 'encoding_result.npz').is_file():
        raise FileNotFoundError(f'{matches[0]}: matching run has no encoding_result.npz yet')
    return matches[0]


def load_session_result(eid: str, variant_path: Path) -> SessionResult:
    """Load the per-neuron test BPS and the keypoint names of one finished variant.

    Args:
        eid: session id.
        variant_path: folder holding `encoding_result.npz`.

    Returns:
        The `SessionResult` of the session.
    """
    data = np.load(variant_path / 'encoding_result.npz', allow_pickle=False)
    bps = {
        (cond, model): np.asarray(data[f'{cond}_{model}_bps_per_neuron'], dtype=np.float64)
        for cond in CONDITIONS
        for model in MODELS
    }
    return SessionResult(
        eid=eid,
        variant_dir=variant_path,
        bps=bps,
        base_keypoints=[str(name) for name in data['base_keypoints']],
        additional_keypoints=[str(name) for name in data['additional_keypoints']],
    )


def list_session_eids(output_root: Path) -> list[str]:
    """List the sessions of an `output_name` folder.

    Args:
        output_root: `<video_preds_dir>/<output_name>`.

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
    output_name: str,
    eids: list[str],
    wanted: dict[str, Any],
) -> list[SessionResult]:
    """Load the matching finished result of every session; all of them must exist.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: user-chosen folder name.
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
        session_dir = session_output_dir(video_preds_dir, output_name, eid)
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
    """Check that all sessions were fitted with the same base and additional keypoints.

    Args:
        results: sessions to compare.

    Raises:
        ValueError: if `results` is empty or the keypoint lists differ between sessions.
    """
    if not results:
        raise ValueError('no sessions to plot')
    first = results[0]
    for result in results[1:]:
        if (
            sorted(result.base_keypoints) != sorted(first.base_keypoints)
            or sorted(result.additional_keypoints) != sorted(first.additional_keypoints)
        ):
            raise ValueError(
                f'keypoints of {result.eid} differ from {first.eid}: '
                f'{result.base_keypoints} + {result.additional_keypoints} vs '
                f'{first.base_keypoints} + {first.additional_keypoints}',
            )


def collect_stats(
    results: list[SessionResult],
) -> dict[tuple[str, str], dict[str, float]]:
    """Pool the finite per-neuron BPS of all sessions and take mean and standard error.

    Args:
        results: sessions with identical keypoints (`check_same_keypoints` is applied).

    Returns:
        `(condition, model)` -> dict with `mean`, `se` (std with ddof=1 over sqrt of the number
        of neurons, 0 for a single neuron) and `n_neurons`; `mean` and `se` are NaN without any
        finite value.

    Raises:
        ValueError: if `results` is empty or the keypoints differ between sessions.
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


def keypoint_text(
    base_keypoints: list[str],
    additional_keypoints: list[str],
    camera: str,
    pca: bool,
    n_sessions: int,
    stats: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Text printed under the axes: the compared keypoints and the data behind the bars.

    Args:
        base_keypoints: base keypoint names.
        additional_keypoints: additional keypoint names.
        camera: camera of the keypoints.
        pca: whether the keypoint speeds were z-scored and PCA-rotated.
        n_sessions: number of pooled sessions.
        stats: output of `collect_stats`.

    Returns:
        Wrapped multi-line text.
    """
    counts = [int(entry['n_neurons']) for entry in stats.values()]
    n_neurons = str(counts[0]) if min(counts) == max(counts) else f'{min(counts)}-{max(counts)}'
    lines = []
    for title, names in (('Base', base_keypoints), ('Additional', additional_keypoints)):
        lines.append(textwrap.fill(
            f'{title}: {", ".join(names)}',
            width=_KEYPOINT_TEXT_WIDTH,
            subsequent_indent='    ',
        ))
    lines.append(f'{camera} camera, pca {"on" if pca else "off"}, {n_sessions} sessions')
    lines.append(f'{n_neurons} neurons; mean ± SE over neurons')
    return '\n'.join(lines)


def plot_encoding_bars(
    stats: dict[tuple[str, str], dict[str, float]],
    text: str,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw BPS bars per encoder (RRR, CNN), base next to base + additional.

    Args:
        stats: output of `collect_stats`.
        text: block printed under the axes (see `keypoint_text`).
        save_path: png path; the figure is not saved when `None`.
        show: show the figure, otherwise close it.

    Returns:
        The figure.
    """
    width = 0.38
    bar_center_spacing = 1.0
    edge_padding = 0.7
    x_centers = np.arange(len(MODELS), dtype=np.float64) * bar_center_spacing
    fig, (ax, ax_text) = plt.subplots(
        2,
        1,
        figsize=(3.4, 4.0),
        dpi=200,
        gridspec_kw={'height_ratios': [3.0, 0.9]},
    )

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

    ax_text.axis('off')
    ax_text.text(0.0, 1.0, text, ha='left', va='top', fontsize=6.5, linespacing=1.4)
    fig.tight_layout(pad=0.4)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def default_output_path(wanted: dict[str, Any]) -> Path:
    """Default png path under the `output_name` folder of the results.

    Args:
        wanted: output of `load_wanted_settings`.

    Returns:
        `<video_preds_dir>/<output_name>/plots/encoding_base_vs_additional_<additional>_<pca>/
        encoding_barplot.png`.
    """
    additional = '_'.join(sorted(wanted['additional_keypoints']))[:60]
    pca = 'pca' if wanted['pca'] else 'nopca'
    return (
        Path(wanted['video_preds_dir'])
        / wanted['output_name']
        / _PLOTS_DIRNAME
        / f'encoding_base_vs_additional_{additional}_{pca}'
        / 'encoding_barplot.png'
    )


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
        'base_keypoints': results[0].base_keypoints,
        'additional_keypoints': results[0].additional_keypoints,
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
        ValueError: if a session has several matching variants or the keypoints differ.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    wanted = load_wanted_settings(
        args.config,
        base_keypoints=args.base_keypoints,
        additional_keypoints=args.additional_keypoints,
        pca=args.pca,
        seed=args.seed,
    )
    output_root = Path(wanted['video_preds_dir']) / wanted['output_name']
    eids = args.eids if args.eids is not None else list_session_eids(output_root)
    results = load_all_sessions(wanted['video_preds_dir'], wanted['output_name'], eids, wanted)
    stats = collect_stats(results)
    for (cond, model), entry in stats.items():
        logger.info(
            f'{cond} {model.upper()}: BPS {entry["mean"]:.4f} ± {entry["se"]:.4f} '
            f'({entry["n_neurons"]} neurons, {len(results)} sessions)',
        )
    save_path = args.output if args.output is not None else default_output_path(wanted)
    if args.no_show:
        plt.switch_backend('Agg')
    text = keypoint_text(
        results[0].base_keypoints,
        results[0].additional_keypoints,
        wanted['camera'],
        wanted['pca'],
        len(results),
        stats,
    )
    plot_encoding_bars(stats, text, save_path=save_path, show=not args.no_show)
    write_summary_json(save_path.with_suffix('.json'), stats, results, wanted)
    logger.info(f'wrote {save_path} and {save_path.with_suffix(".json")}')


if __name__ == '__main__':
    main()
