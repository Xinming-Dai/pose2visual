"""Bar plot of neural encoding BPS: base keypoints vs base + additional keypoints.

Reads the ``encoding_result.npz`` files written by ``step2_run_encoding.sh``
(``pose2visual.encoding.run_litpose_encoding``) and plots the held-out test bits-per-spike (BPS)
of the RRR and CNN encoders for the two input sets. One bar is the mean over all neurons of all
sessions (non-finite BPS dropped), the error bar is the standard error over those neurons. The
base and additional keypoint names are printed under the axes, so a saved figure says which
keypoints it compares.

With ``anatomical_group`` set to a group set of ``body_parts_json`` (e.g.
``nonoverlapping_analysis_groups``), the two keypoint lists of the config hold group names such as
``eye_related``; the figure then prints the group names (the keypoints a group resolved to can
differ between sessions, they are listed per session in the json next to the figure).

Result layout (see ``pose2visual.encoding.litpose_keypoints``)::

    <video_preds_dir>/<output_name>/<eid>/<variant>/encoding_result.npz
    <video_preds_dir>/<output_name>/<eid>/<variant>/encoding_params.json

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pose2visual.encoding.litpose_keypoints import (
    ENCODING_YAML_NESTED_NAMES,
    load_encoding_yaml,
    resolve_group_eids,
    session_output_dir,
)

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
_DEFAULT_PCA_GROUP_SET = 'nonoverlapping_analysis_groups'
# `anatomical_group` value meaning "the keypoint lists hold keypoint names"
_ANATOMICAL_GROUP_ALL = 'all'
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


def load_wanted_settings(
    config_path: str | Path,
    base_keypoints: list[str] | None = None,
    additional_keypoints: list[str] | None = None,
    seed: int | None = None,
    anatomical_group: str | None = None,
    eid_group: str | None = None,
) -> dict[str, Any]:
    """Read the settings that identify the wanted run from the encoding yaml.

    Args:
        config_path: path of the yaml the encoding was run with.
        base_keypoints: replaces `base_keypoints` of the file when given.
        additional_keypoints: replaces `additional_keypoints` of the file when given.
        seed: only runs of this seed match; `None` matches any seed.
        anatomical_group: replaces `anatomical_group` of the file when given.
        eid_group: replaces `session.eid_group` of the file when given.

    Returns:
        Dict with `video_preds_dir`, `output_name`, `camera`, `base_keypoints`,
        `additional_keypoints` (group names when `anatomical_group` is a group set),
        `anatomical_group` (default `all`), `pca`, `pca_group_set`, `step_order`, `seed` and `eids`
        (the sessions of the eid group, `None` when there is no group).

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), a required key is
            missing, or an eid group is set without `paths.eid_groups_json` or is unknown.
    """
    cfg = load_encoding_yaml(config_path)
    missing = [
        ENCODING_YAML_NESTED_NAMES[key] for key in _REQUIRED_CONFIG_KEYS if cfg.get(key) is None
    ]
    if missing:
        raise ValueError(f'{config_path}: missing config keys {missing}')
    group = eid_group if eid_group is not None else cfg.get('eid_group')
    eids = None
    if group is not None:
        if cfg.get('eid_groups_json') is None:
            raise ValueError(f'{config_path}: an eid group needs paths.eid_groups_json')
        eids = resolve_group_eids(cfg['eid_groups_json'], str(group))
    wanted: dict[str, Any] = {
        'video_preds_dir': str(cfg['video_preds_dir']),
        'output_name': str(cfg['output_name']),
        'camera': str(cfg['camera']),
        'base_keypoints': [str(name) for name in (base_keypoints or cfg['base_keypoints'])],
        'additional_keypoints': [
            str(name) for name in (additional_keypoints or cfg['additional_keypoints'])
        ],
        'anatomical_group': str(
            cfg.get('anatomical_group', _ANATOMICAL_GROUP_ALL)
            if anatomical_group is None else anatomical_group,
        ),
        'pca': cfg.get('pca', False),
        'pca_group_set': str(cfg.get('pca_group_set', _DEFAULT_PCA_GROUP_SET)),
        'step_order': str(cfg.get('step_order', 'unknown')),
        'seed': seed,
        'eids': eids,
    }
    return wanted


def signature_matches(signature: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Whether the `run_signature` of a finished run is the wanted run.

    Keypoint lists (group names in group mode) are compared as sets of names, together with the
    anatomical group; a signature without one is an `all` run. The pca group set only matters
    when pca is on, and the seed only when `wanted['seed']` is set.

    Args:
        signature: `run_signature` stored in `encoding_params.json`.
        wanted: output of `load_wanted_settings`.

    Returns:
        `True` if camera, keypoints, anatomical group, pca settings (and seed) agree.
    """
    for key in ('base_keypoints', 'additional_keypoints'):
        if sorted(signature.get(key, [])) != sorted(wanted[key]):
            return False
    if signature.get('camera') != wanted['camera'] or signature.get('pca') != wanted['pca']:
        return False
    if signature.get('anatomical_group', _ANATOMICAL_GROUP_ALL) != wanted['anatomical_group']:
        return False
    if wanted['pca'] and signature.get('pca_group_set') != wanted['pca_group_set']:
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


def unfinished_eids(
    video_preds_dir: str | Path,
    output_name: str,
    eids: list[str],
    wanted: dict[str, Any],
) -> list[str]:
    """List the sessions that have no finished matching result yet.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: user-chosen folder name.
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
            find_variant_dir(session_output_dir(video_preds_dir, output_name, eid), wanted)
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
    data = np.load(variant_path / 'encoding_result.npz', allow_pickle=False)
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

    In group mode the usable keypoints a group resolves to differ between sessions, so the group
    names are compared instead of the keypoints.

    Args:
        results: sessions to compare.

    Raises:
        ValueError: if `results` is empty or the keypoints (groups) differ between sessions.
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
            raise ValueError(
                f'keypoints of {result.eid} differ from {first.eid}: '
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
        results: sessions with identical keypoints or groups (`check_same_keypoints` is applied).

    Returns:
        `(condition, model)` -> dict with `mean`, `se` (std with ddof=1 over sqrt of the number
        of neurons, 0 for a single neuron) and `n_neurons`; `mean` and `se` are NaN without any
        finite value.

    Raises:
        ValueError: if `results` is empty or the keypoints (groups) differ between sessions.
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
    step_order: str,
    n_sessions: int,
    stats: dict[tuple[str, str], dict[str, float]],
) -> str:
    """Text printed under the axes: the compared keypoints and the data behind the bars.

    Args:
        base_keypoints: base keypoint names, or group names in group mode.
        additional_keypoints: additional keypoint names, or group names in group mode.
        camera: camera of the keypoints.
        step_order: step pipeline name that encodes preprocessing (e.g., ensemble_median_01 with PCA).
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
    lines.append(f'{camera} camera, {step_order}, {n_sessions} sessions')
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
        encoding_barplot.png`; `<additional>` is prefixed by the group set in group mode, so the
        same group names of two group sets do not share a folder.
    """
    additional = '_'.join(sorted(wanted['additional_keypoints']))[:60]
    if wanted['anatomical_group'] != _ANATOMICAL_GROUP_ALL:
        additional = f'{wanted["anatomical_group"]}_{additional}'
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
        ValueError: if a session has several matching variants or the keypoints differ.
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
    output_root = Path(wanted['video_preds_dir']) / wanted['output_name']
    if args.eids is not None:
        eids = args.eids
    elif wanted['eids'] is not None:
        eids = wanted['eids']
    else:
        eids = list_session_eids(output_root)
    if args.only_if_complete:
        unfinished = unfinished_eids(
            wanted['video_preds_dir'], wanted['output_name'], eids, wanted,
        )
        if unfinished:
            logger.info(
                f'plot skipped: {len(unfinished)} of {len(eids)} sessions have no finished '
                f'matching result yet: {unfinished}',
            )
            return
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
        results[0].base_groups or results[0].base_keypoints,
        results[0].additional_groups or results[0].additional_keypoints,
        wanted['camera'],
        wanted['step_order'],
        len(results),
        stats,
    )
    plot_encoding_bars(stats, text, save_path=save_path, show=not args.no_show)
    write_summary_json(save_path.with_suffix('.json'), stats, results, wanted)
    logger.info(f'wrote {save_path} and {save_path.with_suffix(".json")}')


if __name__ == '__main__':
    main()
