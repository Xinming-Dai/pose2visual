"""Compare the litpose encoding ("Dense keypoints") with the SABLE zero-shot encoding methods.

The other methods are the results in ``--results-dir`` (default
``/projects/bfsr/xdai3/project3d/iclr_plotting/ibl/SABLE_zero_shot_encoding``), one pickled dict
per method and session, ``<results-dir>/<method>/<eid>/encoding_results_*.npy``, holding the
held-out test spikes ``gt`` and predicted rates ``pred`` of an RRR and a CNN encoder:

* ``random_baseline``: DINO features with permuted frames (Random);
* ``keypoints``: the IBL ``dynamic_vars`` baseline, i.e. keypoint speeds plus the behavior traces
  of the SABLE baseline (identical to the result in ``twoview3d_ckpts/dynamic_vars``);
* ``pca``, ``resnet``, ``beast``: the PCA, ResNet autoencoder and BEAST features;
* ``sable_dino``: SABLE (shown as "SABLE").

The litpose run is found from the encoding yaml exactly as in
``pose2visual.encoding.litpose_results`` (``encoding_predictions.npz`` of the session's variant
folder).

All of them read the same ``<eid>_aligned.npz`` (same trials, split and units), which is checked:
the test ``gt`` of every method must be identical to the litpose one, otherwise the run fails. The
per-neuron bits per spike (BPS) of every method is then recomputed with the same function
(``pose2visual.encoding.utils.bits_per_spike``), so the comparison does not depend on how each
pipeline averaged its own number. The stored R^2 is reported as it is.

The figures follow the SABLE ``encoding_barplot``: ``bps_bars.png`` is one panel with a pair of
bars per method (RRR dark, CNN lightened) holding the mean per-neuron BPS pooled over the
sessions, with the standard error over the pooled neurons; ``bps_scatter.png`` is one panel with
the per-neuron BPS of "Dense keypoints" on x and of every other method on y (one encoder,
``--scatter-encoder``).

A session that lacks the litpose run or the result of any selected method is skipped with a
warning, so every method is pooled over the same sessions. Every result file is about 230 MB, so
run this on a compute node. Loading them needs the numpy version they were saved with (the
``beast`` conda env).

Usage::

    PYTHONPATH=src python scripts/delta/lightningpose_encoding/compare_with_sable_baseline.py \\
      --config configs/encoding/litpose_encoding.yaml --no-show

Without ``--eids`` the sessions are the ``sable`` group of ``paths.eid_groups_json``. Outputs
``comparison.json``, ``comparison.csv`` (one row per session, method and encoder),
``comparison_pooled.csv`` (one row per method and encoder), ``bps_bars.png`` and
``bps_scatter.png`` in ``--output-dir`` (default: the ``plots/compare_with_sable_baseline`` folder
of the run).
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pose2visual.encoding.litpose_keypoints import (
    load_encoding_yaml,
    resolve_group_eids,
    session_output_dir,
)
from pose2visual.encoding.litpose_results import (
    CONDITIONS,
    MODELS,
    RESULT_NAME,
    find_variant_dir,
    load_session_predictions,
    load_wanted_settings,
)
from pose2visual.encoding.utils import bits_per_spike

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(
    '/projects/bfsr/xdai3/project3d/iclr_plotting/ibl/SABLE_zero_shot_encoding',
)
DEFAULT_EID_GROUP = 'sable'
# method folder of `--results-dir` -> (label, base color); order, labels and colors are those of
# DEFAULT_METHODS / DEFAULT_METHOD_LABELS / ENCODING_BASE_COLORS in the SABLE `plot_helpers.py`
METHODS = {
    'random_baseline': ('Random', '#616161'),
    'keypoints': ('Keypoints', '#F8766D'),
    'pca': ('PCA', '#7FCDBB'),
    'resnet': ('ResNet AE', '#4DAF4A'),
    'beast': ('BEAST', '#9E9E9E'),
    'sable_dino': ('SABLE', '#F28E2B'),
}
# the litpose run is shown as "Dense keypoints"; the orange of SABLE is taken, so it is blue
LITPOSE = 'litpose'
LITPOSE_LABEL = 'Dense keypoints'
LITPOSE_COLOR = '#4E79A7'
# same floor as the encoders use before taking the log of the predicted rates
BPS_RATE_THRESHOLD = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Compare the litpose encoding with the SABLE zero-shot encoding methods.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='yaml config the litpose encoding was run with (litpose_encoding.yaml)',
    )
    parser.add_argument(
        '--eids',
        nargs='+',
        default=None,
        help=f'sessions to compare (default: the {DEFAULT_EID_GROUP!r} group of the config)',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help='folder holding <method>/<eid>/encoding_results*.npy (default: %(default)s)',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=list(METHODS),
        default=list(METHODS),
        help='methods to compare, in plot order (default: all of %(choices)s)',
    )
    parser.add_argument(
        '--condition',
        choices=CONDITIONS,
        default=CONDITIONS[0],
        help='litpose condition to compare (default: %(default)s, the only one of a base-only '
        'run)',
    )
    parser.add_argument(
        '--scatter-encoder',
        choices=MODELS,
        default='cnn',
        help='encoder shown in the scatter plot (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='folder for the figures and tables (default: the plots folder of the run)',
    )
    parser.add_argument('--no-show', action='store_true', help='do not open the figures')
    return parser.parse_args(argv)


def method_label(name: str) -> str:
    """Label of a method in the figures and tables.

    Args:
        name: key of `METHODS`, or `LITPOSE`.

    Returns:
        The label, `Dense keypoints` for the litpose run.
    """
    return LITPOSE_LABEL if name == LITPOSE else METHODS[name][0]


def method_color(name: str) -> str:
    """Base color of a method in the figures.

    Args:
        name: key of `METHODS`, or `LITPOSE`.

    Returns:
        Hex color, the one used for the RRR bar.
    """
    return LITPOSE_COLOR if name == LITPOSE else METHODS[name][1]


def lightened_hex(hex_color: str) -> str:
    """Lighten a hex color, as the SABLE bar plots do to tell a CNN bar from its RRR bar.

    Args:
        hex_color: color to lighten.

    Returns:
        The lightened hex color.
    """
    return mcolors.to_hex(tuple(min(1.0, 0.52 + 0.48 * c) for c in mcolors.to_rgb(hex_color)))


def find_method_file(results_dir: Path, method: str, eid: str) -> Path:
    """Find the result file of one method and session.

    Args:
        results_dir: folder holding `<method>/<eid>/encoding_results*.npy`.
        method: key of `METHODS`.
        eid: session id.

    Returns:
        Path of the single `encoding_results*.npy` of the session.

    Raises:
        FileNotFoundError: if the session folder holds no such file.
        ValueError: if it holds more than one.
    """
    matches = sorted((results_dir / method / eid).glob('encoding_results*.npy'))
    if not matches:
        raise FileNotFoundError(f'no encoding_results*.npy in {results_dir / method / eid}')
    if len(matches) > 1:
        raise ValueError(
            f'expected one encoding_results*.npy in {results_dir / method / eid}, found '
            f'{[path.name for path in matches]}',
        )
    return matches[0]


def load_method_results(path: Path, eid: str) -> dict[str, dict[str, Any]]:
    """Load the test spikes, predicted rates and stored scores of both encoders of a method.

    The file is unpickled once; it is about 230 MB.

    Args:
        path: result file of one method and session.
        eid: session id, the key of the pickled dict.

    Returns:
        Encoder name (`'rrr'`, `'cnn'`) -> dict with `gt` and `pred` (float64,
        `[n_trials, n_bins, n_neurons]`) and the stored `bps` and `r2` scalars.

    Raises:
        KeyError: if the file holds no result for the session or an encoder.
    """
    results = np.load(path, allow_pickle=True).item()[eid]
    return {
        model: {
            'gt': np.asarray(results[model]['gt'], dtype=np.float64),
            'pred': np.asarray(results[model]['pred'], dtype=np.float64),
            'bps': float(results[model]['bps']),
            'r2': float(results[model]['r2']),
        }
        for model in MODELS
    }


def load_litpose(variant_path: Path, condition: str, model: str) -> dict[str, Any]:
    """Load the test spikes, predicted rates and stored scores of one litpose encoder.

    Args:
        variant_path: finished variant folder (`litpose_results.find_variant_dir`).
        condition: `'base'` or `'base_plus_additional'`.
        model: `'rrr'` or `'cnn'`.

    Returns:
        Same layout as one encoder of `load_method_results`.

    Raises:
        FileNotFoundError: if the variant has no `encoding_predictions.npz`.
    """
    gt, pred = load_session_predictions(variant_path, condition, model)
    with np.load(variant_path / RESULT_NAME, allow_pickle=False) as data:
        bps = float(data[f'{condition}_{model}_bps'])
        r2 = float(data[f'{condition}_{model}_r2'])
    return {'gt': gt, 'pred': pred, 'bps': bps, 'r2': r2}


def per_neuron_bps(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Test BPS of every neuron, as the encoders compute it.

    Args:
        gt: test spike counts, `[n_trials, n_bins, n_neurons]`.
        pred: predicted rates, same shape.

    Returns:
        Array of shape `[n_neurons]`; infinite BPS becomes NaN.
    """
    bps = np.array([
        bits_per_spike(pred[:, :, [idx]], gt[:, :, [idx]], threshold=BPS_RATE_THRESHOLD)
        for idx in range(gt.shape[-1])
    ])
    bps[np.isinf(bps)] = np.nan
    return bps


def check_same_test_set(
    gt_litpose: np.ndarray,
    gt_method: np.ndarray,
    eid: str,
    method: str,
) -> None:
    """Check that a method was scored on the same trials and units as the litpose run.

    Args:
        gt_litpose: test spike counts of the litpose run.
        gt_method: test spike counts of the method.
        eid: session id, for the error message.
        method: label of the method, for the error message.

    Raises:
        ValueError: if the shapes or the counts differ, which means the `<eid>_aligned.npz` (trial
            split, sampled windows or unit selection) changed between the two runs.
    """
    if gt_litpose.shape != gt_method.shape:
        raise ValueError(
            f'{eid}: test spikes have shape {gt_method.shape} for {method} but '
            f'{gt_litpose.shape} for the litpose run; the two used other aligned npz files',
        )
    if not np.array_equal(gt_litpose, gt_method):
        raise ValueError(
            f'{eid}: test spikes of {method} differ from those of the litpose run; the two used '
            'other aligned npz files',
        )


def _score(
    eid: str,
    name: str,
    model: str,
    loaded: dict[str, Any],
) -> dict[str, Any]:
    """Per-neuron BPS and summary of one encoder, checked against its stored BPS.

    Args:
        eid: session id, for the log.
        name: key of `METHODS`, or `LITPOSE`.
        model: `'rrr'` or `'cnn'`.
        loaded: one encoder of `load_method_results` / `load_litpose`.

    Returns:
        Dict with `bps` (per neuron), `mean_bps`, `n_neurons` and the stored `r2`.
    """
    bps = per_neuron_bps(loaded['gt'], loaded['pred'])
    mean_bps = float(np.nanmean(bps)) if np.isfinite(bps).any() else float('nan')
    gap = abs(mean_bps - loaded['bps'])
    if gap > 1e-6:
        logger.warning(
            f'{eid} {method_label(name)} {model}: recomputed mean BPS differs from the stored '
            f'one by {gap:.2e}',
        )
    return {
        'bps': bps,
        'mean_bps': mean_bps,
        'n_neurons': int(bps.size),
        'r2': loaded['r2'],
    }


def compare_session(
    eid: str,
    results_dir: Path,
    methods: list[str],
    variant_path: Path,
    condition: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score the litpose run and every method of one session, for both encoders.

    Args:
        eid: session id.
        results_dir: folder holding `<method>/<eid>/encoding_results*.npy`.
        methods: keys of `METHODS` to compare.
        variant_path: finished litpose variant folder of the session.
        condition: litpose condition to compare.

    Returns:
        Name (`LITPOSE` or a method) -> encoder -> dict with `bps` (per neuron), `mean_bps`,
        `n_neurons` and the stored `r2`. The entries of the methods also hold `delta_mean_bps`
        (litpose minus the method) and `frac_improved` (fraction of neurons where litpose is
        better).

    Raises:
        FileNotFoundError: if the litpose run or the file of any method is missing; nothing is
            loaded then.
        ValueError: if a method was not scored on the same test set as the litpose run.
    """
    files = {method: find_method_file(results_dir, method, eid) for method in methods}
    litpose = {model: load_litpose(variant_path, condition, model) for model in MODELS}
    out: dict[str, dict[str, dict[str, Any]]] = {
        LITPOSE: {model: _score(eid, LITPOSE, model, litpose[model]) for model in MODELS},
    }
    for method in methods:
        loaded = load_method_results(files[method], eid)
        out[method] = {}
        for model in MODELS:
            check_same_test_set(
                litpose[model]['gt'], loaded[model]['gt'], eid, method_label(method),
            )
            entry = _score(eid, method, model, loaded[model])
            delta = out[LITPOSE][model]['bps'] - entry['bps']
            finite = delta[np.isfinite(delta)]
            entry['delta_mean_bps'] = out[LITPOSE][model]['mean_bps'] - entry['mean_bps']
            entry['frac_improved'] = float(np.mean(finite > 0)) if finite.size else float('nan')
            out[method][model] = entry
        # drop the 230 MB of this method before the next file is read
        del loaded
    return out


def pooled_stats(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    name: str,
    model: str,
) -> dict[str, float | int]:
    """Mean per-neuron BPS of a method pooled over the sessions, as in the SABLE bar plot.

    Args:
        results: eid -> output of `compare_session`.
        name: `LITPOSE` or a key of `METHODS`.
        model: `'rrr'` or `'cnn'`.

    Returns:
        Dict with `mean`, the standard error `se` (`std / sqrt(n)` over the pooled neurons, 0 for a
        single neuron), `n_neurons` (finite values only) and `n_sessions`.
    """
    values = np.concatenate([session[name][model]['bps'] for session in results.values()])
    values = values[np.isfinite(values)]
    return {
        'mean': float(np.mean(values)) if values.size else float('nan'),
        'se': float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
        'n_neurons': int(values.size),
        'n_sessions': len(results),
    }


def summary_rows(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    names: list[str],
) -> list[dict[str, Any]]:
    """Flatten the per-session comparison into table rows.

    Args:
        results: eid -> output of `compare_session`.
        names: `LITPOSE` and the methods, in table order.

    Returns:
        One row per session, name and encoder without the per-neuron arrays. `delta_mean_bps` and
        `frac_improved` (litpose compared with the method) are `None` for the litpose rows.
    """
    return [
        {
            'eid': eid,
            'method': name,
            'label': method_label(name),
            'model': model,
            'n_neurons': session[name][model]['n_neurons'],
            'mean_bps': session[name][model]['mean_bps'],
            'delta_mean_bps': session[name][model].get('delta_mean_bps'),
            'frac_improved': session[name][model].get('frac_improved'),
            'r2': session[name][model]['r2'],
        }
        for eid, session in results.items()
        for name in names
        for model in MODELS
    ]


def pooled_rows(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    names: list[str],
) -> list[dict[str, Any]]:
    """One table row per name and encoder with the BPS pooled over the sessions.

    Args:
        results: eid -> output of `compare_session`.
        names: `LITPOSE` and the methods, in table order.

    Returns:
        Rows with `method`, `label`, `model` and the output of `pooled_stats`.
    """
    return [
        {
            'method': name,
            'label': method_label(name),
            'model': model,
            **pooled_stats(results, name, model),
        }
        for name in names
        for model in MODELS
    ]


def write_tables(
    rows: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write `comparison.json`, `comparison.csv` and `comparison_pooled.csv`.

    Args:
        rows: output of `summary_rows`.
        pooled: output of `pooled_rows`.
        output_dir: folder to write into (created if missing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'comparison.json').write_text(
        json.dumps({'per_session': rows, 'pooled': pooled}, indent=2),
    )
    for name, table in (('comparison.csv', rows), ('comparison_pooled.csv', pooled)):
        with (output_dir / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def _finish_figure(fig: plt.Figure, save_path: Path, show: bool) -> None:
    """Save a figure as png and show or close it.

    Args:
        fig: the figure.
        save_path: png path; its folder is created if missing.
        show: show the figure, otherwise close it.
    """
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_bars(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    methods: list[str],
    save_path: Path,
    show: bool,
) -> plt.Figure:
    """One panel with a pair of bars (RRR dark, CNN lightened) per method and "Dense keypoints".

    Args:
        results: eid -> output of `compare_session`.
        methods: keys of `METHODS` in plot order; "Dense keypoints" comes last.
        save_path: png path.
        show: show the figure, otherwise close it.

    Returns:
        The figure.
    """
    names = [*methods, LITPOSE]
    positions = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(names) + 1.5), 4.0))
    for idx_model, model in enumerate(MODELS):
        offset = (idx_model - (len(MODELS) - 1) / 2) * width
        stats = [pooled_stats(results, name, model) for name in names]
        colors = [
            method_color(name) if model == 'rrr' else lightened_hex(method_color(name))
            for name in names
        ]
        means = np.array([entry['mean'] for entry in stats])
        errors = np.array([entry['se'] for entry in stats])
        ax.bar(positions + offset, means, width, yerr=errors, color=colors, capsize=2)
        for position, mean, error in zip(positions + offset, means, errors):
            ax.text(
                position, mean + error, f'{mean:.3f}',
                ha='center', va='bottom', fontsize='x-small',
            )
    ax.set_xticks(positions)
    ax.set_xticklabels([method_label(name) for name in names], rotation=30, ha='right')
    ax.set_ylabel('mean test BPS')
    ax.set_title(f'Test BPS pooled over {len(results)} session(s)')
    ax.legend(
        handles=[
            mpatches.Patch(color='#595959', label='RRR'),
            mpatches.Patch(color=lightened_hex('#595959'), label='CNN'),
        ],
        frameon=False,
    )
    _finish_figure(fig, save_path, show)
    return fig


def plot_scatter(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    methods: list[str],
    encoder: str,
    save_path: Path,
    show: bool,
) -> plt.Figure:
    """One panel scattering every method's per-neuron BPS against "Dense keypoints".

    Neurons of all sessions are pooled; a neuron with a NaN BPS in either method is left out of
    that method's points. Points above the diagonal are neurons where the method beats "Dense
    keypoints".

    Args:
        results: eid -> output of `compare_session`.
        methods: keys of `METHODS` in plot order.
        encoder: `'rrr'` or `'cnn'`, the encoder shown.
        save_path: png path.
        show: show the figure, otherwise close it.

    Returns:
        The figure.
    """
    x_all = np.concatenate([session[LITPOSE][encoder]['bps'] for session in results.values()])
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    high = 0.0
    for method in methods:
        y_all = np.concatenate([session[method][encoder]['bps'] for session in results.values()])
        keep = np.isfinite(x_all) & np.isfinite(y_all)
        if not keep.any():
            continue
        ax.scatter(
            x_all[keep],
            y_all[keep],
            s=5,
            alpha=0.25,
            color=method_color(method),
            edgecolors='none',
            rasterized=True,
            label=f'{method_label(method)} ({np.mean(y_all[keep]):.3f})',
        )
        high = max(high, float(np.max(y_all[keep])), float(np.max(x_all[keep])))
    ax.plot([0, high], [0, high], linestyle='--', color='0.55', linewidth=1.0, zorder=0)
    ax.set_xlabel(
        f'{LITPOSE_LABEL} BPS ({encoder.upper()}, mean {np.nanmean(x_all):.3f})',
    )
    ax.set_ylabel(f'method BPS ({encoder.upper()})')
    ax.set_title(f'Per-neuron test BPS, {len(results)} session(s)')
    ax.legend(frameon=False, fontsize='small', markerscale=3, title='method (mean BPS)')
    _finish_figure(fig, save_path, show)
    return fig


def select_eids(args: argparse.Namespace) -> list[str]:
    """Sessions to compare: `--eids`, else the default group of the config's eid groups file.

    Args:
        args: parsed command-line arguments.

    Returns:
        The session ids.

    Raises:
        ValueError: if no `--eids` are given and the config has no `paths.eid_groups_json`.
    """
    if args.eids:
        return list(dict.fromkeys(args.eids))
    eid_groups_json = load_encoding_yaml(args.config).get('eid_groups_json')
    if eid_groups_json is None:
        raise ValueError(f'{args.config}: no paths.eid_groups_json; pass --eids')
    return resolve_group_eids(eid_groups_json, DEFAULT_EID_GROUP)


def main(argv: list[str] | None = None) -> None:
    """Compare the litpose encoding with the SABLE methods and write the tables and figures.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        RuntimeError: if no session has both a finished litpose run and the result of every method.
        ValueError: if a session was scored on a different test set by a method.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    methods = list(dict.fromkeys(args.methods))
    wanted = load_wanted_settings(args.config)
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for eid in select_eids(args):
        session_dir = session_output_dir(
            wanted['video_preds_dir'], wanted['output_folder'], eid,
        )
        try:
            variant_path = find_variant_dir(session_dir, wanted)
            results[eid] = compare_session(
                eid, args.results_dir, methods, variant_path, args.condition,
            )
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(f'{eid} skipped: {exc}')
    if not results:
        raise RuntimeError(
            'no session has both a finished litpose run and the result of every method',
        )

    names = [LITPOSE, *methods]
    pooled = pooled_rows(results, names)
    for row in pooled:
        logger.info(
            f'{row["label"]} {row["model"].upper()}: mean BPS {row["mean"]:.4f} +- '
            f'{row["se"]:.4f} ({row["n_neurons"]} neurons, {row["n_sessions"]} sessions)',
        )
    output_dir = args.output_dir or (
        Path(wanted['video_preds_dir'])
        / wanted['output_folder']
        / 'plots'
        / 'compare_with_sable_baseline'
    )
    write_tables(summary_rows(results, names), pooled, output_dir)
    if args.no_show:
        plt.switch_backend('Agg')
    plot_bars(results, methods, output_dir / 'bps_bars.png', show=not args.no_show)
    plot_scatter(
        results, methods, args.scatter_encoder, output_dir / 'bps_scatter.png',
        show=not args.no_show,
    )
    logger.info(f'wrote the comparison to {output_dir}')


if __name__ == '__main__':
    main()
