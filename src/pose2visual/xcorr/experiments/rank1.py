"""Experiment A: subtract the dominant shared pattern (rank 1) from the baseline xcorrs.

Question: do the neuron x lag cross-correlation patterns of different body parts look alike because
one pattern is shared by all of them? The baseline tensor of one camera, ``(n_neurons,
n_behaviors, n_lags)``, is reshaped to a matrix ``M`` of shape ``(n_neurons * n_lags,
n_behaviors)`` whose columns are complete neuron x lag heatmaps. Its singular value decomposition
(without centering: the shared mean pattern is exactly what is tested) gives the rank-1
approximation ``M1 = s_0 u_0 v_0^T``: ``u_0`` is the shared neuron x lag pattern, ``v_0`` how
strongly each behavior carries it (its loadings). The residual ``M - M1`` is what the behaviors do
not share to first order.

What is removed: the best single neuron x lag pattern shared across the behaviors of a camera
(including its per-behavior scale and sign). What remains: everything orthogonal to it,
behavior-specific patterns and any further shared patterns (ranks 2 and up). Rank 1 is a linear
algebra statement, not a physical one: the loadings are reported, not interpreted, and nothing
here assumes that the leading component is global motion. The residual is an SVD residual with
the units of a correlation but is not a Pearson correlation (it is not bounded by 1).

The baseline is only read (``load_baseline``). Everything is written to
``<session>/experiment_A_rank1/``: ``rank1_arrays.<eid>.npz``, ``rank1_behaviors.<eid>.csv``,
``rank1_report.<eid>.{json,md}`` and, in a timestamped subfolder, the heatmaps (original, rank-1
component, residual on the shared colour scale, residual on its own scale), the diagnostics and
example curves. Neuron order, lag axis and colour scale are those of the baseline.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.rank1 \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --exclude 'd*'
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.xcorr.experiments.common import (
    EXPERIMENT_A,
    experiment_dir,
    load_baseline,
    relabel_overview,
)
from pose2visual.xcorr.plots import DEFAULT_N_EXAMPLES, timestamp_subdir
from pose2visual.xcorr.settings import freeze_config, resolve_eids

logger = logging.getLogger(__name__)

# number of singular values reported / drawn in the scree plot
N_SINGULAR_REPORTED = 5
RANK1_COLORBAR = 'rank-1 component (not a Pearson r)'
RESIDUAL_COLORBAR = 'SVD residual (not a Pearson r)'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Experiment A: subtract the rank-1 component from the baseline xcorrs.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=None,
        help='cameras to decompose (each separately); default: all of the baseline run.',
    )
    parser.add_argument(
        '--exclude', type=str, nargs='*', default=[], metavar='PATTERN',
        help='glob patterns of behaviors to leave out of the decomposition and the figures; '
        'quote them.',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='shared colour scale +-VMAX of original / rank-1 / residual; default: the '
        "baseline's rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--n-examples', type=int, default=DEFAULT_N_EXAMPLES,
        help='number of example curves to draw (default 0 = none; 12 was the former default).',
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='figure folder inside experiment_A_rank1 (default: New York time of the run).',
    )
    return parser.parse_args(argv)


def rank1_decompose(tensor: np.ndarray) -> dict[str, np.ndarray | float]:
    """Rank-1 approximation of a ``(neurons, behaviors, lags)`` xcorr tensor.

    Args:
        tensor: finite array ``[N, B, L]`` with ``B >= 2``.

    Returns:
        Dict with `rank1` and `residual` (``[N, B, L]``, `rank1 + residual == tensor`),
        `singular_values` (all ``min(N * L, B)`` of them), `frac_rank1` (``s_0^2 / ||M||_F^2``),
        `loadings` (``[B]`` unit-norm right singular vector ``v_0``), `pattern` (``[N, L]``
        left singular vector ``u_0`` reshaped) and `frac_behavior` (``[B]``, share of each
        behavior's own squared norm that the rank-1 term explains).
        The sign of ``(u_0, v_0)`` is fixed so that ``sum(v_0) >= 0``; the product, hence
        the rank-1 term, does not depend on it.

    Raises:
        ValueError: if the tensor is not three-dimensional, has fewer than two behaviors, is not
            finite or is all zero.
    """
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f'expected a [neurons, behaviors, lags] tensor, got shape {tensor.shape}')
    n_neurons, n_behaviors, n_lags = tensor.shape
    if n_behaviors < 2:
        raise ValueError('a rank-1 decomposition needs at least two behaviors')
    if not np.all(np.isfinite(tensor)):
        raise ValueError('the tensor contains NaN or inf; pass only the valid behaviors')
    # columns = behaviors, rows = (neuron, lag)
    matrix = tensor.transpose(0, 2, 1).reshape(n_neurons * n_lags, n_behaviors)
    total = float(np.sum(matrix ** 2))
    if total == 0.0:
        raise ValueError('the tensor is all zero')
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    u0, v0 = u[:, 0], vt[0]
    if v0.sum() < 0:
        u0, v0 = -u0, -v0
    approx = s[0] * np.outer(u0, v0)
    residual = matrix - approx

    def back(flat: np.ndarray) -> np.ndarray:
        return flat.reshape(n_neurons, n_lags, n_behaviors).transpose(0, 2, 1)

    column_norm = np.sum(matrix ** 2, axis=0)
    # (s_0 v_0[b])^2 / ||M[:, b]||^2, in [0, 1] because M^T u_0 = s_0 v_0
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_behavior = np.where(column_norm > 0, (s[0] * v0) ** 2 / column_norm, 0.0)
    return {
        'rank1': back(approx),
        'residual': back(residual),
        'singular_values': s,
        'frac_rank1': float(s[0] ** 2 / total),
        'loadings': v0,
        'pattern': u0.reshape(n_neurons, n_lags),
        'frac_behavior': frac_behavior,
    }


def summarize_loadings(
    names: list[str],
    groups: list[str],
    loadings: np.ndarray,
) -> dict[str, Any]:
    """Descriptive statistics of the leading behavioral loadings (no interpretation).

    Args:
        names: the ``B`` behavior names.
        groups: body-part group of each behavior.
        loadings: unit-norm loadings ``v_0`` ``[B]``.

    Returns:
        Dict with `table` (dataframe: behavior, group, loading, abs_loading, share = loading^2),
        `group_share` (group -> summed share), `sign_coherence` (fraction of loadings with the
        majority sign) and `participation_ratio` (``1 / sum(v^4)``: about 1 when one behavior
        carries the component, ``B`` when all carry it equally).
    """
    loadings = np.asarray(loadings, dtype=np.float64)
    share = loadings ** 2
    table = pd.DataFrame({
        'behavior': names,
        'group': groups,
        'loading': loadings,
        'abs_loading': np.abs(loadings),
        'share': share,
    })
    n_positive = int(np.sum(loadings > 0))
    return {
        'table': table,
        'group_share': table.groupby('group', sort=False)['share'].sum().to_dict(),
        'sign_coherence': max(n_positive, loadings.size - n_positive) / loadings.size,
        'participation_ratio': float(1.0 / np.sum(share ** 2)),
    }


def median_peak(r: np.ndarray) -> np.ndarray:
    """Median over neurons of each neuron's peak absolute value over lags.

    Args:
        r: array ``[N, B, L]``.

    Returns:
        Array ``[B]``.
    """
    return np.median(np.abs(r).max(axis=2), axis=0)


def select_residual_examples(
    original: np.ndarray,
    rank1: np.ndarray,
    residual: np.ndarray,
    cameras: list[str],
    names: list[str],
    valid: np.ndarray,
    n_examples: int,
) -> list[dict[str, Any]]:
    """The neuron with the largest residual peak for each camera-behavior pair, strongest first.

    Args:
        original: baseline xcorrs ``[C, N, B, L]``.
        rank1: rank-1 component, same shape (NaN where invalid).
        residual: SVD residual, same shape (NaN where invalid).
        cameras: the ``C`` camera names.
        names: the ``B`` behavior names.
        valid: boolean ``[C, B]``.
        n_examples: number of pairs to return.

    Returns:
        List of dicts with `title`, `idx_neuron`, `original`, `rank1` and `residual` curves.
    """
    candidates = []
    for idx_c, camera in enumerate(cameras):
        for idx_b in np.flatnonzero(valid[idx_c]):
            peaks = np.abs(residual[idx_c, :, idx_b, :]).max(axis=1)
            idx_n = int(np.argmax(peaks))
            candidates.append((float(peaks[idx_n]), idx_c, idx_n, int(idx_b)))
    candidates.sort(reverse=True)
    examples = []
    for peak, idx_c, idx_n, idx_b in candidates[:n_examples]:
        examples.append({
            'title': f'{cameras[idx_c]}: {names[idx_b]} | neuron {idx_n}\n'
            f'residual peak {peak:.2f}',
            'idx_neuron': idx_n,
            'original': original[idx_c, idx_n, idx_b],
            'rank1': rank1[idx_c, idx_n, idx_b],
            'residual': residual[idx_c, idx_n, idx_b],
        })
    return examples


def explanation_text(report: dict[str, Any]) -> str:
    """Short markdown explanation plus the numbers of one run.

    Args:
        report: the `rank1_report` dict built by `run_session`.

    Returns:
        Markdown text.
    """
    lines = [
        f'# Experiment A: rank-1 subtraction ({report["eid"]})',
        '',
        '**Removed**: the single neuron x lag pattern `u_0`, scaled per behavior by its loading '
        '`v_0`, that best fits all behaviors of a camera at once (SVD, no centering).',
        '**Remains**: everything orthogonal to it, i.e. behavior-specific patterns and any '
        'further shared patterns (rank 2 and up). The residual is an SVD residual, not a Pearson '
        'correlation. The loadings are descriptive; whether the component is global motion has '
        'to be judged from them.',
        '',
    ]
    for camera, info in report['cameras'].items():
        lines += [
            f'## {camera} camera',
            '',
            f'* rank 1 explains **{100 * info["frac_rank1"]:.1f} %** of the squared Frobenius '
            'norm',
            '* leading singular values (fraction of the squared norm): '
            + ', '.join(f'{100 * f:.1f} %' for f in info['singular_fractions']),
            f'* loadings with the majority sign: {100 * info["sign_coherence"]:.0f} %, '
            f'participation ratio {info["participation_ratio"]:.1f} of {info["n_behaviors"]} '
            'behaviors',
            '* squared-loading share by group: '
            + ', '.join(f'{g} {100 * s:.0f} %' for g, s in info['group_share'].items()),
            '',
            '| behavior | group | loading | rank-1 explains its norm |',
            '|---|---|---|---|',
        ]
        for row in info['behaviors']:
            lines.append(
                f'| {row["behavior"]} | {row["group"]} | {row["loading"]:+.3f} | '
                f'{100 * row["frac_explained"]:.0f} % |',
            )
        lines.append('')
    return '\n'.join(lines)


def run_session(
    config_path: str | Path,
    eid: str,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    vmax: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    plots_subdir: str | None = None,
) -> dict[str, Path]:
    """Run Experiment A on the saved baseline of one session.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        cameras: cameras to decompose (each on its own); `None` = all of the baseline run.
        exclude: glob patterns of behaviors to leave out.
        vmax: shared half-range of the colour scale; `None` applies the baseline's rule to the
            original xcorrs.
        n_examples: number of residual example curves; 0 (default) draws none.
        plots_subdir: figure folder name; `None` = New York time of the run.

    Returns:
        Dict of the written paths: `arrays`, `behaviors`, `report`, `report_md` and `plots`
        (the figure folder).

    Raises:
        FileNotFoundError: if the baseline run is missing.
        ValueError: if a camera has fewer than two behaviors or non-finite valid xcorrs.
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.experiments.diagnostics import (
        plot_rank1_diagnostics,
        plot_residual_examples,
    )
    from pose2visual.xcorr.figures import heatmap_vmax, plot_xcorr_overview, save_figure

    base = load_baseline(config_path, eid, cameras, exclude)
    original, valid = base['r'], base['valid']
    names, groups, cams = base['names'], base['groups'], base['cameras']
    lags_sec, idx_order = base['lags_sec'], np.asarray(base['neuron_order'])
    out_dir = experiment_dir(base['session_dir'], EXPERIMENT_A)
    plots_dir = out_dir / (plots_subdir or timestamp_subdir())

    rank1 = np.full_like(original, np.nan, dtype=np.float64)
    residual = rank1.copy()
    camera_info: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    rows: list[pd.DataFrame] = []
    diag = []
    for idx_c, camera in enumerate(cams):
        idx_b = np.flatnonzero(valid[idx_c])
        sub_names = [names[i] for i in idx_b]
        sub_groups = [groups[i] for i in idx_b]
        res = rank1_decompose(original[idx_c][:, idx_b, :])
        rank1[idx_c][:, idx_b, :] = res['rank1']
        residual[idx_c][:, idx_b, :] = res['residual']
        loadings = summarize_loadings(sub_names, sub_groups, res['loadings'])
        singular_fractions = (res['singular_values'] ** 2) / np.sum(res['singular_values'] ** 2)
        singular_fractions = singular_fractions[:N_SINGULAR_REPORTED]
        table = loadings['table'].assign(
            camera=camera,
            frac_explained=res['frac_behavior'],
            median_peak_original=median_peak(original[idx_c][:, idx_b, :]),
            median_peak_rank1=median_peak(res['rank1']),
            median_peak_residual=median_peak(res['residual']),
        )
        rows.append(table)
        logger.info(
            f'{camera}: rank 1 explains {100 * res["frac_rank1"]:.1f} % of the squared norm of '
            f'{len(idx_b)} behaviors; {100 * loadings["sign_coherence"]:.0f} % of the loadings '
            f'share a sign; group shares {loadings["group_share"]}',
        )
        camera_info[camera] = {
            'n_behaviors': len(idx_b),
            'frac_rank1': res['frac_rank1'],
            'singular_fractions': singular_fractions.tolist(),
            'sign_coherence': loadings['sign_coherence'],
            'participation_ratio': loadings['participation_ratio'],
            'group_share': loadings['group_share'],
            'behaviors': [
                {
                    'behavior': row.behavior, 'group': row.group, 'loading': row.loading,
                    'frac_explained': row.frac_explained,
                }
                for row in table.itertuples()
            ],
        }
        arrays[f'{camera}.singular_values'] = res['singular_values']
        arrays[f'{camera}.loadings'] = res['loadings']
        arrays[f'{camera}.pattern'] = res['pattern']
        arrays[f'{camera}.behaviors'] = np.asarray(sub_names)
        diag.append({
            'camera': camera,
            'singular_fractions': singular_fractions,
            'names': sub_names,
            'groups': sub_groups,
            'loadings': res['loadings'],
            'pattern': res['pattern'][idx_order],
            'frac_behavior': res['frac_behavior'],
        })

    arrays_path = out_dir / f'rank1_arrays.{eid}.npz'
    np.savez_compressed(
        arrays_path, rank1=rank1.astype(np.float32), residual=residual.astype(np.float32),
        lags_sec=lags_sec, cameras=np.asarray(cams), behavior_names=np.asarray(names),
        behavior_groups=np.asarray(groups), valid=valid, neuron_order=idx_order, **arrays,
    )
    behaviors_path = out_dir / f'rank1_behaviors.{eid}.csv'
    pd.concat(rows, ignore_index=True).to_csv(behaviors_path, index=False)
    report = {'eid': eid, 'cameras': camera_info, 'plots_subdir': plots_dir.name}
    report_path = out_dir / f'rank1_report.{eid}.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    report_md_path = out_dir / f'rank1_report.{eid}.md'
    report_md_path.write_text(explanation_text(report), encoding='utf-8')

    # figures: the baseline's overview for each of the three tensors, in the baseline's order
    scale = vmax if vmax is not None else heatmap_vmax(original)
    ylabel = f'neurons ({base["order_label"]} order)'
    panels = [
        ('original', original, scale, None, 'original xcorr (Pearson r)'),
        ('rank1', rank1, scale, RANK1_COLORBAR, 'rank-1 component of the xcorrs'),
        ('residual', residual, scale, RESIDUAL_COLORBAR, 'SVD residual after removing rank 1'),
        (
            'residual_own_scale', residual, heatmap_vmax(residual), RESIDUAL_COLORBAR,
            'SVD residual after removing rank 1, colour scale fitted to the residual',
        ),
    ]
    for stem, data, panel_scale, colorbar, label in panels:
        fig = plot_xcorr_overview(
            data[:, idx_order], lags_sec, names, cams, groups, valid, eid, panel_scale,
            ylabel=ylabel,
        )
        if colorbar is not None:
            relabel_overview(
                fig, f'{eid}: {label}; colour scale +-{panel_scale:.2f}; each strip: '
                'neurons x lag, number = max |value|', colorbar,
            )
        save_figure(fig, plots_dir / f'heatmap.{stem}.{eid}')
    save_figure(
        plot_rank1_diagnostics(diag, lags_sec, f'{eid} | rank-1 component'),
        plots_dir / f'diagnostics.rank1.{eid}',
    )
    examples = select_residual_examples(
        original, rank1, residual, cams, names, valid, n_examples,
    )
    if examples:
        save_figure(
            plot_residual_examples(examples, lags_sec, f'{eid} | residual example curves'),
            plots_dir / f'example_curves.residual.{eid}',
        )
    return {
        'arrays': arrays_path, 'behaviors': behaviors_path, 'report': report_path,
        'report_md': report_md_path, 'plots': plots_dir,
    }


def main(argv: list[str] | None = None) -> None:
    """Run Experiment A for every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    plots_subdir = args.plots_subdir or timestamp_subdir()
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                paths = run_session(
                    config, eid, args.cameras, args.exclude, args.vmax, args.n_examples,
                    plots_subdir,
                )
                logger.info(f'wrote {[str(p) for p in paths.values()]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
