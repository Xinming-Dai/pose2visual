"""Redraw the figures of a finished cross-correlation run, without recomputing anything.

Reads the files ``run_xcorr`` left in ``<video_preds_dir>/<xcorr.output_name>/<eid>/``
(``xcorr_arrays.<eid>.npz``, ``xcorr_summary.<eid>.csv`` and ``xcorr_params.json``) and writes the
overview heatmap (and, with ``--n-examples N``, the example curves) into a new subfolder of that
folder named by the New York time of the replot (``YYYYmmdd_HHMMSS``, so earlier figures are kept)
with the same code as the run itself (``plots.write_plots``). Nothing else is written, so a replot
never changes the results. It takes seconds: no neural data, no behavior traces and no
permutations are involved, and the neuron order stored by the run is reused (rastermap is not run
again), which also makes the figure reproducible.

What can be changed: ``--cameras`` (draw a subset of the rows), ``--exclude`` (hide behaviors by
glob pattern, e.g. ``'tongue_*'``), ``--vmax`` (fixed colour scale, Figure 20 uses 0.6),
``--order`` (``saved``, ``depth`` or ``none``), ``--n-examples`` and ``--plots-subdir`` (to write
into a folder of your own name instead). If ``peth.<eid>.npz`` exists
(``pose2visual.xcorr.peth``) the reward / movement-onset event strips are added to the overview
(cross-correlations on the colour scale of the behaviors, then PETHs), matched to the neurons by
id; ``--no-peth`` hides them and ``--peth-vmax`` sets the colour scale of the PETHs. A run made
before the null 95th percentile was saved has no threshold line in the example curves.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.replot \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --vmax 0.6 \\
      --exclude 'tongue_*' --plots-subdir plots_vmax06
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.xcorr.behavior import excluded_names
from pose2visual.xcorr.peth import load_peth
from pose2visual.xcorr.plots import DEFAULT_N_EXAMPLES, timestamp_subdir, write_plots
from pose2visual.xcorr.settings import (
    freeze_config,
    load_xcorr_settings,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

REPLOT_ORDERS = ('saved', 'depth', 'none')
_REQUIRED_KEYS = (
    'r', 'lags_sec', 'cameras', 'behavior_names', 'behavior_groups', 'valid', 'neuron_order',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Redraw the cross-correlation figures of a finished run (no recomputation).',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=None,
        help='cameras (rows) to draw; default: all cameras of the run.',
    )
    parser.add_argument(
        '--exclude', type=str, nargs='*', default=[], metavar='PATTERN',
        help="glob patterns of behaviors to hide, e.g. 'tongue_*'; quote them.",
    )
    parser.add_argument(
        '--order', type=str, choices=REPLOT_ORDERS, default='saved',
        help='neuron order: the one stored by the run (default), probe depth, or file order.',
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='fix the colour scale to +-VMAX (Figure 20 uses 0.6); default: from the data.',
    )
    parser.add_argument(
        '--n-examples', type=int, default=DEFAULT_N_EXAMPLES,
        help='number of example curves to draw (default 0 = none; 12 was the former default).',
    )
    parser.add_argument(
        '--no-peth', action='store_true',
        help='hide the reward / movement-onset event strips (xcorr and PETH) even if '
        'peth.<eid>.npz exists.',
    )
    parser.add_argument(
        '--peth-vmax', type=float, default=1.0,
        help='half-range of the PETH colour scale (default: 1.0 = a doubling of the rate).',
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='folder of the figures inside the session folder (default: a new folder named by '
        'the New York time of the run, YYYYmmdd_HHMMSS, so no earlier figure is overwritten).',
    )
    return parser.parse_args(argv)


def load_results(session_dir: str | Path, eid: str) -> dict[str, Any]:
    """Load the saved results of one session.

    Args:
        session_dir: `<video_preds_dir>/<xcorr.output_name>/<eid>`.
        eid: session id.

    Returns:
        Dict with `r` ``[C, N, B, L]``, `lags_sec`, `cameras`, `names`, `groups`, `valid`
        ``[C, B]``, `neuron_order`, `null_p95` (``[C, N, B]`` or `None` for a run that did not
        save it), `table` (the summary csv) and `order_label` (the neuron order the run used).

    Raises:
        FileNotFoundError: if the arrays or the summary csv are missing.
        ValueError: if the arrays file lacks a required entry.
    """
    session_dir = Path(session_dir)
    arrays_path = session_dir / f'xcorr_arrays.{eid}.npz'
    summary_path = session_dir / f'xcorr_summary.{eid}.csv'
    for path in (arrays_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; run pose2visual.xcorr.run_xcorr first')
    data = np.load(arrays_path)
    missing = [key for key in _REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(f'{arrays_path} lacks {missing}; it was made by an older version')
    params_path = session_dir / 'xcorr_params.json'
    order_label = 'saved'
    if params_path.is_file():
        order_label = json.loads(params_path.read_text(encoding='utf-8')).get('order', 'saved')
    return {
        'r': data['r'],
        'lags_sec': data['lags_sec'],
        'cameras': [str(c) for c in data['cameras']],
        'names': [str(n) for n in data['behavior_names']],
        'groups': [str(g) for g in data['behavior_groups']],
        'valid': data['valid'],
        'neuron_order': data['neuron_order'],
        'neuron_ids': data['neuron_ids'] if 'neuron_ids' in data.files else None,
        'null_p95': data['null_p95'] if 'null_p95' in data.files else None,
        'table': pd.read_csv(summary_path),
        'order_label': order_label,
    }


def select_results(
    results: dict[str, Any],
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Keep a subset of the cameras and hide behaviors, consistently across all arrays.

    Args:
        results: output of `load_results`.
        cameras: cameras to keep, in the order to draw them; `None` keeps all.
        exclude: glob patterns of behaviors to hide. A behavior that no kept camera has is
            dropped as well.

    Returns:
        A new results dict with `r`, `valid`, `names`, `groups`, `null_p95`, `cameras` and `table`
        restricted to the selection (`neuron_order`, `lags_sec` and `order_label` unchanged).

    Raises:
        ValueError: if a camera is not in the results, or nothing is left to draw.
    """
    have = results['cameras']
    wanted = list(dict.fromkeys(cameras or have))
    unknown = [camera for camera in wanted if camera not in have]
    if unknown:
        raise ValueError(f'cameras {unknown} are not in the results; the run has {have}')
    idx_cam = [have.index(camera) for camera in wanted]
    patterns = list(exclude or [])
    names = results['names']
    for pattern in patterns:
        if not excluded_names(names, [pattern]):
            logger.warning(f'--exclude pattern {pattern!r} matches no behavior')
    hidden = set(excluded_names(names, patterns))
    valid = results['valid'][idx_cam]
    idx_b = [
        idx for idx, name in enumerate(names) if name not in hidden and valid[:, idx].any()
    ]
    if not idx_b:
        raise ValueError(f'nothing left to draw for cameras {wanted} and --exclude {patterns}')
    kept_names = [names[idx] for idx in idx_b]
    table = results['table']
    table = table[table['camera'].isin(wanted) & table['behavior'].isin(kept_names)]
    null_p95 = results['null_p95']
    return {
        **results,
        'cameras': wanted,
        'r': results['r'][idx_cam][:, :, idx_b, :],
        'valid': valid[:, idx_b],
        'names': kept_names,
        'groups': [results['groups'][idx] for idx in idx_b],
        'null_p95': None if null_p95 is None else null_p95[idx_cam][:, :, idx_b],
        'table': table,
    }


def neuron_depths(table: pd.DataFrame, n_neurons: int) -> np.ndarray:
    """Depth of every neuron, read from the summary table.

    Args:
        table: summary csv with `neuron_idx` and `depth`.
        n_neurons: number of neurons ``N`` of the run.

    Returns:
        Array ``[N]`` of depths indexed by `neuron_idx`.

    Raises:
        ValueError: if the table does not cover every neuron.
    """
    per_neuron = table.drop_duplicates('neuron_idx').set_index('neuron_idx')['depth']
    if set(per_neuron.index) != set(range(n_neurons)):
        raise ValueError('the summary table does not cover every neuron; cannot order by depth')
    return per_neuron.sort_index().to_numpy(dtype=np.float64)


def replot_session(
    config_path: str | Path,
    eid: str,
    order: str = 'saved',
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    vmax: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    plots_subdir: str | None = None,
    show_peth: bool = True,
    peth_vmax: float = 1.0,
) -> dict[str, Any]:
    """Redraw the figures of one finished session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        order: `'saved'` (the neuron order the run stored), `'depth'` or `'none'`.
        cameras: cameras to draw; `None` = all of the run.
        exclude: glob patterns of behaviors to hide.
        vmax: half-range of the colour scale; `None` takes it from the data.
        n_examples: number of example curves; 0 (default) draws none.
        plots_subdir: folder of the figures inside the session folder; `None` makes a new
            folder named by the current New York time (`plots.timestamp_subdir`).
        show_peth: draw the PETH strips when `peth.<eid>.npz` exists in the session folder.
        peth_vmax: half-range of the PETH colour scale.

    Returns:
        The result of `plots.write_plots` (`heatmap`, `examples`, `vmax`).

    Raises:
        FileNotFoundError: if the session has no saved results.
        ValueError: for an unknown order or camera, or if nothing is left to draw.
    """
    if order not in REPLOT_ORDERS:
        raise ValueError(f'unknown order {order!r}; use one of {list(REPLOT_ORDERS)}')
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    plots_subdir = plots_subdir or timestamp_subdir()
    full = load_results(session_dir, eid)
    n_neurons = full['r'].shape[1]
    if order == 'saved':
        idx_order = np.asarray(full['neuron_order'])
    elif order == 'depth':
        idx_order = np.argsort(neuron_depths(full['table'], n_neurons), kind='stable')
    else:
        idx_order = np.arange(n_neurons)
    results = select_results(full, cameras, exclude)
    if results['null_p95'] is None:
        logger.info('this run did not save the null threshold: example curves have no dashed line')
    label = full['order_label'] if order == 'saved' else order
    peth = None
    if show_peth and full['neuron_ids'] is not None:
        peth = load_peth(session_dir, eid, full['neuron_ids'])
        if peth is None:
            logger.info('no peth.<eid>.npz: run pose2visual.xcorr.peth to add the PETH strips')
    return write_plots(
        session_dir / plots_subdir, eid, results['r'], results['lags_sec'], results['names'],
        results['groups'], results['cameras'], results['valid'], idx_order, results['table'],
        label, vmax, n_examples, results['null_p95'], peth, peth_vmax,
    )


def main(argv: list[str] | None = None) -> None:
    """Redraw the figures of every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    # one folder name for the whole job, so every session's figures carry the same run time
    plots_subdir = args.plots_subdir or timestamp_subdir()
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                paths = replot_session(
                    config, eid, args.order, args.cameras, args.exclude, args.vmax,
                    args.n_examples, plots_subdir, not args.no_peth, args.peth_vmax,
                )
                logger.info(f'wrote {[str(p) for p in paths["heatmap"] + paths["examples"]]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
