"""Redraw Experiment C (relative motion) figures from the saved arrays; nothing is recomputed.

Counterpart of `pose2visual.xcorr.replot` for `pose2visual.xcorr.experiments.relative_motion`:
loads `relative_arrays.<eid>.npz`, `relative_summary.<eid>.csv`, `relative_groups.<eid>.json` and
`relative_centroids.<eid>.pkl` (written by a finished
`pose2visual.xcorr.experiments.relative_motion` run) and calls the same
`relative_motion.plot_results` the compute step uses, so a redrawn figure looks exactly like the
original one except for the `--cameras`/`--vmax`/`--plots-subdir` override.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.replot_relative_motion \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --vmax 0.6
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.xcorr.experiments.common import resolve_order
from pose2visual.xcorr.experiments.relative_motion import plot_results
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.replot import REPLOT_ORDERS
from pose2visual.xcorr.settings import (
    EXPERIMENT_RELMOTION,
    freeze_config,
    latest_results_dir,
    load_xcorr_settings,
    plot_video_dir,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Redraw Experiment C (relative motion) figures from the saved arrays.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=None,
        help='cameras to draw; default: all of the run.',
    )
    parser.add_argument(
        '--order', type=str, choices=REPLOT_ORDERS, default='saved',
        help="neuron order of the heatmaps: 'saved' (Experiment A's saved order when A ran on "
        "the same neurons, else depth; default), 'depth' (ascending probe depth) or 'none'.",
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='shared colour scale +-VMAX of the original and relative heatmaps; default: the '
        "run's own rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='figure folder inside plot_video/experiment_C_relative_motion (default: New York '
        'time); the neuron order is appended, e.g. <name>_rastermap.',
    )
    return parser.parse_args(argv)


def load_results(out_dir: str | Path, eid: str) -> dict[str, Any]:
    """Load the saved results of one finished Experiment C run.

    Args:
        out_dir: results folder of the run,
            `<session_dir>/results/experiment_C_relative_motion/<run>`.
        eid: session id.

    Returns:
        Dict with `r_original`, `r_relative` ``[C, N, B, L]``, `corrected`, `valid` ``[C, B]``,
        `lags_sec`, `cameras`, `names`, `groups`, `neuron_ids`, `neuron_depths`, `summary` (the csv),
        `no_centroid`, `group_rows` (the flattened per-camera group plans) and `centroids`
        (the pickled frame-rate trajectory data).

    Raises:
        FileNotFoundError: if any of the four saved files is missing.
    """
    out_dir = Path(out_dir)
    arrays_path = out_dir / f'relative_arrays.{eid}.npz'
    summary_path = out_dir / f'relative_summary.{eid}.csv'
    groups_path = out_dir / f'relative_groups.{eid}.json'
    centroids_path = out_dir / f'relative_centroids.{eid}.pkl'
    for path in (arrays_path, summary_path, groups_path, centroids_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} not found; run pose2visual.xcorr.experiments.relative_motion first',
            )
    data = np.load(arrays_path)
    report = json.loads(groups_path.read_text(encoding='utf-8'))
    camera_report = report['cameras']
    group_rows = [
        {'camera': camera, **entry}
        for camera, info in camera_report.items() for entry in info['groups']
    ]
    with centroids_path.open('rb') as f:
        centroids = pickle.load(f)
    return {
        'r_original': data['r_original'],
        'r_relative': data['r_relative'],
        'corrected': data['corrected'],
        'valid': data['valid'],
        'lags_sec': data['lags_sec'],
        'cameras': [str(c) for c in data['cameras']],
        'names': [str(n) for n in data['behavior_names']],
        'groups': [str(g) for g in data['behavior_groups']],
        'neuron_ids': data['neuron_ids'].astype(str),
        'neuron_depths': data['neuron_depths'],
        'summary': pd.read_csv(summary_path),
        'no_centroid': list(report['no_centroid_groups']),
        'group_rows': group_rows,
        'centroids': centroids,
    }


def select_cameras(results: dict[str, Any], cameras: list[str] | None) -> dict[str, Any]:
    """Keep a subset of the cameras, consistently across all arrays.

    Args:
        results: output of `load_results`.
        cameras: cameras to keep, in the order to draw them; `None` keeps all.

    Returns:
        A new results dict restricted to the selection.

    Raises:
        ValueError: if a camera is not in the results.
    """
    have = results['cameras']
    wanted = list(dict.fromkeys(cameras or have))
    unknown = [camera for camera in wanted if camera not in have]
    if unknown:
        raise ValueError(f'cameras {unknown} are not in the results; the run has {have}')
    idx_cam = [have.index(camera) for camera in wanted]
    wanted_set = set(wanted)
    return {
        **results,
        'cameras': wanted,
        'r_original': results['r_original'][idx_cam],
        'r_relative': results['r_relative'][idx_cam],
        'corrected': results['corrected'][idx_cam],
        'valid': results['valid'][idx_cam],
        'summary': results['summary'][results['summary']['camera'].isin(wanted)],
        'group_rows': [row for row in results['group_rows'] if row['camera'] in wanted_set],
        'centroids': [c for c in results['centroids'] if c['camera'] in wanted_set],
    }


def replot_session(
    config_path: str | Path,
    eid: str,
    cameras: list[str] | None = None,
    order: str = 'saved',
    vmax: float | None = None,
    plots_subdir: str | None = None,
) -> list[Path]:
    """Redraw the Experiment C figures of one finished session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        cameras: cameras to draw; `None` = all of the run.
        order: neuron order of the heatmaps, one of `REPLOT_ORDERS`.
        vmax: half-range of the colour scale; `None` takes it from the data.
        plots_subdir: run name of the figure folder inside
            `plot_video/experiment_C_relative_motion`, to which the neuron order is appended;
            `None` makes a new name from the current New York time.

    Returns:
        The written figure files, png and pdf of each figure.

    Raises:
        FileNotFoundError: if the session has no saved Experiment C results.
        ValueError: if a requested camera is not in the results, or `order` is unknown.
    """
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    try:
        out_dir = latest_results_dir(
            session_dir, EXPERIMENT_RELMOTION, f'relative_arrays.{eid}.npz',
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'{error}; run pose2visual.xcorr.experiments.relative_motion first',
        ) from error
    results = select_cameras(load_results(out_dir, eid), cameras)
    idx_order, order_label = resolve_order(
        config_path, eid, order, results['neuron_ids'], results['neuron_depths'],
    )
    plots_dir = plot_video_dir(session_dir, EXPERIMENT_RELMOTION) / order_subdir(
        plots_subdir, order_label,
    )
    return plot_results(
        plots_dir, eid, results['lags_sec'], results['cameras'], results['names'],
        results['groups'], results['valid'], results['r_original'], results['r_relative'],
        idx_order, order_label, vmax, results['corrected'], results['no_centroid'],
        results['summary'], results['group_rows'], results['centroids'],
    )


def main(argv: list[str] | None = None) -> None:
    """Redraw the Experiment C figures of every selected session; failures are listed at the end.

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
                paths = replot_session(
                    config, eid, args.cameras, args.order, args.vmax, plots_subdir,
                )
                logger.info(f'wrote {[str(p) for p in paths]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
