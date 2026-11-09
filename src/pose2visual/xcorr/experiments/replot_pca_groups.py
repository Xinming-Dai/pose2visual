"""Redraw Experiment D (PCA of each group) figures from the saved arrays; nothing is recomputed.

Counterpart of `pose2visual.xcorr.replot` for `pose2visual.xcorr.experiments.pca_groups`: loads
`pca_groups_speed_arrays.<eid>.npz`, `pca_groups_position_arrays.<eid>.npz`,
`pca_groups_summary.<eid>.csv` and `pca_groups_report.<eid>.json` (written by a finished
`pose2visual.xcorr.experiments.pca_groups` run) and calls the same `pca_groups.plot_results` the
compute step uses, so a redrawn figure looks exactly like the original one except for the
`--cameras`/`--vmax`/`--plots-subdir` override.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.replot_pca_groups \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --vmax 0.6
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.xcorr.experiments.common import resolve_order
from pose2visual.xcorr.experiments.pca_groups import plot_results
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.replot import REPLOT_ORDERS
from pose2visual.xcorr.settings import (
    EXPERIMENT_PCA_GROUPS,
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
        description='Redraw Experiment D (PCA of each group) figures from the saved arrays.',
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
        help='shared colour scale +-VMAX of the original and pca-speed heatmaps; default: the '
        "run's own rule applied to the original xcorrs.",
    )
    parser.add_argument(
        '--diff-vmax', type=float, default=None,
        help='half-range of the one colour scale of every difference strip of '
        'heatmap.signed_difference; fix it to compare sessions (default: from the data, once per '
        'figure).',
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='figure folder inside plot_video/experiment_D_pca_groups (default: New York time); '
        'the neuron order is appended, e.g. <name>_rastermap.',
    )
    return parser.parse_args(argv)


def load_results(out_dir: str | Path, eid: str) -> dict[str, Any]:
    """Load the saved results of one finished Experiment D run.

    Args:
        out_dir: results folder of the run,
            `<session_dir>/results/experiment_D_pca_groups/<run>`.
        eid: session id.

    Returns:
        Dict with `r_original`, `r_pca` ``[C, N, B, L]``, `pca_applied`, `valid` ``[C, B]``,
        `lags_sec`, `cameras`, `names`, `groups`, `neuron_ids`, `neuron_depths`, `r_position`
        ``[C, N, P, L]``, `valid_position` ``[C, P]``, `position_names`, `position_groups`,
        `summary` (the csv) and `camera_report` (the report json's `cameras` entry).

    Raises:
        FileNotFoundError: if any of the four saved files is missing.
    """
    out_dir = Path(out_dir)
    speed_path = out_dir / f'pca_groups_speed_arrays.{eid}.npz'
    position_path = out_dir / f'pca_groups_position_arrays.{eid}.npz'
    summary_path = out_dir / f'pca_groups_summary.{eid}.csv'
    groups_path = out_dir / f'pca_groups_report.{eid}.json'
    for path in (speed_path, position_path, summary_path, groups_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} not found; run pose2visual.xcorr.experiments.pca_groups first',
            )
    speed = np.load(speed_path)
    position = np.load(position_path)
    report = json.loads(groups_path.read_text(encoding='utf-8'))
    return {
        'r_original': speed['r_original'],
        'r_pca': speed['r_pca'],
        'pca_applied': speed['pca_applied'],
        'valid': speed['valid'],
        'lags_sec': speed['lags_sec'],
        'cameras': [str(c) for c in speed['cameras']],
        'names': [str(n) for n in speed['behavior_names']],
        'groups': [str(g) for g in speed['behavior_groups']],
        'neuron_ids': speed['neuron_ids'].astype(str),
        'neuron_depths': speed['neuron_depths'],
        'r_position': position['r_position'],
        'valid_position': position['valid'],
        'position_names': [str(n) for n in position['pc_names']],
        'position_groups': [str(g) for g in position['pc_groups']],
        'summary': pd.read_csv(summary_path),
        'camera_report': report['cameras'],
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
    return {
        **results,
        'cameras': wanted,
        'r_original': results['r_original'][idx_cam],
        'r_pca': results['r_pca'][idx_cam],
        'pca_applied': results['pca_applied'][idx_cam],
        'valid': results['valid'][idx_cam],
        'r_position': results['r_position'][idx_cam],
        'valid_position': results['valid_position'][idx_cam],
        'summary': results['summary'][results['summary']['camera'].isin(wanted)],
        'camera_report': {c: v for c, v in results['camera_report'].items() if c in wanted},
    }


def replot_session(
    config_path: str | Path,
    eid: str,
    cameras: list[str] | None = None,
    order: str = 'saved',
    vmax: float | None = None,
    plots_subdir: str | None = None,
    diff_vmax: float | None = None,
) -> list[Path]:
    """Redraw the Experiment D figures of one finished session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        cameras: cameras to draw; `None` = all of the run.
        order: neuron order of the heatmaps, one of `REPLOT_ORDERS`.
        vmax: half-range of the colour scale; `None` takes it from the data.
        plots_subdir: run name of the figure folder inside `plot_video/experiment_D_pca_groups`,
            to which the neuron order is appended; `None` makes a new name from the current New
            York time.
        diff_vmax: fixed half-range of the difference strips; `None` = from the data.

    Returns:
        The written png files and the figure guide.

    Raises:
        FileNotFoundError: if the session has no saved Experiment D results.
        ValueError: if a requested camera is not in the results, or `order` is unknown.
    """
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    try:
        out_dir = latest_results_dir(
            session_dir, EXPERIMENT_PCA_GROUPS, f'pca_groups_speed_arrays.{eid}.npz',
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'{error}; run pose2visual.xcorr.experiments.pca_groups first',
        ) from error
    results = select_cameras(load_results(out_dir, eid), cameras)
    idx_order, order_label = resolve_order(
        config_path, eid, order, results['neuron_ids'], results['neuron_depths'],
    )
    plots_dir = plot_video_dir(session_dir, EXPERIMENT_PCA_GROUPS) / order_subdir(
        plots_subdir, order_label,
    )
    return plot_results(
        plots_dir, eid, results['lags_sec'], results['cameras'], results['names'],
        results['groups'], results['valid'], results['r_original'], results['r_pca'],
        idx_order, order_label, vmax, results['position_names'], results['position_groups'],
        results['r_position'], results['valid_position'], results['pca_applied'],
        results['summary'], results['camera_report'],
        diff_vmax=diff_vmax,
    )


def main(argv: list[str] | None = None) -> None:
    """Redraw the Experiment D figures of every selected session; failures are listed at the end.

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
                    args.diff_vmax,
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
