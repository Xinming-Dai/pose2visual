"""Redraw Experiment B (PCA) figures from the saved arrays; nothing is recomputed.

Counterpart of `pose2visual.xcorr.replot` for `pose2visual.xcorr.experiments.pca`: loads
`pca_speed_arrays.<eid>.npz`, `pca_position_arrays.<eid>.npz`, `pca_summary.<eid>.csv` and
`pca_groups.<eid>.json` (written by a finished `pose2visual.xcorr.experiments.pca` run) and calls
the same `pca.plot_results` the compute step uses, so a redrawn figure looks exactly like the
original one except for the `--cameras`/`--vmax`/`--plots-subdir` override.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.replot_pca \\
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
from pose2visual.xcorr.experiments.pca import anatomical_groups, plot_results
from pose2visual.xcorr.experiments.pca_reconstruction import (
    DECOMPOSITION_SUFFIX,
    HEATMAP_PERCENTILE,
    HEATMAP_STEP,
    decomposition_text,
    draw_decomposition_heatmap,
)
from pose2visual.xcorr.plots import order_subdir, timestamp_subdir
from pose2visual.xcorr.replot import REPLOT_ORDERS
from pose2visual.xcorr.settings import (
    EXPERIMENT_PCA,
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
        description='Redraw Experiment B (PCA) figures from the saved arrays.',
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
        '--plots-subdir', type=str, default=None,
        help='figure folder inside plot_video/experiment_B_pca (default: New York time); the '
        'neuron order is appended, e.g. <name>_rastermap.',
    )
    return parser.parse_args(argv)


def load_results(out_dir: str | Path, eid: str) -> dict[str, Any]:
    """Load the saved results of one finished Experiment B run.

    Args:
        out_dir: results folder of the run, `<session_dir>/results/experiment_B_pca/<run>`.
        eid: session id.

    Returns:
        Dict with `r_original`, `r_pca` ``[C, N, B, L]``, `pca_applied`, `valid` ``[C, B]``,
        `lags_sec`, `cameras`, `names`, `groups`, `neuron_ids`, `neuron_depths`, `r_position` ``[C, N, K, L]``,
        `valid_position` ``[C, K]``, `position_names`, `summary` (the csv) and `camera_report`
        (the groups json's `cameras` entry).

    Raises:
        FileNotFoundError: if any of the four saved files is missing.
    """
    out_dir = Path(out_dir)
    speed_path = out_dir / f'pca_speed_arrays.{eid}.npz'
    position_path = out_dir / f'pca_position_arrays.{eid}.npz'
    summary_path = out_dir / f'pca_summary.{eid}.csv'
    groups_path = out_dir / f'pca_groups.{eid}.json'
    for path in (speed_path, position_path, summary_path, groups_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} not found; run pose2visual.xcorr.experiments.pca first',
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
) -> list[Path]:
    """Redraw the Experiment B figures of one finished session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        cameras: cameras to draw; `None` = all of the run.
        order: neuron order of the heatmaps, one of `REPLOT_ORDERS`.
        vmax: half-range of the colour scale; `None` takes it from the data.
        plots_subdir: run name of the figure folder inside `plot_video/experiment_B_pca`, to which
            the neuron order is appended; `None` makes a new name from the current New York time.

    Returns:
        The written png files and the figure guide.

    Raises:
        FileNotFoundError: if the session has no saved Experiment B results.
        ValueError: if a requested camera is not in the results, or `order` is unknown.
    """
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    try:
        out_dir = latest_results_dir(session_dir, EXPERIMENT_PCA, f'pca_speed_arrays.{eid}.npz')
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f'{error}; run pose2visual.xcorr.experiments.pca first',
        ) from error
    results = select_cameras(load_results(out_dir, eid), cameras)
    idx_order, order_label = resolve_order(
        config_path, eid, order, results['neuron_ids'], results['neuron_depths'],
    )
    plots_dir = plot_video_dir(session_dir, EXPERIMENT_PCA) / order_subdir(
        plots_subdir, order_label,
    )
    return plot_results(
        plots_dir, eid, results['lags_sec'], results['cameras'], results['names'],
        results['groups'], results['valid'], results['r_original'], results['r_pca'],
        idx_order, order_label, vmax, results['position_names'],
        results['r_position'], results['valid_position'], results['pca_applied'],
        results['summary'], results['camera_report'],
        anatomical_groups(config_path, results['names']),
    )


def replot_decomposition(
    config_path: str | Path,
    eid: str,
    cameras: list[str] | None = None,
    plots_subdir: str | None = None,
) -> list[Path]:
    """Redraw the PCA decomposition heatmap of one finished Experiment B run (png only).

    Reads the `pca_decomposition.<eid>.<camera>.npz` files the run saved; the colour scale is the
    run's own (its `pca_decomposition.<eid>.json`, else the data's percentile). The video is not
    redrawn (`pca_reconstruction`).

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        cameras: cameras to draw; `None` = all that were saved.
        plots_subdir: run name; the figures go to `<plots_subdir>_decomposition`; `None` makes a
            new name from the current New York time.

    Returns:
        The written png files and `decomposition.<eid>.md` (how to read them); empty (with a
        warning) if the run saved no decomposition.

    Raises:
        FileNotFoundError: if the session has no saved Experiment B results.
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.figures import heatmap_vmax

    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    out_dir = latest_results_dir(session_dir, EXPERIMENT_PCA, f'pca_speed_arrays.{eid}.npz')
    report_path = out_dir / f'pca_decomposition.{eid}.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    saved = {
        path.name.split('.')[-2]: path for path in out_dir.glob(f'pca_decomposition.{eid}.*.npz')
    }
    if not saved:
        logger.warning(f'{out_dir} has no saved PCA decomposition; its heatmap is skipped')
        return []
    wanted = [c for c in saved if cameras is None or c in cameras]
    if not wanted:
        logger.warning(f'no saved PCA decomposition for cameras {cameras}; skipped')
        return []
    plots_dir = plot_video_dir(session_dir, EXPERIMENT_PCA) / (
        f'{plots_subdir or timestamp_subdir()}_{DECOMPOSITION_SUFFIX}'
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    vmax_used: dict[str, float] = {}
    for camera in wanted:
        data = np.load(saved[camera])
        ratio = data['explained_variance_ratio']
        # runs from before the window heatmap was dropped saved this as `vmax_full`
        vmax = report.get('vmax', report.get('vmax_full', {})).get(camera)
        if vmax is None:
            vmax = heatmap_vmax(data['full_centered'], HEATMAP_PERCENTILE, HEATMAP_STEP)
        vmax_used[camera] = float(vmax)
        paths.append(draw_decomposition_heatmap(
            plots_dir, eid, camera, [str(k) for k in data['keypoints']], data['full_times'],
            data['full_centered'], data['full_rank'], data['full_residual'], float(np.sum(ratio)),
            len(ratio), float(vmax), (float(data['times'][0]), float(data['times'][-1])),
        ))
    if report:
        report = {
            **report, 'vmax': vmax_used,
            'cameras': {c: info for c, info in report['cameras'].items() if c in wanted},
        }
        md_path = plots_dir / f'decomposition.{eid}.md'
        md_path.write_text(decomposition_text(report), encoding='utf-8')
        paths.append(md_path)
    return paths


def main(argv: list[str] | None = None) -> None:
    """Redraw the Experiment B figures of every selected session; failures are listed at the end.

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
                paths += replot_decomposition(config, eid, args.cameras, plots_subdir)
                logger.info(f'wrote {[str(p) for p in paths]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
