"""All figures of the cross-correlation, built from the saved results.

``write_plots`` is the one place that turns the correlation arrays and the summary table into
figures. ``run_xcorr`` calls it right after the computation and ``replot`` calls it on the files a
finished run left behind, so the two can never produce different figures. Matplotlib is imported
inside the functions that draw.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# number of example curves when asked for (--n-examples 12); none are drawn by default
N_EXAMPLES = 12
DEFAULT_N_EXAMPLES = 0
# same clock and format as the run folders of fit_eks_latent_model
RUN_TIMEZONE = ZoneInfo('America/New_York')


def timestamp_subdir() -> str:
    """Name of a new figure folder: the current New York time, `YYYYmmdd_HHMMSS`.

    Returns:
        The folder name, e.g. `'20261021_153012'`.
    """
    return datetime.now(RUN_TIMEZONE).strftime('%Y%m%d_%H%M%S')


def select_examples(
    table: pd.DataFrame,
    n_examples: int,
) -> list[tuple[str, int, str]]:
    """Pick the camera-neuron-behavior triples shown as example curves.

    The best neuron of each camera-behavior pair (by peak absolute correlation), restricted to
    significant pairs when any pair is significant, then the `n_examples` with the largest peaks.

    Args:
        table: output of `summarize_peaks` over all cameras (with a `camera` column), and
            `significant` when a test was run.
        n_examples: number of triples to return.

    Returns:
        List of `(camera, neuron_idx, behavior)`, strongest first.
    """
    pool = table
    if 'significant' in table and table['significant'].any():
        pool = table[table['significant']]
    best = pool.loc[pool.groupby(['camera', 'behavior'])['peak_abs_r'].idxmax()]
    best = best.sort_values('peak_abs_r', ascending=False).head(n_examples)
    return [(row.camera, int(row.neuron_idx), row.behavior) for row in best.itertuples()]


def write_plots(
    plots_dir: str | Path,
    eid: str,
    r: np.ndarray,
    lags_sec: np.ndarray,
    names: list[str],
    groups: list[str],
    cameras: list[str],
    valid: np.ndarray,
    idx_order: np.ndarray,
    table: pd.DataFrame,
    order_label: str,
    vmax: float | None = None,
    n_examples: int = N_EXAMPLES,
    null_p95: np.ndarray | None = None,
    peth: dict[str, Any] | None = None,
    peth_vmax: float = 1.0,
) -> dict[str, Any]:
    """Draw and save the overview heatmap and the example curves of one session.

    Args:
        plots_dir: folder the figures are written to; it is created.
        eid: session id, used in the titles and file names.
        r: correlations ``[C, N, B, L]`` in the neuron order of `table` (`neuron_idx`).
        lags_sec: lag in seconds of each of the ``L`` entries.
        names: the ``B`` behavior names, grouped.
        groups: body-part group of each behavior.
        cameras: the ``C`` camera names.
        valid: boolean ``[C, B]``, whether a camera has a behavior.
        idx_order: permutation ``[N]`` of the neurons for the heatmap rows.
        table: per camera-neuron-behavior table (`camera`, `neuron_idx`, `behavior`, `peak_r`,
            `peak_abs_r`, `peak_lag_sec`, `region`, and `q_value` / `significant` after a test).
        order_label: name of the neuron order, written on the y axis.
        vmax: half-range of the colour scale; `None` takes it from the data.
        n_examples: number of example curves.
        null_p95: optional ``[C, N, B]`` 95th percentile of the shift-null peak |r|, drawn as a
            dashed threshold in the example curves; `None` leaves it out.
        peth: optional event-aligned PETHs (see `figures.plot_xcorr_overview`; `values` ``[E, N,
            Lp]`` in the neuron order of `r` **before** `idx_order` is applied; an optional `xcorr`
            entry holds the event cross-correlations ``[E, N, Lx]`` in that order too, drawn on the
            colour scale of `r`); `None` draws none.
        peth_vmax: half-range of the PETH colour scale.

    Returns:
        Dict with `heatmap` and `examples` (each the list of written png / pdf paths; `examples`
        is empty when the table has no pair) and `vmax`, the colour scale used.
    """
    from pose2visual.xcorr.figures import (
        heatmap_vmax,
        plot_example_curves,
        plot_xcorr_overview,
        save_figure,
    )

    plots_dir = Path(plots_dir)
    scale = vmax if vmax is not None else heatmap_vmax(r)
    logger.info(f'heatmap colour scale +-{scale:.2f}')
    shown_peth = None
    if peth is not None:
        shown_peth = {**peth, 'values': np.asarray(peth['values'])[:, idx_order]}
        if peth.get('xcorr') is not None:
            event_r = np.asarray(peth['xcorr']['values'])[:, idx_order]
            shown_peth['xcorr'] = {**peth['xcorr'], 'values': event_r}
            peak = float(np.nanmax(np.abs(event_r)))
            if peak > scale:
                logger.warning(
                    f'event cross-correlation peaks at {peak:.2f}, above the colour scale '
                    f'+-{scale:.2f}; pass a larger vmax to avoid saturation',
                )
    fig = plot_xcorr_overview(
        r[:, idx_order], lags_sec, names, cameras, groups, valid, eid, scale,
        ylabel=f'neurons ({order_label} order)', peth=shown_peth, peth_vmax=peth_vmax,
    )
    heatmap = save_figure(fig, plots_dir / f'heatmap.all.{eid}')

    examples = []
    for camera, idx_n, name in select_examples(table, n_examples):
        idx_cam, idx_b = cameras.index(camera), names.index(name)
        pair = table[
            (table['camera'] == camera) & (table['neuron_idx'] == idx_n)
            & (table['behavior'] == name)
        ].iloc[0]
        label = f'{camera}: {name} | neuron {idx_n} ({pair["region"]})'
        label += f'\npeak {pair["peak_r"]:+.2f} @ {pair["peak_lag_sec"]:+.2f} s'
        flag = None
        if 'significant' in pair:
            flag = bool(pair['significant'])
            label += f', q={pair["q_value"]:.3g}'
        examples.append({
            'curve': r[idx_cam, idx_n, idx_b],
            'threshold': None if null_p95 is None else float(null_p95[idx_cam, idx_n, idx_b]),
            'title': label,
            'significant': flag,
        })
    example_paths: list[Path] = []
    if examples:
        fig = plot_example_curves(examples, lags_sec, f'{eid} | example cross-correlations')
        example_paths = save_figure(fig, plots_dir / f'example_curves.{eid}')
    return {'heatmap': heatmap, 'examples': example_paths, 'vmax': scale}
