"""Shared pieces of the optional xcorr experiments (PCA, relative motion).

Experiments B and C do not depend on Experiment A (baseline, ``pose2visual.xcorr.run_xcorr``):
`compute_original` recomputes the original keypoint-speed cross-correlations they are compared
against with the baseline's own functions (`run_xcorr.run_cameras`, same yaml settings, same
neural smoothing and neuron selection), so B, C and A can run at the same time. Only the neuron
order of the figures is borrowed from A when its arrays exist (`resolve_order`). This module also
relabels the baseline overview figure for quantities that are not Pearson correlations and writes
the guide to the figure names of a plot folder (`write_figure_guide`). Nothing here writes into
the baseline's files.
"""

import json
import logging
import pickle
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.encoding.litpose_keypoints import load_encoding_yaml
from pose2visual.xcorr.crosscorr import constant_columns, smooth_neural
from pose2visual.xcorr.figures import TEXT_COLOR
from pose2visual.xcorr.replot import REPLOT_ORDERS
from pose2visual.xcorr.run_xcorr import CAMERAS, grouped_behaviors, run_cameras
from pose2visual.xcorr.settings import (
    EXPERIMENT_BASELINE,
    latest_results_dir,
    load_xcorr_settings,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

BASELINE_COLORBAR_LABEL = 'correlation'
# characters of an 8 pt title that fit in one inch of figure width
TITLE_CHARS_PER_INCH = 14


def load_neural(
    session_dir: str | Path,
    eid: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Build the smoothed neural activity exactly as the baseline run does.

    Same steps as `run_xcorr.run_session`: the spike bundle of step 0, gaussian smoothing with
    `settings['smooth_sigma_bins']` over the whole recording, then removal of neurons without
    variance.

    Args:
        session_dir: session folder (holds ``neural_continuous.npz`` and ``<eid>_meta.pkl``).
        eid: session id.
        settings: xcorr settings (`settings.load_xcorr_settings`).

    Returns:
        Dict with `neural` ``[T, N]``, `neuron_ids` ``[N]``, `neuron_depths` ``[N]``,
        `intervals`, `bin_right_edges`, `binsize`, `max_lag_bins` and `lags_sec`.

    Raises:
        FileNotFoundError: if step 0 has not been run.
        ValueError: if no neuron is left.
    """
    session_dir = Path(session_dir)
    npz_path = session_dir / 'neural_continuous.npz'
    meta_path = session_dir / f'{eid}_meta.pkl'
    for path in (npz_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; run extract_neural_continuous first')
    bundle = np.load(npz_path)
    binsize = float(bundle['binsize'])
    neural = smooth_neural(bundle['spikes'], settings['smooth_sigma_bins'])
    keep = ~constant_columns(neural)
    if not keep.all():
        logger.warning(f'dropping {int((~keep).sum())} neurons without variance')
    neural = neural[:, keep]
    if neural.shape[1] == 0:
        raise ValueError('no neuron left to correlate')
    with meta_path.open('rb') as f:
        meta = pickle.load(f)
    max_lag_bins = int(round(settings['max_lag_sec'] / binsize))
    return {
        'neural': neural,
        'neuron_ids': np.asarray(meta['uuids'], dtype=object)[keep].astype(str),
        'neuron_depths': np.asarray(meta['cluster_depths'], dtype=np.float64)[keep],
        'intervals': bundle['intervals'],
        'bin_right_edges': bundle['bin_right_edges'],
        'binsize': binsize,
        'max_lag_bins': max_lag_bins,
        'lags_sec': np.arange(-max_lag_bins, max_lag_bins + 1) * binsize,
    }


def compute_original(
    config_path: str | Path,
    eid: str,
    video_timestamps_dir: str | Path,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Original keypoint-speed cross-correlations of one session, computed like the baseline.

    Runs the baseline's own per-camera code (`run_xcorr.run_cameras`, without the permutation
    test) on the neural activity of `load_neural`, so the result equals the `r` of an Experiment A
    run with the same yaml, cameras and `exclude`, without reading or waiting for that run.

    Args:
        config_path: path of the encoding yaml (use `settings.freeze_config` to protect it).
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `run_xcorr.CAMERAS`.
        exclude: glob patterns of keypoints / features to leave out (e.g. `['d*']`).
        n_jobs: cameras processed in parallel.

    Returns:
        Dict with `r` ``[C, N, B, L]`` (NaN where a camera lacks a behavior), `lags_sec`,
        `cameras`, `names`, `groups`, `valid` ``[C, B]``, `neuron_ids`, `neuron_depths`,
        `neural` (the dict of `load_neural`), `session_dir`, `settings` and `exclude_patterns`.

    Raises:
        FileNotFoundError: if step 0 has not been run.
        ValueError: if `cameras` has unknown names.
        RuntimeError: if a camera's behavior cannot be built (the message names the camera).
    """
    cameras = list(dict.fromkeys(cameras or CAMERAS))
    unknown = sorted(set(cameras) - set(CAMERAS))
    if unknown:
        raise ValueError(f'unknown cameras {unknown}; use {list(CAMERAS)}')
    exclude = list(exclude or [])
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    neural = load_neural(session_dir, eid, settings)
    results = run_cameras(
        cameras, n_jobs,
        config_path=config_path, eid=eid, video_timestamps_dir=video_timestamps_dir,
        intervals=neural['intervals'], bin_right_edges=neural['bin_right_edges'],
        binsize=neural['binsize'], exclude=exclude, neural=neural['neural'],
        max_lag_bins=neural['max_lag_bins'], shifts=None,
    )
    names, groups = grouped_behaviors(
        load_encoding_yaml(config_path), [res['names'] for res in results],
    )
    lags_sec = neural['lags_sec']
    r = np.full((len(results), neural['neural'].shape[1], len(names), lags_sec.size), np.nan)
    for idx_cam, res in enumerate(results):
        for idx_b, name in enumerate(res['names']):
            r[idx_cam, :, names.index(name), :] = res['r'][:, idx_b, :]
    # same float32 rounding as the baseline's saved arrays
    r = r.astype(np.float32)
    logger.info(
        f'original xcorr: {r.shape[1]} neurons, behaviors per camera '
        f'{ {res["camera"]: len(res["names"]) for res in results} }',
    )
    return {
        'r': r,
        'lags_sec': lags_sec,
        'cameras': cameras,
        'names': names,
        'groups': groups,
        'valid': ~np.isnan(r).all(axis=(1, 3)),
        'neuron_ids': neural['neuron_ids'],
        'neuron_depths': neural['neuron_depths'],
        'neural': neural,
        'session_dir': session_dir,
        'settings': settings,
        'exclude_patterns': exclude,
    }


def baseline_order(
    config_path: str | Path,
    eid: str,
    neuron_ids: np.ndarray,
) -> tuple[np.ndarray, str] | None:
    """The neuron order saved by Experiment A, when it ran on the same neurons.

    Args:
        config_path: path of the encoding yaml (locates the baseline folder).
        eid: session id.
        neuron_ids: neuron ids of the experiment, in its neuron order.

    Returns:
        `(idx_order, order_label)` (e.g. the rastermap order and `'rastermap'`), or `None` if
        there is no baseline run or it has other neurons.
    """
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    try:
        base_dir = latest_results_dir(session_dir, EXPERIMENT_BASELINE, f'xcorr_arrays.{eid}.npz')
    except FileNotFoundError:
        return None
    arrays_path = base_dir / f'xcorr_arrays.{eid}.npz'
    data = np.load(arrays_path)
    if 'neuron_ids' not in data.files or not np.array_equal(
        data['neuron_ids'].astype(str), np.asarray(neuron_ids).astype(str),
    ):
        logger.warning(f'{arrays_path} has other neurons; its neuron order is not used')
        return None
    order_label = 'saved'
    params_path = base_dir / 'xcorr_params.json'
    if params_path.is_file():
        order_label = json.loads(params_path.read_text(encoding='utf-8')).get('order', 'saved')
    return np.asarray(data['neuron_order']), order_label


def resolve_order(
    config_path: str | Path,
    eid: str,
    order: str,
    neuron_ids: np.ndarray,
    neuron_depths: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Neuron order of the experiment figures and its label.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        order: one of `REPLOT_ORDERS`. `'saved'` takes Experiment A's saved order (usually
            rastermap) so the figures line up with the baseline's, and falls back to depth order
            when A has not run or has other neurons.
        neuron_ids: neuron ids of the experiment.
        neuron_depths: probe depth of each neuron.

    Returns:
        `(idx_order, order_label)`.

    Raises:
        ValueError: for an unknown order.
    """
    if order not in REPLOT_ORDERS:
        raise ValueError(f'unknown order {order!r}; use one of {list(REPLOT_ORDERS)}')
    if order == 'saved':
        saved = baseline_order(config_path, eid, neuron_ids)
        if saved is not None:
            return saved
        logger.info('no matching Experiment A run: neurons are drawn in depth order')
        order = 'depth'
    if order == 'depth':
        return np.argsort(np.asarray(neuron_depths, dtype=np.float64), kind='stable'), 'depth'
    return np.arange(len(neuron_ids)), 'none'


def write_figure_guide(
    plots_dir: str | Path,
    eid: str,
    title: str,
    order_label: str,
    scale: float,
    corrected_stem: str,
    figures: list[tuple[str, str]],
) -> Path:
    """Write `figures.<eid>.md` next to the figures: what every file name in the folder means.

    Args:
        plots_dir: folder the figures were written into.
        eid: session id.
        title: experiment title, e.g. `'Experiment B: PCA'`.
        order_label: name of the neuron order the heatmaps are drawn in.
        scale: shared half-range of the original and corrected heatmaps.
        corrected_stem: stem of the corrected heatmap, e.g. `'pca_speed'`.
        figures: `(stem, meaning)` of each figure drawn, in drawing order; `stem` excludes the
            eid and the suffix.

    Returns:
        The written markdown file.
    """
    lines = [
        f'# {title}: figures ({eid})',
        '',
        f'Every figure is saved as `<stem>.{eid}.png` (png only); the parts of a stem are '
        'separated by `.`, so `heatmap.original_speed` is a `heatmap` of variant '
        '`original_speed`.',
        '',
        '## Reading a `heatmap.*` figure',
        '',
        '- One row per camera; one narrow strip per keypoint, neurons on y and lag on x. Columns '
        'of one body-part group sit together under one group label; a left- and a right-camera '
        'keypoint of the same body part share a column.',
        f'- Neurons are drawn in **{order_label}** order (the same order in every strip, and in '
        f'`heatmap.original_speed` and `heatmap.{corrected_stem}`).',
        '- The number on a strip is the max |value| of that strip.',
        f'- `heatmap.original_speed` and `heatmap.{corrected_stem}` have the same neurons, '
        f'keypoints, lags and colour scale (+-{scale:.2f}), so they can be compared by eye; a '
        'column that was not corrected is identical in both.',
        '- Values are neuron-behavior cross-correlations (Pearson r at each lag).',
        '',
        '## Figures',
        '',
        '| file | what it shows |',
        '|---|---|',
    ]
    lines += [f'| `{stem}.{eid}` | {meaning} |' for stem, meaning in figures]
    path = Path(plots_dir) / f'figures.{eid}.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def relabel_overview(fig, suptitle: str, colorbar_label: str) -> None:
    """Replace the Pearson wording of a baseline overview figure.

    `figures.plot_xcorr_overview` titles its figure with the correlation definition and labels the
    colorbar ``correlation``. For rank-1 components, SVD residuals and differences those words
    would be wrong, so the baseline function is left as it is and its figure is relabelled.

    Args:
        fig: figure returned by `plot_xcorr_overview`.
        suptitle: new figure title; wrapped to the figure width, since narrow overviews (few
            columns) are much narrower than a one-line title.
        colorbar_label: new label of the colorbar(s) that read ``correlation``.
    """
    width = max(40, int(fig.get_figwidth() * TITLE_CHARS_PER_INCH))
    fig.suptitle(textwrap.fill(suptitle, width), fontsize=8, color=TEXT_COLOR)
    for ax in fig.axes:
        if ax.get_ylabel() == BASELINE_COLORBAR_LABEL:
            ax.set_ylabel(colorbar_label)
