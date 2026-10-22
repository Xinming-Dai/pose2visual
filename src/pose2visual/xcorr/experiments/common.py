"""Shared pieces of the optional xcorr experiments (rank-1 subtraction, relative motion).

Both experiments start from a finished baseline run (``pose2visual.xcorr.run_xcorr``) and must not
change it. This module reads the baseline's saved arrays, rebuilds exactly the baseline's neural
activity (same smoothing, same neuron selection, checked against the saved neuron ids and lag
axis), creates the experiment's own output folder next to the baseline files and relabels the
baseline overview figure for quantities that are not Pearson correlations. Nothing here writes
into the baseline's files.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.xcorr.crosscorr import constant_columns, smooth_neural
from pose2visual.xcorr.replot import load_results, select_results
from pose2visual.xcorr.settings import load_xcorr_settings, xcorr_dir

logger = logging.getLogger(__name__)

EXPERIMENT_A = 'experiment_A_rank1'
EXPERIMENT_B = 'experiment_B_relative_motion'
BASELINE_COLORBAR_LABEL = 'correlation'


def load_baseline(
    config_path: str | Path,
    eid: str,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Read the saved results of the baseline xcorr run of one session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        cameras: cameras to keep; `None` = all of the run.
        exclude: glob patterns of behaviors to leave out.

    Returns:
        The dict of `replot.select_results` (`r` ``[C, N, B, L]``, `lags_sec`, `cameras`,
        `names`, `groups`, `valid`, `neuron_order`, `neuron_ids`, `table`, `order_label`) plus
        `session_dir`, `settings` (the xcorr settings the baseline was run with, from its
        ``xcorr_params.json`` so a later edit of the yaml cannot change the comparison; the yaml
        values if the run did not save them) and `exclude_patterns` (the keypoint patterns the
        baseline left out, `[]` if it did not save them).

    Raises:
        FileNotFoundError: if the baseline arrays or summary are missing.
        ValueError: if the baseline lacks a required entry or nothing is left after the selection.
    """
    yaml_settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, yaml_settings['output_name'])
    results = select_results(load_results(session_dir, eid), cameras, exclude)
    settings = dict(yaml_settings)
    exclude_patterns: list[str] = []
    params_path = session_dir / 'xcorr_params.json'
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding='utf-8'))
        saved = params.get('settings', {})
        settings.update({k: v for k, v in saved.items() if k in yaml_settings})
        exclude_patterns = list(params.get('exclude_patterns', []))
    return {
        **results, 'session_dir': session_dir, 'settings': settings,
        'exclude_patterns': exclude_patterns,
    }


def load_neural(
    session_dir: str | Path,
    eid: str,
    settings: dict[str, Any],
    neuron_ids: np.ndarray | None,
    lags_sec: np.ndarray,
) -> dict[str, Any]:
    """Rebuild the smoothed neural activity of the baseline run.

    Same steps as `run_xcorr.run_session`: the spike bundle of step 0, gaussian smoothing with
    `settings['smooth_sigma_bins']` over the whole recording, then removal of neurons without
    variance.

    Args:
        session_dir: baseline session folder (holds ``neural_continuous.npz``).
        eid: session id.
        settings: xcorr settings of the baseline (`load_baseline`).
        neuron_ids: neuron ids saved by the baseline, in its neuron order; `None` skips the id
            check (an old run).
        lags_sec: lag axis saved by the baseline.

    Returns:
        Dict with `neural` ``[T, N]``, `intervals`, `bin_right_edges`, `binsize` and
        `max_lag_bins`.

    Raises:
        FileNotFoundError: if step 0 has not been run.
        ValueError: if the rebuilt neurons differ from the baseline's (ids or order) or the lag
            axis implied by the settings differs from the saved one.
    """
    session_dir = Path(session_dir)
    npz_path = session_dir / 'neural_continuous.npz'
    if not npz_path.is_file():
        raise FileNotFoundError(f'{npz_path} not found; run extract_neural_continuous first')
    bundle = np.load(npz_path)
    binsize = float(bundle['binsize'])
    neural = smooth_neural(bundle['spikes'], settings['smooth_sigma_bins'])
    keep = ~constant_columns(neural)
    neural = neural[:, keep]
    if neuron_ids is not None:
        with (session_dir / f'{eid}_meta.pkl').open('rb') as f:
            meta = pickle.load(f)
        ids = np.asarray(meta['uuids'], dtype=object)[keep].astype(str)
        if ids.shape != np.asarray(neuron_ids).shape or not np.array_equal(ids, neuron_ids):
            raise ValueError(
                'the neurons rebuilt from neural_continuous.npz differ from those of the '
                'baseline xcorr arrays; rerun the baseline or the experiment on the same data',
            )
    max_lag_bins = int(round(settings['max_lag_sec'] / binsize))
    lags_check = np.arange(-max_lag_bins, max_lag_bins + 1) * binsize
    if lags_check.shape != np.asarray(lags_sec).shape or not np.allclose(lags_check, lags_sec):
        raise ValueError('the lag axis of the settings differs from the baseline lag axis')
    return {
        'neural': neural,
        'intervals': bundle['intervals'],
        'bin_right_edges': bundle['bin_right_edges'],
        'binsize': binsize,
        'max_lag_bins': max_lag_bins,
    }


def experiment_dir(session_dir: str | Path, name: str) -> Path:
    """Create and return the output folder of one experiment inside the session folder.

    Args:
        session_dir: baseline session folder.
        name: `EXPERIMENT_A` or `EXPERIMENT_B`.

    Returns:
        `<session_dir>/<name>`, created if missing. Only this folder is ever written.
    """
    path = Path(session_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def relabel_overview(fig, suptitle: str, colorbar_label: str) -> None:
    """Replace the Pearson wording of a baseline overview figure.

    `figures.plot_xcorr_overview` titles its figure with the correlation definition and labels the
    colorbar ``correlation``. For rank-1 components, SVD residuals and differences those words
    would be wrong, so the baseline function is left as it is and its figure is relabelled.

    Args:
        fig: figure returned by `plot_xcorr_overview`.
        suptitle: new figure title.
        colorbar_label: new label of the colorbar(s) that read ``correlation``.
    """
    fig.suptitle(suptitle)
    for ax in fig.axes:
        if ax.get_ylabel() == BASELINE_COLORBAR_LABEL:
            ax.set_ylabel(colorbar_label)
