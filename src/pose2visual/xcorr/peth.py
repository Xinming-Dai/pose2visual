"""Event-aligned PETHs (peri-event time histograms) of every neuron, as in Figure 20.

Figure 20 of ``spike_assignments.pdf`` shows, next to the correlation strips, a reward-aligned
and a movement-onset-aligned PETH: neurons by time from the event, in the same neuron order as
every other strip. Following the report, the PETH is the *fractional rate change* against the
pre-event baseline, not a z-score: averaging many trials shrinks the baseline standard deviation
(about 30x) and a z-score saturates every panel, whereas ``(rate - baseline) / baseline`` is
readable (+1 is a doubling) and puts high- and low-rate units on the same footing.

The events come from the IBL trials table (``extract_trial_events``): reward is the feedback time
of the correct trials (``feedbackType == 1``) and movement onset the ``firstMovement_times``.
Both are session time, the same clock as the spike bins of ``neural_continuous.npz``. Only events
whose whole window lies inside the binned recording are used, the others are counted and
dropped, never padded with zeros. Neural activity is smoothed like in the correlations
(``xcorr.smooth_neural``, same sigma) and turned into spikes per second before averaging.

The same events are also cross-correlated with every neuron in the style of the behavior strips
(``compute_event_xcorr``): each event type is binned onto the spike bin grid (a count per bin) and
``r(tau) = Corr(neural(t), event(t + tau))`` is taken over ``+-xcorr.max_lag_sec``, so positive lag
means the neural activity leads the event. A neuron that fires after the event therefore responds
at a negative lag, the time reversal of the PETH. The correlations use the same colour scale as the
behavior strips instead of a change in firing rate.

Writes ``peth.<eid>.npz`` and ``peth_params.json`` into
``<video_preds_dir>/<xcorr.output_name>/<eid>/``, which ``run_xcorr`` and ``replot`` draw as the
``events`` columns of the overview figure (event cross-correlations, then PETHs). It reads only
files, so it takes about a minute.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.peth \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid>
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.xcorr.crosscorr import constant_columns, cross_correlation, smooth_neural
from pose2visual.xcorr.settings import (
    freeze_config,
    load_xcorr_settings,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

# event name -> label drawn above the strip
EVENT_LABELS = {'reward': 'reward', 'movement_onset': 'movement onset'}


def events_from_trials(trials: pd.DataFrame) -> dict[str, Any]:
    """Reward and movement-onset times from an IBL trials table.

    Reward is the feedback time of the correct trials (`feedbackType == 1`, the valve opening) and
    movement onset the `firstMovement_times` of any trial. Trials with a missing time are dropped
    (and counted); no other trial filter is applied.

    Args:
        trials: trials table with `feedback_times`, `feedbackType` and `firstMovement_times`.

    Returns:
        Dict with `reward` and `movement_onset` (sorted float arrays of session times in seconds)
        and `counts` (`n_trials`, `n_reward`, `n_reward_missing_time`, `n_movement`,
        `n_movement_missing`).

    Raises:
        ValueError: if a required column is missing.
    """
    required = ('feedback_times', 'feedbackType', 'firstMovement_times')
    missing = [col for col in required if col not in trials.columns]
    if missing:
        raise ValueError(f'trials table lacks columns {missing}; has {list(trials.columns)}')
    rewarded = trials['feedbackType'].to_numpy(dtype=float) == 1
    reward_all = trials['feedback_times'].to_numpy(dtype=np.float64)[rewarded]
    movement_all = trials['firstMovement_times'].to_numpy(dtype=np.float64)
    reward = np.sort(reward_all[np.isfinite(reward_all)])
    movement = np.sort(movement_all[np.isfinite(movement_all)])
    return {
        'reward': reward,
        'movement_onset': movement,
        'counts': {
            'n_trials': int(len(trials)),
            'n_reward': int(rewarded.sum()),
            'n_reward_missing_time': int(rewarded.sum() - reward.size),
            'n_movement': int(movement.size),
            'n_movement_missing': int(movement_all.size - movement.size),
        },
    }


def lag_grid(window_sec: tuple[float, float], binsize: float) -> np.ndarray:
    """Lags (seconds) of the PETH columns: multiples of the bin size inside the window.

    Args:
        window_sec: `(start, end)` relative to the event, containing 0.
        binsize: bin size in seconds.

    Returns:
        Array of `k * binsize` for the integers `k` with `start <= k * binsize <= end`.
    """
    k_start = int(np.ceil(window_sec[0] / binsize - 1e-9))
    k_end = int(np.floor(window_sec[1] / binsize + 1e-9))
    return np.arange(k_start, k_end + 1) * binsize


def peth_from_rates(
    rates: np.ndarray,
    bin_right_edges: np.ndarray,
    event_times: np.ndarray,
    lags_sec: np.ndarray,
    binsize: float,
) -> tuple[np.ndarray, int, int]:
    """Trial-averaged rate of every neuron around the events.

    The bin holding a time `x` is the one whose half-open interval `[edge - binsize, edge)`
    contains it. Only events whose whole window lies inside the binned recording are used.

    Args:
        rates: rate of every neuron ``[T, N]`` (spikes per second).
        bin_right_edges: right edge of every bin ``[T]`` in session time, increasing.
        event_times: event times in seconds.
        lags_sec: lags of the window in seconds ``[L]`` (see `lag_grid`).
        binsize: bin size in seconds.

    Returns:
        Tuple `(peth, n_used, n_outside)`: the mean rate ``[N, L]``, the number of events used and
        the number dropped because their window is not covered.

    Raises:
        ValueError: if `rates` and `bin_right_edges` differ in length.
    """
    rates = np.asarray(rates)
    edges = np.asarray(bin_right_edges, dtype=np.float64)
    if rates.shape[0] != edges.shape[0]:
        raise ValueError(f'{rates.shape[0]} rate bins but {edges.shape[0]} bin edges')
    first_covered, last_covered = edges[0] - binsize, edges[-1]
    events = np.asarray(event_times, dtype=np.float64)
    covered = (
        (events + lags_sec[0] >= first_covered) & (events + lags_sec[-1] < last_covered)
    )
    events = events[covered]
    if events.size == 0:
        return np.full((rates.shape[1], lags_sec.size), np.nan), 0, int(covered.size)
    idx = np.searchsorted(edges, events[:, None] + lags_sec[None, :], side='right')
    idx = np.clip(idx, 0, edges.size - 1)
    peth = rates[idx].mean(axis=0).T  # [E, L, N] -> [L, N] -> [N, L]
    return peth, int(events.size), int(covered.size - events.size)


def fractional_change(
    peth: np.ndarray,
    lags_sec: np.ndarray,
    baseline_sec: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rate change relative to the pre-event baseline, `(rate - baseline) / baseline`.

    Args:
        peth: trial-averaged rate ``[N, L]``.
        lags_sec: lags in seconds ``[L]``.
        baseline_sec: `(start, end)` of the baseline window; the bins with a lag in `[start, end)`
            are averaged.

    Returns:
        Tuple `(change, baseline)`: ``[N, L]`` and the baseline rate ``[N]``. Neurons whose
        baseline is not positive get NaN (a fractional change against zero is undefined).

    Raises:
        ValueError: if no lag falls in the baseline window.
    """
    in_base = (lags_sec >= baseline_sec[0] - 1e-9) & (lags_sec < baseline_sec[1] - 1e-9)
    if not in_base.any():
        raise ValueError(f'no PETH bin in the baseline window {list(baseline_sec)}')
    baseline = peth[:, in_base].mean(axis=1)
    safe = np.where(baseline > 0, baseline, np.nan)
    return (peth - safe[:, None]) / safe[:, None], baseline


def compute_peth(
    counts: np.ndarray,
    bin_right_edges: np.ndarray,
    binsize: float,
    events: dict[str, np.ndarray],
    smooth_sigma_bins: float,
    window_sec: tuple[float, float],
    baseline_sec: tuple[float, float],
    min_trials: int,
) -> dict[str, Any]:
    """PETH of every neuron for every event type.

    Args:
        counts: spike counts per bin ``[T, N]``.
        bin_right_edges: right edge of every bin ``[T]`` in session time.
        binsize: bin size in seconds.
        events: event name -> event times in seconds (`EVENT_LABELS` keys).
        smooth_sigma_bins: gaussian sigma (bins) applied to the counts before averaging.
        window_sec: PETH window around the event.
        baseline_sec: baseline window inside the PETH window.
        min_trials: events needed per type.

    Returns:
        Dict with `lags_sec` ``[L]`` and, per event name, a dict with `frac` ``[N, L]``,
        `rate_hz` ``[N, L]``, `base_hz` ``[N]``, `n_used`, `n_outside` and `n_total`.

    Raises:
        ValueError: if an event type has fewer than `min_trials` usable events; the message tells
            how many events there were and how many fell outside the recording (many outside
            can mean the event times and the spike bins are on different clocks).
    """
    rates = smooth_neural(counts, smooth_sigma_bins) / binsize
    lags = lag_grid(window_sec, binsize)
    out: dict[str, Any] = {'lags_sec': lags}
    for name, times in events.items():
        peth, n_used, n_outside = peth_from_rates(rates, bin_right_edges, times, lags, binsize)
        if n_used < min_trials:
            raise ValueError(
                f'{name}: only {n_used} usable events of {len(times)} ({n_outside} have their '
                f'window outside the binned recording [{bin_right_edges[0]:.1f}, '
                f'{bin_right_edges[-1]:.1f}] s); need at least {min_trials}',
            )
        frac, baseline = fractional_change(peth, lags, baseline_sec)
        out[name] = {
            'frac': frac, 'rate_hz': peth, 'base_hz': baseline, 'n_used': n_used,
            'n_outside': n_outside, 'n_total': int(len(times)),
        }
    return out


def event_trace(
    bin_right_edges: np.ndarray,
    binsize: float,
    event_times: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Number of events in every spike bin, as a regressor on the spike bin grid.

    The bin holding a time `x` is the one whose half-open interval `[edge - binsize, edge)`
    contains it. The grid is a concatenation of intervals, so an event that falls outside the
    recording or into a gap between intervals has no bin; it is dropped and counted, never
    assigned to a neighbouring bin.

    Args:
        bin_right_edges: right edge of every bin ``[T]`` in session time, increasing.
        binsize: bin size in seconds.
        event_times: event times in seconds.

    Returns:
        Tuple `(trace, n_used, n_outside)`: events per bin ``[T]``, the number of events placed
        and the number dropped.
    """
    edges = np.asarray(bin_right_edges, dtype=np.float64)
    events = np.asarray(event_times, dtype=np.float64)
    idx = np.searchsorted(edges, events, side='right')
    inside = idx < edges.size
    # tolerance for the float error of edge - binsize
    inside[inside] = events[inside] >= edges[idx[inside]] - binsize - 1e-9
    trace = np.bincount(idx[inside], minlength=edges.size).astype(np.float64)
    return trace, int(inside.sum()), int((~inside).sum())


def compute_event_xcorr(
    neural: np.ndarray,
    bin_right_edges: np.ndarray,
    binsize: float,
    events: dict[str, np.ndarray],
    max_lag_bins: int,
    min_trials: int,
) -> dict[str, Any]:
    """Cross-correlation of every neuron with every event type, like the behavior strips.

    Args:
        neural: smoothed neural activity ``[T, N]`` (`smooth_neural`).
        bin_right_edges: right edge of every bin ``[T]`` in session time.
        binsize: bin size in seconds.
        events: event name -> event times in seconds (`EVENT_LABELS` keys).
        max_lag_bins: largest lag in bins.
        min_trials: events needed per type.

    Returns:
        Dict with `lags_sec` ``[L]``, `r` ``[N, E, L]`` (events in the order of `events`; NaN rows
        for neurons without variance, which `run_xcorr` drops too), `n_used` and `n_outside`
        (lists of ``E`` ints).

    Raises:
        ValueError: if an event type has fewer than `min_trials` events inside the binned
            recording; the message tells how many were dropped.
    """
    traces, n_used, n_outside = [], [], []
    for name, times in events.items():
        trace, used, outside = event_trace(bin_right_edges, binsize, times)
        if used < min_trials:
            raise ValueError(
                f'{name}: only {used} events of {len(times)} fall into a spike bin ({outside} '
                f'outside the recording or in a gap); need at least {min_trials}',
            )
        traces.append(trace)
        n_used.append(used)
        n_outside.append(outside)
    keep = ~constant_columns(neural)
    r = np.full((neural.shape[1], len(traces), 2 * max_lag_bins + 1), np.nan)
    r[keep] = cross_correlation(neural[:, keep], np.stack(traces, axis=1), max_lag_bins)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * binsize
    return {'lags_sec': lags, 'r': r, 'n_used': n_used, 'n_outside': n_outside}


def compute_peth_session(config_path: str | Path, eid: str) -> Path:
    """PETHs of one session from its saved neural bundle and trial events.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.

    Returns:
        Path of the written `peth.<eid>.npz`.

    Raises:
        FileNotFoundError: if step 0 or the trial-event extraction has not been run.
        ValueError: see `compute_peth`.
    """
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    neural_path = session_dir / 'neural_continuous.npz'
    events_path = session_dir / 'trial_events.npz'
    meta_path = session_dir / f'{eid}_meta.pkl'
    for path, step in (
        (neural_path, 'extract_neural_continuous'), (meta_path, 'extract_neural_continuous'),
        (events_path, 'extract_trial_events'),
    ):
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found; run pose2visual.xcorr.{step} first')
    neural = np.load(neural_path)
    with meta_path.open('rb') as f:
        uuids = [str(u) for u in pickle.load(f)['uuids']]
    if len(uuids) != neural['spikes'].shape[1]:
        raise ValueError(f'{len(uuids)} unit ids but {neural["spikes"].shape[1]} spike columns')
    events_file = np.load(events_path)
    events = {name: events_file[f'{name}_times'] for name in EVENT_LABELS}
    binsize = float(neural['binsize'])
    result = compute_peth(
        neural['spikes'], neural['bin_right_edges'], binsize, events,
        settings['smooth_sigma_bins'], tuple(settings['peth_window_sec']),
        tuple(settings['peth_baseline_sec']), settings['peth_min_trials'],
    )
    event_xcorr = compute_event_xcorr(
        smooth_neural(neural['spikes'], settings['smooth_sigma_bins']), neural['bin_right_edges'],
        binsize, events, int(round(settings['max_lag_sec'] / binsize)),
        settings['peth_min_trials'],
    )
    arrays: dict[str, Any] = {
        'lags_sec': result['lags_sec'], 'neuron_ids': np.asarray(uuids),
        'xcorr_lags_sec': event_xcorr['lags_sec'],
    }
    params: dict[str, Any] = {
        'eid': eid,
        'window_sec': settings['peth_window_sec'],
        'baseline_sec': settings['peth_baseline_sec'],
        'smooth_sigma_bins': settings['smooth_sigma_bins'],
        'max_lag_sec': settings['max_lag_sec'],
        'binsize': binsize,
        'recording_start': float(neural['bin_right_edges'][0] - binsize),
        'recording_end': float(neural['bin_right_edges'][-1]),
        'events': {},
        'trial_events_json': str(session_dir / 'trial_events.json'),
    }
    for idx_e, name in enumerate(EVENT_LABELS):
        res = result[name]
        arrays[f'{name}_xcorr'] = event_xcorr['r'][:, idx_e].astype(np.float32)
        arrays[f'{name}_xcorr_n_used'] = np.asarray(event_xcorr['n_used'][idx_e])
        arrays[f'{name}_frac'] = res['frac'].astype(np.float32)
        arrays[f'{name}_rate_hz'] = res['rate_hz'].astype(np.float32)
        arrays[f'{name}_base_hz'] = res['base_hz']
        arrays[f'{name}_n_used'] = np.asarray(res['n_used'])
        arrays[f'{name}_n_total'] = np.asarray(res['n_total'])
        n_nan = int(np.isnan(res['frac'][:, 0]).sum())
        params['events'][name] = {
            'n_total': res['n_total'], 'n_used': res['n_used'],
            'n_outside_recording': res['n_outside'], 'n_neurons_without_baseline': n_nan,
            'xcorr_n_used': event_xcorr['n_used'][idx_e],
            'xcorr_n_outside': event_xcorr['n_outside'][idx_e],
        }
        logger.info(
            f'{name}: {res["n_used"]} / {res["n_total"]} events used '
            f'({res["n_outside"]} outside the recording), {n_nan} neurons without a baseline; '
            f'xcorr: {event_xcorr["n_used"][idx_e]} events binned',
        )
    out_path = session_dir / f'peth.{eid}.npz'
    np.savez_compressed(out_path, **arrays)
    (session_dir / 'peth_params.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    return out_path


def load_peth(
    session_dir: str | Path,
    eid: str,
    neuron_ids: np.ndarray,
) -> dict[str, Any] | None:
    """Load the saved PETHs and arrange the neurons like a cross-correlation run.

    Args:
        session_dir: `<video_preds_dir>/<xcorr.output_name>/<eid>`.
        eid: session id.
        neuron_ids: unit ids ``[N]`` of the cross-correlation arrays, in their order. The run
            drops neurons without variance, so these can be a subset of the PETH's neurons.

    Returns:
        `None` when no `peth.<eid>.npz` exists, else a dict with `labels` (`EVENT_LABELS` values),
        `values` ``[E, N, L]`` (the fractional change, neurons in the order of `neuron_ids`),
        `lags_sec`, `n` (events used per type) and `xcorr`: a dict with `values` ``[E, N, Lx]``
        (event cross-correlations in the same neuron order) and `lags_sec` ``[Lx]``, or `None`
        when the file was written before they existed (rerun `compute_peth_session`).

    Raises:
        ValueError: if a neuron of `neuron_ids` has no PETH (the files belong to different
            extractions) or the file lacks an event.
    """
    path = Path(session_dir) / f'peth.{eid}.npz'
    if not path.is_file():
        return None
    data = np.load(path)
    missing = [n for n in EVENT_LABELS if f'{n}_frac' not in data.files]
    if missing:
        raise ValueError(f'{path} lacks events {missing}')
    position = {str(uuid): idx for idx, uuid in enumerate(data['neuron_ids'])}
    absent = [str(n) for n in neuron_ids if str(n) not in position]
    if absent:
        raise ValueError(
            f'{len(absent)} neurons of the correlation run have no PETH in {path} '
            f'(e.g. {absent[:3]}); the files come from different extractions',
        )
    idx = [position[str(n)] for n in neuron_ids]
    xcorr = None
    if 'xcorr_lags_sec' in data.files and all(f'{n}_xcorr' in data.files for n in EVENT_LABELS):
        xcorr = {
            'values': np.stack([data[f'{name}_xcorr'][idx] for name in EVENT_LABELS]),
            'lags_sec': data['xcorr_lags_sec'],
        }
    else:
        logger.info(f'{path} has no event cross-correlations: rerun pose2visual.xcorr.peth')
    return {
        'labels': [EVENT_LABELS[name] for name in EVENT_LABELS],
        'values': np.stack([data[f'{name}_frac'][idx] for name in EVENT_LABELS]),
        'lags_sec': data['lags_sec'],
        'n': [int(data[f'{name}_n_used']) for name in EVENT_LABELS],
        'xcorr': xcorr,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Reward / movement-onset PETHs of every neuron (fractional rate change).',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Compute the PETHs of every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                logger.info(f'wrote {compute_peth_session(config, eid)}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
