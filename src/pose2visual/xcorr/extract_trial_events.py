"""Step 0b: reward and movement-onset times of one session, for the event-aligned PETHs.

Reads the IBL trials table through ``ONE`` (the same cache and credentials as
``extract_neural_continuous``; it uses ``ibl_data_utils.load_trials_and_mask`` and keeps the
unfiltered table), or directly from a ``_ibl_trials.table.pqt`` with ``--trials-table`` (pandas
only, no ``ONE``). Reward = ``feedback_times`` of the trials with ``feedbackType == 1``, movement
onset = ``firstMovement_times`` (see ``peth.events_from_trials``). The times are session time, the
clock of the spike bins; nothing is shifted.

Writes ``trial_events.npz`` (``reward_times``, ``movement_onset_times``) and
``trial_events.json`` (counts, source) under ``<video_preds_dir>/<xcorr.output_name>/<eid>/``.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.extract_trial_events \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --one-cache-path <dir>

    # without ONE, from a cached table (one session only):
    PYTHONPATH=src python -m pose2visual.xcorr.extract_trial_events \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> --trials-table <pqt>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.xcorr.peth import events_from_trials
from pose2visual.xcorr.settings import (
    freeze_config,
    load_xcorr_settings,
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
        description='Save the reward and movement-onset times of an IBL session.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument('--one-cache-path', type=str, default=None, help='ONE cache_dir.')
    parser.add_argument(
        '--trials-table', type=str, default=None,
        help='read this _ibl_trials.table.pqt instead of using ONE (a single session).',
    )
    args = parser.parse_args(argv)
    if (args.one_cache_path is None) == (args.trials_table is None):
        parser.error('give exactly one of --one-cache-path or --trials-table')
    return args


def load_trials_table(eid: str, one_cache_path: str | Path) -> pd.DataFrame:
    """The unfiltered IBL trials table of a session, through ONE.

    Args:
        eid: session id.
        one_cache_path: ONE cache directory.

    Returns:
        The trials dataframe (one row per trial).
    """
    # lazy: avoids importing ONE / brainbox unless the table is read through ONE
    from one.api import ONE

    from pose2visual.encoding.ibl_data_utils import load_trials_and_mask

    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        username='intbrainlab',
        password='international',
        silent=True,
        cache_dir=str(one_cache_path),
    )
    trials, _ = load_trials_and_mask(one, eid)
    return trials


def extract_session(
    config_path: str | Path,
    eid: str,
    one_cache_path: str | Path | None = None,
    trials_table: str | Path | None = None,
) -> Path:
    """Save the event times of one session.

    Args:
        config_path: path of the encoding yaml (locates the session folder).
        eid: session id.
        one_cache_path: ONE cache directory; used when `trials_table` is `None`.
        trials_table: path of a `_ibl_trials.table.pqt` to read instead of using ONE.

    Returns:
        Path of the written `trial_events.npz`.

    Raises:
        ValueError: if neither source is given, or the table lacks a required column.
    """
    if trials_table is not None:
        trials = pd.read_parquet(trials_table)
        source = str(trials_table)
    elif one_cache_path is not None:
        trials = load_trials_table(eid, one_cache_path)
        source = f'ONE {one_cache_path}'
    else:
        raise ValueError('give a ONE cache path or a trials table')
    events = events_from_trials(trials)
    settings = load_xcorr_settings(config_path)
    out_dir = xcorr_dir(config_path, eid, settings['output_name'])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / 'trial_events.npz'
    np.savez(
        npz_path,
        reward_times=events['reward'], movement_onset_times=events['movement_onset'],
    )
    info = {
        'eid': eid,
        'source': source,
        'counts': events['counts'],
        'reward_time_range': [float(events['reward'].min()), float(events['reward'].max())],
        'movement_onset_time_range': [
            float(events['movement_onset'].min()), float(events['movement_onset'].max()),
        ],
    }
    (out_dir / 'trial_events.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    logger.info(f'{eid}: {events["counts"]} -> {npz_path}')
    return npz_path


def main(argv: list[str] | None = None) -> None:
    """Save the event times of every selected session; failures are listed at the end.

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
        if args.trials_table is not None and len(eids) != 1:
            raise SystemExit('--trials-table is one session\'s table: give exactly one eid')
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                extract_session(config, eid, args.one_cache_path, args.trials_table)
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
