"""Step 0: spikes of one session as one continuous 60 Hz series aligned to the video.

The whole-recording counterpart of ``pose2visual.encoding.extract_neural_data``. Same session
load (``ONE``), same camera-synchronization check, same integer-second interval rule and same
bin edges, but instead of sampling 400 random 1 s trials it bins **every** contiguous 1 s interval
of the video and concatenates them, so the counts form one time series that lines up bin by bin
with the behavioral traces of ``pose2visual.xcorr.behavior``.

Units are kept when their mean firing rate over the video exceeds ``xcorr.min_fr_hz`` of the
yaml. The spikes do not depend on the camera, so the bundle is written camera-free under
``<video_preds_dir>/<xcorr.output_name>/<eid>/``:

* ``neural_continuous.npz`` — ``spikes`` (uint16 ``[T, N]``), ``bin_right_edges`` (``[T]``
  seconds), ``intervals`` (``[n_intervals, 2]``), ``binsize``.
* ``<eid>_meta.pkl`` — metadata of the kept units (regions, channels, depths, uuids).
* ``neural_params.json`` — settings and alignment diagnostics.

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.extract_neural_continuous \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --one-cache-path <dir> --video-timestamps-dir <dir>
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from pose2visual.xcorr.neural import (
    check_spike_coverage,
    concat_intervals,
    contiguous_intervals,
    interval_bin_grid,
    select_units,
    video_time_range,
)
from pose2visual.xcorr.settings import (
    freeze_config,
    load_xcorr_settings,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

BINSIZE = 1 / 60  # same bin width as the encoding pipeline (params of extract_neural_data)
META_KEYS = ('cluster_regions', 'cluster_channels', 'cluster_depths', 'good_clusters', 'uuids')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Bin the spikes of an IBL session continuously over the video (60 Hz).',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument('--one-cache-path', type=str, required=True, help='ONE cache_dir.')
    parser.add_argument(
        '--video-timestamps-dir', type=str, required=True,
        help='directory containing _ibl_{left,right}Camera.times.<eid>.npy.',
    )
    parser.add_argument('--n-workers', type=int, default=1)
    return parser.parse_args(argv)


def load_camera_timestamps(video_timestamps_dir: str | Path, eid: str) -> list[np.ndarray]:
    """Load the timestamps of every camera that has a file for the session.

    Args:
        video_timestamps_dir: directory with `_ibl_{left,right}Camera.times.<eid>.npy`.
        eid: session id.

    Returns:
        One array per existing camera file (left first).

    Raises:
        FileNotFoundError: if neither camera has a file.
    """
    arrays = []
    for camera in ('left', 'right'):
        path = Path(video_timestamps_dir) / f'_ibl_{camera}Camera.times.{eid}.npy'
        if path.is_file():
            arrays.append(np.load(path))
    if not arrays:
        raise FileNotFoundError(f'no camera timestamps for {eid} in {video_timestamps_dir}')
    return arrays


def extract_session(
    config_path: str | Path,
    eid: str,
    one_cache_path: str | Path,
    video_timestamps_dir: str | Path,
    n_workers: int = 1,
) -> Path:
    """Bin the spikes of one session over the whole video and write the bundle.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        one_cache_path: ONE cache directory.
        video_timestamps_dir: directory with the camera timestamp files.
        n_workers: worker processes for the per-interval binning.

    Returns:
        Path of the written `neural_continuous.npz`.

    Raises:
        ValueError: if the session has no spike data, the spikes do not cover the video, no unit
            passes the firing-rate threshold, or the unit metadata does not match the counts.
        AssertionError: if the left and right camera timestamps differ.
    """
    # lazy: avoids importing ONE / brainbox unless a session is extracted
    from one.api import ONE

    from pose2visual.encoding.ibl_data_utils import (
        bin_spiking_data,
        list_brain_regions,
        prepare_data,
        select_brain_regions,
    )

    settings = load_xcorr_settings(config_path)
    out_dir = xcorr_dir(config_path, eid, settings['output_name'])

    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        username='intbrainlab',
        password='international',
        silent=True,
        cache_dir=str(one_cache_path),
    )
    params = {'interval_len': 1, 'binsize': BINSIZE, 'single_region': False}
    neural_dict, _, meta_dict, _, _ = prepare_data(one, eid, params, n_workers=n_workers)
    if neural_dict is None:
        raise ValueError(f'EID {eid} has no spike data')

    min_timestamp, max_timestamp = video_time_range(
        load_camera_timestamps(video_timestamps_dir, eid),
    )
    intervals = contiguous_intervals(min_timestamp, max_timestamp)
    check_spike_coverage(neural_dict['spike_times'], intervals)
    logger.info(
        f'{len(intervals)} contiguous 1 s intervals, [{intervals[0, 0]:.1f}, '
        f'{intervals[-1, 1]:.1f}] s',
    )

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
    binned, _ = bin_spiking_data(
        region_cluster_ids, neural_dict, intervals=intervals, n_workers=n_workers, **params,
    )
    counts = concat_intervals(binned)
    n_bins = binned.shape[1]
    logger.info(f'binned spike data: {binned.shape} -> {counts.shape}')

    # the unit axis of the counts must be the cluster axis of the metadata
    if counts.shape[1] != len(meta_dict['uuids']):
        raise ValueError(
            f'{counts.shape[1]} binned units but metadata of {len(meta_dict["uuids"])} clusters',
        )
    idx_keep = select_units(counts, BINSIZE, settings['min_fr_hz'])
    if idx_keep.size == 0:
        raise ValueError(f'no unit has a mean rate above {settings["min_fr_hz"]} Hz')
    logger.info(f'# units above {settings["min_fr_hz"]} Hz: {idx_keep.size} / {counts.shape[1]}')
    n_units_total = counts.shape[1]
    counts = counts[:, idx_keep]
    if counts.max() > np.iinfo(np.uint16).max:
        raise ValueError('a spike count exceeds the uint16 range')
    meta = {key: [meta_dict[key][idx] for idx in idx_keep] for key in META_KEYS}

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / 'neural_continuous.npz'
    np.savez_compressed(
        npz_path,
        spikes=counts.astype(np.uint16),
        bin_right_edges=interval_bin_grid(intervals, BINSIZE, n_bins),
        intervals=intervals,
        binsize=np.asarray(BINSIZE),
    )
    with (out_dir / f'{eid}_meta.pkl').open('wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    spike_times = np.asarray(neural_dict['spike_times'])
    with (out_dir / 'neural_params.json').open('w', encoding='utf-8') as f:
        json.dump(
            {
                'eid': eid,
                # the path is a temporary frozen copy, so the settings themselves are recorded
                'config_yaml': Path(config_path).read_text(encoding='utf-8'),
                'binsize': BINSIZE,
                'n_bins_per_interval': n_bins,
                'min_fr_hz': settings['min_fr_hz'],
                'n_units_total': int(n_units_total),
                'n_units_kept': int(idx_keep.size),
                'video_bins_start': float(intervals[0, 0]),
                'video_bins_end': float(intervals[-1, 1]),
                'first_spike': float(spike_times.min()),
                'last_spike': float(spike_times.max()),
                'video_timestamps_dir': str(video_timestamps_dir),
            },
            f,
            indent=2,
        )
    logger.info(f'wrote {npz_path}')
    return npz_path


def main(argv: list[str] | None = None) -> None:
    """Run step 0 for every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    # a private copy of the yaml: editing the original while the job runs changes nothing
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                extract_session(
                    config, eid, args.one_cache_path, args.video_timestamps_dir, args.n_workers,
                )
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
