"""Build ``<eid>_aligned.npz`` for one IBL session via ``ONE``.

Port of E-RayZer-private's ``extract_neural_data.py`` (see
``/u/xdai3/project3d/erayzer_cls/E-RayZer-private/docs/ibl_neural_behavior_extraction.md``).
Pulls spikes and continuous behaviors for one IBL session from the public IBL database via
``ONE``, bins them into 1-second intervals synchronized to the two-view camera timestamps,
filters units by mean firing rate, aligns spikes with behaviors, and splits trials into
train/val/test.

Writes three artifacts under ``--output-dir/<eid>/``:

* ``<eid>_aligned.npz`` — ``train/val/test_spikes``, ``train/val/test_intervals``, and
  ``train/val/test_<behavior>`` (hyphens replaced with underscores), the input of
  ``pose2visual.encoding.extract_dynamic_vars_trials``.
* ``<eid>_meta.pkl`` — cluster metadata (regions, channels, depths, uuids) for units kept
  after firing-rate filtering.
* ``params.json`` — binning/selection params used for this extraction.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.extract_neural_data \\
      --eid <eid> --one-cache-path <dir> --video-timestamps-dir <dir> --output-dir <dir>
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from one.api import ONE

from pose2visual.encoding.constants import DYNAMIC_VARS
from pose2visual.encoding.ibl_data_utils import (
    align_data,
    bin_behaviors,
    bin_spiking_data,
    create_intervals,
    list_brain_regions,
    prepare_data,
    select_brain_regions,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build <eid>_aligned.npz from an IBL session via ONE.',
    )
    parser.add_argument('--eid', type=str, required=True)
    parser.add_argument(
        '--one-cache-path', type=str, required=True, help='ONE cache_dir.',
    )
    parser.add_argument(
        '--video-timestamps-dir', type=str, required=True,
        help='directory containing _ibl_{left,right}Camera.times.{eid}.npy.',
    )
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument(
        '--num-trials', type=int, default=400,
        help='number of 1s intervals to sample before alignment (default: 400).',
    )
    parser.add_argument(
        '--fr-thresh', type=float, default=0.2,
        help='unit is kept if its mean firing rate exceeds 1 / fr_thresh Hz (default: 0.2).',
    )
    parser.add_argument(
        '--nan-thresh', type=float, default=0.3,
        help='drop a behavior if this fraction or more of its trials contain NaN/missing '
        'values; otherwise only those trials are dropped (default: 0.3).',
    )
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-workers', type=int, default=1)
    return parser.parse_args(argv)


def _beh_to_npz_keys(prefix: str, beh_dict: dict) -> dict:
    """Rename a split's behavior arrays to their ``<split>_<behavior>`` npz keys.

    Args:
        prefix: split name (``'train'``, ``'val'``, or ``'test'``).
        beh_dict: behavior name -> array, with hyphens in the name.

    Returns:
        dict of ``f'{prefix}_{name}'`` (hyphens replaced with underscores) -> float array.
    """
    return {
        f"{prefix}_{str(name).replace('-', '_')}": np.asarray(arr, dtype=float)
        for name, arr in beh_dict.items()
    }


def _save_extract_bundle(
    *,
    out_root: Path,
    eid: str,
    spikes: np.ndarray,
    split_idxs: dict[str, np.ndarray],
    split_intervals: dict[str, np.ndarray],
    split_behaviors: dict[str, dict],
    meta: dict,
    params: dict,
) -> None:
    """Write the ``<eid>_aligned.npz`` / ``<eid>_meta.pkl`` / ``params.json`` bundle.

    Args:
        out_root: root output directory; a subdirectory named ``eid`` is created under it.
        eid: session UUID.
        spikes: aligned spike array, indexable by each split's trial indices.
        split_idxs: split name -> trial indices into ``spikes``/``aligned_intervals``.
        split_intervals: split name -> interval array for that split's trials.
        split_behaviors: split name -> behavior name -> array for that split's trials.
        meta: cluster metadata to pickle.
        params: extraction params to write as JSON.
    """
    out_dir = out_root / eid
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = {}
    for split, idx_split in split_idxs.items():
        npz_payload[f'{split}_spikes'] = spikes[idx_split]
        npz_payload[f'{split}_intervals'] = np.asarray(split_intervals[split], dtype=np.float64)
        npz_payload.update(_beh_to_npz_keys(split, split_behaviors[split]))

    np.savez_compressed(out_dir / f'{eid}_aligned.npz', **npz_payload)

    with (out_dir / f'{eid}_meta.pkl').open('wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    with (out_dir / 'params.json').open('w') as f:
        json.dump(params, f, indent=2, default=str)

    logger.info(f'saved bundle under {out_dir}')


def main(argv: list[str] | None = None) -> None:
    """Extract spikes, behaviors, and intervals for one IBL session and write the bundle.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Raises:
        AssertionError: if the left and right camera timestamps differ, or the recording has
            fewer 1s intervals than ``--num-trials``.
        RuntimeError: if the aligned intervals and spikes disagree in length.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    eid = args.eid

    params = {
        'interval_len': 1,
        'binsize': 1 / 60,  # 60 frames per second
        'single_region': False,
        'fr_thresh': args.fr_thresh,
    }

    # public IBL read-only credentials
    one = ONE(
        base_url='https://openalyx.internationalbrainlab.org',
        username='intbrainlab',
        password='international',
        silent=True,
        cache_dir=args.one_cache_path,
    )

    logger.info(f'EID {eid}')

    neural_dict, behave_dict, meta_dict, _, _ = prepare_data(
        one, eid, params, n_workers=args.n_workers,
    )
    if neural_dict is None:
        logger.error(f'skip EID {eid} due to missing spike data')
        return

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    video_timestamps_dir = Path(args.video_timestamps_dir)
    left_ts_path = video_timestamps_dir / f'_ibl_leftCamera.times.{eid}.npy'
    right_ts_path = video_timestamps_dir / f'_ibl_rightCamera.times.{eid}.npy'
    left_timestamps = np.load(left_ts_path) if left_ts_path.is_file() else None
    right_timestamps = np.load(right_ts_path) if right_ts_path.is_file() else None

    ts_arrays = [a for a in (left_timestamps, right_timestamps) if a is not None]
    if not ts_arrays:
        logger.error(f'skip EID {eid}: missing {left_ts_path.name} and {right_ts_path.name}')
        return
    if len(ts_arrays) == 2 and not np.array_equal(ts_arrays[0], ts_arrays[1]):
        raise AssertionError('left and right camera timestamps must be synchronized.')
    min_timestamp = int(round(min(a.min() for a in ts_arrays))) + 1
    max_timestamp = int(round(max(a.max() for a in ts_arrays))) - 1

    intervals = create_intervals(min_timestamp, max_timestamp, params['interval_len'])
    assert len(intervals) >= args.num_trials, 'not enough intervals to sample from.'

    rng = np.random.default_rng(args.seed)
    idx_trial = rng.choice(np.arange(len(intervals)), args.num_trials, replace=False)
    intervals = intervals[idx_trial]

    bin_spikes, _ = bin_spiking_data(
        region_cluster_ids, neural_dict, intervals=intervals, n_workers=args.n_workers, **params,
    )
    logger.info(f'binned spike data: {bin_spikes.shape}')

    mean_fr = bin_spikes.sum(1).mean(0) / params['interval_len']
    idx_unit_keep = np.argwhere(mean_fr > 1 / params['fr_thresh']).flatten()
    bin_spikes = bin_spikes[..., idx_unit_keep]
    logger.info(f'# responsive units: {bin_spikes.shape[-1]} / {len(mean_fr)}')

    for key in ('cluster_regions', 'cluster_channels', 'cluster_depths', 'good_clusters', 'uuids'):
        meta_dict[key] = [meta_dict[key][idx] for idx in idx_unit_keep]

    bin_beh, _ = bin_behaviors(
        one, eid, list(DYNAMIC_VARS), intervals=intervals, allow_nans=True,
        n_workers=args.n_workers, **params,
    )

    try:
        align_bin_spikes, align_bin_beh, _, idx_trial_bad = align_data(
            bin_spikes, bin_beh, list(bin_beh.keys()), nan_thresh=args.nan_thresh,
        )
    except ValueError as e:
        logger.error(f'skip EID {eid} due to error: {e}')
        return

    idx_trial_bad = np.asarray(idx_trial_bad, dtype=np.intp).ravel()
    aligned_intervals = np.delete(np.asarray(intervals), idx_trial_bad, axis=0)
    if aligned_intervals.shape[0] != len(align_bin_spikes):
        raise RuntimeError(
            f'aligned_intervals length {aligned_intervals.shape[0]} != '
            f'align_bin_spikes trials {len(align_bin_spikes)}',
        )

    num_trials = len(aligned_intervals)
    rng = np.random.default_rng(args.seed)
    perm = rng.choice(np.arange(num_trials), num_trials, replace=False)
    n_train = int(args.train_frac * num_trials)
    n_val = int((args.train_frac + args.val_frac) * num_trials)
    split_idxs = {'train': perm[:n_train], 'val': perm[n_train:n_val], 'test': perm[n_val:]}

    split_intervals = {
        split: aligned_intervals[idx_split] for split, idx_split in split_idxs.items()
    }
    split_behaviors = {
        split: {beh: align_bin_beh[beh][idx_split] for beh in align_bin_beh}
        for split, idx_split in split_idxs.items()
    }

    _save_extract_bundle(
        out_root=Path(args.output_dir),
        eid=eid,
        spikes=align_bin_spikes,
        split_idxs=split_idxs,
        split_intervals=split_intervals,
        split_behaviors=split_behaviors,
        meta=meta_dict,
        params={**params, 'nan_thresh': args.nan_thresh},
    )
    logger.info(f'finished EID: {eid}')


if __name__ == '__main__':
    main()
