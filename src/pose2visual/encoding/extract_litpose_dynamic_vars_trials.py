"""Step B: build ``dynamic_vars.npz`` from Lightning Pose keypoints aligned to neural data.

Replaces the ONE-derived ``DYNAMIC_VARS`` of ``extract_dynamic_vars_trials`` by the speed of
every usable keypoint of one camera, predicted by your own Lightning Pose checkpoint. One call
handles several sessions (``--eids``) and cameras (``--cameras``). The speed
(pixels per second; low-likelihood frames masked and interpolated) is binned onto the same
``train/val/test_intervals`` grid as the spikes in ``<eid>_aligned.npz`` (``linspace(beg + binsize,
end, n_bins)``), so the output has the same contract as ``dynamic_vars_z_trials.npz``.

Reads ``<video-preds-dir>/encoding/keypoint_names/<camera>_<eid>.json`` (step A) to decide which
keypoints are usable and which likelihood threshold masks frames (set in step A). Writes
``<video-preds-dir>/<output-name>/<eid>/dynamic_vars.npz`` and
``params.json`` (which records the Lightning Pose checkpoint directory). ``--output-name`` is a
template whose ``{camera}`` placeholder is replaced by the camera, so the cameras get separate
folders (default ``{camera}_encoding``, e.g. ``left_encoding``).

A failing session/camera does not stop the others; all failures are listed at the end and make
the command exit non-zero.

Frame timestamps come from ``_ibl_<camera>Camera.times.<eid>.npy`` when it exists, otherwise from
the left camera file (the IBL cameras are synchronized and step 0 treats them as identical).

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \\
      --eids <eid1> <eid2> --cameras left right --video-preds-dir <model_dir>/video_preds \\
      --output-name '{camera}_encoding' --neural-input-dir <dir> --video-timestamps-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    bin_to_intervals,
    compute_keypoint_speed,
    csv_filename,
    get_keypoint_coords,
    keypoint_names_json_path,
    load_keypoint_csv,
    session_output_dir,
)

logger = logging.getLogger(__name__)

_SPLITS = ('train', 'val', 'test')
_CAMERA_PLACEHOLDER = '{camera}'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build dynamic_vars.npz from Lightning Pose keypoint speeds aligned to '
        'the neural intervals in <eid>_aligned.npz.',
    )
    parser.add_argument('--eids', type=str, nargs='+', required=True, help='session ids')
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        choices=CAMERAS,
        default=list(CAMERAS),
        help='cameras to process (default: left right)',
    )
    parser.add_argument(
        '--video-preds-dir',
        type=str,
        required=True,
        help='directory holding the Lightning Pose prediction csvs (<model_dir>/video_preds)',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='{camera}_encoding',
        help='folder name template; {camera} is replaced by the camera, outputs go to '
        '<video-preds-dir>/<output-name>/<eid>/ (default: {camera}_encoding)',
    )
    parser.add_argument(
        '--neural-input-dir',
        type=str,
        required=True,
        help='root dir containing <eid>/<eid>_aligned.npz and <eid>/params.json',
    )
    parser.add_argument(
        '--video-timestamps-dir',
        type=str,
        required=True,
        help='directory containing _ibl_{left,right}Camera.times.<eid>.npy',
    )
    return parser.parse_args(argv)


def load_frame_times(
    video_timestamps_dir: str | Path,
    camera: str,
    eid: str,
) -> tuple[np.ndarray, Path]:
    """Load the frame timestamps of a camera, falling back to the left camera.

    Args:
        video_timestamps_dir: directory containing `_ibl_<camera>Camera.times.<eid>.npy`.
        camera: `'left'` or `'right'`.
        eid: session id.

    Returns:
        Tuple `(times, path)` of the timestamps in seconds and the file they came from.

    Raises:
        FileNotFoundError: if neither the camera's nor the left camera's file exists.
    """
    video_timestamps_dir = Path(video_timestamps_dir)
    for cam in dict.fromkeys((camera, 'left')):
        path = video_timestamps_dir / f'_ibl_{cam}Camera.times.{eid}.npy'
        if path.is_file():
            if cam != camera:
                logger.warning(f'{camera} camera timestamps not found; using {path.name}')
            return np.load(path), path
    raise FileNotFoundError(
        f'no _ibl_{camera}Camera.times.{eid}.npy or _ibl_leftCamera.times.{eid}.npy in '
        f'{video_timestamps_dir}',
    )


def build_dynamic_vars(
    df: pd.DataFrame,
    keypoints: list[str],
    times: np.ndarray,
    intervals: dict[str, np.ndarray],
    likelihood_threshold: float,
    binsize: float,
    n_bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Bin the speed of each keypoint onto every split's intervals.

    Args:
        df: dataframe from `load_keypoint_csv`.
        keypoints: keypoint names to extract, in output column order.
        times: frame timestamps, one per row of `df`.
        intervals: split name -> interval array `[K_split, 2]`.
        likelihood_threshold: frames below this likelihood are masked.
        binsize: bin size in seconds.
        n_bins: bins per interval.

    Returns:
        Tuple `(z_trials_time, masked_fraction)`: split name -> `[K_split, n_bins, 1, D]` float32
        array, and keypoint name -> fraction of masked frames.
    """
    splits = list(intervals)
    all_intervals = np.concatenate([intervals[split] for split in splits], axis=0)
    binned = np.zeros((len(all_intervals), n_bins, len(keypoints)), dtype=np.float32)
    masked_fraction: dict[str, float] = {}
    for idx_kp, name in enumerate(keypoints):
        x, y, likelihood = get_keypoint_coords(df, name)
        speed, masked_fraction[name] = compute_keypoint_speed(
            x, y, likelihood, times, likelihood_threshold,
        )
        binned[:, :, idx_kp] = bin_to_intervals(speed, times, all_intervals, binsize, n_bins)

    out: dict[str, np.ndarray] = {}
    start = 0
    for split in splits:
        n_split = len(intervals[split])
        out[split] = binned[start:start + n_split, :, None, :]
        start += n_split
    return out, masked_fraction


def extract_session(
    eid: str,
    camera: str,
    video_preds_dir: str | Path,
    output_name: str,
    neural_input_dir: str | Path,
    video_timestamps_dir: str | Path,
) -> Path:
    """Build `dynamic_vars.npz` for one session and camera.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: output folder name (no placeholder left in it).
        neural_input_dir: root dir containing `<eid>/<eid>_aligned.npz` and `<eid>/params.json`.
        video_timestamps_dir: directory containing `_ibl_{left,right}Camera.times.<eid>.npy`.

    Returns:
        Path of the written `dynamic_vars.npz`.

    Raises:
        FileNotFoundError: if the step A json is missing.
        ValueError: if the csv row count differs from the number of frame timestamps, or the
            number of bins disagrees with the spikes in `<eid>_aligned.npz`.
    """
    video_preds_dir = Path(video_preds_dir)

    json_path = keypoint_names_json_path(video_preds_dir, camera, eid)
    if not json_path.is_file():
        raise FileNotFoundError(
            f'{json_path} not found; run pose2visual.encoding.list_litpose_keypoints first',
        )
    with json_path.open(encoding='utf-8') as f:
        keypoint_info = json.load(f)
    keypoints = keypoint_info['usable']
    likelihood_threshold = keypoint_info['likelihood_threshold']

    csv_path = video_preds_dir / csv_filename(camera, eid)
    logger.info(f'loading keypoints from {csv_path}')
    logger.info(f'lightning pose checkpoint dir: {keypoint_info["model_dir"]}')
    df = load_keypoint_csv(csv_path)
    times, times_path = load_frame_times(video_timestamps_dir, camera, eid)
    if len(df) != len(times):
        raise ValueError(
            f'{csv_path.name} has {len(df)} rows but {times_path.name} has {len(times)} '
            'timestamps',
        )

    neural_dir = Path(neural_input_dir) / eid
    neural_aligned_npz = neural_dir / f'{eid}_aligned.npz'
    logger.info(f'loading {neural_aligned_npz}')
    neural_data_dict = np.load(neural_aligned_npz, allow_pickle=True)
    with (neural_dir / 'params.json').open(encoding='utf-8') as f:
        neural_params = json.load(f)
    binsize = float(neural_params['binsize'])
    # same rule as ibl_data_utils.get_behavior_per_interval
    n_bins = int(np.ceil(neural_params['interval_len'] / binsize))
    n_bins_spikes = neural_data_dict['train_spikes'].shape[1]
    if n_bins != n_bins_spikes:
        raise ValueError(f'{n_bins} bins from params.json but spikes have {n_bins_spikes}')
    intervals = {
        split: np.asarray(neural_data_dict[f'{split}_intervals'], dtype=np.float64)
        for split in _SPLITS
    }

    z_trials_time, masked_fraction = build_dynamic_vars(
        df, keypoints, times, intervals, likelihood_threshold, binsize, n_bins,
    )

    output_dir = session_output_dir(video_preds_dir, output_name, eid)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / 'dynamic_vars.npz'
    params_path = output_dir / 'params.json'
    if params_path.is_file():
        with params_path.open(encoding='utf-8') as f:
            previous_camera = json.load(f).get('camera')
        if previous_camera != camera:
            logger.warning(
                f'overwriting {npz_path} previously built from the {previous_camera} camera '
                f'with the {camera} camera',
            )

    npz_kwargs: dict[str, np.ndarray] = {}
    split_sizes: dict[str, int] = {}
    neural_trial_idx_parts: list[np.ndarray] = []
    for split in _SPLITS:
        npz_kwargs[f'{split}_z_trials_time'] = z_trials_time[split]
        npz_kwargs[f'{split}_intervals'] = intervals[split]
        n_split = z_trials_time[split].shape[0]
        split_sizes[split] = n_split
        # local index into this split's own *_intervals, as required by
        # run_encoding._maybe_assert_latent_intervals_match_neural
        neural_trial_idx_parts.append(np.arange(n_split, dtype=np.int64))
        logger.info(f'{split}: z_trials_time shape={z_trials_time[split].shape}')
    npz_kwargs['neural_trial_idx'] = np.concatenate(neural_trial_idx_parts)
    npz_kwargs['keypoint_names'] = np.asarray(keypoints)
    npz_kwargs['camera'] = np.asarray(camera)
    np.savez(npz_path, **npz_kwargs)
    logger.info(f'wrote {npz_path}')

    params = {
        'eid': eid,
        'camera': camera,
        'csv_path': str(csv_path),
        'video_preds_dir': str(video_preds_dir),
        'model_dir': keypoint_info['model_dir'],
        'timestamps_path': str(times_path),
        'neural_aligned_npz': str(neural_aligned_npz),
        'likelihood_threshold': likelihood_threshold,
        'speed_unit': 'pixels/second of the downsampled video frames',
        'binsize': binsize,
        'n_bins': n_bins,
        'keypoints': keypoints,
        'masked_fraction': masked_fraction,
        'split_sizes': split_sizes,
    }
    with params_path.open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    logger.info(f'wrote {params_path}')
    return npz_path


def main(argv: list[str] | None = None) -> None:
    """Build `dynamic_vars.npz` for every requested session and camera.

    A failing session/camera is logged and skipped; the others still run.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        ValueError: if several cameras are requested but `--output-name` has no `{camera}`
            placeholder (the cameras would overwrite each other).
        RuntimeError: after all runs, if any session/camera failed; the message lists them.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    # drop duplicates, keep order
    eids = list(dict.fromkeys(args.eids))
    cameras = list(dict.fromkeys(args.cameras))
    if len(cameras) > 1 and _CAMERA_PLACEHOLDER not in args.output_name:
        raise ValueError(
            f'--output-name {args.output_name!r} must contain {_CAMERA_PLACEHOLDER} when '
            f'several cameras are requested ({cameras}); they would overwrite each other',
        )

    failures: list[str] = []
    n_total = len(eids) * len(cameras)
    for eid in eids:
        for camera in cameras:
            output_name = args.output_name.replace(_CAMERA_PLACEHOLDER, camera)
            logger.info(f'[{eid} {camera}] extracting keypoint dynamic vars -> {output_name}')
            try:
                extract_session(
                    eid,
                    camera,
                    args.video_preds_dir,
                    output_name,
                    args.neural_input_dir,
                    args.video_timestamps_dir,
                )
            except Exception as exc:  # keep going; failures are reported below
                logger.exception(f'[{eid} {camera}] failed')
                failures.append(f'{eid}/{camera}: {type(exc).__name__}: {exc}')

    logger.info(f'{n_total - len(failures)}/{n_total} runs succeeded')
    if failures:
        raise RuntimeError(f'{len(failures)} of {n_total} runs failed:\n' + '\n'.join(failures))


if __name__ == '__main__':
    main()
