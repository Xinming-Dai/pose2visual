"""Step B: build ``dynamic_vars.npz`` from Lightning Pose keypoints aligned to neural data.

Replaces the ONE-derived ``DYNAMIC_VARS`` of ``extract_dynamic_vars_trials`` by the speed of
every usable keypoint of one camera, predicted by your own Lightning Pose checkpoint. One call
handles several sessions and cameras (``--cameras``). The sessions are ``--eids``, else
``--eid-group`` (a named group of an ``ibl_session_ids.json``-style file, ``--eid-groups-json``),
else the ``session.eid_group`` of ``--config`` (file: ``paths.eid_groups_json``); the group of the
yaml is a default, not a constraint.
The speed (pixels per second; low-likelihood frames masked and interpolated) is binned onto the
same ``train/val/test_intervals`` grid as the spikes in ``<eid>_aligned.npz`` (``linspace(beg +
binsize, end, n_bins)``), so the output has the same contract as ``dynamic_vars_z_trials.npz``.

Each keypoint goes through likelihood filtering (frames below ``filter.likelihood_threshold`` or
with NaN x/y are masked; keypoints of ``filter.group_set.unfiltered_groups`` skip the likelihood
test and only lose their NaN x/y frames), linear interpolation over the masked frames,
optionally a PCA reprojection of the x/y of each body-part group (when the yaml's ``step_order`` pipeline contains
``2d_pose_pca``, see ``litpose_step_pipelines`` and ``litpose_pca``; the group counts come from ``PCA_n_components`` in the yaml's
``paths.body_parts_json``), then speed extraction and binning. The speed is taken in the original
pixel space on the reprojected positions, so the variables keep their keypoint names.
Groups of the yaml's ``visibility`` section (e.g. the tongue, often hidden) take none of these
steps: each becomes one binary ``<group>_visible`` column, 1 in a frame where any of its keypoints
has a likelihood above ``visibility.likelihood_threshold`` (computed on the raw likelihood, a NaN
one counts as not visible), 0 otherwise. The bin takes the value of the nearest frame, so it is
exactly 0 or 1 (no interpolation). The columns come after the speed columns in
``keypoint_names`` and no PCA touches them.
Traces of the yaml's ``motion_energy`` section (e.g. ``whisker``) are not computed from the
keypoints: the binned IBL motion energy of the camera, already on the spikes' grid, is copied from
``<eid>_aligned.npz`` (``<split>_<camera>_<trace>_motion_energy``) as the column
``<trace>_motion_energy``, after the visibility columns. No filtering, interpolation or PCA
applies.
``filter.likelihood_threshold``, ``filter.max_masked_frac`` and the PCA settings come from
``--config`` (the nested encoding yaml, shared by all steps). ``step_order`` is required there and
also decides whether PCA runs, so a missing or unknown ``step_order`` fails before any session is
processed; the ``pca`` flag is stored in ``dynamic_vars.npz`` for step C to check. Reads
``<video-preds-dir>/encoding/keypoint_names/<camera>_<eid>.json`` (step A) to decide which
keypoints are usable, and fails if that json was written with other settings than the yaml (rerun
step A). Writes
``<video-preds-dir>/<camera>_<output-name>/<eid>/dynamic_vars.npz`` and
``params.json`` (which records the Lightning Pose checkpoint directory, the ``merge`` info when
``video_preds_dir`` holds csvs merged from several models by ``merge_litpose_keypoints``, and the
config's ``step_order``; ``step_order`` does not change how the csv at ``--video-preds-dir`` is read, it
only decides the PCA). ``--output-name`` is camera-free (default: ``output.output_name`` of
``--config``); the camera is always prefixed (see ``output_folder_name``), so the cameras get
separate folders, e.g. ``left_encoding`` and ``right_encoding`` for ``--output-name encoding``.

A failing session/camera does not stop the others; all failures are listed at the end and make
the command exit non-zero.

Frame timestamps come from ``_ibl_<camera>Camera.times.<eid>.npy`` when it exists, otherwise from
the left camera file (the IBL cameras are synchronized and step 0 treats them as identical).

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \\
      --eids <eid1> <eid2> --cameras left right --video-preds-dir <model_dir>/video_preds \\
      --output-name encoding --neural-input-dir <dir> --video-timestamps-dir <dir> \\
      --config configs/encoding/litpose_encoding.yaml

    # or select a named group of eids instead of --eids; with neither flag the yaml's
    # session.eid_group is used:
    PYTHONPATH=src python -m pose2visual.encoding.extract_litpose_dynamic_vars_trials \\
      --eid-group sable --eid-groups-json scripts/delta/data/ibl_session_ids.json ...
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    DEFAULT_POSITION,
    POSITION_COLUMNS,
    bin_to_intervals,
    check_filter_settings_match,
    csv_filename,
    impute_keypoint_positions,
    keypoint_names_json_path,
    load_filter_settings,
    load_keypoint_csv,
    load_motion_energy_settings,
    load_output_name,
    load_pca_settings,
    load_position_columns,
    load_step_order,
    load_unfiltered_keypoints,
    load_video_preds_dir,
    load_visibility_features,
    load_visibility_settings,
    merge_info_path,
    motion_energy_feature_name,
    motion_energy_npz_key,
    output_folder_name,
    resolve_session_eids,
    sample_nearest_to_intervals,
    session_output_dir,
    speed_from_positions,
    visibility_features_from_df,
)
from pose2visual.encoding.litpose_pca import load_pca_n_components, reproject_keypoints

logger = logging.getLogger(__name__)

_SPLITS = ('train', 'val', 'test')


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
    parser.add_argument(
        '--eids',
        type=str,
        nargs='+',
        default=None,
        help='session ids; mutually exclusive with --eid-group '
        '(default: session.eid_group of --config)',
    )
    parser.add_argument(
        '--eid-group',
        type=str,
        default=None,
        help='name of a group in --eid-groups-json\'s `neural_sessions` (e.g. `sable`, `all`, '
        '`ear_both`, `ear_left`); mutually exclusive with --eids',
    )
    parser.add_argument(
        '--eid-groups-json',
        type=str,
        default=None,
        help='path to an `ibl_session_ids.json`-style file; '
        'default: paths.eid_groups_json of --config',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='encoding yaml holding likelihood_threshold and max_masked_frac (the same numbers '
        'step A used)',
    )
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
        default=None,
        help='directory holding the Lightning Pose prediction csvs (<model_dir>/video_preds); '
        'default: video_preds_dir of --config',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='camera-free output name; outputs go to '
        '<video-preds-dir>/<camera>_<output-name>/<eid>/ '
        '(default: output.output_name of --config)',
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
        # the IBL cameras are synchronized, so using the left file for the right camera is the
        # normal case and not worth a warning; the file used is recorded in params.json
        if path.is_file():
            return np.load(path), path
    raise FileNotFoundError(
        f'no _ibl_{camera}Camera.times.{eid}.npy or _ibl_leftCamera.times.{eid}.npy in '
        f'{video_timestamps_dir}',
    )


def resolve_pca(config_path: str | Path) -> dict[str, Any] | None:
    """Read the PCA settings of the yaml and load the groups they refer to.

    PCA is on exactly when the `step_order` pipeline contains the `2d_pose_pca` step. Groups of
    the yaml's `visibility` section are left out when they are groups of the PCA group set too:
    their keypoints are not speed features, so there is nothing to reproject.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        `None` when the `step_order` pipeline has no `2d_pose_pca` step, otherwise a dict with `body_parts_json`, `group_set`,
        `groups` (group name -> tokens) and `n_components` (group name -> requested count).

    Raises:
        ValueError: if the yaml settings or the `PCA_n_components` of the json are invalid.
    """
    settings = load_pca_settings(config_path)
    if not settings['enabled']:
        return None
    groups, n_components = load_pca_n_components(
        settings['body_parts_json'], settings['group_set'],
    )
    visibility = load_visibility_settings(config_path)
    if visibility is not None and visibility['group_set'] == settings['group_set']:
        groups = {
            name: tokens for name, tokens in groups.items()
            if name not in visibility['groups']
        }
        n_components = {name: n_components[name] for name in groups}
    return {
        'body_parts_json': settings['body_parts_json'],
        'group_set': settings['group_set'],
        'groups': groups,
        'n_components': n_components,
    }


def build_dynamic_vars(
    df: pd.DataFrame,
    keypoints: list[str],
    times: np.ndarray,
    intervals: dict[str, np.ndarray],
    likelihood_threshold: float,
    binsize: float,
    n_bins: int,
    pca: dict[str, Any] | None = None,
    no_filter_keypoints: list[str] | None = None,
    visibility: dict[str, Any] | None = None,
    position_columns: tuple[str, str] = POSITION_COLUMNS[DEFAULT_POSITION],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any] | None]:
    """Bin the speed of each keypoint and the binary visibility features onto every split.

    Args:
        df: dataframe from `load_keypoint_csv`.
        keypoints: keypoint names to extract, in output column order.
        times: frame timestamps, one per row of `df`.
        intervals: split name -> interval array `[K_split, 2]`.
        likelihood_threshold: frames below this likelihood are masked.
        binsize: bin size in seconds.
        n_bins: bins per interval.
        pca: output of `resolve_pca`; `None` takes the speed of the interpolated positions as
            they are, otherwise the positions are first reprojected per body-part group.
        no_filter_keypoints: keypoints exempt from confidence filtering; their likelihood is
            ignored, so only NaN x/y frames are masked and interpolated. `None` = none.
        visibility: output of `load_visibility_features`; each feature becomes one extra column
            after the speed columns, with the value of the nearest frame in every bin. `None` =
            no binary feature.
        position_columns: `(x_column, y_column)` of the csv holding the keypoint positions, from
            `load_position_columns`; the likelihood column is always `likelihood`.

    Returns:
        Tuple `(z_trials_time, masked_fraction, pca_info)`: split name -> `[K_split, n_bins, 1,
        D]` float32 array (`D = len(keypoints)` plus the number of visibility features),
        keypoint name -> fraction of masked frames, and the `reproject_keypoints` info (`None`
        without pca).
    """
    splits = list(intervals)
    all_intervals = np.concatenate([intervals[split] for split in splits], axis=0)
    features = {} if visibility is None else visibility_features_from_df(df, visibility)
    binned = np.zeros(
        (len(all_intervals), n_bins, len(keypoints) + len(features)), dtype=np.float32,
    )
    positions, masked_fraction = impute_keypoint_positions(
        df, keypoints, times, likelihood_threshold, position_columns, no_filter_keypoints,
    )
    pca_info = None
    if pca is not None:
        positions, pca_info = reproject_keypoints(positions, pca['groups'], pca['n_components'])
    for idx_kp, name in enumerate(keypoints):
        speed = speed_from_positions(*positions[name], times)
        binned[:, :, idx_kp] = bin_to_intervals(speed, times, all_intervals, binsize, n_bins)
    for idx_feature, values in enumerate(features.values(), start=len(keypoints)):
        binned[:, :, idx_feature] = sample_nearest_to_intervals(
            values, times, all_intervals, binsize, n_bins,
        )

    out: dict[str, np.ndarray] = {}
    start = 0
    for split in splits:
        n_split = len(intervals[split])
        out[split] = binned[start:start + n_split, :, None, :]
        start += n_split
    return out, masked_fraction, pca_info


def append_motion_energy(
    z_trials_time: dict[str, np.ndarray],
    neural_data_dict: Any,
    camera: str,
    traces: list[str],
) -> dict[str, np.ndarray]:
    """Append the binned motion-energy traces of a camera as extra feature columns.

    The traces are already binned onto the grid of the spikes in `<eid>_aligned.npz`, so they are
    copied as they are: no filtering, interpolation or PCA.

    Args:
        z_trials_time: split name -> `[K_split, n_bins, 1, D]` features from `build_dynamic_vars`.
        neural_data_dict: loaded `<eid>_aligned.npz`.
        camera: `'left'` or `'right'`, the camera whose traces are used.
        traces: `motion_energy.traces` of the config; one column each, in this order.

    Returns:
        Split name -> `[K_split, n_bins, 1, D + len(traces)]` float32 array.

    Raises:
        KeyError: if a trace is missing from `neural_data_dict`.
        ValueError: if a trace does not have shape `[K_split, n_bins, 1]`.
    """
    out: dict[str, np.ndarray] = {}
    for split, z in z_trials_time.items():
        columns = []
        for trace in traces:
            key = motion_energy_npz_key(split, camera, trace)
            if key not in neural_data_dict:
                raise KeyError(f'{key} not found in the aligned npz')
            values = np.asarray(neural_data_dict[key], dtype=np.float32)
            if values.shape != (*z.shape[:2], 1):
                raise ValueError(
                    f'{key} has shape {values.shape}, expected {(*z.shape[:2], 1)}',
                )
            columns.append(values[:, :, None, :])
        out[split] = np.concatenate([z, *columns], axis=-1)
    return out


def extract_session(
    eid: str,
    camera: str,
    video_preds_dir: str | Path,
    output_folder: str,
    neural_input_dir: str | Path,
    video_timestamps_dir: str | Path,
    likelihood_threshold: float,
    max_masked_frac: float,
    step_order: str | None = None,
    config_path: str | Path | None = None,
    pca: dict[str, Any] | None = None,
) -> Path:
    """Build `dynamic_vars.npz` for one session and camera.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        video_preds_dir: directory holding the Lightning Pose prediction csvs. Point this at the
            ensemble-median or EKS csv's folder to use those predictions; this function does not
            build that csv itself. Which columns are the position is the yaml's
            `position.columns`.
        output_folder: folder name under `video_preds_dir`, see `output_folder_name`.
        neural_input_dir: root dir containing `<eid>/<eid>_aligned.npz` and `<eid>/params.json`.
        video_timestamps_dir: directory containing `_ibl_{left,right}Camera.times.<eid>.npy`.
        likelihood_threshold: frames below this likelihood are masked and interpolated over;
            from the config, must equal the value in the step A json.
        max_masked_frac: from the config, must equal the value in the step A json.
        step_order: the `--config` yaml's `step_order` (from `load_step_order`), recorded in
            `params.json`; `None` when called outside the cli.
        config_path: encoding yaml the settings came from, recorded in `params.json`. Its
            `filter.group_set.unfiltered_groups` and `visibility` section must agree with the step A
            json; `None` (outside the cli) means no keypoint is exempt from confidence filtering
            and there is no binary feature. Its `position.columns` picks the csv columns used as
            keypoint position and must agree with the step A json (`None` = `x`, `y`).
        pca: output of `resolve_pca` (`None` = no pca); the flag and the effective number of
            components of each group are stored for step C.

    Returns:
        Path of the written `dynamic_vars.npz`.

    Raises:
        FileNotFoundError: if the step A json is missing.
        ValueError: if the step A json was written with other filter settings (thresholds or
            exempt keypoints, visibility features) than the config, the csv row count differs from the number of frame
            timestamps, or the
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
    no_filter_keypoints = (
        [] if config_path is None
        else load_unfiltered_keypoints(config_path, keypoint_info['keypoints'])
    )
    visibility = (
        None if config_path is None
        else load_visibility_features(config_path, keypoint_info['keypoints'])
    )
    position_columns = (
        POSITION_COLUMNS[DEFAULT_POSITION] if config_path is None
        else load_position_columns(config_path)
    )
    check_filter_settings_match(
        keypoint_info, likelihood_threshold, max_masked_frac, json_path, no_filter_keypoints,
        visibility, position_columns,
    )
    keypoints = keypoint_info['usable']

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

    z_trials_time, masked_fraction, pca_info = build_dynamic_vars(
        df, keypoints, times, intervals, likelihood_threshold, binsize, n_bins, pca,
        no_filter_keypoints, visibility, position_columns,
    )
    feature_names = [] if visibility is None else list(visibility['features'])
    motion_energy = None if config_path is None else load_motion_energy_settings(config_path)
    if motion_energy is not None:
        z_trials_time = append_motion_energy(
            z_trials_time, neural_data_dict, camera, motion_energy,
        )
        feature_names += [motion_energy_feature_name(trace) for trace in motion_energy]
        logger.info(f'appended {camera} motion-energy traces: {motion_energy}')
    visible_fraction = (
        {}
        if visibility is None
        else {
            feature: float(values.astype(float).mean())
            for feature, values in visibility_features_from_df(df, visibility).items()
        }
    )
    if visible_fraction:
        logger.info(f'visible fraction of the binary features: {visible_fraction}')

    output_dir = session_output_dir(video_preds_dir, output_folder, eid)
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
    npz_kwargs['keypoint_names'] = np.asarray(keypoints + feature_names)
    npz_kwargs['camera'] = np.asarray(camera)
    npz_kwargs['pca'] = np.asarray(pca is not None)
    np.savez(npz_path, **npz_kwargs)
    logger.info(f'wrote {npz_path}')

    merge_info = None
    merge_path = merge_info_path(video_preds_dir)
    if merge_path.is_file():
        # the csv was merged from several models: record which one gave which keypoint
        with merge_path.open(encoding='utf-8') as f:
            merge_info = json.load(f)
        merge_info['keypoints_per_source'] = merge_info['keypoints_per_source'].get(
            f'{camera}_{eid}',
        )
    params = {
        'eid': eid,
        'camera': camera,
        'csv_path': str(csv_path),
        'video_preds_dir': str(video_preds_dir),
        'model_dir': keypoint_info['model_dir'],
        'merge': merge_info,
        'timestamps_path': str(times_path),
        'neural_aligned_npz': str(neural_aligned_npz),
        'config': None if config_path is None else str(Path(config_path).resolve()),
        'likelihood_threshold': likelihood_threshold,
        'max_masked_frac': max_masked_frac,
        'no_filter_keypoints': no_filter_keypoints,
        'position_columns': list(position_columns),
        'speed_unit': 'pixels/second of the downsampled video frames',
        'binsize': binsize,
        'n_bins': n_bins,
        'keypoints': keypoints,
        'visibility': (
            None if visibility is None else {**visibility, 'visible_fraction': visible_fraction}
        ),
        'motion_energy': (
            None if motion_energy is None else {
                'traces': motion_energy,
                'npz_keys': [motion_energy_npz_key('train', camera, t) for t in motion_energy],
            }
        ),
        'masked_fraction': masked_fraction,
        'split_sizes': split_sizes,
        'step_order': step_order,
        'pca': (
            {'enabled': False} if pca is None else {
                'enabled': True,
                'body_parts_json': pca['body_parts_json'],
                'group_set': pca['group_set'],
                **pca_info,
            }
        ),
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
        ValueError: if the config lacks a valid `likelihood_threshold` / `max_masked_frac`, its
            `filter.group_set.unfiltered_groups` is invalid or not in `filter.group_set.name`, its
            `visibility` section is invalid, its `step_order` is missing or not a known pipeline (this also decides the pca), its
            `paths.body_parts_json` is missing for a pca pipeline or its `PCA_n_components` are
            invalid, both `--eids` and `--eid-group` are
            given, none of `--eids`, `--eid-group` and the config's `session.eid_group` is given,
            `--eid-group` is given without `--eid-groups-json` or `paths.eid_groups_json`, the
            group is unknown, neither `--video-preds-dir` nor the config's `video_preds_dir` is
            given, or neither `--output-name` nor the config's `output.output_name` is given.
        RuntimeError: after all runs, if any session/camera failed; the message lists them.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    likelihood_threshold, max_masked_frac = load_filter_settings(args.config)
    step_order = load_step_order(args.config)
    pca = resolve_pca(args.config)
    video_preds_dir = args.video_preds_dir or load_video_preds_dir(args.config)
    if video_preds_dir is None:
        raise ValueError(
            f'no video_preds_dir: pass --video-preds-dir or set video_preds_dir in {args.config}',
        )
    eids = resolve_session_eids(args.config, args.eids, args.eid_group, args.eid_groups_json)
    if eids is None:
        raise ValueError(
            f'no sessions: pass --eids or --eid-group, or set session.eid_group in '
            f'{args.config}',
        )
    output_name = args.output_name or load_output_name(args.config)
    if output_name is None:
        raise ValueError(
            f'no output name: pass --output-name or set output.output_name in {args.config}',
        )
    cameras = list(dict.fromkeys(args.cameras))

    failures: list[str] = []
    n_total = len(eids) * len(cameras)
    for eid in eids:
        for camera in cameras:
            output_folder = output_folder_name(camera, output_name)
            logger.info(f'[{eid} {camera}] extracting keypoint dynamic vars -> {output_folder}')
            try:
                extract_session(
                    eid,
                    camera,
                    video_preds_dir,
                    output_folder,
                    args.neural_input_dir,
                    args.video_timestamps_dir,
                    likelihood_threshold,
                    max_masked_frac,
                    step_order=step_order,
                    config_path=args.config,
                    pca=pca,
                )
            except Exception as exc:  # keep going; failures are reported below
                logger.exception(f'[{eid} {camera}] failed')
                failures.append(f'{eid}/{camera}: {type(exc).__name__}: {exc}')

    logger.info(f'{n_total - len(failures)}/{n_total} runs succeeded')
    if failures:
        raise RuntimeError(f'{len(failures)} of {n_total} runs failed:\n' + '\n'.join(failures))


if __name__ == '__main__':
    main()
