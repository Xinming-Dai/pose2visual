"""Step A, part 2: save the PCA-reprojected keypoint positions of every listed session and camera.

``list_litpose_keypoints`` (step 0) calls ``save_pca_sessions`` after it listed the usable
keypoints, when the yaml's ``step_order`` pipeline contains ``2d_pose_pca``. Step B
(``extract_litpose_dynamic_vars_trials``) reads the result with ``load_pca_csv`` instead of
computing the PCA itself, so the PCA is computed once and every script sees the same positions.

Per session and camera, the usable keypoints of the step A json go through:

1. likelihood masking and linear interpolation over the frame timestamps
   (``litpose_keypoints.impute_keypoint_positions``, exactly as step B and ``pose2visual.xcorr``);
2. per body-part group, a projection of the x/y pixel positions onto the top ``PCA_n_components``
   principal components of the group, mapped back to pixels (``litpose_pca.reproject_keypoints``).
   The groups are those of ``pca.group_set`` in ``paths.body_parts_json``, without the groups of
   the yaml's ``visibility`` section when they belong to the same group set (see ``resolve_pca``).

Output, in ``<video_preds_dir>/encoding/pca/<pca.group_set>/``:

* ``_iblrig_<camera>Camera.downsampled.<eid>.csv`` (the name of the prediction csv it was made
  from; the folder says it is the PCA): a Lightning Pose style csv (``scorer`` / ``bodyparts`` /
  ``coords`` header rows) with ``x``, ``y`` and ``likelihood`` for what the encoder sees: first
  every usable keypoint, in the order of the step A json, with the reprojected x/y (keypoints of
  no group: only interpolated), then the keypoints of the yaml's ``visibility`` groups with
  their raw x/y, unmasked and not reprojected. Always plain ``x``/``y`` columns whatever
  ``position.columns`` selected as input; ``likelihood`` is the original one everywhere, so step
  B turns the visibility keypoints into their binary feature from this file. Unusable keypoints
  are not in it.
* ``pca_config.<eid>.<camera>.json``: how the csv was made (command, yaml text, ``settings``
  that step B compares with its own, groups and effective component counts, explained variance).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    get_keypoint_coords,
    get_keypoint_names,
    impute_keypoint_positions,
    load_frame_times,
    load_keypoint_csv,
    load_pca_settings,
    load_visibility_settings,
    pca_config_json_path,
    pca_dir_path,
    pca_positions_csv_path,
    run_parallel,
    write_keypoint_csv,
)
from pose2visual.encoding.litpose_pca import load_pca_n_components, reproject_keypoints

logger = logging.getLogger(__name__)

PIPELINE = (
    'mask frames with likelihood < likelihood_threshold or NaN x/y (exempt keypoints: NaN x/y '
    'only)',
    'linearly interpolate the masked frames over the frame timestamps',
    'per body-part group: center the x/y pixel positions, project onto the top n_components '
    'principal components (fit on all frames), map back to pixels',
)


def resolve_pca(config_path: str | Path) -> dict[str, Any] | None:
    """Read the PCA settings of the yaml and load the groups they refer to.

    PCA is on exactly when the `step_order` pipeline contains the `2d_pose_pca` step. Groups of
    the yaml's `visibility` section are left out when they are groups of the PCA group set too:
    their keypoints are not speed features, so there is nothing to reproject.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        `None` when the `step_order` pipeline has no `2d_pose_pca` step, otherwise a dict with
        `body_parts_json`, `group_set`, `groups` (group name -> tokens) and `n_components` (group
        name -> requested count).

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


def visibility_keypoints(keypoint_info: dict[str, Any]) -> list[str]:
    """Keypoints behind the binary visibility features, in feature order.

    Args:
        keypoint_info: the step A json of the session and camera.

    Returns:
        The keypoint names of all `visibility` features; empty without a `visibility` section.
    """
    visibility = keypoint_info.get('visibility')
    if visibility is None:
        return []
    return list(dict.fromkeys(
        name for members in visibility['features'].values() for name in members
    ))


def pca_csv_keypoints(keypoint_info: dict[str, Any]) -> list[str]:
    """Keypoints of the PCA csv: the usable ones, then those of the visibility features.

    Args:
        keypoint_info: the step A json of the session and camera.

    Returns:
        Keypoint names in csv order.
    """
    return list(keypoint_info['usable']) + visibility_keypoints(keypoint_info)


def pca_settings_record(keypoint_info: dict[str, Any], pca: dict[str, Any]) -> dict[str, Any]:
    """Everything the PCA csv depends on, as stored in its json and compared by step B.

    Args:
        keypoint_info: the step A json of the session and camera (see
            `list_litpose_keypoints.build_keypoint_info`).
        pca: output of `resolve_pca`.

    Returns:
        Json-serializable dict with `usable`, `visibility_keypoints`, `likelihood_threshold`,
        `max_masked_frac`, `no_filter_keypoints`, `position_columns`, `group_set`, `groups` and
        `n_components_requested`.
    """
    return {
        'usable': list(keypoint_info['usable']),
        'visibility_keypoints': visibility_keypoints(keypoint_info),
        'likelihood_threshold': keypoint_info['likelihood_threshold'],
        'max_masked_frac': keypoint_info['max_masked_frac'],
        'no_filter_keypoints': list(keypoint_info['no_filter_keypoints']),
        'position_columns': list(keypoint_info['position_columns']),
        'group_set': pca['group_set'],
        'groups': {name: list(tokens) for name, tokens in pca['groups'].items()},
        'n_components_requested': dict(pca['n_components']),
    }


def pca_positions_frame(
    df: pd.DataFrame,
    keypoint_info: dict[str, Any],
    pca: dict[str, Any],
    times: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute and PCA-reproject the usable keypoints of one session and camera.

    Args:
        df: dataframe from `load_keypoint_csv`, the csv the step A json describes.
        keypoint_info: the step A json of the session and camera.
        pca: output of `resolve_pca`.
        times: frame timestamps in seconds, one per row of `df`.

    Returns:
        Tuple `(frame, info)`: a dataframe with the `(scorer, bodypart, coord)` columns `x`, `y`
        and `likelihood` of every usable keypoint (reprojected), then of every keypoint of the
        visibility features (raw), and the `reproject_keypoints` info.

    Raises:
        ValueError: if a usable keypoint belongs to several groups, or has fewer than two
            unmasked frames.
    """
    keypoints = list(keypoint_info['usable'])
    position_columns = tuple(keypoint_info['position_columns'])
    positions, _ = impute_keypoint_positions(
        df, keypoints, times, keypoint_info['likelihood_threshold'], position_columns,
        keypoint_info['no_filter_keypoints'],
    )
    positions_hat, info = reproject_keypoints(positions, pca['groups'], pca['n_components'])
    scorer = df.columns.get_level_values(0)[0]
    data = {}
    for name in keypoints:
        x, y = positions_hat[name]
        likelihood = get_keypoint_coords(df, name, position_columns)[2]
        data[(scorer, name, 'x')] = x
        data[(scorer, name, 'y')] = y
        data[(scorer, name, 'likelihood')] = likelihood
    # the visibility keypoints skip masking, interpolation and the PCA
    for name in visibility_keypoints(keypoint_info):
        x, y, likelihood = get_keypoint_coords(df, name, position_columns)
        data[(scorer, name, 'x')] = x
        data[(scorer, name, 'y')] = y
        data[(scorer, name, 'likelihood')] = likelihood
    frame = pd.DataFrame(data, index=df.index)
    frame.columns = pd.MultiIndex.from_tuples(list(data))
    return frame, info


def save_pca_session(
    config_path: str | Path,
    camera: str,
    eid: str,
    keypoint_info: dict[str, Any],
    pca: dict[str, Any],
    video_preds_dir: str | Path,
    video_timestamps_dir: str | Path,
) -> Path:
    """Compute the PCA of one session and camera and write its csv and config json.

    Args:
        config_path: encoding yaml the settings came from; its text is stored in the json.
        camera: `'left'` or `'right'`.
        eid: session id.
        keypoint_info: the step A json of the session and camera.
        pca: output of `resolve_pca`.
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        video_timestamps_dir: directory containing `_ibl_{left,right}Camera.times.<eid>.npy`.

    Returns:
        Path of the written csv.

    Raises:
        FileNotFoundError: if no frame timestamps exist.
        ValueError: if the csv and the timestamps differ in length, or see `pca_positions_frame`.
    """
    csv_path = Path(keypoint_info['csv_path'])
    df = load_keypoint_csv(csv_path)
    times, times_path = load_frame_times(video_timestamps_dir, camera, eid)
    if len(df) != len(times):
        raise ValueError(
            f'{csv_path.name} has {len(df)} rows but {times_path.name} has {len(times)} '
            'timestamps',
        )
    frame, info = pca_positions_frame(df, keypoint_info, pca, times)

    group_set = pca['group_set']
    out_csv = pca_positions_csv_path(video_preds_dir, group_set, eid, camera)
    out_json = pca_config_json_path(video_preds_dir, group_set, eid, camera)
    write_keypoint_csv(out_csv, frame)
    config = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'eid': eid,
        'camera': camera,
        'csv_path': str(csv_path),
        'timestamps_path': str(times_path),
        'n_frames': int(len(df)),
        'config_path': str(Path(config_path).resolve()),
        'config_yaml': Path(config_path).read_text(encoding='utf-8'),
        'body_parts_json': pca['body_parts_json'],
        'pipeline': list(PIPELINE),
        'settings': pca_settings_record(keypoint_info, pca),
        'pca': info,
        'pca_csv': {
            'path': str(out_csv),
            'columns': 'x, y and likelihood (original) per keypoint: the usable ones with '
            'reprojected pixels, then the visibility ones with raw pixels',
        },
    }
    out_json.write_text(json.dumps(config, indent=2), encoding='utf-8')
    logger.info(f'  wrote {out_csv} and {out_json.name}')
    return out_csv


def save_pca_sessions(
    config_path: str | Path,
    items: list[tuple[str, str, dict[str, Any]]],
    video_preds_dir: str | Path,
    video_timestamps_dir: str | Path,
    n_jobs: int = 1,
) -> None:
    """Save the PCA of every session and camera, as set by the yaml's `step_order`.

    Does nothing when the `step_order` pipeline has no `2d_pose_pca` step. A failing
    session/camera is logged and skipped; the others still run.

    Args:
        config_path: encoding yaml.
        items: `(camera, eid, keypoint_info)` of every listed session and camera, with
            `keypoint_info` the step A json content.
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        video_timestamps_dir: directory containing `_ibl_{left,right}Camera.times.<eid>.npy`.
        n_jobs: number of session/camera csvs processed in parallel processes.

    Raises:
        ValueError: if the yaml or the `PCA_n_components` of its `paths.body_parts_json` are
            invalid, or `n_jobs` is below 1.
        RuntimeError: after all runs, if any session/camera failed; the message lists them.
    """
    pca = resolve_pca(config_path)
    if pca is None:
        logger.info('step_order has no 2d_pose_pca step; no PCA saved')
        return
    out_dir = pca_dir_path(video_preds_dir, pca['group_set'])
    logger.info(f'PCA of group set {pca["group_set"]} -> {out_dir}')
    failures: list[str] = []
    jobs = [
        {
            'config_path': config_path,
            'camera': camera,
            'eid': eid,
            'keypoint_info': keypoint_info,
            'pca': pca,
            'video_preds_dir': video_preds_dir,
            'video_timestamps_dir': video_timestamps_dir,
        }
        for camera, eid, keypoint_info in items
    ]
    for (camera, eid, _), (_, error) in zip(
        items, run_parallel(save_pca_session, jobs, n_jobs), strict=True,
    ):
        if error is not None:  # keep going; failures are reported below
            logger.error(f'[{eid} {camera}] PCA failed', exc_info=error)
            failures.append(f'{eid}/{camera}: {type(error).__name__}: {error}')
    logger.info(f'PCA saved for {len(items) - len(failures)}/{len(items)} session/camera csvs')
    if failures:
        raise RuntimeError(f'{len(failures)} of {len(items)} PCAs failed:\n' + '\n'.join(failures))


def load_pca_csv(
    video_preds_dir: str | Path,
    pca: dict[str, Any],
    keypoint_info: dict[str, Any],
    eid: str,
    camera: str,
    source_csv_path: str | Path,
    n_frames: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the PCA csv of one session and camera, checking it belongs to the current settings.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        pca: output of `resolve_pca`.
        keypoint_info: the step A json of the session and camera.
        eid: session id.
        camera: `'left'` or `'right'`.
        source_csv_path: the prediction csv the PCA was computed from.
        n_frames: number of rows of that csv.

    Returns:
        Tuple `(frame, info)`: the dataframe from `load_keypoint_csv` (x/y/likelihood of every
        usable keypoint, then of the visibility keypoints) and the `reproject_keypoints` info
        stored in the json.

    Raises:
        FileNotFoundError: if the csv or its json is missing.
        ValueError: if the json was written with other settings than the current yaml and step A
            json, the prediction csv is newer than the PCA csv, or the csv has other keypoints or
            another number of frames.
    """
    group_set = pca['group_set']
    csv_path = pca_positions_csv_path(video_preds_dir, group_set, eid, camera)
    json_path = pca_config_json_path(video_preds_dir, group_set, eid, camera)
    for path in (csv_path, json_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} not found; run pose2visual.encoding.list_litpose_keypoints (step 0) '
                'with this config first',
            )
    rerun = 'rerun pose2visual.encoding.list_litpose_keypoints (step 0) with this config'
    with json_path.open(encoding='utf-8') as f:
        record = json.load(f)
    expected = pca_settings_record(keypoint_info, pca)
    differing = [key for key in expected if record['settings'].get(key) != expected[key]]
    if differing:
        raise ValueError(
            f'{json_path} was written with other settings than the config ({differing}); '
            f'{rerun}',
        )
    if csv_path.stat().st_mtime < Path(source_csv_path).stat().st_mtime:
        raise ValueError(f'{csv_path} is older than {source_csv_path}; {rerun}')
    frame = load_keypoint_csv(csv_path)
    if len(frame) != n_frames:
        raise ValueError(f'{csv_path.name} has {len(frame)} rows but the csv has {n_frames}')
    if get_keypoint_names(frame) != pca_csv_keypoints(keypoint_info):
        raise ValueError(
            f'{csv_path.name} holds other keypoints than the usable and visibility ones of the '
            f'step A json; {rerun}',
        )
    return frame, record['pca']
