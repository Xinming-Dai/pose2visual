"""Part of step 0: merge the keypoints of several Lightning Pose models into one prediction csv.

Different models are good at different body parts (e.g. a model trained for the ears and a
general face model). The encoding pipeline reads one csv per session and camera, so this step
builds ``<video_preds_dir>/_iblrig_<camera>Camera.downsampled.<eid>.csv`` of the encoding yaml
(``paths.video_preds_dir``) from the csvs of the models listed in its ``merge.sources``. Step 0
(``list_litpose_keypoints``) calls ``merge_sessions`` itself when the yaml has a ``merge`` section,
so steps 1 and 2 run unchanged on the merged directory.

Each source takes the keypoints of its ``groups`` (names of ``merge.group_set`` in
``paths.body_parts_json``; a keypoint belongs to a group when the first ``_``-separated segment
of its name is one of the group's tokens, the rule of ``expand_groups``). One source may use
``groups: rest``: it takes every keypoint whose token is claimed by no other source, so the
keypoints of the other models' groups are dropped from it. Every keypoint comes from exactly one
source. All sources must hold the same frames (same row count and index).

Writes, next to the merged csvs, ``encoding/merge_info.json`` with the sources and the keypoints
each one contributed per session and camera.

Merging is slow, so a session and camera is skipped when its merged csv already exists, is
recorded in ``merge_info.json`` under the same ``merge.sources`` and ``merge.group_set`` as the
yaml, and is not older than any source csv. Changing the yaml's ``merge`` or regenerating a source
therefore re-merges automatically; ``--force`` rebuilds everything.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.merge_litpose_keypoints \\
      --config configs/encoding/litpose_encoding.yaml \\
      [--eids <eid1> <eid2> | --eid-group ear_left] [--cameras left right] [--force]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    MERGE_REST,
    csv_filename,
    get_keypoint_names,
    load_keypoint_csv,
    load_merge_settings,
    load_video_preds_dir,
    merge_info_path,
    resolve_session_eids,
    run_parallel,
    write_keypoint_csv,
)
from pose2visual.lightningpose.fit_eks_latent_model import load_groups

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Merge the keypoints of several Lightning Pose models into one csv per '
        'session and camera, as set by the merge section of the encoding yaml.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml')
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
        help='name of a group in --eid-groups-json\'s `neural_sessions`; mutually exclusive '
        'with --eids',
    )
    parser.add_argument(
        '--eid-groups-json',
        type=str,
        default=None,
        help='path to an `ibl_session_ids.json`-style file; '
        'default: paths.eid_groups_json of --config',
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
        '--force',
        action='store_true',
        help='rebuild merged csvs even when an up-to-date one exists',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='session/camera csvs merged in parallel processes (default: 1)',
    )
    return parser.parse_args(argv)


def select_source_keypoints(
    keypoints_per_source: list[list[str]],
    groups_per_source: list[list[str] | str],
    groups: dict[str, list[str]],
) -> list[list[str]]:
    """Pick the keypoints each source contributes.

    Args:
        keypoints_per_source: keypoint names (csv header order) of each source.
        groups_per_source: for each source, a list of group names or `'rest'`.
        groups: group name -> tokens, as from `load_groups`.

    Returns:
        For each source, the keypoints it contributes, in its header order.

    Raises:
        ValueError: if a group is unknown or has no keypoint in its source, a keypoint is
            claimed by two sources, or two sources would contribute the same keypoint name.
    """
    claimed_tokens: set[str] = set()
    for group_names in groups_per_source:
        if group_names == MERGE_REST:
            continue
        unknown = sorted(set(group_names) - set(groups))
        if unknown:
            raise ValueError(f'unknown groups {unknown}; available: {sorted(groups)}')
        claimed_tokens.update(token for name in group_names for token in groups[name])

    selected: list[list[str]] = []
    for keypoints, group_names in zip(keypoints_per_source, groups_per_source, strict=True):
        if group_names == MERGE_REST:
            selected.append([kp for kp in keypoints if kp.split('_')[0] not in claimed_tokens])
            continue
        for name in group_names:
            if not any(kp.split('_')[0] in groups[name] for kp in keypoints):
                raise ValueError(
                    f'group {name!r} (tokens {groups[name]}) has no keypoint in its source; '
                    f'keypoints: {keypoints}',
                )
        tokens = {token for name in group_names for token in groups[name]}
        selected.append([kp for kp in keypoints if kp.split('_')[0] in tokens])

    owner: dict[str, int] = {}
    for idx_source, keypoints in enumerate(selected):
        for kp in keypoints:
            if kp in owner:
                raise ValueError(
                    f'keypoint {kp!r} would come from source {owner[kp]} and source '
                    f'{idx_source}; every keypoint must have exactly one source',
                )
            owner[kp] = idx_source
    return selected


def merge_session(
    eid: str,
    camera: str,
    source_dirs: list[str],
    groups_per_source: list[list[str] | str],
    groups: dict[str, list[str]],
    output_dir: str | Path,
) -> list[list[str]]:
    """Write the merged prediction csv of one session and camera.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        source_dirs: `video_preds` directories of the models, one per source.
        groups_per_source: for each source, a list of group names or `'rest'`.
        groups: group name -> tokens, as from `load_groups`.
        output_dir: directory the merged csv is written to.

    Returns:
        For each source, the keypoints it contributed.

    Raises:
        FileNotFoundError: if a source has no csv for this session and camera.
        ValueError: if the sources disagree on the frames, or the selection is invalid (see
            `select_source_keypoints`).
    """
    name = csv_filename(camera, eid)
    dfs: list[pd.DataFrame] = []
    for source_dir in source_dirs:
        csv_path = Path(source_dir) / name
        if not csv_path.is_file():
            raise FileNotFoundError(f'{csv_path} not found')
        dfs.append(load_keypoint_csv(csv_path))
    for source_dir, df in zip(source_dirs[1:], dfs[1:], strict=True):
        if not df.index.equals(dfs[0].index):
            raise ValueError(
                f'{name}: {source_dir} has {len(df)} rows but {source_dirs[0]} has '
                f'{len(dfs[0])}, or their frame indices differ',
            )
    selected = select_source_keypoints(
        [get_keypoint_names(df) for df in dfs], groups_per_source, groups,
    )
    parts = [
        df.loc[:, df.columns.get_level_values(1).isin(keypoints)]
        for df, keypoints in zip(dfs, selected, strict=True)
    ]
    merged = pd.concat(parts, axis=1)
    out_path = Path(output_dir) / name
    write_keypoint_csv(out_path, merged)
    logger.info(
        f'{camera} {eid}: {len(get_keypoint_names(merged))} keypoints -> {out_path}; '
        + '; '.join(f'{src}: {kps}' for src, kps in zip(source_dirs, selected, strict=True)),
    )
    return selected


def write_merge_info(
    output_dir: str | Path,
    settings: dict[str, Any],
    sessions: dict[str, list[list[str]]],
) -> Path:
    """Record the merge settings and the keypoints each source gave, updating earlier runs.

    Args:
        output_dir: the merged `video_preds` directory.
        settings: output of `load_merge_settings`.
        sessions: `<camera>_<eid>` -> keypoints contributed by each source (in source order).

    Returns:
        Path of `merge_info.json`.
    """
    path = merge_info_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if path.is_file():
        with path.open(encoding='utf-8') as f:
            previous = json.load(f)
    info = {
        'group_set': settings['group_set'],
        'sources': settings['sources'],
        # sessions merged under other settings would be misleading, so keep them only when the
        # sources are unchanged
        'keypoints_per_source': (
            previous.get('keypoints_per_source', {})
            if previous.get('sources') == settings['sources']
            and previous.get('group_set') == settings['group_set']
            else {}
        ),
    }
    info['keypoints_per_source'].update(sessions)
    with path.open('w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    return path


def merged_csv_is_current(
    eid: str,
    camera: str,
    source_dirs: list[str],
    output_dir: str | Path,
    settings: dict[str, Any],
    previous_info: dict[str, Any],
) -> bool:
    """Whether the merged csv of a session and camera can be kept instead of rebuilt.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        source_dirs: `video_preds` directories of the models, one per source.
        output_dir: directory the merged csv is written to.
        settings: output of `load_merge_settings`.
        previous_info: content of the existing `merge_info.json`, empty when there is none.

    Returns:
        `True` when the merged csv exists, `merge_info.json` records it under the same
        `sources` and `group_set` as `settings`, and the csv is not older than any source csv.
    """
    name = csv_filename(camera, eid)
    out_path = Path(output_dir) / name
    if not out_path.is_file():
        return False
    if (
        previous_info.get('sources') != settings['sources']
        or previous_info.get('group_set') != settings['group_set']
        or f'{camera}_{eid}' not in previous_info.get('keypoints_per_source', {})
    ):
        return False
    out_mtime = out_path.stat().st_mtime
    source_paths = [Path(source_dir) / name for source_dir in source_dirs]
    return all(path.is_file() and path.stat().st_mtime <= out_mtime for path in source_paths)


def _merge_job(
    eid: str,
    camera: str,
    source_dirs: list[str],
    groups_per_source: list[list[str] | str],
    groups: dict[str, list[str]],
    output_dir: str | Path,
    settings: dict[str, Any],
    previous_info: dict[str, Any],
    force: bool,
) -> list[list[str]] | None:
    """Merge one session and camera unless its merged csv is up to date.

    Runs in a worker process of `merge_sessions`; it does not touch `merge_info.json`, which the
    parent writes once.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        source_dirs: `video_preds` directories of the models, one per source.
        groups_per_source: for each source, a list of group names or `'rest'`.
        groups: group name -> tokens, as from `load_groups`.
        output_dir: directory the merged csv is written to.
        settings: output of `load_merge_settings`.
        previous_info: content of the previous `merge_info.json` (empty if none).
        force: rebuild even if an up-to-date merged csv exists.

    Returns:
        The keypoints each source contributed, or `None` if the csv was up to date and skipped.
    """
    if not force and merged_csv_is_current(
        eid, camera, source_dirs, output_dir, settings, previous_info,
    ):
        logger.info(f'{camera} {eid}: merged csv is up to date, skipping')
        return None
    return merge_session(eid, camera, source_dirs, groups_per_source, groups, output_dir)


def merge_sessions(
    config_path: str | Path,
    eids: list[str],
    cameras: list[str],
    force: bool = False,
    n_jobs: int = 1,
) -> None:
    """Merge the prediction csvs of every session and camera, as set by the yaml's `merge`.

    A session/camera whose merged csv already exists is skipped (see `merged_csv_is_current`)
    unless `force` is set. A failing session/camera is logged and skipped; the others still run.

    Args:
        config_path: encoding yaml with a `merge.sources` section and `paths.video_preds_dir`.
        eids: session ids to merge.
        cameras: cameras to merge, each `'left'` or `'right'`.
        force: rebuild every merged csv even if an up-to-date one exists.
        n_jobs: number of session/camera csvs merged in parallel processes.

    Raises:
        ValueError: if the yaml has no `merge.sources` (see `load_merge_settings`), its group set
            is missing from `paths.body_parts_json`, a group is unknown, or `n_jobs` is below 1.
        RuntimeError: after all runs, if any session/camera failed; the message lists them.
    """
    settings = load_merge_settings(config_path)
    if settings is None:
        raise ValueError(f'{config_path} has no merge.sources; nothing to merge')
    output_dir = load_video_preds_dir(config_path)
    groups = load_groups(settings['body_parts_json'], settings['group_set'])
    source_dirs = [source['video_preds_dir'] for source in settings['sources']]
    groups_per_source = [source['groups'] for source in settings['sources']]
    previous_info: dict[str, Any] = {}
    info_path = merge_info_path(output_dir)
    if info_path.is_file():
        with info_path.open(encoding='utf-8') as f:
            previous_info = json.load(f)

    sessions: dict[str, list[list[str]]] = {}
    failures: list[str] = []
    n_skipped = 0
    keys = [(eid, camera) for eid in eids for camera in cameras]
    jobs = [
        {
            'eid': eid,
            'camera': camera,
            'source_dirs': source_dirs,
            'groups_per_source': groups_per_source,
            'groups': groups,
            'output_dir': output_dir,
            'settings': settings,
            'previous_info': previous_info,
            'force': force,
        }
        for eid, camera in keys
    ]
    for (eid, camera), (selected, error) in zip(
        keys, run_parallel(_merge_job, jobs, n_jobs), strict=True,
    ):
        if error is not None:  # keep going; failures are reported below
            logger.error(f'[{eid} {camera}] failed', exc_info=error)
            failures.append(f'{eid}/{camera}: {type(error).__name__}: {error}')
        elif selected is None:
            n_skipped += 1
        else:
            sessions[f'{camera}_{eid}'] = selected
    if sessions:
        logger.info(f'wrote {write_merge_info(output_dir, settings, sessions)}')
    n_total = len(eids) * len(cameras)
    logger.info(
        f'merged {len(sessions)}/{n_total} session/camera csvs, {n_skipped} up to date '
        f'(skipped), {len(failures)} failed',
    )
    if failures:
        raise RuntimeError(
            f'{len(failures)} of {n_total} merges failed:\n' + '\n'.join(failures),
        )


def main(argv: list[str] | None = None) -> None:
    """Merge the prediction csvs of every requested session and camera.

    Normally step 0 (`list_litpose_keypoints`) does this itself when the yaml has a `merge`
    section; this entry point merges without listing the keypoints.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        ValueError: if the yaml has no `merge.sources`, or no sessions are selected (see
            `resolve_session_eids`).
        RuntimeError: if any session/camera failed (see `merge_sessions`).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    eids = resolve_session_eids(args.config, args.eids, args.eid_group, args.eid_groups_json)
    if eids is None:
        raise ValueError(
            f'no sessions: pass --eids or --eid-group, or set session.eid_group in {args.config}',
        )
    merge_sessions(
        args.config, eids, list(dict.fromkeys(args.cameras)), args.force, args.n_jobs,
    )


if __name__ == '__main__':
    main()
