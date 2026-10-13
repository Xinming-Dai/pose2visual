"""Shared helpers for encoding from Lightning Pose keypoint predictions.

Lightning Pose writes one csv per camera video, ``_iblrig_<camera>Camera.downsampled.<eid>.csv``,
with three header rows (``scorer`` / ``bodyparts`` / ``coords``) and ``x, y, likelihood``
columns per keypoint, one row per video frame. This module holds the file naming conventions,
csv reading, keypoint speed computation and trial binning shared by
``list_litpose_keypoints``, ``extract_litpose_dynamic_vars_trials`` and
``run_litpose_encoding``.

Directory layout under ``video_preds_dir`` (``<model_dir>/video_preds``)::

    encoding/merge_info.json                        optional, written when the yaml has a
        ``merge`` section (``merge_litpose_keypoints`` builds the csvs of this directory from
        several models)
    encoding/keypoint_names/<camera>_<eid>.json     step A, one per csv
    <camera>_<output_name>/<eid>/dynamic_vars.npz   step B
    <camera>_<output_name>/<eid>/base/encoding_result.npz
        step C, base fit, reused by variants
    <camera>_<output_name>/<eid>/<variant_name>/encoding_result.npz
        step C, one per additional set (the default name is the sorted additional keypoints
        joined by ``_``, plus a hash suffix when that folder holds a run with other settings)

``<camera>_<output_name>`` is built by ``output_folder_name`` from the camera and the yaml's
camera-free ``output.output_name``.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pose2visual.encoding.litpose_step_pipelines import (
    load_step_pipelines,
    resolve_step_pipeline,
)
from pose2visual.lightningpose.fit_eks_latent_model import load_groups

CAMERAS = ('left', 'right')
DEFAULT_PCA_GROUP_SET = 'nonoverlapping_analysis_groups'
DEFAULT_FILTER_GROUP_SET = 'nonoverlapping_analysis_groups'
DEFAULT_MERGE_GROUP_SET = 'nonoverlapping_analysis_groups'
# `merge.sources[*].groups` value meaning "every keypoint no other source claims"
MERGE_REST = 'rest'
_PCA_STEP = '2d_pose_pca'
# `anatomical_group` value meaning "base / additional lists hold explicit keypoint names"
ANATOMICAL_GROUP_ALL = 'all'
# nested yaml layout: section -> {key in the section: key of the flat dict `load_encoding_yaml`
# returns}; `step_order` is the only top-level key outside the sections
ENCODING_YAML_SECTIONS = {
    'session': {'eid': 'eid', 'eid_group': 'eid_group', 'camera': 'camera'},
    'paths': {
        'video_preds_dir': 'video_preds_dir',
        'body_parts_json': 'body_parts_json',
        'eid_groups_json': 'eid_groups_json',
    },
    'output': {'output_name': 'output_name', 'variant_name': 'variant_name'},
    'filter': {
        'likelihood_threshold': 'likelihood_threshold',
        'max_masked_frac': 'max_masked_frac',
    },
    'pca': {'group_set': 'pca_group_set'},
    'merge': {'group_set': 'merge_group_set', 'sources': 'merge_sources'},
    'anatomical_group': {
        'name': 'anatomical_group',
        'base_keypoints': 'base_keypoints',
        'additional_keypoints': 'additional_keypoints',
    },
}
# keys of a section that are themselves mappings: (section, key) -> {key in the mapping: key of
# the flat dict}, e.g. `filter.group_set.unfiltered_groups`
ENCODING_YAML_SUBSECTIONS = {
    ('filter', 'group_set'): {
        'name': 'filter_group_set',
        'unfiltered_groups': 'unfiltered_groups',
    },
}
# key of the flat dict -> its place in the yaml, e.g. `output_name` -> `output.output_name`
ENCODING_YAML_NESTED_NAMES = {
    **{
        flat: f'{section}.{key}'
        for section, keys in ENCODING_YAML_SECTIONS.items()
        for key, flat in keys.items()
    },
    **{
        flat: f'{section}.{key}.{sub_key}'
        for (section, key), sub_keys in ENCODING_YAML_SUBSECTIONS.items()
        for sub_key, flat in sub_keys.items()
    },
}

# <eid> never contains an underscore, which keeps ``*_temporal_norm.csv`` files out
_CSV_PATTERN = re.compile(r'^_iblrig_(left|right)Camera\.downsampled\.([^_]+)\.csv$')


def csv_filename(camera: str, eid: str) -> str:
    """Name of the Lightning Pose prediction csv for one camera and session.

    Args:
        camera: `'left'` or `'right'`.
        eid: session id.

    Returns:
        File name such as `_iblrig_leftCamera.downsampled.<eid>.csv`.
    """
    return f'_iblrig_{camera}Camera.downsampled.{eid}.csv'


def list_keypoint_csvs(video_preds_dir: str | Path) -> list[tuple[str, str, Path]]:
    """Find every prediction csv in a ``video_preds`` directory.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.

    Returns:
        Sorted list of `(camera, eid, csv_path)`; `*_temporal_norm.csv` files are skipped.
    """
    found = []
    for path in Path(video_preds_dir).iterdir():
        match = _CSV_PATTERN.match(path.name)
        if match:
            found.append((match.group(1), match.group(2), path))
    return sorted(found, key=lambda item: (item[1], item[0]))


def keypoint_names_json_path(video_preds_dir: str | Path, camera: str, eid: str) -> Path:
    """Path of the step A json listing the keypoints of one camera and session.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        camera: `'left'` or `'right'`.
        eid: session id.

    Returns:
        `<video_preds_dir>/encoding/keypoint_names/<camera>_<eid>.json`.
    """
    return Path(video_preds_dir) / 'encoding' / 'keypoint_names' / f'{camera}_{eid}.json'


def merge_info_path(video_preds_dir: str | Path) -> Path:
    """Path of the json recording which model each merged keypoint came from.

    Args:
        video_preds_dir: directory holding the merged Lightning Pose prediction csvs.

    Returns:
        `<video_preds_dir>/encoding/merge_info.json`, written by `merge_litpose_keypoints`.
    """
    return Path(video_preds_dir) / 'encoding' / 'merge_info.json'


def output_folder_name(camera: str, output_name: str) -> str:
    """Name of the folder under ``video_preds_dir`` that holds the results of one camera.

    This is the one place that knows the rule: the yaml's camera-free `output.output_name` gets
    the camera as prefix, so both cameras of a step B job get their own folder.

    Args:
        camera: `'left'` or `'right'`.
        output_name: camera-free name from the yaml (`output.output_name`).

    Returns:
        `<camera>_<output_name>`, e.g. `left_encoding`.
    """
    return f'{camera}_{output_name}'


def session_output_dir(video_preds_dir: str | Path, output_folder: str, eid: str) -> Path:
    """Directory holding the step B / step C outputs of one session.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eid: session id.

    Returns:
        `<video_preds_dir>/<output_folder>/<eid>`.
    """
    return Path(video_preds_dir) / output_folder / eid


def base_dir(
    video_preds_dir: str | Path,
    output_folder: str,
    eid: str,
    key_hash: str | None = None,
) -> Path:
    """Directory holding the base-keypoints fits that the variants of a session reuse.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eid: session id.
        key_hash: hash of the base cache key; each distinct base input (pca on / off, number of
            components, seed, ...) gets its own subfolder so one fit never replaces another.

    Returns:
        `<video_preds_dir>/<output_folder>/<eid>/base`, plus `/<key_hash>` when given.
    """
    root = session_output_dir(video_preds_dir, output_folder, eid) / 'base'
    return root if key_hash is None else root / key_hash


def variant_dir(
    video_preds_dir: str | Path,
    output_folder: str,
    eid: str,
    variant_name: str,
) -> Path:
    """Directory holding the results of one `additional_keypoints` set.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eid: session id.
        variant_name: name of the additional-keypoints variant.

    Returns:
        `<video_preds_dir>/<output_folder>/<eid>/<variant_name>`.
    """
    return session_output_dir(video_preds_dir, output_folder, eid) / variant_name


def validate_filter_settings(
    likelihood_threshold: Any,
    max_masked_frac: Any,
    source: str | Path,
) -> tuple[float, float]:
    """Check the two keypoint filter settings shared by steps A, B and C.

    Args:
        likelihood_threshold: frames below this likelihood are masked.
        max_masked_frac: keypoints with a larger masked fraction are not usable.
        source: where the values came from, used in error messages.

    Returns:
        Tuple `(likelihood_threshold, max_masked_frac)` as floats.

    Raises:
        ValueError: if a value is missing, not a number (booleans included) or outside `[0, 1]`.
    """
    values = {
        'likelihood_threshold': likelihood_threshold,
        'max_masked_frac': max_masked_frac,
    }
    for key, value in values.items():
        if value is None:
            raise ValueError(f'{source}: missing {key}')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{source}: {key} must be a number, got {value!r}')
        if not 0 <= value <= 1:
            raise ValueError(f'{source}: {key} must be in [0, 1], got {value!r}')
    return float(likelihood_threshold), float(max_masked_frac)


def _flatten_subsection(
    config_path: str | Path,
    section: str,
    key: str,
    value: Any,
) -> dict[str, Any]:
    """Flatten one mapping-valued key of a yaml section (see `ENCODING_YAML_SUBSECTIONS`).

    Args:
        config_path: path of the encoding yaml, used in error messages.
        section: top-level section, e.g. `'filter'`.
        key: key of the section holding the mapping, e.g. `'group_set'`.
        value: the yaml value; `None` (an empty `group_set:`) counts as an empty mapping.

    Returns:
        Flat dict of the keys of the mapping that are present.

    Raises:
        ValueError: if the value is not a mapping (for example the string of an older layout)
            or holds an unknown key.
    """
    allowed = ENCODING_YAML_SUBSECTIONS[(section, key)]
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f'{config_path}: `{section}.{key}` must be a mapping with keys {sorted(allowed)}, '
            f'got {value!r}',
        )
    unknown = sorted(str(k) for k in value if k not in allowed)
    if unknown:
        raise ValueError(
            f'{config_path}: unknown keys {unknown} in `{section}.{key}`; '
            f'allowed: {sorted(allowed)}',
        )
    return {flat: value[name] for name, flat in allowed.items() if name in value}


def load_encoding_yaml(config_path: str | Path) -> dict[str, Any]:
    """Read the nested encoding yaml and flatten it.

    This is the one place that knows the yaml layout (`ENCODING_YAML_SECTIONS`); every step reads
    the yaml through it. Sections and keys that are absent are simply left out, so callers decide
    which ones they require and which defaults apply.

    Args:
        config_path: path of the encoding yaml (`configs/encoding/litpose_encoding.yaml`).

    Returns:
        Flat dict of the keys present: `step_order`, `eid`, `eid_group`, `camera`,
        `video_preds_dir`, `body_parts_json`, `eid_groups_json`, `output_name`, `variant_name`,
        `likelihood_threshold`, `max_masked_frac`, `unfiltered_groups`
        (`filter.group_set.unfiltered_groups`), `filter_group_set` (`filter.group_set.name`),
        `pca_group_set` (`pca.group_set`), `merge_group_set` and `merge_sources` (`merge.*`),
        `anatomical_group` (`anatomical_group.name`), `base_keypoints`, `additional_keypoints`
        and `pca`. `pca` is not a yaml key: it is derived from `step_order` (True when the named
        pipeline contains the `2d_pose_pca` step), so the two can never disagree.

    Raises:
        ValueError: if the yaml is not a mapping, a section is not a mapping, a key is unknown, a
            key of the old flat layout is used (the error names its new place), a removed
            `pca.enabled` or `pca_n_components*` key is present, or `step_order` is missing or
            not a known pipeline name (see `litpose_step_pipelines`).
    """
    with Path(config_path).open(encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'{config_path}: expected a yaml mapping')
    cfg: dict[str, Any] = {}
    for key, value in raw.items():
        if key == 'step_order':
            cfg[key] = value
        elif key in ENCODING_YAML_SECTIONS and isinstance(value, dict):
            if key == 'pca' and 'enabled' in value:
                raise ValueError(
                    f'{config_path}: pca.enabled was removed; delete it and set `step_order` '
                    f'instead: PCA runs exactly when the pipeline contains the {_PCA_STEP!r} '
                    'step (see litpose_step_pipelines.json)',
                )
            mappings = {name for section, name in ENCODING_YAML_SUBSECTIONS if section == key}
            allowed = sorted(set(ENCODING_YAML_SECTIONS[key]) | mappings)
            unknown = sorted(str(k) for k in value if k not in allowed)
            if unknown:
                raise ValueError(
                    f'{config_path}: unknown keys {unknown} in `{key}`; allowed: {allowed}',
                )
            for sub_key, flat in ENCODING_YAML_SECTIONS[key].items():
                if sub_key in value:
                    cfg[flat] = value[sub_key]
            for name in sorted(mappings & set(value)):
                cfg.update(_flatten_subsection(config_path, key, name, value[name]))
        elif key in ENCODING_YAML_SECTIONS:
            # `pca: true` and `anatomical_group: <name>` of the old flat layout
            raise ValueError(
                f'{config_path}: `{key}` must be a mapping with keys '
                f'{sorted(ENCODING_YAML_SECTIONS[key])}, got {value!r}',
            )
        elif str(key).startswith('pca_n_components'):
            raise ValueError(
                f'{config_path}: {key} was removed; the number of components of each body-part '
                'group is now `PCA_n_components` in the body_parts_json file',
            )
        elif key in ENCODING_YAML_NESTED_NAMES:
            raise ValueError(
                f'{config_path}: top-level `{key}` is no longer supported; '
                f'move it to `{ENCODING_YAML_NESTED_NAMES[key]}`',
            )
        else:
            raise ValueError(f'{config_path}: unknown top-level key {key!r}')
    step_order = cfg.get('step_order')
    if step_order is None:
        raise ValueError(
            f'{config_path}: step_order is required; available: '
            f'{sorted(load_step_pipelines())}',
        )
    try:
        cfg['pca'] = resolve_step_pipeline(str(step_order))['use_pca']
    except ValueError as exc:
        raise ValueError(f'{config_path}: {exc}') from exc
    return cfg


def resolve_group_eids(eid_groups_json: str | Path, group: str) -> list[str]:
    """Look up a named group's eids from an `ibl_session_ids.json`-style file.

    Args:
        eid_groups_json: path to the json file (top-level `neural_sessions` key maps a group
            name to a flat list of eid strings).
        group: group name, e.g. `'sable'`, `'all'`, `'ear_both'`, `'ear_left'`.

    Returns:
        eids in the file's order.

    Raises:
        ValueError: if `group` is not a key of `neural_sessions`.
    """
    with Path(eid_groups_json).open(encoding='utf-8') as f:
        groups = json.load(f)['neural_sessions']
    if group not in groups:
        raise ValueError(f'unknown group {group!r}; available: {sorted(groups)}')
    return list(groups[group])


def load_config_eids(config_path: str | Path) -> list[str] | None:
    """Read the sessions of the yaml's `session.eid_group`.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        The eids of the group in the file's order, or `None` when the yaml sets no group.

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), a group is set without
            `paths.eid_groups_json`, or the group is not in that file.
    """
    cfg = load_encoding_yaml(config_path)
    group = cfg.get('eid_group')
    if group is None:
        return None
    if cfg.get('eid_groups_json') is None:
        raise ValueError(
            f'{config_path}: session.eid_group is set but paths.eid_groups_json is missing',
        )
    return resolve_group_eids(cfg['eid_groups_json'], str(group))


def resolve_session_eids(
    config_path: str | Path,
    eids: list[str] | None,
    eid_group: str | None,
    eid_groups_json: str | Path | None,
) -> list[str] | None:
    """Pick the sessions of a run: explicit eids, else a named group, else the yaml's group.

    Args:
        config_path: path of the encoding yaml.
        eids: session ids given on the command line, or `None`.
        eid_group: group name given on the command line, or `None`.
        eid_groups_json: `ibl_session_ids.json`-style file for `eid_group`; `None` uses
            `paths.eid_groups_json` of the yaml.

    Returns:
        The eids without duplicates in the given order, or `None` when neither `eids`, `eid_group`
        nor the yaml's `session.eid_group` selects any.

    Raises:
        ValueError: if both `eids` and `eid_group` are given, `eid_group` is given without a
            groups file, or a group is unknown (see `load_config_eids`).
    """
    if eids is not None and eid_group is not None:
        raise ValueError('exactly one of --eids or --eid-group must be given')
    if eids is not None:
        return list(dict.fromkeys(eids))
    if eid_group is not None:
        eid_groups_json = eid_groups_json or load_encoding_yaml(config_path).get(
            'eid_groups_json',
        )
        if eid_groups_json is None:
            raise ValueError(
                '--eid-groups-json is required when --eid-group is given '
                f'(or set paths.eid_groups_json in {config_path})',
            )
        return resolve_group_eids(eid_groups_json, eid_group)
    return load_config_eids(config_path)


def nest_encoding_config(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `load_encoding_yaml`: arrange a flat settings dict into the nested yaml layout.

    Args:
        flat: settings keyed as `load_encoding_yaml` returns them, except `pca`: it is derived
            from `step_order`, so pass the `step_order` you want instead.

    Returns:
        Dict that `yaml.safe_dump` writes as a valid encoding yaml; keys of `flat` that are
        absent stay absent.

    Raises:
        ValueError: if a key of `flat` is not a setting of the encoding yaml.
    """
    unknown = sorted(set(flat) - set(ENCODING_YAML_NESTED_NAMES) - {'step_order'})
    if unknown:
        raise ValueError(f'unknown encoding settings {unknown}')
    nested: dict[str, Any] = {}
    if 'step_order' in flat:
        nested['step_order'] = flat['step_order']
    for section, keys in ENCODING_YAML_SECTIONS.items():
        entries = {key: flat[name] for key, name in keys.items() if name in flat}
        for (sub_section, key), sub_keys in ENCODING_YAML_SUBSECTIONS.items():
            sub_entries = {k: flat[name] for k, name in sub_keys.items() if name in flat}
            if sub_section == section and sub_entries:
                entries[key] = sub_entries
        if entries:
            nested[section] = entries
    return nested


def load_filter_settings(config_path: str | Path) -> tuple[float, float]:
    """Read the shared keypoint filter settings from the encoding yaml.

    The yaml is the single source of these two numbers for the whole pipeline: step A uses them
    to pick the usable keypoints, step B to mask and interpolate low-likelihood frames, and step C
    checks that they still match what steps A and B used.

    Args:
        config_path: path of the encoding yaml (`configs/encoding/litpose_encoding.yaml`).

    Returns:
        Tuple `(likelihood_threshold, max_masked_frac)`.

    Raises:
        ValueError: if the yaml is not a mapping, or a setting is missing or invalid.
    """
    cfg = load_encoding_yaml(config_path)
    return validate_filter_settings(
        cfg.get('likelihood_threshold'), cfg.get('max_masked_frac'), config_path,
    )


def validate_unfiltered_groups(value: Any, source: str | Path) -> list[str]:
    """Check the `filter.group_set.unfiltered_groups` setting.

    Args:
        value: the yaml value; `None` (absent or null) means no group is exempt.
        source: where the value came from, used in error messages.

    Returns:
        The group names as a list of str, empty when nothing is exempt.

    Raises:
        ValueError: if the value is not a list of non-empty strings.
    """
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(n, str) and n for n in value):
        raise ValueError(
            f'{source}: filter.group_set.unfiltered_groups must be a list of group names, '
            f'got {value!r}',
        )
    return list(dict.fromkeys(value))


def load_unfiltered_keypoints(config_path: str | Path, keypoints: list[str]) -> list[str]:
    """Resolve `filter.group_set.unfiltered_groups` to the keypoints exempt from filtering.

    A keypoint belongs to a group when the first `_`-separated segment of its name is one of the
    group's tokens, the rule of `expand_groups`. Unlike `expand_groups`, a group without a
    keypoint in `keypoints` is not an error: the exemption just has nothing to apply to.

    Args:
        config_path: path of the encoding yaml.
        keypoints: keypoint names of the prediction csv, in header order.

    Returns:
        The keypoints of any named group, in `keypoints` order; empty when the yaml names no
        group.

    Raises:
        ValueError: if the yaml is invalid, `unfiltered_groups` is not a list of group names,
            `paths.body_parts_json` is missing, the group set is missing from it, or a group
            name is not in the group set.
    """
    cfg = load_encoding_yaml(config_path)
    group_names = validate_unfiltered_groups(cfg.get('unfiltered_groups'), config_path)
    if not group_names:
        return []
    body_parts_json = cfg.get('body_parts_json')
    if body_parts_json is None:
        raise ValueError(
            f'{config_path}: filter.group_set.unfiltered_groups needs paths.body_parts_json to '
            'look the groups up',
        )
    group_set = str(cfg.get('filter_group_set') or DEFAULT_FILTER_GROUP_SET)
    groups = load_groups(body_parts_json, group_set)
    unknown = sorted(set(group_names) - set(groups))
    if unknown:
        raise ValueError(
            f'{config_path}: filter.group_set.unfiltered_groups {unknown} not in group set '
            f'{group_set!r} of {body_parts_json}; available: {sorted(groups)}',
        )
    tokens = {token for name in group_names for token in groups[name]}
    return [name for name in keypoints if name.split('_')[0] in tokens]


def load_merge_settings(config_path: str | Path) -> dict[str, Any] | None:
    """Read and validate the optional `merge` section of the encoding yaml.

    The section lists several Lightning Pose models whose keypoints are combined into one csv per
    session and camera (see `merge_litpose_keypoints`), so each body part can come from the model
    that predicts it best.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        `None` when the yaml has no `merge.sources`, otherwise a dict with `group_set`
        (`merge.group_set`, default `'nonoverlapping_analysis_groups'`), `body_parts_json`
        (`paths.body_parts_json`) and `sources`: a list of `{'video_preds_dir': str, 'groups':
        list[str] | 'rest'}` in yaml order.

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), `merge.sources` is not a
            non-empty list of mappings with exactly the keys `video_preds_dir` and `groups`,
            `groups` is neither `'rest'` nor a non-empty list of group names, more than one
            source uses `'rest'`, a group is listed by two sources, `paths.video_preds_dir`
            (the merged output) is also a source, or `paths.body_parts_json` /
            `paths.video_preds_dir` is missing.
    """
    cfg = load_encoding_yaml(config_path)
    sources = cfg.get('merge_sources')
    if sources is None:
        return None
    if not isinstance(sources, list) or not sources:
        raise ValueError(f'{config_path}: merge.sources must be a non-empty list')
    body_parts_json = cfg.get('body_parts_json')
    output_dir = cfg.get('video_preds_dir')
    if body_parts_json is None or output_dir is None:
        raise ValueError(
            f'{config_path}: merge.sources needs paths.body_parts_json (to resolve the groups) '
            'and paths.video_preds_dir (where the merged csvs are written)',
        )
    parsed: list[dict[str, Any]] = []
    for idx_source, source in enumerate(sources):
        where = f'{config_path}: merge.sources[{idx_source}]'
        if not isinstance(source, dict) or set(source) != {'video_preds_dir', 'groups'}:
            raise ValueError(
                f'{where} must be a mapping with exactly the keys video_preds_dir and groups, '
                f'got {source!r}',
            )
        groups = source['groups']
        if groups != MERGE_REST and (
            not isinstance(groups, list) or not groups
            or not all(isinstance(g, str) and g for g in groups)
        ):
            raise ValueError(
                f'{where}.groups must be {MERGE_REST!r} or a non-empty list of group names, '
                f'got {groups!r}',
            )
        source_dir = str(source['video_preds_dir'])
        if Path(source_dir).resolve() == Path(str(output_dir)).resolve():
            raise ValueError(
                f'{where}.video_preds_dir equals paths.video_preds_dir; the merged csvs would '
                'overwrite the source',
            )
        parsed.append({
            'video_preds_dir': source_dir,
            'groups': groups if groups == MERGE_REST else list(dict.fromkeys(groups)),
        })
    if sum(source['groups'] == MERGE_REST for source in parsed) > 1:
        raise ValueError(f'{config_path}: at most one merge source may use groups: {MERGE_REST}')
    named = [g for source in parsed if source['groups'] != MERGE_REST for g in source['groups']]
    duplicated = sorted({g for g in named if named.count(g) > 1})
    if duplicated:
        raise ValueError(f'{config_path}: groups {duplicated} are listed by several merge sources')
    return {
        'group_set': str(cfg.get('merge_group_set') or DEFAULT_MERGE_GROUP_SET),
        'body_parts_json': str(body_parts_json),
        'sources': parsed,
    }


def load_video_preds_dir(config_path: str | Path) -> str | None:
    """Read the optional `video_preds_dir` from the encoding yaml.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        The `video_preds_dir` value, or `None` if absent/null.

    Raises:
        ValueError: if the yaml is not a mapping.
    """
    value = load_encoding_yaml(config_path).get('video_preds_dir')
    return None if value is None else str(value)


def load_output_name(config_path: str | Path) -> str | None:
    """Read the optional camera-free `output_name` from the encoding yaml.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        The `output.output_name` value, or `None` if absent/null.

    Raises:
        ValueError: if the yaml is not a mapping.
    """
    value = load_encoding_yaml(config_path).get('output_name')
    return None if value is None else str(value)


def load_pca_settings(config_path: str | Path) -> dict[str, Any]:
    """Read the PCA reprojection settings from the encoding yaml.

    Step B applies them (see `litpose_pca`); step C only checks that its `pca` flag agrees with
    the one step B stored in `dynamic_vars.npz`. Whether PCA runs is not a setting of its own: it
    follows from `step_order`, whose pipeline either contains the `2d_pose_pca` step or not.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        Dict with `enabled` (bool, whether the `step_order` pipeline has the `2d_pose_pca`
        step), `body_parts_json` (`paths.body_parts_json` as written in the yaml, `None` when
        absent) and `group_set` (`pca.group_set`, default `'nonoverlapping_analysis_groups'`).

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), or the `step_order`
            pipeline has PCA but `paths.body_parts_json` is missing.
    """
    cfg = load_encoding_yaml(config_path)
    enabled = cfg['pca']
    body_parts_json = cfg.get('body_parts_json')
    if enabled and body_parts_json is None:
        raise ValueError(
            f'{config_path}: step_order {cfg["step_order"]!r} includes {_PCA_STEP} but '
            'paths.body_parts_json is missing',
        )
    return {
        'enabled': enabled,
        'body_parts_json': None if body_parts_json is None else str(body_parts_json),
        'group_set': str(cfg.get('pca_group_set', DEFAULT_PCA_GROUP_SET)),
    }


def expand_groups(
    group_names: list[str],
    groups: dict[str, list[str]],
    available: list[str],
) -> list[str]:
    """Resolve body-part group names to the usable keypoints they contain.

    A keypoint belongs to a group when the first `_`-separated segment of its name is one of the
    group's tokens (`ear_top_left` -> `ear`), the rule `litpose_pca.assign_keypoints_to_groups`
    uses. Unlike that function, groups may overlap: a keypoint of several named groups is listed
    once.

    Args:
        group_names: names of the groups to expand, e.g. `['eye_related', 'mouth_related']`.
        groups: group name -> keypoint name tokens, as from `load_groups`.
        available: usable keypoint names (those of `dynamic_vars.npz`).

    Returns:
        The keypoints of `available` that belong to any of the groups, in `available` order.

    Raises:
        ValueError: if a group name is unknown or a group has no keypoint in `available`.
    """
    unknown = sorted(set(group_names) - set(groups))
    if unknown:
        raise ValueError(f'unknown groups {unknown}; available: {sorted(groups)}')
    selected: set[str] = set()
    for name in group_names:
        members = [kp for kp in available if kp.split('_')[0] in groups[name]]
        if not members:
            raise ValueError(
                f'group {name!r} (tokens {groups[name]}) has no usable keypoint; '
                f'usable keypoints: {available}',
            )
        selected.update(members)
    return [kp for kp in available if kp in selected]


def load_step_order(config_path: str | Path) -> str:
    """Read the required `step_order` pipeline name from the encoding yaml.

    This is the single source of `step_order` for the whole pipeline, the same way
    `load_filter_settings` is for `likelihood_threshold` / `max_masked_frac`: step B reads it
    here instead of taking its own separate flag, so it can never drift from what step C's
    `run_litpose_encoding.load_config` sees. It also decides whether PCA runs (see
    `load_pca_settings`).

    Args:
        config_path: path of the encoding yaml.

    Returns:
        The `step_order` value.

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), or `step_order` is
            missing or not a known pipeline name (see `litpose_step_pipelines`).
    """
    return str(load_encoding_yaml(config_path)['step_order'])


def check_filter_settings_match(
    keypoint_info: dict[str, Any],
    likelihood_threshold: float,
    max_masked_frac: float,
    json_path: str | Path,
    no_filter_keypoints: list[str] | None = None,
) -> None:
    """Fail when the step A json was written with other filter settings than the yaml holds.

    Args:
        keypoint_info: contents of the step A json.
        likelihood_threshold: value from the yaml.
        max_masked_frac: value from the yaml.
        json_path: path of the step A json, used in the error message.
        no_filter_keypoints: keypoints the yaml exempts from confidence filtering (see
            `load_unfiltered_keypoints`); `None` or empty means none. A json from before the
            exemption existed has no such entry and counts as none.

    Raises:
        ValueError: if the json's `likelihood_threshold`, `max_masked_frac` or
            `no_filter_keypoints` differs from the yaml; the usable-keypoint list in the json is
            then stale, so step A must be rerun.
    """
    mismatches = [
        f'{key}: yaml {expected} vs json {keypoint_info.get(key)}'
        for key, expected in (
            ('likelihood_threshold', likelihood_threshold),
            ('max_masked_frac', max_masked_frac),
        )
        if keypoint_info.get(key) != expected
    ]
    expected_exempt = sorted(no_filter_keypoints or [])
    json_exempt = sorted(keypoint_info.get('no_filter_keypoints', []))
    if json_exempt != expected_exempt:
        mismatches.append(
            f'no_filter_keypoints: yaml {expected_exempt} vs json {json_exempt}',
        )
    if mismatches:
        raise ValueError(
            f'{json_path} was written with other filter settings than the config ('
            f'{"; ".join(mismatches)}); rerun pose2visual.encoding.list_litpose_keypoints '
            '(step 0) with this config',
        )


def load_keypoint_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a Lightning Pose prediction csv with its three header rows.

    Args:
        csv_path: path of the csv.

    Returns:
        Dataframe with `(scorer, bodypart, coord)` column levels, one row per frame.
    """
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def get_keypoint_names(df: pd.DataFrame) -> list[str]:
    """Keypoint names (second header row) deduplicated in file order.

    Args:
        df: dataframe from `load_keypoint_csv`.

    Returns:
        Keypoint names in header order.
    """
    return list(dict.fromkeys(df.columns.get_level_values(1)))


def get_keypoint_coords(
    df: pd.DataFrame,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract `x`, `y` and `likelihood` of one keypoint.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.

    Returns:
        Tuple of float64 arrays `(x, y, likelihood)`, each of shape `[n_frames]`.
    """
    sub = df.xs(name, axis=1, level=1)
    # the remaining column level is the coordinate name
    sub.columns = pd.Index(sub.columns.get_level_values(-1))
    return (
        sub['x'].to_numpy(dtype=np.float64),
        sub['y'].to_numpy(dtype=np.float64),
        sub['likelihood'].to_numpy(dtype=np.float64),
    )


def masked_frames(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    likelihood_threshold: float | None,
) -> np.ndarray:
    """Boolean mask of unusable frames: NaN coordinate/likelihood or low likelihood.

    Args:
        x: x position per frame.
        y: y position per frame.
        likelihood: likelihood per frame.
        likelihood_threshold: frames below this likelihood are masked; `None` ignores the
            likelihood (also a NaN one) and masks only frames with a NaN x or y.

    Returns:
        Boolean array `[n_frames]`, `True` for masked frames.
    """
    if likelihood_threshold is None:
        return np.isnan(x) | np.isnan(y)
    # NaN likelihood compares False in the threshold test, so mask it explicitly
    return (likelihood < likelihood_threshold) | np.isnan(x) | np.isnan(y) | np.isnan(likelihood)


def frame_fractions(
    df: pd.DataFrame,
    name: str,
    likelihood_threshold: float | None,
) -> dict[str, float]:
    """Split the unusable frames of a keypoint by cause.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.
        likelihood_threshold: frames below this likelihood count as low-likelihood; `None`
            ignores the likelihood, so only NaN x/y frames count (`low_likelihood` is 0).

    Returns:
        Dict with `masked` (any cause, as `masked_frames`), `nan` (NaN x, y or likelihood) and
        `low_likelihood` (finite likelihood below the threshold), each a fraction in `[0, 1]`.
    """
    x, y, likelihood = get_keypoint_coords(df, name)
    if likelihood_threshold is None:
        nan = np.isnan(x) | np.isnan(y)
        low = np.zeros_like(nan)
    else:
        nan = np.isnan(x) | np.isnan(y) | np.isnan(likelihood)
        low = ~np.isnan(likelihood) & (likelihood < likelihood_threshold)
    return {
        'masked': float(np.mean(masked_frames(x, y, likelihood, likelihood_threshold))),
        'nan': float(np.mean(nan)),
        'low_likelihood': float(np.mean(low)),
    }


def masked_fraction(df: pd.DataFrame, name: str, likelihood_threshold: float | None) -> float:
    """Fraction of frames of a keypoint that `masked_frames` marks unusable.

    Keypoints on the far side of the animal are typically predicted with likelihood near 0 and
    therefore have a masked fraction near 1.

    Args:
        df: dataframe from `load_keypoint_csv`.
        name: keypoint name.
        likelihood_threshold: frames below this likelihood are masked; `None` masks only NaN
            x/y frames.

    Returns:
        Fraction in `[0, 1]`.
    """
    x, y, likelihood = get_keypoint_coords(df, name)
    return float(np.mean(masked_frames(x, y, likelihood, likelihood_threshold)))


def interpolate_keypoint(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    times: np.ndarray,
    likelihood_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mask the unusable frames of one keypoint and interpolate its position over them.

    Frames flagged by `masked_frames` (low likelihood or NaN) are masked and the position is
    linearly interpolated over them.

    Args:
        x: x position per frame.
        y: y position per frame.
        likelihood: likelihood per frame.
        times: frame timestamps in seconds, strictly increasing.
        likelihood_threshold: frames below this likelihood are masked; `None` masks only NaN
            x/y frames (the keypoint is exempt from confidence filtering).

    Returns:
        Tuple `(x_interp, y_interp, masked_fraction)`: finite positions per frame
        `[n_frames]` and the fraction of masked frames.

    Raises:
        ValueError: if fewer than two frames remain unmasked.
    """
    masked = masked_frames(x, y, likelihood, likelihood_threshold)
    idx_valid = np.flatnonzero(~masked)
    if idx_valid.size < 2:
        criterion = (
            'finite x/y' if likelihood_threshold is None
            else f'likelihood >= {likelihood_threshold}'
        )
        raise ValueError(f'fewer than 2 frames with {criterion}; cannot compute keypoint speed')
    x_interp = np.interp(times, times[idx_valid], x[idx_valid])
    y_interp = np.interp(times, times[idx_valid], y[idx_valid])
    return x_interp, y_interp, float(masked.mean())


def speed_from_positions(x: np.ndarray, y: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Speed of a keypoint trajectory in pixels per second.

    Args:
        x: finite x position per frame.
        y: finite y position per frame.
        times: frame timestamps in seconds, strictly increasing.

    Returns:
        Speed per frame `[n_frames]`.
    """
    return np.hypot(np.gradient(x, times), np.gradient(y, times))


def compute_keypoint_speed(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    times: np.ndarray,
    likelihood_threshold: float | None,
) -> tuple[np.ndarray, float]:
    """Speed of one keypoint in pixels per second.

    Frames flagged by `masked_frames` (low likelihood or NaN) are masked and the
    position is linearly interpolated over them before differentiating.

    Args:
        x: x position per frame.
        y: y position per frame.
        likelihood: likelihood per frame.
        times: frame timestamps in seconds, strictly increasing.
        likelihood_threshold: frames below this likelihood are masked; `None` masks only NaN
            x/y frames.

    Returns:
        Tuple `(speed, masked_fraction)`: speed per frame `[n_frames]` and the fraction of
        masked frames.

    Raises:
        ValueError: if fewer than two frames remain unmasked.
    """
    x_interp, y_interp, masked_fraction = interpolate_keypoint(
        x, y, likelihood, times, likelihood_threshold,
    )
    return speed_from_positions(x_interp, y_interp, times), masked_fraction


def bin_to_intervals(
    values: np.ndarray,
    times: np.ndarray,
    intervals: np.ndarray,
    binsize: float,
    n_bins: int,
) -> np.ndarray:
    """Linearly interpolate a time series onto the per-interval bin grid used for neural data.

    The grid of each interval is `linspace(beg + binsize, end, n_bins)`, identical to
    `ibl_data_utils.get_behavior_per_interval`.

    Args:
        values: signal sampled at `times`, shape `[n_frames]`.
        times: sample timestamps in seconds, strictly increasing.
        intervals: interval start/end times, shape `[n_intervals, 2]`.
        binsize: bin size in seconds.
        n_bins: number of bins per interval.

    Returns:
        Array of shape `[n_intervals, n_bins]`.

    Raises:
        ValueError: if an interval lies outside the range of `times`.
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size and (intervals.min() < times[0] or intervals.max() > times[-1]):
        raise ValueError(
            f'intervals [{intervals.min():.3f}, {intervals.max():.3f}] fall outside the '
            f'frame times [{times[0]:.3f}, {times[-1]:.3f}]',
        )
    grid = np.linspace(intervals[:, 0] + binsize, intervals[:, 1], n_bins, axis=1)
    return np.interp(grid.ravel(), times, values).reshape(grid.shape)
