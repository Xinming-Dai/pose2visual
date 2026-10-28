"""Step C: neural encoding from Lightning Pose keypoints, base vs base + additional keypoints.

Reads the yaml config (see ``configs/encoding/litpose_encoding.yaml``) and
``<video_preds_dir>/<camera>_<output_name>/<eid>/dynamic_vars.npz`` from
``extract_litpose_dynamic_vars_trials``, then fits RRR and CNN encoders (Ray Tune, as in
``run_encoding``) on two input sets; the train/val/test split is the one already stored in
``<eid>_aligned.npz``, so it is identical for both:

* ``base``: the speed of ``base_keypoints``;
* ``base_plus_additional``: the speed of ``base_keypoints + additional_keypoints`` (skipped when
  ``additional_keypoints`` is empty).

Keypoints listed in neither list are ``unused``. The held-out test R^2 and bits-per-spike (BPS),
overall and per neuron, are compared; no significance test is run. With an empty
``additional_keypoints`` only the ``base`` condition is fitted (a base-only run): the variant folder
is ``base_only`` by default, the result holds the ``base_*`` metrics and predictions without deltas,
and the comparison plots, which need both conditions, do not apply.

``session.camera: both`` fits one model (one RRR and one CNN) on the features of both cameras:
the step B outputs of ``left_<output_name>`` and ``right_<output_name>`` are loaded, the lists are
resolved against each camera's own keypoints and the selected columns are concatenated, named
``left:<name>`` and ``right:<name>``. A trace that does not depend on the camera (``wheel_speed``,
``licks``) is taken once, from the left camera, and keeps its plain name. With explicit keypoint
names (``anatomical_group.name: all``) each name must exist in both cameras, except the keypoints
the yaml's ``same_side_filter`` drops for a camera (``side_excluded`` of its step A json): the left
camera then takes ``wrist_left`` and the right camera ``wrist_right`` of the same list. The results
go to ``both_<output_name>``.

The ``anatomical_group`` section holds the two lists and ``anatomical_group.name`` (default
``all``) says what they hold. With ``all`` they are explicit keypoint names. With a group set of
``paths.body_parts_json`` (e.g. ``nonoverlapping_analysis_groups``) they are group names, e.g.
``base_keypoints: [eye_related]`` and ``additional_keypoints: [mouth_related]``: each group is resolved to its usable keypoints of
the session (those in ``dynamic_vars.npz``, see ``litpose_keypoints.expand_groups``), so the
resolved keypoints can differ between sessions. A group of the yaml's ``visibility`` section
(e.g. ``tongue``) resolves to its binary column ``<group>_visible`` (0/1, see
``extract_litpose_dynamic_vars_trials``) instead of keypoint speeds; its settings enter the run
signature, so another threshold never reuses a result. A trace of the yaml's ``one_api`` section
(``whisker_motion_energy``, ``wheel_speed``, ``licks``) is selected the same way and resolves to the
column of the same name (copied from ``<eid>_aligned.npz`` by step B). The resolved names are
what the encoders see and what ``encoding_params.json`` / ``encoding_result.npz`` store as
``base_keypoints`` / ``additional_keypoints``; the group names are stored as ``base_groups`` /
``additional_groups``.

Before fitting, the features of each condition are checked to be finite (step B already
interpolates over masked frames, so there is nothing to impute). PCA is not done here: when the yaml's
``step_order`` pipeline contains ``2d_pose_pca``, step B already reprojected the x/y of each body-part group onto its top
principal components before taking the speed (see ``litpose_pca``; the number of components of
each group is ``PCA_n_components`` in the ``paths.body_parts_json`` of the yaml), and the features
keep their keypoint names. This step only checks that the PCA flag the yaml's ``step_order`` implies
equals the one stored in ``dynamic_vars.npz`` and records the group set and the effective number of components of step B's
``params.json``; they enter the run signature and the base cache key, so a pca and a no-pca run
(or other component counts) never replace each other.

The base fit is written once to ``<eid>/base/<hash8>/encoding_result.npz`` (with
``encoding_predictions.npz``, ``base_key.json`` and ``ray_tune/``), ``<hash8>`` being a hash of the base inputs (keypoints,
camera, seed, pca settings, data), and reused by every variant whose base inputs match, so trying
another ``additional_keypoints`` set only needs a new ``additional_keypoints`` list;
``--rerun_base`` forces a refit. Each variant
writes to ``<eid>/<variant_name>/``: ``encoding_result.npz`` (metrics and deltas, loadable
without pickle), ``encoding_predictions.npz`` (float32 held-out test ``gt`` / ``pred`` of every
condition and model, ``[trials, bins, neurons]``, read by the psth / raster plot),
``encoding_params.json`` and ``encoding.log``, both recording the Lightning
Pose checkpoint location and the base / additional / unused keypoints. A cached base without
``encoding_predictions.npz`` (made before it was saved) is refitted.

``output.variant_name`` defaults (null or absent) to the sorted ``additional_keypoints`` (names or group
names, as written in the config) joined by ``_``; a string in the config overrides it. Either way an earlier result is never overwritten by
a run with other settings: if ``<variant_name>/`` already holds an ``encoding_result.npz`` whose
``run_signature`` (camera, base / additional keypoints, pca settings, seed) differs from this
run's, or that predates ``run_signature``, the run goes to ``<variant_name>_<hash8>/`` instead.
Re-running identical settings overwrites its own folder. The chosen folder and the reason are
logged. A base fit is never replaced by a run with other inputs (e.g. a ``step_order`` without ``2d_pose_pca``), it gets
its own ``base/<hash8>/``; only ``--rerun_base`` refits one in place. Every variant result also keeps its
own copy of the base metrics. A flat ``<eid>/base/encoding_result.npz`` from an earlier version
is left untouched and no longer read.

Usage::

    PYTHONPATH=src python -u -m pose2visual.encoding.run_litpose_encoding \\
      --config configs/encoding/litpose_encoding.yaml --neural_input_dir <dir>
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import ray
from sklearn.metrics import r2_score

from pose2visual.encoding.encoder import train_cnn_encoder_with_tune, train_rrr_encoder_with_tune
from pose2visual.encoding.litpose_keypoints import (
    ANATOMICAL_GROUP_ALL,
    CAMERA_BOTH,
    DEFAULT_FILTER_GROUP_SET,
    DEFAULT_PCA_GROUP_SET,
    ENCODING_CAMERAS,
    ENCODING_YAML_NESTED_NAMES,
    ONE_API_TRACES,
    base_dir,
    cameras_of,
    check_filter_settings_match,
    expand_groups,
    keypoint_names_json_path,
    load_encoding_yaml,
    load_one_api_trace_settings,
    load_position_columns,
    load_same_side_excluded,
    load_unfiltered_keypoints,
    load_visibility_features,
    load_visibility_settings,
    output_folder_name,
    session_output_dir,
    validate_filter_settings,
    validate_same_side_groups,
    validate_unfiltered_groups,
    variant_dir,
)
from pose2visual.encoding.litpose_preprocess import check_finite
from pose2visual.encoding.run_encoding import (
    _flatten_views,
    _maybe_assert_latent_intervals_match_neural,
    _resolve_ray_session_temp_dir,
)
from pose2visual.encoding.utils import bits_per_spike, set_seed, setup_logging
from pose2visual.lightningpose.fit_eks_latent_model import load_groups

logger = logging.getLogger(__name__)

_CONDITIONS = ('base', 'base_plus_additional')
_MODELS = ('rrr', 'cnn')
_REQUIRED_CONFIG_KEYS = (
    'eid',
    'video_preds_dir',
    'output_name',
    'camera',
    'base_keypoints',
    'additional_keypoints',
    'likelihood_threshold',
    'max_masked_frac',
)
_SPLITS = ('train', 'val', 'test')
_BASE_RESERVED_NAME = 'base'
# default variant folder of a run without additional keypoints (only the base is fitted)
_BASE_ONLY_NAME = 'base_only'
# derived variant names stay well below the 255-byte file name limit
_MAX_VARIANT_NAME_LEN = 100
_NAME_HASH_LEN = 8
# same floor as the encoders use before taking log of the predicted rates
_BPS_RATE_THRESHOLD = 1e-3
_FROZEN_CONFIG_NAME = 'encoding_config.yaml'
# test-set gt / pred of every condition and model, read by the psth / raster plot
_PREDICTIONS_NAME = 'encoding_predictions.npz'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Neural encoding from Lightning Pose keypoints: base vs base + additional.',
    )
    parser.add_argument('--config', type=str, required=True, help='yaml encoding config')
    parser.add_argument(
        '--neural_input_dir',
        type=str,
        required=True,
        help='root dir containing <eid>/<eid>_aligned.npz',
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--eid', type=str, default=None, help='overrides `eid` in the config')
    parser.add_argument(
        '--rerun_base',
        action='store_true',
        help='refit the base keypoints even if a matching base result is cached',
    )
    return parser.parse_args(argv)


def _short_hash(text: str) -> str:
    """First hex digits of the sha256 of a text, used to disambiguate folder names.

    Args:
        text: text to hash.

    Returns:
        `_NAME_HASH_LEN` hex characters.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:_NAME_HASH_LEN]


def derive_variant_name(additional: list[str]) -> str:
    """Default variant folder name: the sorted additional keypoints joined by `_`.

    Names longer than `_MAX_VARIANT_NAME_LEN` are cut and ended by a hash of the full name. The
    name alone is not unique (`['a_b']` and `['a', 'b']` give the same string), which
    `resolve_variant_dir` handles through the run signature.

    Args:
        additional: `additional_keypoints` from the config; empty for a base-only run.

    Returns:
        Folder name; `_BASE_ONLY_NAME` when `additional` is empty.
    """
    if not additional:
        return _BASE_ONLY_NAME
    name = '_'.join(sorted(additional))
    if len(name) > _MAX_VARIANT_NAME_LEN:
        keep = _MAX_VARIANT_NAME_LEN - _NAME_HASH_LEN - 1
        name = f'{name[:keep]}_{_short_hash(name)}'
    return name


def run_signature(
    cfg: dict[str, Any],
    seed: int,
    pca_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Settings that make two runs with the same variant name different.

    Args:
        cfg: validated config from `load_config`.
        seed: random seed of the run.
        pca_record: `pca` entry of step B's `params.json` (`read_pca_record`); its group set and
            effective number of components per group are part of the signature when pca is on.

    Returns:
        Json-serializable dict, stored in `encoding_params.json` and compared with `==`.
        `base_keypoints` / `additional_keypoints` are the entries of the config (group names in
        group mode); `anatomical_group` is only present when it is not `all`, and
        `unfiltered_groups` / `filter_group_set` only when groups are exempt from confidence
        filtering, `same_side_groups` only when the yaml has a `same_side_filter`, and
        `visibility` (group set, groups, threshold) only when the yaml has a
        `visibility` section, and `one_api` (sorted traces) only when the yaml has a `one_api`
        section, so results from before they existed keep matching.
    """
    signature = {
        'camera': cfg['camera'],
        'base_keypoints': list(cfg['base_keypoints']),
        'additional_keypoints': sorted(cfg['additional_keypoints']),
        'pca': cfg['pca'],
        'pca_group_set': cfg['pca_group_set'] if cfg['pca'] else None,
        'pca_n_components': (
            None if not cfg['pca'] or pca_record is None else pca_record['n_components']
        ),
        'seed': int(seed),
    }
    if cfg['anatomical_group'] != ANATOMICAL_GROUP_ALL:
        signature['anatomical_group'] = cfg['anatomical_group']
    if cfg.get('unfiltered_groups'):
        signature['unfiltered_groups'] = sorted(cfg['unfiltered_groups'])
        signature['filter_group_set'] = cfg['filter_group_set']
    if cfg.get('same_side_groups'):
        signature['same_side_groups'] = sorted(cfg['same_side_groups'])
    if cfg.get('visibility'):
        signature['visibility'] = {
            'group_set': cfg['visibility']['group_set'],
            'groups': sorted(cfg['visibility']['groups']),
            'likelihood_threshold': cfg['visibility']['likelihood_threshold'],
        }
    if cfg.get('one_api'):
        signature['one_api'] = sorted(cfg['one_api'])
    return signature


def read_pca_record(
    video_preds_dir: str | Path,
    output_folder: str,
    eid: str,
) -> dict[str, Any] | None:
    """Read the PCA settings that step B stored next to `dynamic_vars.npz`.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eid: session id.

    Returns:
        `None` when step B's `params.json` is missing, has no `pca` entry (written before PCA
        moved to step B) or pca was off; otherwise a dict with `group_set` and `n_components`
        (group name -> effective number of components, `0` for a skipped group).
    """
    params_path = session_output_dir(video_preds_dir, output_folder, eid) / 'params.json'
    if not params_path.is_file():
        return None
    with params_path.open(encoding='utf-8') as f:
        record = json.load(f).get('pca')
    if not record or not record.get('enabled'):
        return None
    return {
        'group_set': record['group_set'],
        'n_components': {
            name: group['n_components'] for name, group in record['groups'].items()
        },
    }


def read_camera_pca_record(
    video_preds_dir: str | Path,
    output_name: str,
    camera: str,
    eid: str,
) -> dict[str, Any] | None:
    """Read the PCA settings of step B for the cameras of a step C run.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_name: camera-free `output.output_name`.
        camera: `session.camera`: `'left'`, `'right'` or `'both'`.
        eid: session id.

    Returns:
        `read_pca_record` of the camera for a single camera. For `'both'`: `None` when pca is off
        for both cameras, otherwise `{'group_set': ..., 'n_components': {'left': {...},
        'right': {...}}}`.

    Raises:
        ValueError: for `'both'`, if only one camera has PCA settings or the two used other group
            sets.
    """
    records = {
        name: read_pca_record(video_preds_dir, output_folder_name(name, output_name), eid)
        for name in cameras_of(camera)
    }
    if camera != CAMERA_BOTH:
        return records[camera]
    if all(record is None for record in records.values()):
        return None
    missing = [name for name, record in records.items() if record is None]
    if missing:
        raise ValueError(
            f'step B stored pca settings for only one camera (none for {missing}); rerun '
            'extract_litpose_dynamic_vars_trials for both cameras with the same config',
        )
    group_sets = {record['group_set'] for record in records.values()}
    if len(group_sets) > 1:
        raise ValueError(f'the cameras of step B used different pca group sets: {group_sets}')
    return {
        'group_set': group_sets.pop(),
        'n_components': {name: record['n_components'] for name, record in records.items()},
    }


def resolve_variant_dir(
    video_preds_dir: str | Path,
    output_folder: str,
    eid: str,
    variant_name: str,
    signature: dict[str, Any],
) -> tuple[Path, str]:
    """Pick the results folder of a run without overwriting a run with other settings.

    The folder `<variant_name>` is used when it holds no result yet or a result of the same
    `signature` (re-running identical settings reproduces it). Otherwise
    `<variant_name>_<hash of the signature>` is used. A result written before `run_signature`
    existed counts as a different run.

    Args:
        video_preds_dir: directory holding the Lightning Pose prediction csvs.
        output_folder: folder name, see `output_folder_name`.
        eid: session id.
        variant_name: wanted folder name (derived or set in the config).
        signature: `run_signature` of this run.

    Returns:
        Tuple `(directory, reason)` with a short text for the log.
    """
    plain = variant_dir(video_preds_dir, output_folder, eid, variant_name)
    if not (plain / 'encoding_result.npz').is_file():
        return plain, 'new folder'
    previous = None
    params_path = plain / 'encoding_params.json'
    if params_path.is_file():
        try:
            with params_path.open(encoding='utf-8') as f:
                previous = json.load(f).get('run_signature')
        except (OSError, ValueError):
            previous = None
    if previous == signature:
        return plain, 'same settings as the existing result, which is overwritten'
    suffixed = f'{variant_name}_{_short_hash(json.dumps(signature, sort_keys=True))}'
    return (
        variant_dir(video_preds_dir, output_folder, eid, suffixed),
        f'{variant_name!r} already holds a result with other settings, kept untouched',
    )


def load_config(path: str | Path, eid: str | None = None) -> dict[str, Any]:
    """Load and validate the nested encoding yaml config.

    The yaml is read by `load_encoding_yaml`; the returned dict is flat, keyed as
    `load_encoding_yaml` documents.

    Args:
        path: path of the yaml file.
        eid: if given, overrides `session.eid` of the file.

    Returns:
        Config dict with `base_keypoints` and `additional_keypoints` as lists of str (keypoint
        names, or group names when `anatomical_group` is a group set),
        `anatomical_group` as str (default `all`: the lists hold keypoint names; otherwise a
        group-set key of `body_parts_json`),
        `variant_name` as str (the sorted additional keypoints joined by `_` when the file has
        none or null; `base_only` when there are none), `output_folder` as str
        (`<camera>_<output_name>`, with `camera` possibly `both`; the folder under
        `video_preds_dir`), `likelihood_threshold` / `max_masked_frac` as floats (shared by all
        steps; step A and B must have used the same values), `step_order` as str (required; a name from
        `litpose_step_pipelines.json`), `unfiltered_groups` as a list of str (groups exempt from
        confidence filtering, default empty), `filter_group_set` as str, `pca` as bool (derived
        from `step_order`: whether the pipeline has the `2d_pose_pca` step; step B applies it,
        step C checks it), `pca_group_set` as str and `visibility` (`load_visibility_settings`:
        `None` without a `visibility` section) and `one_api` (`load_one_api_trace_settings`:
        `None` without a `one_api` section). `additional_keypoints` may be empty: only the base is
        fitted then. The actual keypoint source is still `video_preds_dir`.

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`: not nested, unknown key,
            removed `pca_n_components*` key), a required key is missing,
            `filter.likelihood_threshold` / `filter.max_masked_frac` is not a number in `[0, 1]`, the camera is unknown, a
            given `output.variant_name` is not a plain folder name (or is `base`), the base list
            is empty or the additional list is not a list, `anatomical_group.name` is a group set
            but `paths.body_parts_json` is missing or does not hold it, a listed name is not a group of it, `pca.enabled` is still
            in the yaml (removed), `step_order` is missing or an unknown pipeline name, or the
            `visibility` or `one_api` section is invalid.
    """
    cfg = load_encoding_yaml(path)
    if eid is not None:
        cfg['eid'] = eid
    missing = [
        ENCODING_YAML_NESTED_NAMES[key] for key in _REQUIRED_CONFIG_KEYS if cfg.get(key) is None
    ]
    if missing:
        raise ValueError(f'{path}: missing config keys {missing}')
    cfg['likelihood_threshold'], cfg['max_masked_frac'] = validate_filter_settings(
        cfg['likelihood_threshold'], cfg['max_masked_frac'], path,
    )
    if cfg['camera'] not in ENCODING_CAMERAS:
        raise ValueError(
            f'{path}: session.camera must be one of {ENCODING_CAMERAS}, got {cfg["camera"]!r}',
        )
    cfg['output_folder'] = output_folder_name(cfg['camera'], str(cfg['output_name']))
    # an empty additional list is a base-only run
    for key, allow_empty in (('base_keypoints', False), ('additional_keypoints', True)):
        if not isinstance(cfg[key], list) or not (cfg[key] or allow_empty):
            raise ValueError(
                f'{path}: anatomical_group.{key} must be a '
                f'{"list" if allow_empty else "non-empty list"}',
            )
        cfg[key] = [str(name) for name in cfg[key]]
    if cfg.get('variant_name') is None:
        cfg['variant_name'] = derive_variant_name(cfg['additional_keypoints'])
    else:
        variant_name = str(cfg['variant_name'])
        if (
            not variant_name
            or variant_name == _BASE_RESERVED_NAME
            or variant_name.startswith('.')
            or Path(variant_name).name != variant_name
        ):
            raise ValueError(
                f'{path}: output.variant_name must be null or a plain folder name other than '
                f'{_BASE_RESERVED_NAME!r}, got {variant_name!r}',
            )
        cfg['variant_name'] = variant_name
    cfg['unfiltered_groups'] = validate_unfiltered_groups(cfg.get('unfiltered_groups'), path)
    cfg['filter_group_set'] = str(cfg.get('filter_group_set') or DEFAULT_FILTER_GROUP_SET)
    if cfg['unfiltered_groups'] and cfg.get('body_parts_json') is None:
        raise ValueError(
            f'{path}: filter.group_set.unfiltered_groups needs paths.body_parts_json to look the '
            'groups up',
        )
    cfg['same_side_groups'] = validate_same_side_groups(cfg.get('same_side_groups'), path)
    cfg['visibility'] = load_visibility_settings(path)
    cfg['one_api'] = load_one_api_trace_settings(path)
    cfg['pca_group_set'] = str(cfg.get('pca_group_set', DEFAULT_PCA_GROUP_SET))
    cfg['anatomical_group'] = str(cfg.get('anatomical_group', ANATOMICAL_GROUP_ALL))
    if cfg['anatomical_group'] != ANATOMICAL_GROUP_ALL:
        if cfg.get('body_parts_json') is None:
            raise ValueError(
                f'{path}: anatomical_group.name {cfg["anatomical_group"]!r} needs '
                'paths.body_parts_json',
            )
        groups = load_groups(cfg['body_parts_json'], cfg['anatomical_group'])
        # a one_api trace is selected by its name, like a group
        selectable = set(groups) | set(cfg['one_api'] or [])
        for key in ('base_keypoints', 'additional_keypoints'):
            unknown = sorted(set(cfg[key]) - selectable)
            if unknown:
                raise ValueError(
                    f'{path}: anatomical_group.{key} {unknown} are not groups of '
                    f'{cfg["anatomical_group"]!r} or one_api.behavior_traces; '
                    f'groups: {sorted(groups)}',
                )
    return cfg


def expand_selection(
    names: list[str],
    groups: dict[str, list[str]],
    traces: list[str] | None,
    available: list[str],
) -> list[str]:
    """Resolve group names and one_api trace names to feature columns.

    Args:
        names: `base_keypoints` or `additional_keypoints` of the config in group mode.
        groups: group name -> keypoint name tokens, as from `load_groups`.
        traces: `one_api.behavior_traces` of the config, `None` without that section.
        available: feature names stored in `dynamic_vars.npz`.

    Returns:
        The columns of the groups (see `expand_groups`) followed by the column of every trace in
        `names` (named like the trace), in `names` order.

    Raises:
        ValueError: if a group is unknown or has no usable keypoint, or a trace column is not
            in `available` (step 1 was run without that trace).
    """
    traces = traces or []
    selected = expand_groups([name for name in names if name not in traces], groups, available)
    for name in names:
        if name not in traces:
            continue
        if name not in available:
            raise ValueError(
                f'{name!r} is not in dynamic_vars.npz (features: {available}); rerun '
                'extract_litpose_dynamic_vars_trials (step 1) with it in one_api.behavior_traces',
            )
        selected.append(name)
    return selected


def split_keypoints(
    base: list[str],
    additional: list[str],
    keypoint_info: dict[str, Any],
    available: list[str],
) -> tuple[list[str], list[str]]:
    """Validate the base / additional selection and return the usable additional and unused ones.

    A masked `additional` keypoint is dropped with a warning, so a session where it is not
    tracked reliably still runs; a masked `base` keypoint is an error.

    Args:
        base: `base_keypoints` from the config.
        additional: `additional_keypoints` from the config.
        keypoint_info: step A json of this camera and session.
        available: keypoint names stored in `dynamic_vars.npz`.

    Returns:
        Tuple `(additional, unused)`. `additional` is the input without the keypoints skipped as
        mostly masked, in the input order. `unused` holds the keypoints of the csv header that
        are in neither `base` nor `additional`, including those skipped as mostly masked.

    Raises:
        ValueError: if a keypoint appears in both lists, twice in a list, does not exist, or a
            `base` keypoint was skipped as masked.
    """
    for key, names in (('base_keypoints', base), ('additional_keypoints', additional)):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f'{key} lists keypoints more than once: {duplicated}')
    overlap = sorted(set(base) & set(additional))
    if overlap:
        raise ValueError(f'keypoints in both base_keypoints and additional_keypoints: {overlap}')
    skipped_masked = keypoint_info['skipped_masked']
    additional_kept = []
    for name in base + additional:
        if name in skipped_masked:
            message = (
                f'keypoint {name!r} was skipped as mostly masked (masked fraction '
                f'{skipped_masked[name]:.3f} > {keypoint_info["max_masked_frac"]}, likelihood '
                f'threshold {keypoint_info["likelihood_threshold"]}) in '
                f'{keypoint_info["csv_path"]}'
            )
            if name in base:
                raise ValueError(message)
            logger.warning(f'dropping {name!r} from additional_keypoints: {message}')
            continue
        if name not in available:
            raise ValueError(f'unknown keypoint {name!r}; available: {available}')
        if name not in base:
            additional_kept.append(name)
    selected = set(base) | set(additional_kept)
    unused = [name for name in keypoint_info['keypoints'] if name not in selected]
    return additional_kept, unused


def select_keypoints(
    z_trials_time: np.ndarray,
    available: list[str],
    names: list[str],
) -> np.ndarray:
    """Pick the feature columns of some keypoints.

    Args:
        z_trials_time: array of shape `[K, T, 1, D]`, `D = len(available)`.
        available: keypoint names of the last axis.
        names: keypoints to keep, in output order.

    Returns:
        Array of shape `[K, T, 1, len(names)]`.
    """
    idx = [available.index(name) for name in names]
    return z_trials_time[..., idx]


def per_neuron_metrics(result: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-neuron R^2 (normalized rates) and BPS (raw rates) of an encoder result.

    Args:
        result: one `train_*_encoder` result with `gt`, `pred`, `norm_gt`, `norm_pred`.

    Returns:
        Dict with `r2` and `bps`, each of shape `[n_neurons]`; infinite BPS becomes NaN as in the
        encoders.
    """
    n_neurons = result['norm_gt'].shape[-1]
    r2 = r2_score(
        result['norm_gt'].reshape(-1, n_neurons),
        result['norm_pred'].reshape(-1, n_neurons),
        multioutput='raw_values',
    )
    bps = np.array([
        bits_per_spike(
            result['pred'][:, :, [idx]],
            result['gt'][:, :, [idx]],
            threshold=_BPS_RATE_THRESHOLD,
        )
        for idx in range(n_neurons)
    ])
    bps[np.isinf(bps)] = np.nan
    return {'r2': np.asarray(r2, dtype=np.float64), 'bps': bps}


def _make_train_data(
    eid: str,
    z_trials_time: dict[str, np.ndarray],
    neural_data_dict: Any,
) -> dict[str, Any]:
    """Build a fresh `train_data` dict (the RRR encoder mutates it in place).

    Args:
        eid: session id.
        z_trials_time: split name -> `[K, T, 1, D]` features.
        neural_data_dict: loaded `<eid>_aligned.npz`.

    Returns:
        Mapping `eid -> {'X': [train, val, test], 'y': [...], 'setup': {}}`.
    """
    splits = ('train', 'val', 'test')
    return {
        eid: {
            'X': [_flatten_views(z_trials_time[split]) for split in splits],
            'y': [np.array(neural_data_dict[f'{split}_spikes']) for split in splits],
            'setup': {},
        },
    }


def compare_conditions(metrics: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    """Compute full - base deltas for each model and metric.

    Args:
        metrics: flat dict with `{cond}_{model}_{r2,bps}` scalars and
            `{cond}_{model}_{r2,bps}_per_neuron` arrays.

    Returns:
        Dict with `delta_{model}_{metric}`, `delta_{model}_{metric}_per_neuron` and
        `frac_neurons_improved_{model}_{metric}` (over neurons with a finite delta).
    """
    base, full = _CONDITIONS
    out: dict[str, np.ndarray | float] = {}
    for model in _MODELS:
        for metric in ('r2', 'bps'):
            delta_per_neuron = (
                np.asarray(metrics[f'{full}_{model}_{metric}_per_neuron'])
                - np.asarray(metrics[f'{base}_{model}_{metric}_per_neuron'])
            )
            finite = delta_per_neuron[np.isfinite(delta_per_neuron)]
            out[f'delta_{model}_{metric}'] = float(
                metrics[f'{full}_{model}_{metric}'] - metrics[f'{base}_{model}_{metric}'],
            )
            out[f'delta_{model}_{metric}_per_neuron'] = delta_per_neuron
            out[f'frac_neurons_improved_{model}_{metric}'] = (
                float(np.mean(finite > 0)) if finite.size else float('nan')
            )
    return out


def format_summary_table(metrics: dict[str, np.ndarray | float]) -> str:
    """Format the held-out test BPS and R^2 of every condition and model as a text table.

    Args:
        metrics: output of `fit_condition` / `load_or_fit_base` merged with `compare_conditions`.

    Returns:
        Multi-line, space-aligned table with one row per fitted condition and model, followed by
        the `base_plus_additional - base` deltas when both conditions were fitted.
    """
    rows = [('Condition', 'Model', 'Test BPS', 'Test R2')]
    for cond in _CONDITIONS:
        if f'{cond}_{_MODELS[0]}_bps' not in metrics:
            continue
        for model in _MODELS:
            rows.append((
                cond,
                model.upper(),
                f'{metrics[f"{cond}_{model}_bps"]:.4f}',
                f'{metrics[f"{cond}_{model}_r2"]:.4f}',
            ))
    for model in _MODELS:
        if f'delta_{model}_bps' not in metrics:
            continue
        rows.append((
            f'delta ({_CONDITIONS[1]} - {_CONDITIONS[0]})',
            model.upper(),
            f'{metrics[f"delta_{model}_bps"]:+.4f}',
            f'{metrics[f"delta_{model}_r2"]:+.4f}',
        ))
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    return '\n'.join(
        '  '.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


def fit_condition(
    cond: str,
    eid: str,
    z_trials_time: dict[str, np.ndarray],
    neural_data_dict: Any,
    tune_storage_path: str,
    seed: int,
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray]]:
    """Fit RRR and CNN on one input set and collect their test metrics and predictions.

    Args:
        cond: condition name used as prefix of the returned keys.
        eid: session id.
        z_trials_time: split name -> `[K, T, 1, D]` features of the selected keypoints.
        neural_data_dict: loaded `<eid>_aligned.npz`.
        tune_storage_path: Ray Tune experiment root.
        seed: random seed forwarded to the CNN trials.

    Returns:
        Tuple `(metrics, predictions)`. `metrics` holds the `{cond}_{model}_{r2,bps}` scalars and
        `{cond}_{model}_{r2,bps}_per_neuron` arrays for `rrr` and `cnn`. `predictions` holds the
        float32 held-out test arrays `{cond}_{model}_gt` and `{cond}_{model}_pred`, each of shape
        `[n_trials, n_bins, n_neurons]`.
    """
    metrics: dict[str, np.ndarray | float] = {}
    predictions: dict[str, np.ndarray] = {}
    for model, train_fn in (
        ('rrr', train_rrr_encoder_with_tune),
        ('cnn', train_cnn_encoder_with_tune),
    ):
        train_data = _make_train_data(eid, z_trials_time, neural_data_dict)
        for split_name, x, y in zip(
            ('Train', 'Val', 'Test'), train_data[eid]['X'], train_data[eid]['y'],
        ):
            logger.info(f'{cond} {split_name} X Shape: {x.shape}, Y Shape: {y.shape}')
        kwargs = {'seed': seed} if model == 'cnn' else {}
        result = train_fn(
            train_data, num_samples=30, tune_storage_path=tune_storage_path, **kwargs,
        )[eid]
        logger.info(
            f'{cond} {model.upper()} Encoding {eid} Test BPS: {result["bps"]} '
            f'Test R2: {result["r2"]}',
        )
        per_neuron = per_neuron_metrics(result)
        metrics[f'{cond}_{model}_r2'] = float(result['r2'])
        metrics[f'{cond}_{model}_bps'] = float(result['bps'])
        metrics[f'{cond}_{model}_r2_per_neuron'] = per_neuron['r2']
        metrics[f'{cond}_{model}_bps_per_neuron'] = per_neuron['bps']
        predictions[f'{cond}_{model}_gt'] = np.asarray(result['gt'], dtype=np.float32)
        predictions[f'{cond}_{model}_pred'] = np.asarray(result['pred'], dtype=np.float32)
    return metrics, predictions


def base_cache_key(
    base: list[str],
    camera: str,
    seed: int,
    likelihood_threshold: float,
    z_base: dict[str, np.ndarray],
    neural_data_dict: Any,
    pca_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe every input of the base fit, to decide whether a cached base can be reused.

    The trial split is part of `<eid>_aligned.npz`, so it enters through the hash of the spikes.

    Args:
        base: `base_keypoints` in column order.
        camera: `'left'` or `'right'`.
        seed: random seed (model init / Ray Tune).
        likelihood_threshold: threshold used by step A / step B.
        z_base: split name -> `[K, T, 1, D]` features fed to the encoder (after preprocessing).
        neural_data_dict: loaded `<eid>_aligned.npz`.
        pca_info: `read_pca_record` of step B, `None` when pca is off; the `pca` entry is only
            added when pca is on, so a cache from before pca existed is still reused by
            runs without pca.

    Returns:
        Json-serializable dict, compared with `==` against the cached `base_key.json`.
    """
    digest = hashlib.sha256()
    for split in _SPLITS:
        z = np.ascontiguousarray(z_base[split], dtype=np.float32)
        spikes = np.ascontiguousarray(neural_data_dict[f'{split}_spikes'])
        for arr in (z, spikes):
            digest.update(f'{split}{arr.shape}{arr.dtype}'.encode())
            digest.update(arr.tobytes())
    key: dict[str, Any] = {
        'base_keypoints': list(base),
        'camera': camera,
        'seed': int(seed),
        'likelihood_threshold': float(likelihood_threshold),
        'data_sha256': digest.hexdigest(),
    }
    if pca_info is not None:
        key['pca'] = {
            'group_set': pca_info['group_set'],
            'n_components': pca_info['n_components'],
        }
    return key


def _read_base_metrics(result_path: Path) -> dict[str, np.ndarray | float]:
    """Load the `base_{model}_{metric}` entries of a cached base result.

    Args:
        result_path: `<eid>/base/encoding_result.npz`.

    Returns:
        Dict with scalars as `float` and per-neuron entries as arrays.
    """
    data = np.load(result_path)
    out: dict[str, np.ndarray | float] = {}
    for model in _MODELS:
        for metric in ('r2', 'bps'):
            out[f'base_{model}_{metric}'] = float(data[f'base_{model}_{metric}'])
            out[f'base_{model}_{metric}_per_neuron'] = np.asarray(
                data[f'base_{model}_{metric}_per_neuron'],
            )
    return out


def _read_base_predictions(predictions_path: Path) -> dict[str, np.ndarray]:
    """Load the `base_{model}_{gt,pred}` arrays of a cached base fit.

    Args:
        predictions_path: `<eid>/base/<hash8>/encoding_predictions.npz`.

    Returns:
        Dict of the float32 test arrays.
    """
    with np.load(predictions_path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def load_or_fit_base(
    base_directory: Path,
    key: dict[str, Any],
    fit: Any,
    rerun: bool,
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray], str]:
    """Reuse the cached base fit when its key matches, otherwise fit and cache it.

    A cache from before the predictions were saved (no `encoding_predictions.npz`) cannot be
    reused, since the psth / raster plot needs the test predictions; the base is refitted.

    Args:
        base_directory: `<eid>/base/<hash8>`; holds `encoding_result.npz`,
            `encoding_predictions.npz` and `base_key.json`.
        key: output of `base_cache_key`.
        fit: zero-argument callable returning the base `(metrics, predictions)`
            (`fit_condition('base', ...)`).
        rerun: refit even if the cache matches.

    Returns:
        Tuple `(metrics, predictions, status)`; `status` is `'reused'`, `'fitted (no cache)'`,
        `'fitted (cache key mismatch)'`, `'fitted (no predictions cached)'` or
        `'fitted (--rerun_base)'`.
    """
    result_path = base_directory / 'encoding_result.npz'
    predictions_path = base_directory / _PREDICTIONS_NAME
    key_path = base_directory / 'base_key.json'
    status = 'fitted (no cache)'
    if result_path.is_file() and key_path.is_file():
        with key_path.open(encoding='utf-8') as f:
            cached_key = json.load(f)
        if rerun:
            status = 'fitted (--rerun_base)'
        elif cached_key != key:
            status = 'fitted (cache key mismatch)'
        elif not predictions_path.is_file():
            status = 'fitted (no predictions cached)'
        else:
            return (
                _read_base_metrics(result_path),
                _read_base_predictions(predictions_path),
                'reused',
            )
    metrics, predictions = fit()
    base_directory.mkdir(parents=True, exist_ok=True)
    # write to a temp file and rename, so a half-written result is never read as a cache; the
    # predictions go first because the result file together with the key marks a usable cache
    tmp_predictions = base_directory / 'encoding_predictions.tmp.npz'
    np.savez_compressed(tmp_predictions, **predictions)
    tmp_predictions.replace(predictions_path)
    tmp_path = base_directory / 'encoding_result.tmp.npz'
    np.savez(tmp_path, base_keypoints=np.asarray(key['base_keypoints']), **metrics)
    tmp_path.replace(result_path)
    with key_path.open('w', encoding='utf-8') as f:
        json.dump(key, f, indent=2)
    return metrics, predictions, status


def main(argv: list[str] | None = None) -> None:
    """Run the base vs base + additional keypoint encoding comparison for one session.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if `dynamic_vars.npz` or the step A json is missing.
        ValueError: if the config is invalid or its camera differs from `dynamic_vars.npz`.
    """
    setup_logging()
    args = parse_args(argv)
    cfg = load_config(args.config, eid=args.eid)
    pca_record = read_camera_pca_record(
        cfg['video_preds_dir'], str(cfg['output_name']), cfg['camera'], cfg['eid'],
    )
    out_dir, out_reason = resolve_variant_dir(
        cfg['video_preds_dir'],
        cfg['output_folder'],
        cfg['eid'],
        cfg['variant_name'],
        run_signature(cfg, args.seed, pca_record),
    )
    # the folder name is the variant name from here on (it may carry a hash suffix)
    cfg['variant_name'] = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # freeze the config: every later read (and the plot, via params['config']) uses this copy, so
    # editing the yaml for another task while the job runs cannot change this run
    config_frozen = out_dir / _FROZEN_CONFIG_NAME
    shutil.copyfile(args.config, config_frozen)
    args.config = config_frozen

    file_handler = logging.FileHandler(out_dir / 'encoding.log', mode='w')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s'),
    )
    root_logger = logging.getLogger()
    level_before = root_logger.level
    # make sure INFO records reach the file even if a handler configured the root logger earlier
    root_logger.setLevel(min(level_before or logging.INFO, logging.INFO))
    root_logger.addHandler(file_handler)
    logger.info(f'results folder: {out_dir} ({out_reason})')
    try:
        _run(args, cfg, out_dir, pca_record)
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(level_before)
        file_handler.close()


def _load_camera_inputs(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    camera: str,
) -> dict[str, Any]:
    """Load the step A json and step B npz of one camera and check them against the config.

    Args:
        args: parsed command-line arguments (`args.config` is the frozen config).
        cfg: validated config from `load_config`.
        camera: `'left'` or `'right'`.

    Returns:
        Dict with `camera`, `dynamic_vars_path`, `keypoint_info` (step A json),
        `no_filter_keypoints`, `visibility`, `latent` (loaded `dynamic_vars.npz`) and `available`
        (its feature names).

    Raises:
        FileNotFoundError: if `dynamic_vars.npz` or the step A json is missing.
        ValueError: if the step B outputs were built from another camera or with another pca
            flag than the config implies, or the step A json was written with other filter
            settings than the config.
    """
    eid = cfg['eid']
    video_preds_dir = Path(cfg['video_preds_dir'])
    output_folder = output_folder_name(camera, str(cfg['output_name']))
    session_dir = session_output_dir(video_preds_dir, output_folder, eid)
    dynamic_vars_path = session_dir / 'dynamic_vars.npz'
    if not dynamic_vars_path.is_file():
        raise FileNotFoundError(
            f'{dynamic_vars_path} not found; run '
            'pose2visual.encoding.extract_litpose_dynamic_vars_trials first',
        )
    json_path = keypoint_names_json_path(video_preds_dir, camera, eid)
    with json_path.open(encoding='utf-8') as f:
        keypoint_info = json.load(f)
    no_filter_keypoints = load_unfiltered_keypoints(args.config, keypoint_info['keypoints'])
    visibility = load_visibility_features(args.config, keypoint_info['keypoints'])
    check_filter_settings_match(
        keypoint_info,
        cfg['likelihood_threshold'],
        cfg['max_masked_frac'],
        json_path,
        no_filter_keypoints,
        visibility,
        load_position_columns(args.config),
        load_same_side_excluded(args.config, keypoint_info['keypoints'], camera),
    )

    latent_data_dict = np.load(dynamic_vars_path)
    stored_camera = str(latent_data_dict['camera'])
    if stored_camera != camera:
        raise ValueError(
            f'config camera {cfg["camera"]!r} != camera {stored_camera!r} of {dynamic_vars_path}',
        )
    # an npz from before PCA moved to step B was built without pca
    step_b_pca = bool(latent_data_dict['pca']) if 'pca' in latent_data_dict.files else False
    if step_b_pca != cfg['pca']:
        raise ValueError(
            f'config step_order {cfg["step_order"]!r} implies pca: {cfg["pca"]} but '
            f'{dynamic_vars_path} was built with pca: {step_b_pca}; '
            'rerun extract_litpose_dynamic_vars_trials (step 1) with this config, or use the '
            'output_name of the matching run',
        )
    return {
        'camera': camera,
        'dynamic_vars_path': dynamic_vars_path,
        'keypoint_info': keypoint_info,
        'no_filter_keypoints': no_filter_keypoints,
        'visibility': visibility,
        'latent': latent_data_dict,
        'available': [str(name) for name in latent_data_dict['keypoint_names']],
    }


def select_camera_features(
    cfg: dict[str, Any],
    groups: dict[str, list[str]] | None,
    available: list[str],
    skip_session_traces: bool,
    side_excluded: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve `base_keypoints` and `additional_keypoints` against one camera's features.

    Args:
        cfg: validated config from `load_config`.
        groups: group name -> keypoint name tokens (`load_groups`), `None` when the lists hold
            explicit keypoint names.
        available: feature names stored in this camera's `dynamic_vars.npz`.
        skip_session_traces: drop the one_api traces that do not depend on the camera
            (`wheel_speed`, `licks`), because another camera already provides them.
        side_excluded: keypoints the yaml's `same_side_filter` drops for this camera (`side_excluded`
            of the step A json). With explicit keypoint names (`groups is None`) they are left out
            of the selection instead of failing as unknown, so `wrist_left` and `wrist_right` can
            both be listed and each camera takes its own. Group names need nothing, as a group
            only resolves to the keypoints in `available`.

    Returns:
        Tuple `(base, additional)` of feature names of this camera.

    Raises:
        ValueError: if a group is unknown or has no usable keypoint, a trace is missing from
            `available` (see `expand_selection`), or the side filter leaves no base keypoint.
    """
    if groups is None:
        dropped = set(side_excluded or [])
        base = [name for name in cfg['base_keypoints'] if name not in dropped]
        additional = [name for name in cfg['additional_keypoints'] if name not in dropped]
        left_out = sorted(
            dropped & (set(cfg['base_keypoints']) | set(cfg['additional_keypoints'])),
        )
        if left_out:
            logger.info(f'same_side_filter: not using {left_out} for this camera')
        if not base:
            raise ValueError(
                f'same_side_filter leaves no base keypoint for this camera; '
                f'base_keypoints {cfg["base_keypoints"]}, dropped {sorted(dropped)}',
            )
    else:
        base = expand_selection(cfg['base_keypoints'], groups, cfg['one_api'], available)
        additional = expand_selection(
            cfg['additional_keypoints'], groups, cfg['one_api'], available,
        )
    if skip_session_traces:
        base = [name for name in base if not _is_session_trace(name)]
        additional = [name for name in additional if not _is_session_trace(name)]
    return base, additional


def _is_session_trace(name: str) -> bool:
    """Whether a feature is a one_api trace that is the same for both cameras.

    Args:
        name: feature name.

    Returns:
        `True` for `wheel_speed` and `licks`.
    """
    return name in ONE_API_TRACES and not ONE_API_TRACES[name]


def column_names(camera: str, names: list[str], both: bool) -> list[str]:
    """Name the columns of one camera in the model input.

    Args:
        camera: `'left'` or `'right'`.
        names: feature names of that camera.
        both: whether the model uses both cameras; otherwise the names are kept as they are.

    Returns:
        `<camera>:<name>` for each name of the two-camera model, except the one_api traces that
        do not depend on the camera, which keep their plain name.
    """
    if not both:
        return list(names)
    return [name if _is_session_trace(name) else f'{camera}:{name}' for name in names]


def _concat_selected(
    inputs: list[dict[str, Any]],
    key: str,
) -> dict[str, np.ndarray]:
    """Concatenate the selected columns of all cameras along the feature axis.

    Args:
        inputs: output of `_load_camera_inputs` per camera, each with the feature names to pick
            under `key`.
        key: `'base'` or `'additional'`.

    Returns:
        Split name -> `[K, T, 1, D]` array, the cameras' columns in camera order.
    """
    return {
        split: np.concatenate(
            [
                select_keypoints(
                    item['latent'][f'{split}_z_trials_time'], item['available'], item[key],
                )
                for item in inputs
            ],
            axis=-1,
        )
        for split in _SPLITS
    }


def _check_cameras_aligned(inputs: list[dict[str, Any]]) -> None:
    """Check that the step B outputs of the cameras cover the same trials.

    Args:
        inputs: output of `_load_camera_inputs` per camera.

    Raises:
        ValueError: if the cameras differ in the intervals of any split.
    """
    first = inputs[0]
    for item in inputs[1:]:
        for split in _SPLITS:
            key = f'{split}_intervals'
            if not np.array_equal(first['latent'][key], item['latent'][key]):
                raise ValueError(
                    f'{key} of {item["dynamic_vars_path"]} differs from '
                    f'{first["dynamic_vars_path"]}; rerun step B for both cameras',
                )


def _run(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    out_dir: Path,
    pca_record: dict[str, Any] | None,
) -> None:
    """Body of `main` once the log file handler is attached.

    Args:
        args: parsed command-line arguments.
        cfg: validated config from `load_config`.
        out_dir: `<video_preds_dir>/<camera>_<output_name>/<eid>/<variant_name>`.
        pca_record: `read_camera_pca_record` of step B, `None` when pca is off.

    Raises:
        FileNotFoundError: if `dynamic_vars.npz` or the step A json is missing.
        ValueError: if the config camera or pca flag differs from `dynamic_vars.npz`, the step A
            json was written with other filter settings than the config, the cameras of a
            two-camera run cover different trials, or the selection is invalid.
    """
    eid = cfg['eid']
    video_preds_dir = Path(cfg['video_preds_dir'])
    both = cfg['camera'] == CAMERA_BOTH
    inputs = [_load_camera_inputs(args, cfg, camera) for camera in cameras_of(cfg['camera'])]
    _check_cameras_aligned(inputs)

    if cfg['anatomical_group'] == ANATOMICAL_GROUP_ALL:
        groups = None
        base_groups: list[str] = []
        additional_groups: list[str] = []
    else:
        groups = load_groups(cfg['body_parts_json'], cfg['anatomical_group'])
        base_groups = cfg['base_keypoints']
        additional_groups = cfg['additional_keypoints']
    base: list[str] = []
    additional: list[str] = []
    unused: list[str] = []
    for idx, item in enumerate(inputs):
        # the traces that do not depend on the camera come from the first camera only
        item['base'], item['additional'] = select_camera_features(
            cfg, groups, item['available'], skip_session_traces=idx > 0,
            side_excluded=item['keypoint_info'].get('side_excluded', []),
        )
        camera = item['camera']
        if groups is not None:
            logger.info(f'[{camera}] base groups {base_groups} -> keypoints {item["base"]}')
            logger.info(
                f'[{camera}] additional groups {additional_groups} -> keypoints '
                f'{item["additional"]}',
            )
        item['additional'], item['unused'] = split_keypoints(
            item['base'], item['additional'], item['keypoint_info'], item['available'],
        )
        base += column_names(camera, item['base'], both)
        additional += column_names(camera, item['additional'], both)
        unused += column_names(camera, item['unused'], both)

    def per_camera(key: str, fn: Any) -> Any:
        """Value of every camera for the params, or the plain value for a single camera."""
        values = {item['camera']: fn(item[key]) for item in inputs}
        return values if both else next(iter(values.values()))

    params = {
        'eid': eid,
        'camera': cfg['camera'],
        'variant_name': cfg['variant_name'],
        'run_signature': run_signature(cfg, args.seed, pca_record),
        'seed': args.seed,
        'config': str(Path(args.config).resolve()),
        'model_dir': per_camera('keypoint_info', lambda info: info['model_dir']),
        'video_preds_dir': str(video_preds_dir),
        'csv_path': per_camera('keypoint_info', lambda info: info['csv_path']),
        'dynamic_vars_npz': per_camera('dynamic_vars_path', str),
        'anatomical_group': cfg['anatomical_group'],
        'base_groups': base_groups,
        'additional_groups': additional_groups,
        'base_keypoints': base,
        'additional_keypoints': additional,
        'unused_keypoints': unused,
        'likelihood_threshold': cfg['likelihood_threshold'],
        'max_masked_frac': cfg['max_masked_frac'],
        'no_filter_keypoints': per_camera('no_filter_keypoints', list),
        'visibility': per_camera('visibility', lambda value: value),
        'one_api': cfg['one_api'],
        'skipped_masked_keypoints': per_camera(
            'keypoint_info', lambda info: info['skipped_masked'],
        ),
        'step_order': cfg['step_order'],
    }
    for key, value in params.items():
        logger.info(f'{key}: {value}')

    ray_temp = _resolve_ray_session_temp_dir()
    logger.info(f'Ray session temp dir (_temp_dir): {ray_temp}')
    ray.init(num_cpus=2, num_gpus=1, _temp_dir=ray_temp)
    set_seed(args.seed)

    neural_data_dict = np.load(
        Path(args.neural_input_dir) / eid / f'{eid}_aligned.npz', allow_pickle=True,
    )
    for item in inputs:
        _maybe_assert_latent_intervals_match_neural(item['latent'], neural_data_dict)
    z_base = _concat_selected(inputs, 'base')
    check_finite(z_base, 'base')
    run_full = bool(additional)
    if run_full:
        z_additional = _concat_selected(inputs, 'additional')
        z_full = {
            split: np.concatenate([z_base[split], z_additional[split]], axis=-1)
            for split in _SPLITS
        }
        check_finite(z_full, 'base_plus_additional')
    params['pca'] = {'enabled': cfg['pca'], 'group_set': None, 'n_components': None}
    if pca_record is not None:
        params['pca'].update(pca_record)

    key = base_cache_key(
        base,
        cfg['camera'],
        args.seed,
        cfg['likelihood_threshold'],
        z_base,
        neural_data_dict,
        pca_record,
    )
    # one folder per distinct base input, so a pca / no-pca run never replaces the other's fit
    base_directory = base_dir(
        video_preds_dir, cfg['output_folder'], eid, _short_hash(json.dumps(key, sort_keys=True)),
    )
    params['base_dir'] = str(base_directory)
    logger.info(f'condition base: {len(base)} features')
    metrics, predictions, base_status = load_or_fit_base(
        base_directory,
        key,
        lambda: fit_condition(
            'base', eid, z_base, neural_data_dict, str(base_directory / 'ray_tune'), args.seed,
        ),
        rerun=args.rerun_base,
    )
    logger.info(f'base: {base_status} ({base_directory})')
    params['base_status'] = base_status
    with (out_dir / 'encoding_params.json').open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)

    if run_full:
        logger.info(f'condition base_plus_additional: {len(base + additional)} features')
        metrics_full, predictions_full = fit_condition(
            'base_plus_additional',
            eid,
            z_full,
            neural_data_dict,
            str(out_dir / 'ray_tune'),
            args.seed,
        )
        metrics.update(metrics_full)
        predictions.update(predictions_full)
        metrics.update(compare_conditions(metrics))
        for model in _MODELS:
            logger.info(
                f'{model.upper()}: delta R2 = {metrics[f"delta_{model}_r2"]:.4f} '
                f'(improved {metrics[f"frac_neurons_improved_{model}_r2"]:.2%} of neurons), '
                f'delta BPS = {metrics[f"delta_{model}_bps"]:.4f} '
                f'(improved {metrics[f"frac_neurons_improved_{model}_bps"]:.2%} of neurons)',
            )
    else:
        logger.info('no additional keypoints: only the base condition was fitted')

    logger.info(f'Summary (held-out test set):\n{format_summary_table(metrics)}')

    # predictions first: encoding_result.npz is what marks the run as finished for the plots
    predictions_path = out_dir / _PREDICTIONS_NAME
    np.savez_compressed(predictions_path, **predictions)
    logger.info(f'wrote {predictions_path}')
    result_path = out_dir / 'encoding_result.npz'
    np.savez(
        result_path,
        variant_name=np.asarray(cfg['variant_name']),
        anatomical_group=np.asarray(cfg['anatomical_group']),
        base_groups=np.asarray(base_groups, dtype=str),
        additional_groups=np.asarray(additional_groups, dtype=str),
        base_keypoints=np.asarray(base),
        additional_keypoints=np.asarray(additional, dtype=str),
        unused_keypoints=np.asarray(unused, dtype=str),
        **metrics,
    )
    logger.info(f'wrote {result_path}')


if __name__ == '__main__':
    main()
