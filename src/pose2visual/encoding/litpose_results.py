"""Shared helpers of the encoding plots: find the finished run of a session and read its results.

``plot_encoding_base_vs_additional.py``, ``plot_psth_activity_raster.py`` and
``plot_bps_scatter.py`` all identify the wanted run from the encoding yaml and read the files that
``pose2visual.encoding.run_litpose_encoding`` wrote into the variant folder::

    <video_preds_dir>/<camera>_<output_name>/<eid>/<variant>/encoding_result.npz
    <video_preds_dir>/<camera>_<output_name>/<eid>/<variant>/encoding_predictions.npz
    <video_preds_dir>/<camera>_<output_name>/<eid>/<variant>/encoding_params.json

The variant is chosen by the ``run_signature`` of ``encoding_params.json`` (see
``find_variant_dir``).
All figures go to one folder per run setting, see ``plots_dir``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.encoding.litpose_keypoints import (
    ENCODING_YAML_NESTED_NAMES,
    load_encoding_yaml,
    output_folder_name,
    resolve_group_eids,
)

CONDITIONS = ('base', 'base_plus_additional')
MODELS = ('rrr', 'cnn')
LEGEND_CONDITION_LABELS = {'base': 'Base', 'base_plus_additional': 'Base + additional'}
# base hues of the old figure-3 plot: grey for the reference, orange for the alternative
CONDITION_COLORS = {'base': '#616161', 'base_plus_additional': '#F28E2B'}
RESULT_NAME = 'encoding_result.npz'
PREDICTIONS_NAME = 'encoding_predictions.npz'
_REQUIRED_CONFIG_KEYS = (
    'video_preds_dir',
    'output_name',
    'camera',
    'base_keypoints',
    'additional_keypoints',
)
_DEFAULT_PCA_GROUP_SET = 'nonoverlapping_analysis_groups'
# `anatomical_group` value meaning "the keypoint lists hold keypoint names"
_ANATOMICAL_GROUP_ALL = 'all'
# folder of `output_folder` that holds the plots, never a session
_PLOTS_DIRNAME = 'plots'


def load_wanted_settings(
    config_path: str | Path,
    base_keypoints: list[str] | None = None,
    additional_keypoints: list[str] | None = None,
    seed: int | None = None,
    anatomical_group: str | None = None,
    eid_group: str | None = None,
) -> dict[str, Any]:
    """Read the settings that identify the wanted run from the encoding yaml.

    Args:
        config_path: path of the yaml the encoding was run with.
        base_keypoints: replaces `base_keypoints` of the file when given.
        additional_keypoints: replaces `additional_keypoints` of the file when given.
        seed: only runs of this seed match; `None` matches any seed.
        anatomical_group: replaces `anatomical_group` of the file when given.
        eid_group: replaces `session.eid_group` of the file when given.

    Returns:
        Dict with `video_preds_dir`, `output_name`, `output_folder` (`<camera>_<output_name>`),
        `camera`, `base_keypoints`,
        `additional_keypoints` (group names when `anatomical_group` is a group set),
        `anatomical_group` (default `all`), `same_side_groups` (sorted, empty when the yaml has no
        `same_side_filter`), `pca`, `pca_group_set`, `step_order`, `seed` and `eids`
        (the sessions of the eid group, `None` when there is no group).

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), a required key is
            missing, or an eid group is set without `paths.eid_groups_json` or is unknown.
    """
    cfg = load_encoding_yaml(config_path)
    missing = [
        ENCODING_YAML_NESTED_NAMES[key] for key in _REQUIRED_CONFIG_KEYS if cfg.get(key) is None
    ]
    if missing:
        raise ValueError(f'{config_path}: missing config keys {missing}')
    group = eid_group if eid_group is not None else cfg.get('eid_group')
    eids = None
    if group is not None:
        if cfg.get('eid_groups_json') is None:
            raise ValueError(f'{config_path}: an eid group needs paths.eid_groups_json')
        eids = resolve_group_eids(cfg['eid_groups_json'], str(group))
    wanted: dict[str, Any] = {
        'video_preds_dir': str(cfg['video_preds_dir']),
        'output_name': str(cfg['output_name']),
        'output_folder': output_folder_name(str(cfg['camera']), str(cfg['output_name'])),
        'camera': str(cfg['camera']),
        'base_keypoints': [str(name) for name in (base_keypoints or cfg['base_keypoints'])],
        'additional_keypoints': [
            str(name) for name in (additional_keypoints or cfg['additional_keypoints'])
        ],
        'anatomical_group': str(
            cfg.get('anatomical_group', _ANATOMICAL_GROUP_ALL)
            if anatomical_group is None else anatomical_group,
        ),
        'same_side_groups': sorted(cfg.get('same_side_groups') or []),
        'pca': cfg.get('pca', False),
        'pca_group_set': str(cfg.get('pca_group_set', _DEFAULT_PCA_GROUP_SET)),
        'step_order': str(cfg.get('step_order', 'unknown')),
        'seed': seed,
        'eids': eids,
    }
    return wanted


def signature_differences(signature: dict[str, Any], wanted: dict[str, Any]) -> list[str]:
    """List the settings in which a finished run differs from the wanted run.

    Keypoint lists (group names in group mode) are compared as sets of names, together with the
    anatomical group; a signature without one is an `all` run. The pca group set only matters
    when pca is on, and the seed only when `wanted['seed']` is set.

    Args:
        signature: `run_signature` stored in `encoding_params.json`.
        wanted: output of `load_wanted_settings`.

    Returns:
        One short text per differing setting (empty when the run is the wanted one).
    """
    differences = []
    for key in ('base_keypoints', 'additional_keypoints'):
        if sorted(signature.get(key, [])) != sorted(wanted[key]):
            differences.append(f'{key} differ')
    if signature.get('camera') != wanted['camera']:
        differences.append(f'camera: run {signature.get("camera")!r}, yaml {wanted["camera"]!r}')
    # a signature from before the filter existed has no such entry and counts as none
    run_sides = sorted(signature.get('same_side_groups', []))
    if run_sides != wanted.get('same_side_groups', []):
        differences.append(
            f'same_side_groups: run {run_sides}, yaml {wanted.get("same_side_groups", [])}',
        )
    if signature.get('pca') != wanted['pca']:
        differences.append(
            f'pca: run {signature.get("pca")}, yaml {wanted["pca"]} '
            f'(the yaml\'s step_order {wanted.get("step_order")!r} decides it)',
        )
    run_group = signature.get('anatomical_group', _ANATOMICAL_GROUP_ALL)
    if run_group != wanted['anatomical_group']:
        differences.append(
            f'anatomical_group: run {run_group!r}, yaml {wanted["anatomical_group"]!r}',
        )
    if wanted['pca'] and signature.get('pca_group_set') != wanted['pca_group_set']:
        differences.append(
            f'pca_group_set: run {signature.get("pca_group_set")!r}, '
            f'yaml {wanted["pca_group_set"]!r}',
        )
    if wanted['seed'] is not None and signature.get('seed') != wanted['seed']:
        differences.append(f'seed: run {signature.get("seed")}, wanted {wanted["seed"]}')
    return differences


def signature_matches(signature: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Whether the `run_signature` of a finished run is the wanted run.

    Keypoint lists (group names in group mode) are compared as sets of names, together with the
    anatomical group; a signature without one is an `all` run. The pca group set only matters
    when pca is on, and the seed only when `wanted['seed']` is set.

    Args:
        signature: `run_signature` stored in `encoding_params.json`.
        wanted: output of `load_wanted_settings`.

    Returns:
        `True` if camera, keypoints, anatomical group, pca settings (and seed) agree.
    """
    return not signature_differences(signature, wanted)


def find_variant_dir(session_dir: Path, wanted: dict[str, Any]) -> Path:
    """Find the finished variant folder of one session that matches the wanted run.

    Args:
        session_dir: `<video_preds_dir>/<camera>_<output_name>/<eid>`.
        wanted: output of `load_wanted_settings`.

    Returns:
        The variant folder holding `encoding_result.npz`.

    Raises:
        FileNotFoundError: if no variant matches (the message says how each existing variant
            differs from the yaml), or the matching one has no `encoding_result.npz` yet (the run
            is unfinished or failed).
        ValueError: if more than one variant matches.
    """
    matches = []
    mismatched = []
    for params_path in sorted(session_dir.glob('*/encoding_params.json')):
        try:
            with params_path.open(encoding='utf-8') as f:
                signature = json.load(f).get('run_signature')
        except (OSError, ValueError):
            continue
        # results from before run_signature existed cannot be identified
        if not isinstance(signature, dict):
            continue
        differences = signature_differences(signature, wanted)
        if differences:
            mismatched.append(f'{params_path.parent.name}: {"; ".join(differences)}')
        else:
            matches.append(params_path.parent)
    if not matches:
        found = (
            f'; existing variants differ from the yaml in: {" | ".join(mismatched)}'
            if mismatched else (
                '; the folder holds no variant with a run_signature' if session_dir.is_dir()
                else '; the folder does not exist, so the session was not run'
            )
        )
        raise FileNotFoundError(
            f'{session_dir}: no variant matches the wanted settings{found}',
        )
    if len(matches) > 1:
        raise ValueError(
            f'{session_dir}: {len(matches)} variants match the wanted settings: '
            f'{[path.name for path in matches]}',
        )
    if not (matches[0] / 'encoding_result.npz').is_file():
        raise FileNotFoundError(f'{matches[0]}: matching run has no encoding_result.npz yet')
    return matches[0]



def resolve_eid(config_path: str | Path, eid: str | None) -> str:
    """Session to plot: the given eid, else `session.eid` of the config.

    Args:
        config_path: path of the encoding yaml.
        eid: session id given on the command line, `None` to use the config.

    Returns:
        The session id.

    Raises:
        ValueError: if neither is set.
    """
    if eid is not None:
        return eid
    config_eid = load_encoding_yaml(config_path).get('eid')
    if config_eid is None:
        raise ValueError(f'{config_path}: no session.eid in the config; pass --eid')
    return str(config_eid)


def plots_dir(wanted: dict[str, Any]) -> Path:
    """Folder that holds all figures of one run setting.

    Args:
        wanted: output of `load_wanted_settings`.

    Returns:
        `<video_preds_dir>/<output_folder>/plots/encoding_base_vs_additional_<additional>_<pca>`;
        `<additional>` is prefixed by the group set in group mode, so the same group names of two
        group sets do not share a folder.
    """
    additional = '_'.join(sorted(wanted['additional_keypoints']))[:60]
    if wanted['anatomical_group'] != _ANATOMICAL_GROUP_ALL:
        additional = f'{wanted["anatomical_group"]}_{additional}'
    pca = 'pca' if wanted['pca'] else 'nopca'
    return (
        Path(wanted['video_preds_dir'])
        / wanted['output_folder']
        / _PLOTS_DIRNAME
        / f'encoding_base_vs_additional_{additional}_{pca}'
    )


def load_session_bps(variant_path: Path, condition: str, model: str) -> np.ndarray:
    """Load the held-out test BPS of every neuron of one finished variant.

    Args:
        variant_path: folder holding `encoding_result.npz`.
        condition: `base` or `base_plus_additional`.
        model: `rrr` or `cnn`.

    Returns:
        Array of shape `[n_neurons]`; infinite BPS was stored as NaN.
    """
    with np.load(variant_path / RESULT_NAME, allow_pickle=False) as data:
        return np.asarray(data[f'{condition}_{model}_bps_per_neuron'], dtype=np.float64)


def load_session_predictions(
    variant_path: Path,
    condition: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the held-out test spikes and predicted rates of one condition and model.

    Args:
        variant_path: folder holding `encoding_predictions.npz`.
        condition: `base` or `base_plus_additional`.
        model: `rrr` or `cnn`.

    Returns:
        Tuple `(gt, pred)`, each of shape `[n_trials, n_bins, n_neurons]` (float64).

    Raises:
        FileNotFoundError: if the variant has no `encoding_predictions.npz` (it was run before
            the predictions were saved; rerun step 2 for the session).
        ValueError: if `gt` and `pred` are not matching 3d arrays.
    """
    path = variant_path / PREDICTIONS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} not found: this run was made before the test predictions were saved; '
            'rerun step2_run_encoding.sh for the session (the base is refitted automatically)',
        )
    with np.load(path, allow_pickle=False) as data:
        gt = np.asarray(data[f'{condition}_{model}_gt'], dtype=np.float64)
        pred = np.asarray(data[f'{condition}_{model}_pred'], dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(
            f'{path}: gt / pred must be matching 3d arrays (trials, bins, neurons); '
            f'got {gt.shape}, {pred.shape}',
        )
    return gt, pred
