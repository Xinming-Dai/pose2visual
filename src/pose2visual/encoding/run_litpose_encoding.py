"""Step C: neural encoding from Lightning Pose keypoints, base vs base + additional keypoints.

Reads the yaml config (see ``configs/encoding/litpose_encoding.yaml``) and
``<video_preds_dir>/<output_name>/<eid>/dynamic_vars.npz`` from
``extract_litpose_dynamic_vars_trials``, then fits RRR and CNN encoders (Ray Tune, as in
``run_encoding``) on two input sets; the train/val/test split is the one already stored in
``<eid>_aligned.npz``, so it is identical for both:

* ``base``: the speed of ``base_keypoints``;
* ``base_plus_additional``: the speed of ``base_keypoints + additional_keypoints``.

Keypoints listed in neither list are ``unused``. The held-out test R^2 and bits-per-spike (BPS),
overall and per neuron, are compared; no significance test is run.

Before fitting, the features of each condition are checked to be finite (step B already
interpolates over masked frames, so there is nothing to impute) and, unless ``pca: false``,
z-scored and rotated onto principal components (see ``litpose_preprocess``). The z-score
statistics and the PCA basis are fit once on the train, val and test features pooled, so the
splits share one coordinate system; test features influence the basis, spikes never do.
``n_components`` is the number of keypoints of the condition unless ``pca_n_components_base`` /
``pca_n_components_base_plus_additional`` set it (at most the condition's keypoint count).
The pca settings enter the base cache key, so changing them refits the base.

The base fit is written once to ``<eid>/base/<hash8>/encoding_result.npz`` (with
``base_key.json`` and ``ray_tune/``), ``<hash8>`` being a hash of the base inputs (keypoints,
camera, seed, pca settings, data), and reused by every variant whose base inputs match, so trying another ``additional_keypoints`` set
only needs a new ``additional_keypoints`` list; ``--rerun_base`` forces a refit. Each variant
writes to ``<eid>/<variant_name>/``: ``encoding_result.npz`` (metrics and deltas, loadable
without pickle), ``encoding_params.json`` and ``encoding.log``, both recording the Lightning
Pose checkpoint location and the base / additional / unused keypoints.

``variant_name`` defaults (null or absent) to the sorted ``additional_keypoints`` joined by
``_``; a string in the config overrides it. Either way an earlier result is never overwritten by
a run with other settings: if ``<variant_name>/`` already holds an ``encoding_result.npz`` whose
``run_signature`` (camera, base / additional keypoints, pca settings, seed) differs from this
run's, or that predates ``run_signature``, the run goes to ``<variant_name>_<hash8>/`` instead.
Re-running identical settings overwrites its own folder. The chosen folder and the reason are
logged. A base fit is never replaced by a run with other inputs (e.g. ``pca: false``), it gets its own
``base/<hash8>/``; only ``--rerun_base`` refits one in place. Every variant result also keeps its
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
from pathlib import Path
from typing import Any

import numpy as np
import ray
import yaml
from sklearn.metrics import r2_score

from pose2visual.encoding.encoder import train_cnn_encoder_with_tune, train_rrr_encoder_with_tune
from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    base_dir,
    keypoint_names_json_path,
    session_output_dir,
    variant_dir,
)
from pose2visual.encoding.litpose_preprocess import check_finite, preprocess_splits
from pose2visual.encoding.run_encoding import (
    _flatten_views,
    _maybe_assert_latent_intervals_match_neural,
    _resolve_ray_session_temp_dir,
)
from pose2visual.encoding.utils import bits_per_spike, set_seed, setup_logging

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
)
_SPLITS = ('train', 'val', 'test')
# optional per-condition number of principal components (null = number of keypoints)
_PCA_N_COMPONENTS_KEYS = ('pca_n_components_base', 'pca_n_components_base_plus_additional')
_BASE_RESERVED_NAME = 'base'
# derived variant names stay well below the 255-byte file name limit
_MAX_VARIANT_NAME_LEN = 100
_NAME_HASH_LEN = 8
# same floor as the encoders use before taking log of the predicted rates
_BPS_RATE_THRESHOLD = 1e-3


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
        additional: `additional_keypoints` from the config.

    Returns:
        Folder name.
    """
    name = '_'.join(sorted(additional))
    if len(name) > _MAX_VARIANT_NAME_LEN:
        keep = _MAX_VARIANT_NAME_LEN - _NAME_HASH_LEN - 1
        name = f'{name[:keep]}_{_short_hash(name)}'
    return name


def run_signature(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Settings that make two runs with the same variant name different.

    Args:
        cfg: validated config from `load_config`.
        seed: random seed of the run.

    Returns:
        Json-serializable dict, stored in `encoding_params.json` and compared with `==`.
    """
    return {
        'camera': cfg['camera'],
        'base_keypoints': list(cfg['base_keypoints']),
        'additional_keypoints': sorted(cfg['additional_keypoints']),
        'pca': cfg['pca'],
        'pca_n_components_base': cfg['pca_n_components_base'],
        'pca_n_components_base_plus_additional': cfg['pca_n_components_base_plus_additional'],
        'seed': int(seed),
    }


def resolve_variant_dir(
    video_preds_dir: str | Path,
    output_name: str,
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
        output_name: user-chosen folder name.
        eid: session id.
        variant_name: wanted folder name (derived or set in the config).
        signature: `run_signature` of this run.

    Returns:
        Tuple `(directory, reason)` with a short text for the log.
    """
    plain = variant_dir(video_preds_dir, output_name, eid, variant_name)
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
        variant_dir(video_preds_dir, output_name, eid, suffixed),
        f'{variant_name!r} already holds a result with other settings, kept untouched',
    )


def load_config(path: str | Path, eid: str | None = None) -> dict[str, Any]:
    """Load and validate the encoding yaml config.

    Args:
        path: path of the yaml file.
        eid: if given, overrides the `eid` of the file.

    Returns:
        Config dict with `base_keypoints` and `additional_keypoints` as lists of str,
        `variant_name` as str (the sorted additional keypoints joined by `_` when the file has
        none or null), `pca` as bool (default `True`) and `pca_n_components_base` /
        `pca_n_components_base_plus_additional` as int or `None` (default).

    Raises:
        ValueError: if the yaml is not a mapping, a key is missing, the camera is unknown, a
            given `variant_name` is not a plain folder name (or is `base`), a keypoint list is
            empty,
            `pca` is not a bool, a `pca_n_components_*` is not a positive int or null, or the
            legacy `pca_n_components` key is present.
    """
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'{path}: expected a yaml mapping')
    if eid is not None:
        cfg['eid'] = eid
    missing = [key for key in _REQUIRED_CONFIG_KEYS if cfg.get(key) is None]
    if missing:
        raise ValueError(f'{path}: missing config keys {missing}')
    if cfg['camera'] not in CAMERAS:
        raise ValueError(f'{path}: camera must be one of {CAMERAS}, got {cfg["camera"]!r}')
    for key in ('base_keypoints', 'additional_keypoints'):
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f'{path}: {key} must be a non-empty list')
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
                f'{path}: variant_name must be null or a plain folder name other than '
                f'{_BASE_RESERVED_NAME!r}, got {variant_name!r}',
            )
        cfg['variant_name'] = variant_name
    cfg['pca'] = cfg.get('pca', True)
    if not isinstance(cfg['pca'], bool):
        raise ValueError(f'{path}: pca must be true or false, got {cfg["pca"]!r}')
    if 'pca_n_components' in cfg:
        raise ValueError(
            f'{path}: pca_n_components was replaced by one key per condition: '
            f'{", ".join(_PCA_N_COMPONENTS_KEYS)}',
        )
    for key in _PCA_N_COMPONENTS_KEYS:
        n_components = cfg.get(key)
        if n_components is not None and (
            isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1
        ):
            raise ValueError(f'{path}: {key} must be a positive int or null')
        cfg[key] = n_components
    return cfg


def split_keypoints(
    base: list[str],
    additional: list[str],
    keypoint_info: dict[str, Any],
    available: list[str],
) -> list[str]:
    """Validate the base / additional selection and return the unused keypoints.

    Args:
        base: `base_keypoints` from the config.
        additional: `additional_keypoints` from the config.
        keypoint_info: step A json of this camera and session.
        available: keypoint names stored in `dynamic_vars.npz`.

    Returns:
        Keypoints of the csv header that are in neither `base` nor `additional`, including
        those skipped as mostly masked.

    Raises:
        ValueError: if a keypoint appears in both lists, twice in a list, was skipped as masked,
            or does not exist.
    """
    for key, names in (('base_keypoints', base), ('additional_keypoints', additional)):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f'{key} lists keypoints more than once: {duplicated}')
    overlap = sorted(set(base) & set(additional))
    if overlap:
        raise ValueError(f'keypoints in both base_keypoints and additional_keypoints: {overlap}')
    skipped_masked = keypoint_info['skipped_masked']
    for name in base + additional:
        if name in skipped_masked:
            raise ValueError(
                f'keypoint {name!r} was skipped as mostly masked (masked fraction '
                f'{skipped_masked[name]:.3f} > {keypoint_info["max_masked_frac"]}, likelihood '
                f'threshold {keypoint_info["likelihood_threshold"]}) in '
                f'{keypoint_info["csv_path"]}',

            )
        if name not in available:
            raise ValueError(f'unknown keypoint {name!r}; available: {available}')
    selected = set(base) | set(additional)
    return [name for name in keypoint_info['keypoints'] if name not in selected]


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


def preprocess_features(
    z_trials_time: dict[str, np.ndarray],
    cond: str,
    use_pca: bool,
    n_components: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    """Check the features are finite and, if enabled, z-score and PCA them.

    Args:
        z_trials_time: split name -> `[K, T, 1, D]` features of one condition.
        cond: condition name used in log and error messages.
        use_pca: whether to apply `preprocess_splits`.
        n_components: number of components of this condition; `None` keeps all `D`.

    Returns:
        Tuple `(z, pca_info)`; `z` is the input when `use_pca` is false, `pca_info` is `None`
        then, otherwise the `preprocess_splits` info dict.

    Raises:
        ValueError: if a split holds a non-finite value or `n_components` exceeds the number of
            keypoints of the condition.
    """
    check_finite(z_trials_time, cond)
    if not use_pca:
        return z_trials_time, None
    n_keypoints = z_trials_time['train'].shape[-1]
    if n_components is None:
        n_components = n_keypoints
    elif n_components > n_keypoints:
        raise ValueError(
            f'pca_n_components for {cond} is {n_components} but the condition has only '
            f'{n_keypoints} keypoints',
        )
    z_pca, info = preprocess_splits(z_trials_time, n_components)
    logger.info(
        f'{cond}: pooled z-score + PCA, {n_keypoints} keypoints -> {n_components} components, '
        f'explained variance ratio {np.round(info["explained_variance_ratio"], 4).tolist()}, '
        f'explained variance {np.sum(info["explained_variance_ratio"]):.4f}',
    )
    return z_pca, info


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
        Multi-line, space-aligned table with one row per condition and model, followed by the
        `base_plus_additional - base` deltas.
    """
    rows = [('Condition', 'Model', 'Test BPS', 'Test R2')]
    for cond in _CONDITIONS:
        for model in _MODELS:
            rows.append((
                cond,
                model.upper(),
                f'{metrics[f"{cond}_{model}_bps"]:.4f}',
                f'{metrics[f"{cond}_{model}_r2"]:.4f}',
            ))
    for model in _MODELS:
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
) -> dict[str, np.ndarray | float]:
    """Fit RRR and CNN on one input set and collect their test metrics.

    Args:
        cond: condition name used as prefix of the returned keys.
        eid: session id.
        z_trials_time: split name -> `[K, T, 1, D]` features of the selected keypoints.
        neural_data_dict: loaded `<eid>_aligned.npz`.
        tune_storage_path: Ray Tune experiment root.
        seed: random seed forwarded to the CNN trials.

    Returns:
        Dict with `{cond}_{model}_{r2,bps}` scalars and `{cond}_{model}_{r2,bps}_per_neuron`
        arrays for `rrr` and `cnn`.
    """
    metrics: dict[str, np.ndarray | float] = {}
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
    return metrics


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
        pca_info: `preprocess_splits` info of the base condition, `None` when pca is off; the
            `pca` entry is only added when pca is on, so a cache from before pca existed is
            still reused by `pca: false`.

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
            'n_components': pca_info['n_components'],
            'scaling': pca_info['scaling'],
            'fit_split': pca_info['fit_split'],
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


def load_or_fit_base(
    base_directory: Path,
    key: dict[str, Any],
    fit: Any,
    rerun: bool,
) -> tuple[dict[str, np.ndarray | float], str]:
    """Reuse the cached base fit when its key matches, otherwise fit and cache it.

    Args:
        base_directory: `<eid>/base/<hash8>`; holds `encoding_result.npz` and `base_key.json`.
        key: output of `base_cache_key`.
        fit: zero-argument callable returning the base metrics (`fit_condition('base', ...)`).
        rerun: refit even if the cache matches.

    Returns:
        Tuple `(metrics, status)`; `status` is `'reused'`, `'fitted (no cache)'`,
        `'fitted (cache key mismatch)'` or `'fitted (--rerun_base)'`.
    """
    result_path = base_directory / 'encoding_result.npz'
    key_path = base_directory / 'base_key.json'
    status = 'fitted (no cache)'
    if result_path.is_file() and key_path.is_file():
        with key_path.open(encoding='utf-8') as f:
            cached_key = json.load(f)
        if rerun:
            status = 'fitted (--rerun_base)'
        elif cached_key == key:
            return _read_base_metrics(result_path), 'reused'
        else:
            status = 'fitted (cache key mismatch)'
    metrics = fit()
    base_directory.mkdir(parents=True, exist_ok=True)
    # write to a temp file and rename, so a half-written result is never read as a cache
    tmp_path = base_directory / 'encoding_result.tmp.npz'
    np.savez(tmp_path, base_keypoints=np.asarray(key['base_keypoints']), **metrics)
    tmp_path.replace(result_path)
    with key_path.open('w', encoding='utf-8') as f:
        json.dump(key, f, indent=2)
    return metrics, status


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
    out_dir, out_reason = resolve_variant_dir(
        cfg['video_preds_dir'],
        cfg['output_name'],
        cfg['eid'],
        cfg['variant_name'],
        run_signature(cfg, args.seed),
    )
    # the folder name is the variant name from here on (it may carry a hash suffix)
    cfg['variant_name'] = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

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
        _run(args, cfg, out_dir)
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(level_before)
        file_handler.close()


def _run(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    """Body of `main` once the log file handler is attached.

    Args:
        args: parsed command-line arguments.
        cfg: validated config from `load_config`.
        out_dir: `<video_preds_dir>/<output_name>/<eid>/<variant_name>`.

    Raises:
        FileNotFoundError: if `dynamic_vars.npz` or the step A json is missing.
        ValueError: if the config camera differs from `dynamic_vars.npz`, or the selection is
            invalid.
    """
    eid = cfg['eid']
    video_preds_dir = Path(cfg['video_preds_dir'])
    dynamic_vars_path = (
        session_output_dir(video_preds_dir, cfg['output_name'], eid) / 'dynamic_vars.npz'
    )
    if not dynamic_vars_path.is_file():
        raise FileNotFoundError(
            f'{dynamic_vars_path} not found; run '
            'pose2visual.encoding.extract_litpose_dynamic_vars_trials first',
        )
    json_path = keypoint_names_json_path(video_preds_dir, cfg['camera'], eid)
    with json_path.open(encoding='utf-8') as f:
        keypoint_info = json.load(f)

    latent_data_dict = np.load(dynamic_vars_path)
    camera = str(latent_data_dict['camera'])
    if camera != cfg['camera']:
        raise ValueError(
            f'config camera {cfg["camera"]!r} != camera {camera!r} of {dynamic_vars_path}',
        )
    available = [str(name) for name in latent_data_dict['keypoint_names']]
    base = cfg['base_keypoints']
    additional = cfg['additional_keypoints']
    unused = split_keypoints(base, additional, keypoint_info, available)

    params = {
        'eid': eid,
        'camera': camera,
        'variant_name': cfg['variant_name'],
        'run_signature': run_signature(cfg, args.seed),
        'seed': args.seed,
        'config': str(Path(args.config).resolve()),
        'model_dir': keypoint_info['model_dir'],
        'video_preds_dir': str(video_preds_dir),
        'csv_path': keypoint_info['csv_path'],
        'dynamic_vars_npz': str(dynamic_vars_path),
        'base_keypoints': base,
        'additional_keypoints': additional,
        'unused_keypoints': unused,
        'likelihood_threshold': keypoint_info['likelihood_threshold'],
        'skipped_masked_keypoints': keypoint_info['skipped_masked'],
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
    _maybe_assert_latent_intervals_match_neural(latent_data_dict, neural_data_dict)
    z_all = {split: latent_data_dict[f'{split}_z_trials_time'] for split in _SPLITS}
    z_base = {split: select_keypoints(z_all[split], available, base) for split in _SPLITS}
    z_full = {
        split: select_keypoints(z_all[split], available, base + additional)
        for split in _SPLITS
    }
    z_base, pca_base = preprocess_features(
        z_base, 'base', cfg['pca'], cfg['pca_n_components_base'],
    )
    z_full, pca_full = preprocess_features(
        z_full,
        'base_plus_additional',
        cfg['pca'],
        cfg['pca_n_components_base_plus_additional'],
    )
    params['pca'] = {'enabled': cfg['pca'], 'base': pca_base, 'base_plus_additional': pca_full}

    key = base_cache_key(
        base,
        camera,
        args.seed,
        keypoint_info['likelihood_threshold'],
        z_base,
        neural_data_dict,
        pca_base,
    )
    # one folder per distinct base input, so a pca / no-pca run never replaces the other's fit
    base_directory = base_dir(
        video_preds_dir, cfg['output_name'], eid, _short_hash(json.dumps(key, sort_keys=True)),
    )
    params['base_dir'] = str(base_directory)
    logger.info(f'condition base: {len(base)} keypoints')
    metrics, base_status = load_or_fit_base(
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

    logger.info(f'condition base_plus_additional: {len(base + additional)} keypoints')
    metrics.update(
        fit_condition(
            'base_plus_additional',
            eid,
            z_full,
            neural_data_dict,
            str(out_dir / 'ray_tune'),
            args.seed,
        ),
    )
    metrics.update(compare_conditions(metrics))
    for model in _MODELS:
        logger.info(
            f'{model.upper()}: delta R2 = {metrics[f"delta_{model}_r2"]:.4f} '
            f'(improved {metrics[f"frac_neurons_improved_{model}_r2"]:.2%} of neurons), '
            f'delta BPS = {metrics[f"delta_{model}_bps"]:.4f} '
            f'(improved {metrics[f"frac_neurons_improved_{model}_bps"]:.2%} of neurons)',
        )

    logger.info(f'Summary (held-out test set):\n{format_summary_table(metrics)}')

    result_path = out_dir / 'encoding_result.npz'
    np.savez(
        result_path,
        variant_name=np.asarray(cfg['variant_name']),
        base_keypoints=np.asarray(base),
        additional_keypoints=np.asarray(additional),
        unused_keypoints=np.asarray(unused, dtype=str),
        **metrics,
    )
    logger.info(f'wrote {result_path}')


if __name__ == '__main__':
    main()
