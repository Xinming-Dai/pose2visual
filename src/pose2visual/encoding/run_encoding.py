"""CLI entry point for neural encoding: RRR + CNN, orchestrated by Ray Tune.

Predicts spike rates from the ``dynamic_vars`` latents written by
``pose2visual.encoding.extract_dynamic_vars_trials`` and saves
``<latent_input_dir>/dynamic_vars_z/<eid>/encoding_results_dynamic_vars.npy``.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.run_encoding \\
      --eid <eid> --neural_input_dir <dir> --latent_input_dir <dir> \\
      --eval_task encoding --latent_kind dynamic_vars
"""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import ray

from pose2visual.encoding.encoder import train_cnn_encoder_with_tune, train_rrr_encoder_with_tune
from pose2visual.encoding.utils import get_encoding_args, set_seed, setup_logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RTOL_INTERVALS = 1e-9
_ATOL_INTERVALS = 1e-12

# (subdir under --latent_input_dir, trials npz filename)
_LATENT_KIND_LAYOUT = {
    'dynamic_vars': ('dynamic_vars_z', 'dynamic_vars_z_trials.npz'),
}


def _default_result_basename(latent_kind: str) -> str:
    """Build the default result basename from the latent kind.

    Args:
        latent_kind: parsed `--latent_kind`.

    Returns:
        Basename such as `encoding_results_dynamic_vars`.
    """
    return f'encoding_results_{latent_kind}'


def _latent_session_dir_and_trials_npz(
    latent_root: str,
    eid: str,
    latent_kind: str,
) -> tuple[Path, str]:
    """Return (session directory, trials npz filename) for loading latents.

    Args:
        latent_root: `--latent_input_dir`.
        eid: session id.
        latent_kind: parsed `--latent_kind`.

    Returns:
        Tuple `(session_dir, trials_npz_filename)`.
    """
    subdir, fname = _LATENT_KIND_LAYOUT[latent_kind]
    return Path(latent_root) / subdir / eid, fname


def _resolve_tune_storage_path(
    latent_root: str,
    latent_kind: str,
    explicit: str | None,
) -> str:
    """Ray Tune root: `explicit` if given, else the latent layout subdir.

    Args:
        latent_root: `--latent_input_dir`.
        latent_kind: parsed `--latent_kind`.
        explicit: value of `--tune_storage_path`, or `None`.

    Returns:
        Resolved Ray Tune storage path.
    """
    if explicit is not None:
        return explicit
    return str(Path(latent_root) / _LATENT_KIND_LAYOUT[latent_kind][0])


def _resolve_ray_session_temp_dir() -> str:
    """Ray defaults to /tmp/ray; on some HPC nodes /tmp is missing — use a writable path.

    Returns:
        A writable absolute directory path for Ray's `_temp_dir`.

    Raises:
        RuntimeError: if no writable candidate directory could be found.
    """
    bases: list[Path] = []
    ray_tmp = os.environ.get('RAY_TMPDIR')
    if ray_tmp:
        bases.append(Path(ray_tmp).expanduser().resolve())
    for key in ('SLURM_TMPDIR', 'TMPDIR'):
        v = os.environ.get(key)
        if v:
            bases.append(Path(v).expanduser().resolve())
    bases.append(Path(tempfile.gettempdir()).resolve())
    bases.append(Path.home() / '.cache')
    bases.append(_REPO_ROOT / '.ray_tmp')

    for base in bases:
        ray_dir = base / 'ray'
        try:
            ray_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            probe = ray_dir / '.ray_write_probe'
            probe.write_text('ok', encoding='ascii')
            probe.unlink()
            return str(ray_dir)
        except OSError:
            continue
    raise RuntimeError(
        'Could not create a writable Ray temp directory (see FileNotFoundError for /tmp/ray '
        'on some clusters). Set RAY_TMPDIR to an existing, writable absolute path.',
    )


def _log_interval_mismatch_detail(
    *,
    split_name: str,
    latent_row: int,
    neural_trial_idx: int,
    la: np.ndarray,
    ne: np.ndarray,
) -> None:
    """Log diagnostics when latent vs neural interval rows disagree.

    Args:
        split_name: `'train'`, `'val'`, or `'test'`.
        latent_row: row index within the latent intervals array.
        neural_trial_idx: corresponding row index within the neural intervals array.
        la: latent interval row.
        ne: neural interval row.
    """
    la = np.asarray(la)
    ne = np.asarray(ne)
    lines = [
        '[intervals check] mismatch detail (latent z_trials.npz vs neural *_aligned.npz)',
        f'  split={split_name!r}  latent_row={latent_row}  neural_trial_idx={neural_trial_idx}',
        f'  latent: shape={la.shape} dtype={la.dtype}',
        f'  latent: {np.array2string(la, precision=17, floatmode="unique")}',
        f'  neural: shape={ne.shape} dtype={ne.dtype}',
        f'  neural: {np.array2string(ne, precision=17, floatmode="unique")}',
    ]
    if la.shape != ne.shape:
        lines.append('  reason: shape mismatch (cannot subtract / allclose)')
        logger.warning('\n'.join(lines))
        return
    la64 = la.astype(np.float64, copy=False)
    ne64 = ne.astype(np.float64, copy=False)
    lines.append(
        f'  latent has_nan={bool(np.isnan(la64).any())}  '
        f'neural has_nan={bool(np.isnan(ne64).any())}',
    )
    diff = np.abs(la64 - ne64)
    diff_str = np.array2string(diff, precision=17, floatmode='unique')
    lines.append(f'  abs(latent - neural): {diff_str}')
    max_abs = 'nan (all elements)' if np.all(np.isnan(diff)) else float(np.nanmax(diff))
    lines.append(f'  max abs diff: {max_abs}')
    rel = diff / (np.maximum(np.abs(ne64), np.abs(la64)) + 1e-300)
    max_rel = 'nan (all elements)' if np.all(np.isnan(rel)) else float(np.nanmax(rel))
    lines.append(f'  max relative diff (rough): {max_rel}')
    ok = np.isclose(la64, ne64, rtol=_RTOL_INTERVALS, atol=_ATOL_INTERVALS, equal_nan=True)
    lines.append(f'  np.isclose (equal_nan=True) elementwise: {ok}')
    lines.append(
        f'  tolerance used: rtol={_RTOL_INTERVALS:g} atol={_ATOL_INTERVALS:g} '
        '(equal_nan=False in assert)',
    )
    logger.warning('\n'.join(lines))


def _maybe_assert_latent_intervals_match_neural(
    latent_data_dict: Any,
    neural_data_dict: Any,
) -> None:
    """If z_trials.npz carries intervals + neural_trial_idx, check vs aligned npz.

    Latent outputs may omit incomplete trials, so we index neural intervals by
    `neural_trial_idx` (same row order as `z_trials_time`: train block, then val, then test).

    Mismatches and missing metadata only emit warnings — the run continues.

    Args:
        latent_data_dict: loaded `z_trials.npz` (an `np.lib.npyio.NpzFile`).
        neural_data_dict: loaded `<eid>_aligned.npz` (an `np.lib.npyio.NpzFile`).
    """
    keys = ('train_intervals', 'val_intervals', 'test_intervals', 'neural_trial_idx')
    missing = [k for k in keys if k not in latent_data_dict.files]
    if missing:
        warnings.warn(
            f'z_trials.npz missing keys {missing}; skipping latent vs neural interval check.',
            stacklevel=2,
        )
        return
    nti = np.asarray(latent_data_dict['neural_trial_idx'])
    n_tr = len(latent_data_dict['train_intervals'])
    n_va = len(latent_data_dict['val_intervals'])
    n_te = len(latent_data_dict['test_intervals'])
    if n_tr + n_va + n_te != len(nti):
        warnings.warn(
            f'neural_trial_idx length {len(nti)} != train+val+test interval counts '
            f'({n_tr}+{n_va}+{n_te}); skipping latent vs neural interval check.',
            stacklevel=2,
        )
        return
    logger.info(
        '[intervals check] comparing latent intervals to neural aligned npz — '
        f'train/val/test rows=({n_tr},{n_va},{n_te}), neural_trial_idx len={len(nti)}',
    )
    splits = (
        ('train', n_tr, 0),
        ('val', n_va, n_tr),
        ('test', n_te, n_tr + n_va),
    )
    for name, n_loc, off in splits:
        ne_all = np.asarray(neural_data_dict[f'{name}_intervals'])
        for idx_row in range(n_loc):
            idx_neural = int(nti[off + idx_row])
            if idx_neural < 0 or idx_neural >= len(ne_all):
                warnings.warn(
                    f'neural_trial_idx={idx_neural} out of range for neural {name}_intervals '
                    f'(len={len(ne_all)}); skipping interval check.',
                    stacklevel=2,
                )
                return
            la = np.asarray(latent_data_dict[f'{name}_intervals'][idx_row])
            ne = np.asarray(ne_all[idx_neural])
            if la.shape != ne.shape:
                _log_interval_mismatch_detail(
                    split_name=name,
                    latent_row=idx_row,
                    neural_trial_idx=idx_neural,
                    la=la,
                    ne=ne,
                )
                warnings.warn(
                    f'Interval shape mismatch for {name} latent row {idx_row} '
                    f'(neural_trial_idx={idx_neural}); continuing without interval validation.',
                    stacklevel=2,
                )
                return
            allclose = np.allclose(
                la, ne, rtol=_RTOL_INTERVALS, atol=_ATOL_INTERVALS, equal_nan=False,
            )
            if not allclose:
                _log_interval_mismatch_detail(
                    split_name=name,
                    latent_row=idx_row,
                    neural_trial_idx=idx_neural,
                    la=la,
                    ne=ne,
                )
                warnings.warn(
                    f'Interval mismatch for {name} latent row {idx_row} '
                    f'(neural_trial_idx={idx_neural}); continuing without interval validation.',
                    stacklevel=2,
                )
                return
    logger.info('OK: latent intervals match neural aligned npz via neural_trial_idx.')


def _flatten_views(z: np.ndarray) -> np.ndarray:
    """Flatten embeddings across camera view directions.

    Args:
        z: array of shape `[K, T, V, D]`.

    Returns:
        float32 array of shape `[K, T, V * D]`.
    """
    k0, t0, v0, d0 = z.shape
    return np.asarray(z.reshape(k0, t0, v0 * d0), dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    """Run the neural encoding pipeline end to end (RRR + CNN via Ray Tune).

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        ValueError: if the latent trials npz has the legacy single-stack layout or all splits
            are empty.
        KeyError: if the latent trials npz lacks a `{train,val,test}_z_trials_time` array.
    """
    setup_logging()
    args = get_encoding_args(argv)

    ray_temp = _resolve_ray_session_temp_dir()
    logger.info(f'Ray session temp dir (_temp_dir): {ray_temp}')
    ray.init(num_cpus=2, num_gpus=1, _temp_dir=ray_temp)

    eid = args.eid
    seed = args.seed
    neural_input_dir = Path(args.neural_input_dir) / eid
    latent_session_dir, latent_trials_npz = _latent_session_dir_and_trials_npz(
        args.latent_input_dir, eid, args.latent_kind,
    )
    tune_storage_path = _resolve_tune_storage_path(
        args.latent_input_dir, args.latent_kind, args.tune_storage_path,
    )
    logger.info(f'Ray Tune storage_path: {tune_storage_path}')
    logger.info('Neural Encoding:')

    result_basename = _default_result_basename(args.latent_kind)
    if args.result_name is not None:
        name = args.result_name.strip()
        name = name[:-4] if name.lower().endswith('.npy') else name
        result_basename = name or result_basename
    save_path = latent_session_dir / result_basename
    logger.info(f'Saving results to: {save_path}.npy')
    trials_npz_path = latent_session_dir / latent_trials_npz
    logger.info(f'Loading latent trials from: {trials_npz_path}')

    set_seed(seed)

    logger.info(f'Processing {eid}')
    neural_data_dict = np.load(neural_input_dir / f'{eid}_aligned.npz', allow_pickle=True)
    latent_data_dict = np.load(trials_npz_path, allow_pickle=True)

    if 'z_trials_time' in latent_data_dict.files:
        raise ValueError(
            f'{trials_npz_path}: single-stack z_trials_time is no longer supported; '
            'use train_z_trials_time, val_z_trials_time, and test_z_trials_time.',
        )
    required_split_keys = tuple(f'{s}_z_trials_time' for s in ('train', 'val', 'test'))
    if not all(k in latent_data_dict.files for k in required_split_keys):
        raise KeyError(
            f'{trials_npz_path}: require {", ".join(required_split_keys)}; '
            f'got {sorted(latent_data_dict.files)}',
        )

    embeddings_raw = [
        np.asarray(latent_data_dict[key], dtype=np.float32) for key in required_split_keys
    ]
    if all(x.size == 0 for x in embeddings_raw):
        raise ValueError(f'{trials_npz_path}: all split tensors are empty.')
    train_embedding, val_embedding, test_embedding = (_flatten_views(x) for x in embeddings_raw)

    _maybe_assert_latent_intervals_match_neural(latent_data_dict, neural_data_dict)

    train_data = {
        eid: {
            'X': [train_embedding, val_embedding, test_embedding],
            'y': [
                neural_data_dict['train_spikes'],
                neural_data_dict['val_spikes'],
                neural_data_dict['test_spikes'],
            ],
            'setup': {},
        },
    }
    for split_name, x, y in zip(
        ('Train', 'Val', 'Test'), train_data[eid]['X'], train_data[eid]['y'],
    ):
        logger.info(f'{split_name} X Shape: {x.shape}, {split_name} Y Shape: {y.shape}')

    rrr_result = train_rrr_encoder_with_tune(
        train_data, num_samples=30, tune_storage_path=tune_storage_path,
    )
    cnn_result = train_cnn_encoder_with_tune(
        train_data, num_samples=30, tune_storage_path=tune_storage_path, seed=seed,
    )
    logger.info(
        f"RRR Encoding {eid} Test BPS: {rrr_result[eid]['bps']} Test R2: {rrr_result[eid]['r2']}",
    )
    logger.info(
        f"CNN Encoding {eid} Test BPS: {cnn_result[eid]['bps']} Test R2: {cnn_result[eid]['r2']}",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, {eid: {'rrr': rrr_result[eid], 'cnn': cnn_result[eid]}})


if __name__ == '__main__':
    main()
