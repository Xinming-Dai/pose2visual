"""Build ``dynamic_vars_z_trials.npz`` from IBL ``DYNAMIC_VARS``.

``DYNAMIC_VARS`` (wheel speed, licks, whisker motion energy, nose speed, paw speed — all
keypoint-derived IBL behavior traces) are extracted by an external script (see
``https://github.com/yzhang511/beast/blob/behavior_traces_and_pca_latents/beast/extract_neural_data.py``)
and saved as ``train_<var>`` / ``val_<var>`` / ``test_<var>`` arrays directly inside each
``<eid>_aligned.npz`` file, already binned into the same trials as ``train/val/test_spikes``.

No csv loading or frame/trial re-alignment is needed: this module only concatenates the
``DYNAMIC_VARS`` arrays along the feature axis and reshapes them into the
``train/val/test_z_trials_time`` contract that ``pose2visual.encoding.run_encoding`` expects.

Writes ``<output-dir>/dynamic_vars_z/<eid>/dynamic_vars_z_trials.npz`` and ``params.json``.

Sessions where any ``DYNAMIC_VARS`` entry is missing or contains NaN are skipped (error logged,
nothing written).

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.extract_dynamic_vars_trials \\
      --eid <eid> --neural-input-dir <dir> --output-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from pose2visual.encoding.constants import DYNAMIC_VARS

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
        description='Build dynamic_vars_z_trials.npz from IBL DYNAMIC_VARS behavior traces '
        'already stored inside <eid>_aligned.npz.',
    )
    parser.add_argument('--eid', type=str, required=True, help='session id')
    parser.add_argument(
        '--neural-input-dir',
        type=str,
        required=True,
        help='root dir containing <eid>/<eid>_aligned.npz',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='root dir to write dynamic_vars_z/<eid>/dynamic_vars_z_trials.npz',
    )
    return parser.parse_args(argv)


def _npz_key(split_name: str, dynamic_var: str) -> str:
    """Npz key for one split/variable, e.g. `('train', 'wheel-speed')` -> `'train_wheel_speed'`.

    Args:
        split_name: `'train'`, `'val'`, or `'test'`.
        dynamic_var: one entry of `DYNAMIC_VARS` (hyphenated).

    Returns:
        The corresponding key in `<eid>_aligned.npz`.
    """
    return f"{split_name}_{dynamic_var.replace('-', '_')}"


def build_split_z_trials_time(
    neural_data_dict: np.lib.npyio.NpzFile,
    split_name: str,
) -> np.ndarray:
    """Concatenate one split's `DYNAMIC_VARS` into a `[K, T, 1, D]` array.

    Args:
        neural_data_dict: loaded `<eid>_aligned.npz`.
        split_name: `'train'`, `'val'`, or `'test'`.

    Returns:
        Array of shape `[K, T, 1, D]`, `D = sum` of each `DYNAMIC_VARS` entry's feature dim.

    Raises:
        KeyError: if a `DYNAMIC_VARS` key is missing from `neural_data_dict`.
    """
    missing = [
        var for var in DYNAMIC_VARS if _npz_key(split_name, var) not in neural_data_dict
    ]
    if missing:
        raise KeyError(f'{split_name}: missing DYNAMIC_VARS keys {missing} in aligned npz')
    parts = [
        np.asarray(neural_data_dict[_npz_key(split_name, var)], dtype=np.float32)
        for var in DYNAMIC_VARS
    ]
    combined = np.concatenate(parts, axis=-1)  # [K, T, D]
    return combined[:, :, None, :]  # [K, T, 1, D]


def find_unusable_vars(neural_data_dict: np.lib.npyio.NpzFile) -> list[str]:
    """Describe every `DYNAMIC_VARS` entry that is missing or contains NaN, in any split.

    A single NaN in the latents poisons the z-scoring and the regression in
    `pose2visual.encoding.run_encoding`, so any NaN makes the session unusable.

    Args:
        neural_data_dict: loaded `<eid>_aligned.npz`.

    Returns:
        One `'<split>/<var>: <reason>'` string per problem; empty if the session is usable.
    """
    problems = []
    for split_name in _SPLITS:
        for var in DYNAMIC_VARS:
            key = _npz_key(split_name, var)
            if key not in neural_data_dict:
                problems.append(f'{split_name}/{var}: missing')
                continue
            arr = np.asarray(neural_data_dict[key], dtype=np.float32)
            n_nan = int(np.isnan(arr).sum())
            if n_nan:
                problems.append(f'{split_name}/{var}: {100 * n_nan / arr.size:.0f}% NaN')
    return problems


def main(argv: list[str] | None = None) -> None:
    """Build `dynamic_vars_z_trials.npz` for one IBL session's `DYNAMIC_VARS` traces.

    A session with a missing or NaN-containing variable is skipped: an error is logged and
    nothing is written, so no NaN latents reach the encoding step.

    Args:
        argv: argument list; `None` uses `sys.argv`.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)

    neural_aligned_npz = Path(args.neural_input_dir) / args.eid / f'{args.eid}_aligned.npz'
    logger.info(f'loading {neural_aligned_npz}')
    neural_data_dict = np.load(neural_aligned_npz, allow_pickle=True)

    problems = find_unusable_vars(neural_data_dict)
    if problems:
        logger.error(
            f'skip EID {args.eid}: {len(problems)} unusable DYNAMIC_VARS: {"; ".join(problems)}',
        )
        return

    output_dir = Path(args.output_dir) / 'dynamic_vars_z' / args.eid
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_kwargs: dict[str, np.ndarray] = {}
    split_sizes: dict[str, int] = {}
    neural_trial_idx_parts: list[np.ndarray] = []
    for split_name in _SPLITS:
        z_trials_time = build_split_z_trials_time(neural_data_dict, split_name)
        npz_kwargs[f'{split_name}_z_trials_time'] = z_trials_time
        npz_kwargs[f'{split_name}_intervals'] = np.asarray(
            neural_data_dict[f'{split_name}_intervals'], dtype=np.float64,
        )
        n_split = z_trials_time.shape[0]
        split_sizes[split_name] = n_split
        # local index into this split's own *_intervals, as required by
        # run_encoding._maybe_assert_latent_intervals_match_neural
        neural_trial_idx_parts.append(np.arange(n_split, dtype=np.int64))
        logger.info(f'{split_name}: z_trials_time shape={z_trials_time.shape}')

    npz_kwargs['neural_trial_idx'] = np.concatenate(neural_trial_idx_parts)

    trials_npz_path = output_dir / 'dynamic_vars_z_trials.npz'
    np.savez(trials_npz_path, **npz_kwargs)
    logger.info(f'wrote {trials_npz_path}')

    params = {
        'eid': args.eid,
        'neural_aligned_npz': str(neural_aligned_npz),
        'dynamic_vars': list(DYNAMIC_VARS),
        'split_sizes': split_sizes,
    }
    params_path = output_dir / 'params.json'
    with params_path.open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    logger.info(f'wrote {params_path}')


if __name__ == '__main__':
    main()
