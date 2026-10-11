"""Step A: list the keypoint names of every Lightning Pose prediction csv.

For each ``_iblrig_<camera>Camera.downsampled.<eid>.csv`` in ``--video-preds-dir`` (default: the
``video_preds_dir`` of ``--config``) this writes
``<video-preds-dir>/encoding/keypoint_names/<camera>_<eid>.json`` with the keypoint names (second
csv header row), the keypoints that are usable, and the keypoints skipped because too many of
their frames are masked (NaN ``x``/``y`` or ``likelihood`` below the likelihood threshold;
keypoints on the far side of the animal have likelihood near 0 and are skipped this way). Run once
per ``video_preds`` directory; the json files tell
which keypoints can go in the ``base_keypoints`` / ``additional_keypoints`` lists of the
encoding config, and are read by ``extract_litpose_dynamic_vars_trials``.

``likelihood_threshold`` and ``max_masked_frac`` come from ``--config`` (the encoding yaml), the
single source shared by all steps; ``--likelihood-threshold`` / ``--max-masked-frac`` override them
for this run only. Steps B and C fail when their config disagrees with the json written here, so
rerun this step after changing either number in the yaml.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.list_litpose_keypoints \\
      --config configs/encoding/litpose_encoding.yaml \\
      [--video-preds-dir <model_dir>/video_preds] [--likelihood-threshold 0.9] \\
      [--max-masked-frac 0.3]
"""

import argparse
import json
import logging
from pathlib import Path

from pose2visual.encoding.litpose_keypoints import (
    frame_fractions,
    get_keypoint_names,
    keypoint_names_json_path,
    list_keypoint_csvs,
    load_filter_settings,
    load_keypoint_csv,
    load_video_preds_dir,
    validate_filter_settings,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='List the keypoint names of every Lightning Pose prediction csv.',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='encoding yaml holding likelihood_threshold and max_masked_frac',
    )
    parser.add_argument(
        '--video-preds-dir',
        type=str,
        default=None,
        help='directory holding _iblrig_<camera>Camera.downsampled.<eid>.csv files (default: '
        'video_preds_dir of --config)',
    )
    parser.add_argument(
        '--likelihood-threshold',
        type=float,
        default=None,
        help='overrides likelihood_threshold of --config for this run; frames below this '
        'likelihood count as masked',
    )
    parser.add_argument(
        '--max-masked-frac',
        type=float,
        default=None,
        help='overrides max_masked_frac of --config for this run; skip a keypoint when the '
        'fraction of masked frames (NaN x/y or low likelihood) exceeds this',
    )
    return parser.parse_args(argv)


def build_keypoint_info(
    csv_path: Path,
    camera: str,
    eid: str,
    likelihood_threshold: float,
    max_masked_frac: float,
) -> dict:
    """Summarize the keypoints of one prediction csv.

    Args:
        csv_path: path of the csv.
        camera: `'left'` or `'right'`.
        eid: session id.
        likelihood_threshold: frames below this likelihood count as masked.
        max_masked_frac: keypoints with a larger masked fraction are not usable.

    Returns:
        Json-serializable dict with `csv_path`, `video_preds_dir`, `model_dir`, `camera`, `eid`,
        `n_frames`, `likelihood_threshold`, `max_masked_frac`, `keypoints`, per-keypoint
        `masked_fraction`, `nan_fraction` and `low_likelihood_fraction`, `usable`,
        `unusable` (names, header order), `unusable_reasons` (name -> text) and
        `skipped_masked` (name -> masked fraction, same keys as `unusable`).
    """
    df = load_keypoint_csv(csv_path)
    keypoints = get_keypoint_names(df)
    fractions = {name: frame_fractions(df, name, likelihood_threshold) for name in keypoints}
    masked_fractions = {name: frac['masked'] for name, frac in fractions.items()}
    skipped_masked = {
        name: frac for name, frac in masked_fractions.items() if frac > max_masked_frac
    }
    unusable_reasons = {
        name: (
            f'masked fraction {masked_fractions[name]:.2f} > {max_masked_frac} '
            f'(NaN x/y/likelihood {fractions[name]["nan"]:.2f}, '
            f'likelihood < {likelihood_threshold}: {fractions[name]["low_likelihood"]:.2f})'
        )
        for name in skipped_masked
    }
    video_preds_dir = csv_path.resolve().parent
    return {
        'csv_path': str(csv_path.resolve()),
        'video_preds_dir': str(video_preds_dir),
        # the lightning pose checkpoint directory that produced these predictions
        'model_dir': str(video_preds_dir.parent),
        'camera': camera,
        'eid': eid,
        'n_frames': int(len(df)),
        'likelihood_threshold': likelihood_threshold,
        'max_masked_frac': max_masked_frac,
        'keypoints': keypoints,
        'masked_fraction': masked_fractions,
        'nan_fraction': {name: frac['nan'] for name, frac in fractions.items()},
        'low_likelihood_fraction': {
            name: frac['low_likelihood'] for name, frac in fractions.items()
        },
        'usable': [name for name in keypoints if name not in skipped_masked],
        'unusable': [name for name in keypoints if name in skipped_masked],
        'unusable_reasons': unusable_reasons,
        'skipped_masked': skipped_masked,
    }


def main(argv: list[str] | None = None) -> None:
    """Write one keypoint-names json per prediction csv in a `video_preds` directory.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if the directory contains no prediction csv.
        ValueError: if the config lacks a valid `likelihood_threshold` / `max_masked_frac`, or
            neither `--video-preds-dir` nor the config's `video_preds_dir` is given.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    video_preds_dir_text = args.video_preds_dir or load_video_preds_dir(args.config)
    if video_preds_dir_text is None:
        raise ValueError(
            f'no video_preds_dir: pass --video-preds-dir or set video_preds_dir in {args.config}',
        )
    video_preds_dir = Path(video_preds_dir_text)
    logger.info(f'video_preds_dir={video_preds_dir}')
    likelihood_threshold, max_masked_frac = load_filter_settings(args.config)
    if args.likelihood_threshold is not None or args.max_masked_frac is not None:
        # an override applies to this run only; the config stays the shared default
        if args.likelihood_threshold is not None:
            likelihood_threshold = args.likelihood_threshold
        if args.max_masked_frac is not None:
            max_masked_frac = args.max_masked_frac
        likelihood_threshold, max_masked_frac = validate_filter_settings(
            likelihood_threshold, max_masked_frac, 'command line',
        )
        logger.warning(
            f'overriding {args.config}: likelihood_threshold={likelihood_threshold}, '
            f'max_masked_frac={max_masked_frac}; steps B and C will reject this json unless '
            'their config holds the same values',
        )
    logger.info(
        f'likelihood_threshold={likelihood_threshold}, max_masked_frac={max_masked_frac}',
    )

    csvs = list_keypoint_csvs(video_preds_dir)
    if not csvs:
        raise FileNotFoundError(
            f'no _iblrig_<camera>Camera.downsampled.<eid>.csv found in {video_preds_dir}',
        )
    for camera, eid, csv_path in csvs:
        info = build_keypoint_info(
            csv_path, camera, eid, likelihood_threshold, max_masked_frac,
        )
        json_path = keypoint_names_json_path(video_preds_dir, camera, eid)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        logger.info(
            f'{camera} {eid}: {len(info["usable"])}/{len(info["keypoints"])} usable keypoints, '
            f'unusable: {info["unusable"]} -> {json_path}',
        )


if __name__ == '__main__':
    main()
