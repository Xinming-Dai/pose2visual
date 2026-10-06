"""Step A: list the keypoint names of every Lightning Pose prediction csv.

For each ``_iblrig_<camera>Camera.downsampled.<eid>.csv`` in ``--video-preds-dir`` this writes
``<video-preds-dir>/encoding/keypoint_names/<camera>_<eid>.json`` with the keypoint names (second
csv header row), the keypoints that are usable, and the keypoints skipped because too many of
their frames are masked (NaN ``x``/``y`` or ``likelihood`` below ``--likelihood-threshold``;
keypoints on the far side of the animal have likelihood near 0 and are skipped this way). Run once
per ``video_preds`` directory; the json files tell
which keypoints can go in the ``base_keypoints`` / ``additional_keypoints`` lists of the
encoding config, and are read by ``extract_litpose_dynamic_vars_trials``.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.list_litpose_keypoints \\
      --video-preds-dir <model_dir>/video_preds [--likelihood-threshold 0.9] \\
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
    load_keypoint_csv,
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
        '--video-preds-dir',
        type=str,
        required=True,
        help='directory holding _iblrig_<camera>Camera.downsampled.<eid>.csv files',
    )
    parser.add_argument(
        '--likelihood-threshold',
        type=float,
        default=0.9,
        help='frames below this likelihood count as masked; also used by '
        'extract_litpose_dynamic_vars_trials (default: 0.9)',
    )
    parser.add_argument(
        '--max-masked-frac',
        type=float,
        default=0.3,
        help='skip a keypoint when the fraction of masked frames (NaN x/y or low likelihood) '
        'exceeds this (default: 0.3)',
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
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    video_preds_dir = Path(args.video_preds_dir)

    csvs = list_keypoint_csvs(video_preds_dir)
    if not csvs:
        raise FileNotFoundError(
            f'no _iblrig_<camera>Camera.downsampled.<eid>.csv found in {video_preds_dir}',
        )
    for camera, eid, csv_path in csvs:
        info = build_keypoint_info(
            csv_path, camera, eid, args.likelihood_threshold, args.max_masked_frac,
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
