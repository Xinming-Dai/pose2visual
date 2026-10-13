"""Export Pose-EKS smoothed (x, y) to a DLC-style csv, one per camera.

Combines every body-part group's fitted latent model under a session's eks output directory
(``fit_eks_latent_model.py``'s ``out_root``) into a single csv per camera, in the same
``(scorer, bodyparts, coords)`` 3-level-header format as the raw prediction csvs — a drop-in
for tools that expect that format (e.g. ``pose2visual.qc.overlay``).

Safe to combine every group into one csv because ``nonoverlapping_analysis_groups`` (what
``fit_eks_latent_model.py`` fits by default) is a true partition: no landmark appears in more
than one group. This module still raises if it ever finds a landmark duplicated across groups,
or frame counts that disagree between groups for the same camera, rather than silently merging
inconsistent data.

The EKS smoother doesn't produce a real confidence score, so the ``likelihood`` column is filled
with ``1.0`` for every row — purely to keep the same 3-column-per-keypoint structure as the raw
csvs.

Usage::

    PYTHONPATH=src python -m pose2visual.lightningpose.export_eks_csv \\
      --eks-dir /path/to/eks/<hash8>/<session>

Default output: ``<eks-dir>/smoothed_<camera>.csv``. Pass ``--out-dir`` to override.

Pass ``--standard-naming`` to instead write ``_iblrig_<camera>Camera.downsampled.<eid>.csv``
(``<eid>`` taken from ``<eks-dir>``'s own folder name, since ``fit_eks_latent_model.py`` writes
its output under ``<out_root>/<session>/<group>/...``) -- the same filename
``pose2visual.encoding.list_litpose_keypoints``/``extract_litpose_dynamic_vars_trials`` already
scan for, so the encoding pipeline's ``eks_mean`` step order (see
``pose2visual.encoding.litpose_step_pipelines``) just means pointing ``video_preds_dir`` at this
csv's folder, exactly like the plain single-seed case.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.lightningpose.fit_eks_latent_model import load_latent_model

logger = logging.getLogger(__name__)


def merge_group_npzs(npz_paths: list[Path]) -> tuple[np.ndarray, list[str]]:
    """Merge several groups' latent-model npz files into one (x, y) table.

    Args:
        npz_paths: paths to `<group>_<camera>_latent-model.npz` files for one camera, one per
            group.

    Returns:
        `(xy, landmarks)`: `xy` has shape `(n_frames, sum(n_landmarks per group), 2)`, and
        `landmarks` is the matching list of landmark names, in `npz_paths` order.

    Raises:
        ValueError: if an npz lacks the `columns` array needed to recover landmark names, if
            the same landmark name appears in more than one npz, or if npz files disagree on
            frame count.
    """
    all_xy = []
    all_landmarks: list[str] = []
    n_frames = None
    for npz_path in npz_paths:
        loaded = load_latent_model(npz_path)
        if loaded['landmarks'] is None:
            raise ValueError(
                f'{npz_path} has no `columns` array (predates that field) — refit with the '
                'current fit_eks_latent_model.py to export it.',
            )
        if n_frames is None:
            n_frames = loaded['xy'].shape[0]
        elif loaded['xy'].shape[0] != n_frames:
            raise ValueError(
                f'{npz_path} has {loaded["xy"].shape[0]} frames, expected {n_frames} '
                '(frame count disagrees with an earlier group for this camera)',
            )
        duplicates = set(loaded['landmarks']) & set(all_landmarks)
        if duplicates:
            raise ValueError(f'landmark(s) {duplicates} appear in more than one group\'s npz')
        all_xy.append(loaded['xy'])
        all_landmarks.extend(loaded['landmarks'])

    return np.concatenate(all_xy, axis=1), all_landmarks


def write_smoothed_csv(xy: np.ndarray, landmarks: list[str], out_path: Path) -> None:
    """Write smoothed (x, y) to a DLC-style 3-level-header csv.

    Args:
        xy: `(n_frames, n_landmarks, 2)` smoothed coordinates.
        landmarks: landmark name per `xy`'s second axis.
        out_path: output csv path.

    Likelihood is filled with `1.0` for every row — not a real confidence score, just to keep
    the same 3-column-per-keypoint structure as the raw prediction csvs.
    """
    n_frames = xy.shape[0]
    columns = pd.MultiIndex.from_product(
        [['eks'], landmarks, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = np.empty((n_frames, len(landmarks) * 3))
    data[:, 0::3] = xy[:, :, 0]
    data[:, 1::3] = xy[:, :, 1]
    data[:, 2::3] = 1.0
    df = pd.DataFrame(data, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Export Pose-EKS smoothed (x, y) to a DLC-style csv, one per camera, combining '
            'every body-part group under --eks-dir.'
        ),
    )
    parser.add_argument(
        '--eks-dir',
        required=True,
        help='Session-level eks output directory (fit_eks_latent_model.py\'s out-root), '
        'containing <group>/<group>_<camera>_latent-model.npz subdirectories',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=['left', 'right'],
        help='Camera names to export (default: left right)',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: same as --eks-dir)',
    )
    parser.add_argument(
        '--standard-naming',
        action='store_true',
        help='write _iblrig_<camera>Camera.downsampled.<eid>.csv (<eid> from --eks-dir\'s own '
        'folder name) instead of smoothed_<camera>.csv, so the csv can be pointed at directly '
        'as a video_preds_dir for the encoding pipeline\'s eks_mean step order',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: write one combined smoothed csv per camera.

    Args:
        argv: command line arguments; defaults to `sys.argv[1:]`.

    Raises:
        FileNotFoundError: if no group npz is found for a camera.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)

    eks_dir = Path(args.eks_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else eks_dir
    eid = eks_dir.name

    for camera in args.cameras:
        npz_paths = sorted(eks_dir.glob(f'*/*_{camera}_latent-model.npz'))
        if not npz_paths:
            raise FileNotFoundError(f'no group npz found for camera {camera!r} under {eks_dir}')
        group_names = [p.parent.name for p in npz_paths]
        logger.info(f'{camera}: merging {len(npz_paths)} groups: {group_names}')

        xy, landmarks = merge_group_npzs(npz_paths)
        out_name = (
            f'_iblrig_{camera}Camera.downsampled.{eid}.csv'
            if args.standard_naming
            else f'smoothed_{camera}.csv'
        )
        out_path = out_dir / out_name
        write_smoothed_csv(xy, landmarks, out_path)
        logger.info(f'wrote {out_path} ({xy.shape[0]} frames, {len(landmarks)} landmarks)')


if __name__ == '__main__':
    main()
