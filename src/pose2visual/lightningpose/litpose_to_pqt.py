"""Convert the Brainwide Lightning Pose csvs (EKS, merged, PCA) to ``.pqt`` tables.

Reads the csvs that ``predict_brainwide`` and the encoding step 0 leave under ``<out-root>`` and
writes one lossless ``.pqt`` per session, camera and stage to::

    <out-root>/results/experiment_brainwide_litpose/<run>/pqt/_ibl_<camera>Camera.<stage>.<eid>.pqt

with ``<stage>`` one of

========== ==================================================================================
eks.digits ``<out-root>/models/digits/eks_run/eks/_iblrig_<cam>Camera.downsampled.<eid>.csv``
eks.rest   ``<out-root>/models/rest/eks_run/eks/...``
merged     ``<video_preds_dir>/...``, digits + rest merged by the ``merge`` section of the yaml
pca        ``<video_preds_dir>/encoding/pca/<pca.group_set>/...``
========== ==================================================================================

``video_preds_dir`` and ``pca.group_set`` come from the brainwide encoding yaml (``--config``).

Each csv has ``scorer / bodyparts / coords`` header rows and one row per video frame. The table
keeps one row per frame (same order and index) and one column per ``(bodypart, coord)``, named
``<bodypart>_<coord>`` as in IBL ``lightningPose`` tables (``nose_tip_x``, ``nose_tip_y``,
``nose_tip_likelihood``); the extra EKS coords keep their name (``nose_tip_x_ens_median``,
``nose_tip_y_posterior_var``, ...). The row count is checked against the frame timestamps in
``<out-root>/inputs/timestamps``. Native keypoint names are kept; nothing is renamed or dropped.

Usage::

    conda activate beast
    PYTHONPATH=src python -m pose2visual.lightningpose.litpose_to_pqt \\
      --out-root <out-root> --run <run> \\
      --config configs/encoding/litpose_encoding_brainwide.yaml \\
      --sessionids <eid1> <eid2> [--stages eks.digits eks.rest merged pca] [--overwrite] \\
      [--n-jobs 4]

``--n-jobs`` converts that many csvs at a time in parallel processes (each holds one csv in
memory); the default 1 runs everything inline.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    csv_filename,
    load_frame_times,
    load_keypoint_csv,
    load_pca_settings,
    load_video_preds_dir,
    pca_dir_path,
    run_parallel,
)
from pose2visual.lightningpose.predict_brainwide import DEFAULT_OUT_ROOT

logger = logging.getLogger(__name__)

STAGES = ('eks.digits', 'eks.rest', 'merged', 'pca')
EXPERIMENT_NAME = 'experiment_brainwide_litpose'


def stage_csv_dir(stage: str, out_root: str | Path, config_path: str | Path) -> Path:
    """Return the directory holding the csvs of a stage.

    Args:
        stage: one of ``STAGES``.
        out_root: output root of ``predict_brainwide``.
        config_path: brainwide encoding yaml (gives ``paths.video_preds_dir``, ``pca.group_set``).

    Returns:
        directory with ``_iblrig_<cam>Camera.downsampled.<eid>.csv`` files.

    Raises:
        ValueError: if the stage is unknown, or the yaml has no ``paths.video_preds_dir``.
    """
    out_root = Path(out_root)
    if stage.startswith('eks.') and stage in STAGES:
        return out_root / 'models' / stage.split('.', 1)[1] / 'eks_run' / 'eks'
    if stage not in STAGES:
        raise ValueError(f'unknown stage {stage!r}; expected one of {STAGES}')
    video_preds_dir = load_video_preds_dir(config_path)
    if video_preds_dir is None:
        raise ValueError(f'{config_path} has no paths.video_preds_dir')
    if stage == 'merged':
        return Path(video_preds_dir)
    return pca_dir_path(video_preds_dir, load_pca_settings(config_path)['group_set'])


def pqt_path(out_root: str | Path, run: str, stage: str, camera: str, eid: str) -> Path:
    """Return the output path of one ``.pqt``.

    Args:
        out_root: output root of ``predict_brainwide``.
        run: run name, the folder under ``results/experiment_brainwide_litpose``.
        stage: one of ``STAGES``.
        camera: ``'left'`` or ``'right'``.
        eid: session id.

    Returns:
        ``<out_root>/results/experiment_brainwide_litpose/<run>/pqt/`` plus
        ``_ibl_<camera>Camera.<stage>.<eid>.pqt``; a period (never ``-``) separates the session
        id from the other parts.
    """
    return (
        Path(out_root) / 'results' / EXPERIMENT_NAME / run / 'pqt'
        / f'_ibl_{camera}Camera.{stage}.{eid}.pqt'
    )


def flatten_columns(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Flatten ``(scorer, bodypart, coord)`` columns to ``<bodypart>_<coord>``.

    Args:
        df: dataframe from ``load_keypoint_csv``.
        copy: return a copy and leave `df` untouched; ``False`` renames the columns of `df` in
            place, which saves a full copy of a large csv in memory.

    Returns:
        dataframe with flat string columns, same rows and index.

    Raises:
        ValueError: if two columns map to the same flat name.
    """
    names = [f'{bodypart}_{coord}' for _, bodypart, coord in df.columns]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f'duplicate columns after flattening: {duplicates[:5]}')
    flat = df.copy() if copy else df
    flat.columns = names
    return flat


def csv_to_pqt(
    csv_path: str | Path,
    out_path: str | Path,
    n_frames: int,
    float32: bool = False,
) -> int:
    """Convert one prediction csv to a ``.pqt``.

    Args:
        csv_path: Lightning Pose style csv.
        out_path: output ``.pqt`` path; written through a temporary file, so an interrupted run
            leaves no partial file.
        n_frames: expected number of frames (rows), the length of the frame timestamps.
        float32: store the values as float32 instead of float64.

    Returns:
        number of columns written.

    Raises:
        ValueError: if the csv does not have ``n_frames`` rows.
    """
    df = flatten_columns(load_keypoint_csv(csv_path), copy=False)
    if len(df) != n_frames:
        raise ValueError(f'{csv_path} has {len(df)} frames, timestamps have {n_frames}')
    if float32:
        df = df.astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + '.tmp')
    df.to_parquet(tmp_path, engine='pyarrow')
    tmp_path.replace(out_path)
    return df.shape[1]


def convert_sessions(
    out_root: str | Path,
    run: str,
    config_path: str | Path,
    eids: list[str],
    stages: list[str],
    overwrite: bool = False,
    float32: bool = False,
    n_jobs: int = 1,
) -> list[Path]:
    """Convert the csvs of every session, camera and stage.

    The csvs and frame timestamps are checked first, so a missing file fails before any
    conversion starts. A conversion that fails is logged and the others still run; the first
    failure is raised at the end.

    Args:
        out_root: output root of ``predict_brainwide``.
        run: run name of the output folder.
        config_path: brainwide encoding yaml.
        eids: session ids.
        stages: stages to convert, from ``STAGES``.
        overwrite: rebuild ``.pqt`` files that already exist.
        float32: store float32 values.
        n_jobs: number of csvs converted at the same time, in parallel processes.

    Returns:
        paths of the ``.pqt`` files written (existing ones kept without ``overwrite`` excluded).

    Raises:
        FileNotFoundError: if a csv or the frame timestamps are missing.
        ValueError: if a csv has the wrong number of frames, or `n_jobs` is below 1.
    """
    timestamps_dir = Path(out_root) / 'inputs' / 'timestamps'
    jobs = []
    for stage in stages:
        csv_dir = stage_csv_dir(stage, out_root, config_path)
        for eid in eids:
            for camera in CAMERAS:
                out_path = pqt_path(out_root, run, stage, camera, eid)
                if out_path.is_file() and not overwrite:
                    logger.info(f'exists, skipping: {out_path}')
                    continue
                csv_path = csv_dir / csv_filename(camera, eid)
                if not csv_path.is_file():
                    raise FileNotFoundError(f'{stage}: csv not found: {csv_path}')
                times, _ = load_frame_times(timestamps_dir, camera, eid)
                jobs.append({
                    'csv_path': csv_path,
                    'out_path': out_path,
                    'n_frames': len(times),
                    'float32': float32,
                })
    written = []
    first_error = None
    for job, (n_columns, error) in zip(jobs, run_parallel(csv_to_pqt, jobs, n_jobs), strict=True):
        if error is not None:  # keep going; the first failure is raised below
            logger.error(f'{job["csv_path"]} failed: {error!r}')
            first_error = first_error or error
            continue
        logger.info(f'{job["out_path"].name}: {job["n_frames"]} frames x {n_columns} columns')
        written.append(job['out_path'])
    if first_error is not None:
        raise first_error
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT, help='output root')
    parser.add_argument('--run', required=True, help='run name (folder under the experiment)')
    parser.add_argument('--config', type=Path, required=True, help='brainwide encoding yaml')
    parser.add_argument('--sessionids', nargs='+', required=True, help='session ids')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--overwrite', action='store_true', help='rebuild existing .pqt files')
    parser.add_argument('--float32', action='store_true', help='store float32 instead of float64')
    parser.add_argument(
        '--n-jobs', type=int, default=1, help='csvs converted at the same time (default: 1)',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    Args:
        argv: command line arguments; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    written = convert_sessions(
        args.out_root,
        args.run,
        args.config,
        args.sessionids,
        args.stages,
        args.overwrite,
        args.float32,
        args.n_jobs,
    )
    logger.info(f'wrote {len(written)} pqt files')


if __name__ == '__main__':
    main()
