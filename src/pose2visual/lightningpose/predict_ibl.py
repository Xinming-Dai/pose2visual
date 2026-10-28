"""Predict on IBL-2view sessions with Lightning Pose.

Resolves the camera video paths of each session from the IBL-2view directory layout and calls
``litpose predict`` once per session with all camera videos. Camera names and the video naming
templates are read from a small yaml config. Run in the ``lp`` conda environment.

Usage::

    conda activate lp
    PYTHONPATH=src python -m pose2visual.lightningpose.predict_ibl \\
      --root /work/hdd/bfsr/xdai3/IBL-2view \\
      --model-dir /path/to/lightning_pose_model \\
      --config configs/lightningpose/extraction_pipeline_pose2visual.yaml \\
      [--sessionids <eid1> <eid2>] [--cameras right] \\
      [--skip-existing] [--dry-run] [-- --skip_viz]

``--cameras`` restricts prediction to a subset of the cameras of the config (e.g. to finish only
the camera that did not complete in an earlier job). Arguments after ``--`` are forwarded to
``litpose predict`` unchanged.
"""

import argparse
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SESSION_ID_SENTINEL = '__SESSION_ID__'


@dataclass(frozen=True)
class VideoNaming:
    """Directory layout and file naming of the camera videos under the IBL-2view root.

    Attributes:
        camera_video_subdir: template of the per-camera video subdirectory, formatted with ``cam``.
        video_filename: template of the video file name, formatted with ``cam`` and
            ``session_id``.
    """

    camera_video_subdir: str = '{cam}Camera.video'
    video_filename: str = '_iblrig_{cam}Camera.downsampled.{session_id}.mp4'


@dataclass(frozen=True)
class PredictConfig:
    """Subset of the extraction config needed to locate videos.

    Attributes:
        cameras: camera names, in the order their videos are passed to litpose.
        sessionids: session ids to restrict to, or None for every session found.
        video_naming: video layout and naming templates.
    """

    cameras: list[str]
    sessionids: list[str] | None
    video_naming: VideoNaming


def load_config(path: str | Path) -> PredictConfig:
    """Load the keys needed for prediction from a yaml config.

    Unknown keys are ignored, so the full beast extraction config can be used as well.

    Args:
        path: path to the yaml config.

    Returns:
        parsed config.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if ``cameras`` is missing or empty, or the yaml is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'config file not found: {path}')
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f'config must be a yaml mapping: {path}')

    cameras = raw.get('cameras')
    if not cameras:
        raise ValueError(f'config must define a non-empty `cameras` list: {path}')

    naming_raw = raw.get('video_naming') or {}
    naming_fields = {
        key: naming_raw[key] for key in VideoNaming.__dataclass_fields__ if key in naming_raw
    }
    sessionids = raw.get('sessionids')
    return PredictConfig(
        cameras=[str(cam) for cam in cameras],
        sessionids=[str(sid) for sid in sessionids] if sessionids else None,
        video_naming=VideoNaming(**naming_fields),
    )


def _video_session_re(cam: str, video_naming: VideoNaming) -> re.Pattern:
    """Build a regex that extracts the session id from a video file name for one camera.

    The regex is derived from the ``video_filename`` template so it stays in sync when the
    template changes.

    Args:
        cam: camera name.
        video_naming: video layout and naming templates.

    Returns:
        compiled regex with one capture group for the session id.
    """
    filled = video_naming.video_filename.format(cam=cam, session_id=_SESSION_ID_SENTINEL)
    pattern = re.escape(filled).replace(re.escape(_SESSION_ID_SENTINEL), r'(.+)')
    return re.compile(f'^{pattern}$')


def _discover_sessions(
    root: Path,
    cameras: list[str],
    video_naming: VideoNaming,
    sessionids: list[str] | None = None,
) -> list[str]:
    """Find session ids that have a video for every camera.

    Args:
        root: IBL-2view root directory.
        cameras: camera names to require.
        video_naming: video layout and naming templates.
        sessionids: optional session ids to restrict to.

    Returns:
        sorted list of session ids.

    Raises:
        FileNotFoundError: if a camera video directory is missing.
    """
    ids_per_cam: list[set[str]] = []
    for cam in cameras:
        cam_dir = root / video_naming.camera_video_subdir.format(cam=cam)
        if not cam_dir.is_dir():
            raise FileNotFoundError(f'camera video directory not found: {cam_dir}')
        session_re = _video_session_re(cam, video_naming)
        ids_per_cam.append({
            match.group(1) for f in cam_dir.iterdir() if (match := session_re.match(f.name))
        })

    ids = sorted(set.intersection(*ids_per_cam))
    if sessionids:
        missing = sorted(set(sessionids) - set(ids))
        if missing:
            logger.warning(f'session ids without videos for every camera: {missing}')
        ids = [sid for sid in ids if sid in set(sessionids)]
    return ids


def _video_paths(
    root: Path,
    cameras: list[str],
    session_id: str,
    video_naming: VideoNaming,
) -> dict[str, Path]:
    """Return the video path of every camera for one session.

    Args:
        root: IBL-2view root directory.
        cameras: camera names.
        session_id: session identifier.
        video_naming: video layout and naming templates.

    Returns:
        dict mapping camera name to mp4 path.
    """
    return {
        cam: root
        / video_naming.camera_video_subdir.format(cam=cam)
        / video_naming.video_filename.format(cam=cam, session_id=session_id)
        for cam in cameras
    }


def _should_skip(model_dir: Path, cam_videos: dict[str, Path]) -> bool:
    """Return True if a prediction csv already exists for every camera video.

    Args:
        model_dir: Lightning Pose model directory.
        cam_videos: dict mapping camera name to mp4 path.

    Returns:
        True if all csvs are present.
    """
    video_preds_dir = model_dir / 'video_preds'
    return all(
        (video_preds_dir / f'{video.stem}.csv').is_file() for video in cam_videos.values()
    )


def _select_cameras(cameras: list[str], requested: list[str] | None) -> list[str]:
    """Restrict the configured cameras to the requested ones.

    Args:
        cameras: camera names of the config, in video order.
        requested: camera names to keep, or None to keep all.

    Returns:
        the requested cameras in config order, or all cameras if ``requested`` is None.

    Raises:
        ValueError: if a requested camera is not in ``cameras``.
    """
    if requested is None:
        return cameras
    unknown = sorted(set(requested) - set(cameras))
    if unknown:
        raise ValueError(f'cameras {unknown} not in the config cameras {cameras}')
    return [cam for cam in cameras if cam in requested]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Run litpose predict per IBL-2view session. '
            'Resolves mp4 paths from the IBL-2view directory layout. '
            'Run in the lp conda environment.'
        ),
    )
    parser.add_argument('--root', type=Path, required=True, help='IBL-2view root directory')
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Lightning Pose model directory (the one containing config.yaml)',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='yaml config; cameras, sessionids and video_naming are read from it',
    )
    parser.add_argument(
        '--sessionids',
        nargs='+',
        default=None,
        metavar='SESSION_ID',
        help='session ids to predict; overrides config sessionids if provided',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=None,
        metavar='CAMERA',
        help='cameras to predict, a subset of the config cameras (default: all of them)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print litpose commands without running',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip a session if all prediction csvs already exist',
    )
    parser.add_argument(
        '--litpose-bin',
        default='litpose',
        help='litpose executable name or path (default: litpose)',
    )
    parser.add_argument(
        'litpose_argv',
        nargs=argparse.REMAINDER,
        help='extra args forwarded to litpose predict (use: -- --skip_viz ...)',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point: run ``litpose predict`` for each session.

    Args:
        argv: command line arguments; defaults to ``sys.argv[1:]``.

    Raises:
        FileNotFoundError: if litpose, a camera directory, or a session video is missing.
        RuntimeError: if no session with videos for every camera is found.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    config = load_config(args.config)
    sessionids = args.sessionids if args.sessionids is not None else config.sessionids
    cameras = _select_cameras(config.cameras, args.cameras)

    if not args.dry_run and shutil.which(args.litpose_bin) is None:
        raise FileNotFoundError(
            f'litpose not found ({args.litpose_bin!r}). Activate the lp conda environment '
            'or pass --litpose-bin /path/to/litpose.'
        )

    session_ids = _discover_sessions(root, cameras, config.video_naming, sessionids)
    if not session_ids:
        raise RuntimeError(f'no sessions with videos for cameras {cameras} under {root}')

    extra = list(args.litpose_argv)
    if extra and extra[0] == '--':
        extra = extra[1:]

    for session_id in session_ids:
        cam_videos = _video_paths(root, cameras, session_id, config.video_naming)
        for cam, video in cam_videos.items():
            if not video.is_file():
                raise FileNotFoundError(
                    f'video missing for session_id={session_id} camera={cam}: {video}'
                )

        if args.skip_existing and _should_skip(model_dir, cam_videos):
            logger.info(f'skip (predictions exist): {session_id}')
            continue

        cmd = [
            args.litpose_bin,
            'predict',
            str(model_dir),
            *[str(video) for video in cam_videos.values()],
            *extra,
        ]
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue

        logger.info(f'litpose predict: session_id={session_id}')
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
