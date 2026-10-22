"""Resolve sessions, groups, csvs and videos and render one QC video per session."""

import argparse
import json
import logging
from pathlib import Path

from pose2visual.lightningpose.fit_eks_latent_model import (
    discover_keypoint_names,
    group_columns_for_camera,
    load_groups,
    resolve_group_bases,
)
from pose2visual.preprocessing import resolve_video_path
from pose2visual.qc.video.loading import (
    CameraData,
    load_camera_data,
    load_pca_positions,
    read_run_label,
    resolve_eks_csv,
)
from pose2visual.qc.video.render_compare import VideoJob, render_video

logger = logging.getLogger(__name__)


def build_output_path(
    out_dir: Path,
    session_id: str,
    group_name: str,
    cameras: list[str],
    start_frame: int,
    n_frames: int,
    tag: str | None = None,
) -> Path:
    """Name the video of one session and group; it lives in a folder named after the group.

    The session id is the only part containing `-`; it is separated from the other parts by `.`
    so that it can be split out of the file name.

    Args:
        out_dir: output directory.
        session_id: session UUID.
        group_name: body-part group shown.
        cameras: cameras shown.
        start_frame: first frame rendered.
        n_frames: number of frames rendered.
        tag: optional extra name part.

    Returns:
        `<out_dir>/<group>/qc_video.<session>.<group>.<cameras>.frames<start>_<end>[.<tag>].mp4`.
    """
    parts = [
        'qc_video',
        session_id,
        group_name,
        '_'.join(cameras),
        f'frames{start_frame:06d}_{start_frame + n_frames - 1:06d}',
    ]
    if tag:
        parts.append(tag)
    return out_dir / group_name / ('.'.join(parts) + '.mp4')


def load_crop_boxes(
    groups_file: str | Path,
    group_set: str,
    group_name: str,
) -> dict[str, tuple[int, int, int, int] | None]:
    """Read the zoom boxes of one group from the `crop_boxes` entry of a `body_parts.json` file.

    The entry is `crop_boxes[group_set][group][camera] = [x0, y0, x1, y1]` in source pixels, as
    printed by `scripts/delta/qc/select_crops.py`.

    Args:
        groups_file: path of the json file.
        group_set: top-level group set, e.g. `'nonoverlapping_analysis_groups'`.
        group_name: body-part group.

    Returns:
        camera name -> box; None for a camera without a box (all None when the file has no
        `crop_boxes` entry for the group).

    Raises:
        ValueError: if a box is not four numbers with `x1 > x0` and `y1 > y0`, or names an unknown
            camera.
    """
    raw = json.loads(Path(groups_file).read_text())
    entry = raw.get('crop_boxes', {}).get(group_set, {}).get(group_name, {})
    unknown = sorted(set(entry) - {'left', 'right'})
    if unknown:
        raise ValueError(f'crop_boxes of {group_name!r} has unknown cameras {unknown}')
    boxes: dict[str, tuple[int, int, int, int] | None] = {}
    for camera in ('left', 'right'):
        box = entry.get(camera)
        if box is not None:
            valid = (
                isinstance(box, list)
                and len(box) == 4
                and all(isinstance(value, (int, float)) for value in box)
                and box[2] > box[0]
                and box[3] > box[1]
            )
            if not valid:
                raise ValueError(
                    f'crop_boxes of {group_name!r}/{camera} must be [x0, y0, x1, y1] with '
                    f'x1 > x0 and y1 > y0, got {box!r}',
                )
            box = tuple(round(value) for value in box)
        boxes[camera] = box
    return boxes


def load_session_cameras(
    eks_dir: Path,
    session_id: str,
    cameras: list[str],
    groups: dict[str, list[str]],
    start_frame: int,
    n_frames: int,
    pca_dir: Path | None = None,
) -> tuple[dict[str, CameraData], dict[str, Path]]:
    """Load the Pose-EKS arrays of every camera of one session.

    Args:
        eks_dir: directory of the Pose-EKS csvs.
        session_id: session UUID.
        cameras: cameras to load.
        groups: body-part group name -> tokens; the keypoints of all groups are loaded, once each.
        start_frame: first frame.
        n_frames: number of frames.
        pca_dir: directory of the files written by `pose2visual.lightningpose.pca_eks`; when
            given, each camera's `pca_xy` is filled from its file (it stays None when the file does
            not exist). None loads no PCA.

    Returns:
        the arrays and the csv path, each keyed by camera.

    Raises:
        FileNotFoundError: if a csv is missing.
        ValueError: if a group matches no keypoint, or a csv is too short.
    """
    data = {}
    csv_paths = {}
    for camera in cameras:
        csv_path = resolve_eks_csv(eks_dir, session_id, camera)
        keypoint_names = discover_keypoint_names(csv_path)
        columns: list[str] = []
        landmarks: list[str] = []
        for tokens in groups.values():
            bases = resolve_group_bases(keypoint_names, tokens)
            for (base, _), column in zip(bases, group_columns_for_camera(bases, camera)):
                if column not in columns:
                    columns.append(column)
                    landmarks.append(base)
        data[camera] = load_camera_data(
            csv_path, columns, landmarks, camera, start_frame, n_frames,
        )
        if pca_dir is not None:
            data[camera].pca_xy = load_pca_positions(
                pca_dir, session_id, camera, columns, start_frame, n_frames,
            )
            if data[camera].pca_xy is None:
                logger.info(f'  {camera}: no PCA file in {pca_dir}; no red PCA trace')
        csv_paths[camera] = csv_path
    return data, csv_paths


def run(args: argparse.Namespace) -> list[Path]:
    """Render one video per session and group, each in a folder named after its group.

    Args:
        args: parsed command-line arguments (see `pose2visual.qc.video.cli.parse_args`); when
            `args.groups` is None every group of the group set is rendered, and a group with no
            keypoint in the csvs is skipped with a warning (an explicitly named one is an error).

    Returns:
        the written video paths, in session then group order.

    Raises:
        ValueError: if a requested group, session group or an explicitly named group's keypoints
            do not exist.
        FileNotFoundError: if a csv or video cannot be found.
    """
    all_groups = load_groups(args.groups_file, group_set=args.group_set)
    explicit_groups = args.groups is not None
    group_names = args.groups if explicit_groups else list(all_groups)
    missing = sorted(set(group_names) - set(all_groups))
    if missing:
        raise ValueError(f'groups not found in {args.groups_file}: {missing}')

    if args.sessions is not None:
        sessions = args.sessions
    else:
        session_groups = load_groups(args.sessions_file, group_set='neural_sessions')
        if args.session_group not in session_groups:
            raise ValueError(
                f'session group {args.session_group!r} not found in {args.sessions_file}: '
                f'{sorted(session_groups)}',
            )
        sessions = session_groups[args.session_group]
    logger.info(f'sessions: {sessions}, groups: {group_names}')

    eks_dir = Path(args.eks_dir).resolve()
    out_dir = (
        Path(args.out_dir).resolve() if args.out_dir else eks_dir.parent / 'qc' / 'video_qc'
    )
    run_label = read_run_label(eks_dir)
    # the PCA of `pca_eks` is per keypoint and saved per group set, so it follows --group-set
    pca_dir = (
        None if args.no_pca
        else Path(args.pca_dir) if args.pca_dir else eks_dir.parent / 'pca' / args.group_set
    )
    video_overrides = {'left': args.video_left, 'right': args.video_right}
    cli_crops = {
        'left': tuple(args.crop_left) if args.crop_left else None,
        'right': tuple(args.crop_right) if args.crop_right else None,
    }

    written = []
    for session_idx, session_id in enumerate(sessions, start=1):
        logger.info(f'=== [{session_idx}/{len(sessions)}] session {session_id} ===')
        csv_paths = {
            camera: resolve_eks_csv(eks_dir, session_id, camera) for camera in args.cameras
        }
        video_paths = {}
        for camera in args.cameras:
            if video_overrides[camera]:
                video_paths[camera] = Path(video_overrides[camera]).resolve()
                continue
            video_path = resolve_video_path(str(csv_paths[camera]), video_root=args.video_root)
            if video_path is None:
                raise FileNotFoundError(
                    f'could not find the {camera} video of session {session_id}; '
                    f'pass --video-{camera} or --video-root',
                )
            video_paths[camera] = video_path
            logger.info(f'  {camera}: {video_path}')

        keypoint_names = discover_keypoint_names(csv_paths[args.cameras[0]])
        for group_name in group_names:
            tokens = all_groups[group_name]
            if not explicit_groups:
                try:
                    resolve_group_bases(keypoint_names, tokens)
                except ValueError:
                    logger.warning(f'  skipping group {group_name}: no keypoint in the csv')
                    continue
            logger.info(f'  --- group {group_name} ---')
            # explicit --crop-* win, then the group's boxes in the groups file, then auto/full
            json_crops = (
                {} if args.no_json_crops
                else load_crop_boxes(args.groups_file, args.group_set, group_name)
            )
            crops = {
                camera: cli_crops[camera] or json_crops.get(camera) for camera in cli_crops
            }
            for camera in args.cameras:
                if crops[camera] is not None and not cli_crops[camera]:
                    logger.info(f'  {camera}: crop {crops[camera]} from {args.groups_file}')
            data, _ = load_session_cameras(
                eks_dir, session_id, args.cameras, {group_name: tokens}, args.start_frame,
                args.n_frames, pca_dir,
            )
            job = VideoJob(
                session_id=session_id,
                data=data,
                video_paths=video_paths,
                out_path=build_output_path(
                    out_dir, session_id, group_name, args.cameras, args.start_frame,
                    args.n_frames, args.tag,
                ),
                start_frame=args.start_frame,
                n_frames=args.n_frames,
                conf_threshold=args.conf_threshold,
                bar_scale=args.bar_scale,
                trace_window=args.trace_window,
                crops=crops,
                auto_crop_pad=args.auto_crop_pad,
                save_diagnostics=args.save_diagnostics,
                group_names=[group_name],
                run_label=run_label,
            )
            written.append(render_video(job))
            logger.info(f'  wrote {written[-1]}')
    return written
