"""Resolve csvs and videos per session, load the anchor window and save the figures."""

import argparse
import logging
from pathlib import Path

from pose2visual.preprocessing import resolve_video_path
from pose2visual.qc.showcase.figures import (
    build_frame_figure,
    build_trace_figure,
    keypoint_colors,
    save_figure,
)
from pose2visual.qc.showcase.loading import load_window, match_frame_size, read_frame
from pose2visual.qc.video.loading import resolve_eks_csv

logger = logging.getLogger(__name__)

CAMERAS = ('left', 'right')
MAX_FPS_MISMATCH = 0.01


def build_output_stem(
    out_dir: Path,
    kind: str,
    session_id: str,
    anchor_frame: int,
    half_window_sec: float | None = None,
) -> Path:
    """Name the figure of one session; the suffixes are added when it is saved.

    The session id is the only part containing `-`; it is separated from the other parts by `.`
    so that it can be split out of the file name.

    Args:
        out_dir: output directory.
        kind: figure kind, `kp_frame` or `kp_traces`.
        session_id: session UUID.
        anchor_frame: video frame index of the anchor.
        half_window_sec: seconds shown on each side of the anchor, for the traces figure.

    Returns:
        `<out_dir>/<kind>.<session>.frame<anchor>[.pm<seconds>s]`, the seconds written with a
        `p` for the decimal point (`pm2p5s` for 2.5).
    """
    parts = [kind, session_id, f'frame{anchor_frame:06d}']
    if half_window_sec is not None:
        parts.append(f'pm{half_window_sec:g}s'.replace('.', 'p'))
    return out_dir / '.'.join(parts)


def run(args: argparse.Namespace) -> list[Path]:
    """Save the requested figures of every session.

    Args:
        args: parsed command-line arguments (see `pose2visual.qc.showcase.cli.parse_args`).

    Returns:
        the written paths (png then pdf of each figure), in session order.

    Raises:
        FileNotFoundError: if a csv or video cannot be found or opened.
        ValueError: if the cameras have different frame rates, or the anchor window does not fit
            in the videos or csvs.
    """
    eks_dir = Path(args.eks_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else eks_dir / 'qc' / 'showcase'
    written = []
    for session_idx, session_id in enumerate(args.sessions, start=1):
        logger.info(f'=== [{session_idx}/{len(args.sessions)}] session {session_id} ===')
        csv_paths = {camera: resolve_eks_csv(eks_dir, session_id, camera) for camera in CAMERAS}
        images = {}
        fps_values = {}
        for camera in CAMERAS:
            video_path = resolve_video_path(str(csv_paths[camera]), video_root=args.video_root)
            if video_path is None:
                raise FileNotFoundError(
                    f'could not find the {camera} video of session {session_id}; '
                    'pass --video-root',
                )
            logger.info(f'  {camera}: {video_path}')
            images[camera], fps_values[camera] = read_frame(video_path, args.anchor_frame)
        fps = fps_values[CAMERAS[0]]
        if any(abs(value - fps) > MAX_FPS_MISMATCH for value in fps_values.values()):
            raise ValueError(f'cameras have different frame rates: {fps_values}')

        make_traces = 'traces' in args.figures
        half_window = round(args.half_window_sec * fps) if make_traces else 0
        data = {
            camera: load_window(csv_paths[camera], camera, args.anchor_frame, half_window)
            for camera in CAMERAS
        }
        if not args.native_aspect:
            # every camera takes the frame size of the last one (right), e.g. a portrait left
            # frame is stretched to the landscape shape of the right frame
            height, width = images[CAMERAS[-1]].shape[:2]
            for camera in CAMERAS[:-1]:
                images[camera], data[camera] = match_frame_size(
                    images[camera], data[camera], (width, height),
                )
        n_keypoints = len(data[CAMERAS[0]].columns)
        colors = keypoint_colors(n_keypoints, args.colormap)

        if 'frames' in args.figures:
            fig = build_frame_figure(
                images, data, half_window, colors, args.conf_threshold, args.dot_size,
                args.keep_far_digits,
            )
            stem = build_output_stem(out_dir, 'kp_frame', session_id, args.anchor_frame)
            written += save_figure(fig, stem, args.dpi)
        if make_traces:
            fig = build_trace_figure(
                images, data, half_window, fps, colors, args.conf_threshold, args.dot_size,
                args.keep_far_digits,
            )
            stem = build_output_stem(
                out_dir, 'kp_traces', session_id, args.anchor_frame, args.half_window_sec,
            )
            written += save_figure(fig, stem, args.dpi)
        logger.info(f'  wrote {[str(path) for path in written[-4:]]}')
    return written
