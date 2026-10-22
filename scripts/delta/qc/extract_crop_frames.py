"""Save one frame per camera of a session, to pick the zoom boxes of the QC videos on.

Standalone: needs only opencv, not the `pose2visual` package. Pair it with `select_crops.py`, which
draws one box per body-part group and camera on the saved frames.

Examples:
    python scripts/delta/qc/extract_crop_frames.py --session <uuid> --out-dir crop_frames
    python scripts/delta/qc/extract_crop_frames.py --left l.mp4 --right r.mp4 --frame 500
"""

import argparse
import os
from pathlib import Path

import cv2

CAMERAS = ('left', 'right')
DEFAULT_VIDEO_ROOT = '/work/hdd/bfsr/xdai3/IBL-2view'


def resolve_video(session: str, camera: str, video_root: Path) -> Path:
    """Build the path of a session's video in the IBL-2view layout.

    Args:
        session: session UUID.
        camera: `'left'` or `'right'`.
        video_root: folder holding the `<camera>Camera.video` folders.

    Returns:
        `<root>/<camera>Camera.video/_iblrig_<camera>Camera.downsampled.<session>.mp4`.
    """
    name = f'_iblrig_{camera}Camera.downsampled.{session}.mp4'
    return video_root / f'{camera}Camera.video' / name


def save_frame(video: Path, frame_idx: int, out_path: Path) -> tuple[int, int]:
    """Read one frame of a video and write it as a png.

    Args:
        video: video file.
        frame_idx: zero-based frame to read.
        out_path: png to write.

    Returns:
        frame `(width, height)`.

    Raises:
        FileNotFoundError: if the video cannot be opened.
        ValueError: if the frame cannot be read (e.g. past the end of the video).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open video: {video}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        raise ValueError(f'cannot read frame {frame_idx} of {video} ({n_frames} frames)')
    cv2.imwrite(str(out_path), frame)
    return frame.shape[1], frame.shape[0]


def main() -> None:
    """Save one png per requested camera."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--session', help='session UUID; videos are found under --video-root')
    parser.add_argument(
        '--video-root',
        default=os.environ.get('IBL_2VIEW_ROOT', DEFAULT_VIDEO_ROOT),
        help='root of the IBL-2view video folders (default: $IBL_2VIEW_ROOT)',
    )
    parser.add_argument('--left', help='left video file (instead of --session)')
    parser.add_argument('--right', help='right video file (instead of --session)')
    parser.add_argument('--frame', type=int, default=1000, help='frame to save (default: 1000)')
    parser.add_argument('--out-dir', default='crop_frames', help='output folder')
    args = parser.parse_args()

    videos = {'left': args.left, 'right': args.right}
    if args.session:
        videos = {
            camera: videos[camera] or resolve_video(args.session, camera, Path(args.video_root))
            for camera in CAMERAS
        }
    videos = {camera: Path(path) for camera, path in videos.items() if path}
    if not videos:
        parser.error('give --session, or --left and/or --right')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.session or 'video'
    for camera, video in videos.items():
        out_path = out_dir / f'crop_frame.{stem}.{camera}.png'
        width, height = save_frame(video, args.frame, out_path)
        print(f'{camera}: {video} -> {out_path} ({width}x{height})')


if __name__ == '__main__':
    main()
