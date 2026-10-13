"""Utilities for video handling and path resolution.

This module provides functions for resolving video paths from dataset conventions,
particularly for IBL-2view dataset where videos are organized by camera type.
"""

import os
import re
from pathlib import Path

DEFAULT_VIDEO_ROOT = Path('/work/hdd/bfsr/xdai3/IBL-2view')
VIDEO_ROOT_ENV_VAR = 'IBL_2VIEW_ROOT'


def resolve_video_path(csv_path: str, video_root: str | Path | None = None) -> Path | None:
    """Resolve video path from CSV filename for IBL-2view dataset.

    Extracts camera and UUID from CSV filename and looks for corresponding video.
    Expected CSV stem format: _iblrig_<left|right>Camera.downsampled.<UUID>
    Video location: <video_root>/<left|right>Camera.video/

    Args:
        csv_path: Path to predictions CSV
        video_root: Root holding the `<left|right>Camera.video` folders. Falls back to
            `$IBL_2VIEW_ROOT`, then to `DEFAULT_VIDEO_ROOT`.

    Returns:
        Path to video if found, None otherwise

    Example:
        >>> csv = 'path/_iblrig_leftCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35.csv'
        >>> video = resolve_video_path(csv)
        >>> if video:
        ...     print(f'Found video: {video}')
    """
    csv_stem = Path(csv_path).stem
    match = re.search(r'_iblrig_(left|right)Camera\.downsampled\.([a-f0-9\-]+)', csv_stem)

    if not match:
        return None

    camera = match.group(1)
    uuid = match.group(2)

    root = Path(video_root or os.environ.get(VIDEO_ROOT_ENV_VAR) or DEFAULT_VIDEO_ROOT)
    video_dir = root / f'{camera}Camera.video'
    video_path = video_dir / f'_iblrig_{camera}Camera.downsampled.{uuid}.mp4'

    if video_path.exists():
        return video_path
    return None
