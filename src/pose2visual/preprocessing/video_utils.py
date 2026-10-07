"""Utilities for video handling and path resolution.

This module provides functions for resolving video paths from dataset conventions,
particularly for IBL-2view dataset where videos are organized by camera type.
"""

import re
from pathlib import Path
from typing import Optional


def resolve_video_path(csv_path: str) -> Optional[Path]:
    """Resolve video path from CSV filename for IBL-2view dataset.

    Extracts camera and UUID from CSV filename and looks for corresponding video.
    Expected CSV stem format: _iblrig_<left|right>Camera.downsampled.<UUID>
    Video location: /work/hdd/bfsr/xdai3/IBL-2view/<left|right>Camera.video/

    Args:
        csv_path: Path to predictions CSV

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

    video_dir = Path('/work/hdd/bfsr/xdai3/IBL-2view') / f'{camera}Camera.video'
    video_path = video_dir / f'_iblrig_{camera}Camera.downsampled.{uuid}.mp4'

    if video_path.exists():
        return video_path
    return None
