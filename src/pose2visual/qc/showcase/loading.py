"""Load the video frame and the Pose-EKS window around the anchor frame of one camera."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from pose2visual.lightningpose.fit_eks_latent_model import discover_keypoint_names
from pose2visual.qc.video.loading import CameraData, load_camera_data


def read_frame(video_path: str | Path, frame_idx: int) -> tuple[np.ndarray, float]:
    """Read one frame of a video.

    Args:
        video_path: source video.
        frame_idx: 0-based index of the frame to read.

    Returns:
        the RGB frame, shape (height, width, 3), and the frame rate of the video.

    Raises:
        FileNotFoundError: if the video cannot be opened.
        ValueError: if the video has no frame `frame_idx`.
        RuntimeError: if the frame cannot be decoded.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f'cannot open video: {video_path}')
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= n_frames:
            raise ValueError(f'{video_path} has {n_frames} frames, need frame {frame_idx}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'cannot read frame {frame_idx} of {video_path}')
    finally:
        cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps


def load_window(
    csv_path: str | Path,
    camera: str,
    anchor_frame: int,
    half_window: int,
) -> CameraData:
    """Read the Pose-EKS output of every keypoint around the anchor frame.

    Args:
        csv_path: Pose-EKS csv of one session and camera.
        camera: camera name ('left' or 'right').
        anchor_frame: video frame index of the anchor.
        half_window: number of frames before and after the anchor.

    Returns:
        arrays of `2 * half_window + 1` frames starting at `anchor_frame - half_window`, so row
        `half_window` is the anchor; keypoints are in csv header order.

    Raises:
        ValueError: if the window starts before frame 0 or runs past the end of the csv.
    """
    start_frame = anchor_frame - half_window
    if start_frame < 0:
        raise ValueError(
            f'anchor frame {anchor_frame} is closer than {half_window} frames to the start',
        )
    names = discover_keypoint_names(csv_path)
    return load_camera_data(csv_path, names, names, camera, start_frame, 2 * half_window + 1)


def match_frame_size(
    image: np.ndarray,
    data: CameraData,
    size: tuple[int, int],
) -> tuple[np.ndarray, CameraData]:
    """Resize a frame to `size` and scale the keypoints of its camera to match.

    The frame is stretched independently along x and y, so a frame with another aspect ratio
    takes the aspect ratio of `size`. Positions and SDs are scaled the same way, which makes the
    traces use the pixel units of the resized frame.

    Args:
        image: RGB frame, shape (height, width, 3).
        data: Pose-EKS arrays of the camera the frame belongs to.
        size: target `(width, height)`.

    Returns:
        the resized frame and a copy of `data` in the pixel units of the resized frame; both are
        returned unchanged if the frame already has this size.
    """
    height, width = image.shape[:2]
    if (width, height) == size:
        return image, data
    scale = np.array([size[0] / width, size[1] / height])
    resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return resized, replace(
        data,
        xy_median=data.xy_median * scale,
        xy_eks=data.xy_eks * scale,
        ens_sd=data.ens_sd * scale,
        post_sd=data.post_sd * scale,
    )
