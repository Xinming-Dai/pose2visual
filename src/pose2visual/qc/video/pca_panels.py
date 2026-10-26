"""Render the three-panel PCA decomposition video of one camera.

Panels, left to right, all in the same source-pixel coordinate system and on the same video frame
(the real image is behind all three):

1. the tracked keypoints (`mean + X_centered`; hollow where the position was imputed because the
   frame was unreliable),
2. the rank-k reconstruction (`mean + X_rank`),
3. the residual (`mean + X_residual`).

The mean positions are added back to the last two so the motion is shown in image coordinates; a
ring marks each keypoint's mean position, a filled dot its current position and a thin line the
displacement between them. The crop is fixed for the whole clip (the whole
frame unless a box is given), so a keypoint's panel position never depends on the frame. The
layout, colours and encoder are those of the comparison video (`render_compare`, `style`).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pose2visual.qc.video.render_compare import (
    BANNER_HEIGHT,
    CANVAS_BACKGROUND,
    LOG_EVERY_FRAMES,
    PANEL_FOOTER,
    PANEL_HEADER,
    PANEL_IMAGE_HEIGHT,
    PanelGeometry,
    fit_crop_box,
    make_geometry,
    open_encoder,
)
from pose2visual.qc.video.style import (
    FONT,
    draw_legend_row,
    fit_font_scale,
    keypoint_color,
    put_text,
)

logger = logging.getLogger(__name__)

N_PANELS = 3
RING_RADIUS = 6
DOT_RADIUS = 3
LABEL_SCALE = 0.33
PANEL_TITLES = ('tracked keypoints', 'rank-{rank} reconstruction', 'residual')


@dataclass
class PcaVideoJob:
    """Everything needed to render the video of one camera.

    Attributes:
        eid: session id (banner).
        camera: camera name.
        keypoints: keypoint names, in the column order of the arrays.
        video_path: source video.
        out_path: output mp4.
        start_frame: video frame index of the first row of the arrays.
        mean: mean position of each keypoint, shape (n_keypoints, 2).
        xy_centered: `X_centered` as (n_frames, n_keypoints, 2); only the sum with `mean` is drawn.
        xy_rank: `X_rank`, same shape.
        xy_residual: `X_residual`, same shape.
        reliable: boolean (n_frames, n_keypoints), False where the position was imputed.
        explained_variance: fraction of the variance kept by the rank-k part.
        rank: number of components kept.
        group: body-part group name (banner).
        crop_pad: zoom to the tracked keypoints plus this many source pixels (one fixed box for the
            whole clip); None = the whole frame.
    """

    eid: str
    camera: str
    keypoints: list[str]
    video_path: Path
    out_path: Path
    start_frame: int
    mean: np.ndarray
    xy_centered: np.ndarray
    xy_rank: np.ndarray
    xy_residual: np.ndarray
    reliable: np.ndarray
    explained_variance: float
    rank: int
    group: str = ''
    crop_pad: float | None = None

    @property
    def n_frames(self) -> int:
        """Number of frames rendered."""
        return self.xy_centered.shape[0]


def fixed_crop_box(
    mean: np.ndarray,
    xy: np.ndarray,
    pad: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Fit one crop around every tracked position of the clip and the mean positions.

    Args:
        mean: mean positions (n_keypoints, 2).
        xy: tracked positions (n_frames, n_keypoints, 2).
        pad: margin in source pixels on each side.
        frame_size: source frame `(width, height)`.

    Returns:
        crop box as in `fit_crop_box`; the whole frame if nothing is finite.
    """
    points = np.concatenate([mean, xy.reshape(-1, 2)])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return 0, 0, frame_size[0], frame_size[1]
    x0, y0 = points.min(axis=0) - pad
    x1, y1 = points.max(axis=0) + pad
    return fit_crop_box((x0, y0, x1, y1), frame_size)


def panel_point(
    xy: np.ndarray,
    geometry: PanelGeometry,
) -> tuple[int, int] | None:
    """Map a source-pixel position to panel pixels (below the header).

    Args:
        xy: `(x, y)` in source pixels.
        geometry: the fixed panel geometry.

    Returns:
        `(x, y)` in the panel, or None when the position is not finite.
    """
    if not np.isfinite(xy).all():
        return None
    x = round((xy[0] - geometry.crop[0]) * geometry.scale_x)
    y = round((xy[1] - geometry.crop[1]) * geometry.scale_y) + PANEL_HEADER
    return x, y


def put_outlined_text(
    img: np.ndarray,
    xy: tuple[int, int],
    text: str,
    font_scale: float,
    color: tuple[int, int, int],
) -> None:
    """Draw text with a dark outline so it reads on both bright and dark parts of the frame.

    Args:
        img: BGR image, modified in place.
        xy: bottom-left corner of the text.
        text: string to draw.
        font_scale: cv2 font scale.
        color: BGR colour of the text.
    """
    cv2.putText(img, text, xy, FONT, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, FONT, font_scale, color, 1, cv2.LINE_AA)


def draw_panel(
    image: np.ndarray,
    geometry: PanelGeometry,
    keypoints: list[str],
    xy: np.ndarray,
    title: str,
    footer: str,
    reliable: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Draw one panel: header, the video frame, keypoints and footer.

    Marks, all with a dark outline: a filled dot at each position in `xy`; when `mean` is given
    also a ring at each mean position and a thin line from the ring to the dot (the displacement);
    a keypoint whose position was imputed (`reliable` False) is drawn as a hollow circle.

    Args:
        image: the frame already cropped and resized to `(PANEL_IMAGE_HEIGHT, geometry.width)`.
        geometry: how source pixels map to the panel.
        keypoints: keypoint names.
        xy: positions to draw (n_keypoints, 2) in source pixels.
        title: header text.
        footer: footer text.
        reliable: boolean (n_keypoints,); False keypoints are drawn hollow; None = all solid.
        mean: mean positions (n_keypoints, 2); when given a ring marks each one.

    Returns:
        BGR panel of height `PANEL_HEADER + PANEL_IMAGE_HEIGHT + PANEL_FOOTER`.
    """
    width = geometry.width
    height = PANEL_HEADER + PANEL_IMAGE_HEIGHT + PANEL_FOOTER
    panel = np.zeros((height, width, 3), np.uint8)
    panel[PANEL_HEADER:PANEL_HEADER + PANEL_IMAGE_HEIGHT] = image
    for idx, name in enumerate(keypoints):
        color = keypoint_color(name)
        point = panel_point(xy[idx], geometry)
        if point is None:
            continue
        if mean is not None:
            ring = panel_point(mean[idx], geometry)
            if ring is not None:
                cv2.line(panel, ring, point, color, 1, cv2.LINE_AA)
                cv2.circle(panel, ring, RING_RADIUS + 1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(panel, ring, RING_RADIUS, color, 1, cv2.LINE_AA)
        solid = reliable is None or bool(reliable[idx])
        if solid:
            cv2.circle(panel, point, DOT_RADIUS + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(panel, point, DOT_RADIUS, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(panel, point, DOT_RADIUS + 1, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(panel, point, DOT_RADIUS + 1, color, 1, cv2.LINE_AA)
        put_outlined_text(panel, (point[0] + 5, point[1] - 5), name, LABEL_SCALE, color)
    cv2.rectangle(panel, (0, 0), (width, PANEL_HEADER - 2), (0, 0, 0), -1)
    put_text(panel, (5, 17), title, fit_font_scale(title, width - 10, 0.46))
    cv2.rectangle(panel, (0, height - PANEL_FOOTER), (width, height), (0, 0, 0), -1)
    put_text(panel, (5, height - 8), footer, fit_font_scale(footer, width - 10, 0.42))
    return panel


def rms_displacement(offset: np.ndarray) -> float:
    """Root-mean-square length of per-keypoint displacements.

    Args:
        offset: displacements (n_keypoints, 2); NaN rows are ignored.

    Returns:
        RMS distance in pixels, NaN if nothing is finite.
    """
    norm_squared = np.sum(offset ** 2, axis=1)
    norm_squared = norm_squared[np.isfinite(norm_squared)]
    return float(np.sqrt(norm_squared.mean())) if norm_squared.size else float('nan')


def draw_banner(canvas: np.ndarray, job: PcaVideoJob, fps: float) -> None:
    """Draw the legend and the two description lines at the top.

    Args:
        canvas: full video frame, modified in place.
        job: the render settings.
        fps: video frame rate.
    """
    width = canvas.shape[1]
    x = draw_legend_row(canvas, job.keypoints)
    line = f'{job.group} | {job.camera} camera | session {job.eid}'
    put_text(canvas, (x + 8, 20), line, fit_font_scale(line, width - x - 16, 0.5))
    line = (
        f'LEFT: tracked keypoints | MIDDLE: mean + rank-{job.rank} part '
        f'({100 * job.explained_variance:.1f} % of the variance) | RIGHT: mean + residual '
        '(true pixel scale)'
    )
    put_text(canvas, (10, 46), line, fit_font_scale(line, width - 20, 0.5))
    line = (
        f'ring = mean position | dot = current position (line = displacement) | hollow = imputed '
        f'position | {job.n_frames} frames at {fps:g} fps from frame {job.start_frame} | fixed '
        'image coordinates'
    )
    put_text(canvas, (10, 67), line, fit_font_scale(line, width - 20, 0.48))


def render_pca_video(job: PcaVideoJob) -> Path:
    """Render every frame of the decomposition video.

    Args:
        job: the render settings.

    Returns:
        path of the written mp4.

    Raises:
        FileNotFoundError: if the source video cannot be opened.
        ValueError: if the video is too short for the clip.
        RuntimeError: if a source frame cannot be read, ffmpeg fails, or the written video does
            not have `job.n_frames` frames.
    """
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open video: {job.video_path}')
    n_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_available < job.start_frame + job.n_frames:
        cap.release()
        raise ValueError(
            f'{job.video_path} has {n_available} frames, need {job.start_frame + job.n_frames}',
        )
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if job.crop_pad is None:
        crop = (0, 0, *frame_size)
    else:
        crop = fixed_crop_box(job.mean, job.mean + job.xy_centered, job.crop_pad, frame_size)
    geometry = make_geometry(crop, frame_size)
    logger.info(f'  {job.camera}: frame {frame_size[0]}x{frame_size[1]}, crop {crop}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, job.start_frame)

    width = N_PANELS * geometry.width
    height = BANNER_HEIGHT + PANEL_HEADER + PANEL_IMAGE_HEIGHT + PANEL_FOOTER
    titles = [title.format(rank=job.rank) for title in PANEL_TITLES]
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = job.out_path.with_suffix('.ffmpeg.log')
    process, log_file = open_encoder(job.out_path, (width, height), fps, log_path)
    logger.info(f'  rendering {job.n_frames} frames of {width}x{height} -> {job.out_path}')
    x0, y0, x1, y1 = geometry.crop
    try:
        for idx in range(job.n_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'cannot read {job.camera} frame {job.start_frame + idx}')
            image = cv2.resize(
                frame[y0:y1, x0:x1], (geometry.width, PANEL_IMAGE_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            video_frame = job.start_frame + idx
            xy_tracked = job.mean + job.xy_centered[idx]
            xy_rank = job.mean + job.xy_rank[idx]
            xy_residual = job.mean + job.xy_residual[idx]
            footers = (
                f'frame {video_frame} | {idx + 1}/{job.n_frames} | {video_frame / fps:.2f} s',
                f'RMS displacement from mean {rms_displacement(job.xy_rank[idx]):.2f} px',
                f'RMS displacement from mean {rms_displacement(job.xy_residual[idx]):.2f} px',
            )
            panels = (
                draw_panel(
                    image, geometry, job.keypoints, xy_tracked, titles[0], footers[0],
                    reliable=job.reliable[idx],
                ),
                draw_panel(
                    image, geometry, job.keypoints, xy_rank, titles[1], footers[1],
                    mean=job.mean,
                ),
                draw_panel(
                    image, geometry, job.keypoints, xy_residual, titles[2], footers[2],
                    mean=job.mean,
                ),
            )
            canvas = np.full((height, width, 3), CANVAS_BACKGROUND, np.uint8)
            draw_banner(canvas, job, fps)
            canvas[BANNER_HEIGHT:] = np.concatenate(panels, axis=1)
            process.stdin.write(canvas.tobytes())
            if idx % LOG_EVERY_FRAMES == 0:
                logger.info(f'  rendered {idx}/{job.n_frames}')
    finally:
        process.stdin.close()
        cap.release()
        return_code = process.wait()
        log_file.close()
    if return_code != 0:
        raise RuntimeError(f'ffmpeg failed (exit {return_code}); see {log_path}')
    log_path.unlink()

    check = cv2.VideoCapture(str(job.out_path))
    n_written = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    if n_written != job.n_frames:
        raise RuntimeError(f'{job.out_path} has {n_written} frames, expected {job.n_frames}')
    return job.out_path
