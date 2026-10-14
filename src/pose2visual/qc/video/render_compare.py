"""Render the ensemble median vs Pose-EKS comparison video of one session.

Layout: one column per camera (each with its own image panel and an SD strip), top row the
ensemble median with an ensemble-SD cross, bottom row the Pose-EKS position with a posterior-SD
cross, and the scrolling coordinate traces of `trace_panels` underneath. A keypoint is drawn in
both rows only on frames where its mean ensemble likelihood exceeds the gate. The ±1 SD cross
arms are magnified by `bar_scale`, because the SDs are often a fraction of a pixel; the strips
print the true pixel SDs.

The image panels can show a zoomed crop of the frame (a fixed box per camera, or one fitted around
the drawn keypoints): the crop is cut from the full-resolution frame before it is resized, and the
keypoints are shifted and scaled to match. Frames are piped to ffmpeg as raw BGR and written as
browser-compatible H264.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from pose2visual.qc.video.loading import CameraData
from pose2visual.qc.video.style import (
    draw_legend_row,
    fit_font_scale,
    keypoint_color,
    put_text,
)
from pose2visual.qc.video.trace_panels import (
    TraceData,
    build_trace_data,
    compute_trace_diagnostics,
    gate_mask,
    render_trace_panel,
    trace_panel_height,
)

logger = logging.getLogger(__name__)

PANEL_IMAGE_HEIGHT = 512
PANEL_HEADER = 26
PANEL_FOOTER = 24
STRIP_WIDTH = 250
BANNER_HEIGHT = 76
CANVAS_BACKGROUND = 14
FULL_ROW_HEIGHT = 60
COMPACT_ROW_HEIGHT = 18
STRIP_HEADER_Y = 22
STRIP_FIRST_ROW = 50
MAX_FPS_MISMATCH = 0.01
LOG_EVERY_FRAMES = 250


@dataclass(frozen=True)
class RowSpec:
    """One row of the video: which arrays it draws and how it is labelled.

    Attributes:
        label: panel header text.
        sd_label: name of the SD in the strip, e.g. `ens SD x/y`.
        xy_attr: `CameraData` attribute holding the positions to draw.
        sd_attr: `CameraData` attribute holding the SD of the cross.
    """

    label: str
    sd_label: str
    xy_attr: str
    sd_attr: str


ROWS = (
    RowSpec('Ensemble median', 'ens SD x/y', 'xy_median', 'ens_sd'),
    RowSpec('Pose-EKS (single-camera)', 'post SD x/y', 'xy_eks', 'post_sd'),
)


@dataclass(frozen=True)
class PanelGeometry:
    """Mapping from source-frame pixels to one camera's image panel.

    Attributes:
        crop: `(x0, y0, x1, y1)` region of the source frame shown in the panel.
        width: panel width in pixels (even); the image part is `PANEL_IMAGE_HEIGHT` tall.
        frame_size: source frame `(width, height)`.
    """

    crop: tuple[int, int, int, int]
    width: int
    frame_size: tuple[int, int]

    @property
    def is_cropped(self) -> bool:
        """Whether the panel shows less than the whole source frame."""
        return self.crop != (0, 0, *self.frame_size)

    @property
    def scale_x(self) -> float:
        """Panel pixels per source pixel along x."""
        return self.width / (self.crop[2] - self.crop[0])

    @property
    def scale_y(self) -> float:
        """Panel pixels per source pixel along y."""
        return PANEL_IMAGE_HEIGHT / (self.crop[3] - self.crop[1])


@dataclass
class VideoJob:
    """Everything needed to render the video of one session.

    Attributes:
        session_id: session UUID (shown in the banner).
        data: camera name -> loaded Pose-EKS arrays, drawn left to right in this order.
        video_paths: camera name -> source video.
        out_path: output mp4 path.
        start_frame: first video frame rendered.
        n_frames: number of frames rendered.
        conf_threshold: likelihood gate.
        bar_scale: magnification of the SD cross arms.
        trace_window: number of trailing frames in the trace panel.
        crops: camera name -> explicit `(x0, y0, x1, y1)` crop of the source frame; cameras
            absent or None are shown whole (or auto-cropped).
        auto_crop_pad: if set, cameras without an explicit crop are cropped to the drawn
            keypoints plus this many source pixels on each side.
        group_names: body-part groups shown (banner).
        run_label: model description for the banner, or None.
        save_diagnostics: also write a `.diagnostics.csv` with the same stem beside the mp4.
    """

    session_id: str
    data: dict[str, CameraData]
    video_paths: dict[str, Path]
    out_path: Path
    start_frame: int
    n_frames: int
    conf_threshold: float = 0.7
    bar_scale: float = 5.0
    trace_window: int = 300
    crops: dict[str, tuple[int, int, int, int] | None] = field(default_factory=dict)
    auto_crop_pad: float | None = None
    group_names: list[str] = field(default_factory=list)
    run_label: str | None = None
    save_diagnostics: bool = False


def fit_crop_box(
    box: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Expand a box to the frame's aspect ratio and move it inside the frame.

    Args:
        box: `(x0, y0, x1, y1)` region of interest in source pixels.
        frame_size: source frame `(width, height)`.

    Returns:
        integer box with (up to rounding) the frame's aspect ratio, inside the frame; the whole
        frame if the expanded box would not fit.

    Raises:
        ValueError: if the box is empty.
    """
    frame_w, frame_h = frame_size
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f'crop box {box} is empty')
    aspect = frame_w / frame_h
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    if width >= frame_w or height >= frame_h:
        return 0, 0, frame_w, frame_h
    new_x0 = min(max((x0 + x1) / 2 - width / 2, 0), frame_w - width)
    new_y0 = min(max((y0 + y1) / 2 - height / 2, 0), frame_h - height)
    return (
        round(new_x0),
        round(new_y0),
        min(frame_w, round(new_x0 + width)),
        min(frame_h, round(new_y0 + height)),
    )


def auto_crop_box(
    data: CameraData,
    conf_threshold: float,
    pad: float,
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Fit a fixed crop around every gated position of both methods over the clip.

    Args:
        data: one camera's Pose-EKS arrays.
        conf_threshold: likelihood gate.
        pad: margin in source pixels added on each side.
        frame_size: source frame `(width, height)`.

    Returns:
        crop box as in `fit_crop_box`; the whole frame if nothing is gated.
    """
    gate = gate_mask(data.likelihood, conf_threshold)
    points = np.concatenate([data.xy_median[gate], data.xy_eks[gate]])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return 0, 0, frame_size[0], frame_size[1]
    x0, y0 = points.min(axis=0) - pad
    x1, y1 = points.max(axis=0) + pad
    return fit_crop_box((x0, y0, x1, y1), frame_size)


def make_geometry(
    crop: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> PanelGeometry:
    """Build the panel geometry that shows `crop` at `PANEL_IMAGE_HEIGHT` pixels tall.

    Args:
        crop: `(x0, y0, x1, y1)` region of the source frame.
        frame_size: source frame `(width, height)`.

    Returns:
        the geometry; the panel width follows the crop's aspect ratio, rounded up to even.
    """
    width = round(PANEL_IMAGE_HEIGHT * (crop[2] - crop[0]) / (crop[3] - crop[1]))
    return PanelGeometry(crop=crop, width=width + width % 2, frame_size=frame_size)


def _draw_strip(
    strip: np.ndarray,
    sd_label: str,
    entries: list[tuple[str, tuple[int, int, int], float, float, float]],
) -> None:
    """Fill an SD strip with one entry per drawn keypoint.

    Args:
        strip: BGR image of the strip, modified in place.
        sd_label: name of the SD, printed on every SD value (e.g. `post SD x/y`).
        entries: `(label, color, likelihood, sd_x, sd_y)` per drawn keypoint. One two-line row
            each when they fit, else one compact line each (the last lines become a '+n more').
    """
    height, width = strip.shape[:2]
    available = height - STRIP_FIRST_ROW - 6
    if len(entries) * FULL_ROW_HEIGHT <= available:
        put_text(strip, (8, STRIP_HEADER_Y), 'keypoint', 0.43)
        put_text(strip, (width - 100, STRIP_HEADER_Y), 'likelihood', 0.43)
        for idx, (label, color, likelihood, sd_x, sd_y) in enumerate(entries):
            y = STRIP_FIRST_ROW + idx * FULL_ROW_HEIGHT
            put_text(strip, (8, y), label, fit_font_scale(label, width - 80, 0.47), color)
            put_text(strip, (width - 62, y), f'{likelihood:.3f}', 0.47, color)
            put_text(strip, (8, y + 20), f'{sd_label}  {sd_x:.3f} / {sd_y:.3f}', 0.43)
        return

    put_text(strip, (8, STRIP_HEADER_Y), 'keypoint', 0.4)
    put_text(strip, (124, STRIP_HEADER_Y), 'lik', 0.4)
    put_text(strip, (166, STRIP_HEADER_Y), sd_label, 0.4)
    capacity = available // COMPACT_ROW_HEIGHT
    shown = entries if len(entries) <= capacity else entries[:capacity - 1]
    for idx, (label, color, likelihood, sd_x, sd_y) in enumerate(shown):
        y = STRIP_FIRST_ROW + idx * COMPACT_ROW_HEIGHT
        put_text(strip, (8, y), label, fit_font_scale(label, 112, 0.38), color)
        put_text(strip, (124, y), f'{likelihood:.2f}', 0.36, color)
        put_text(strip, (166, y), f'{sd_x:.2f}/{sd_y:.2f}', 0.36)
    if len(shown) < len(entries):
        y = STRIP_FIRST_ROW + len(shown) * COMPACT_ROW_HEIGHT
        put_text(strip, (8, y), f'+{len(entries) - len(shown)} more', 0.36)


def draw_row(
    image: np.ndarray,
    geometry: PanelGeometry,
    data: CameraData,
    row: RowSpec,
    frame_idx: int,
    job: VideoJob,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one row (image panel and SD strip) of one camera at one frame.

    Args:
        image: the frame already cropped and resized to `(PANEL_IMAGE_HEIGHT, geometry.width)`.
        geometry: how source pixels map to the panel.
        data: this camera's arrays.
        row: which arrays to draw.
        frame_idx: current frame relative to the first rendered frame.
        job: the render settings.
        fps: video frame rate (footer time).

    Returns:
        `(panel, strip)` BGR images of height `PANEL_IMAGE_HEIGHT + PANEL_HEADER + PANEL_FOOTER`.
    """
    width = geometry.width
    height = PANEL_IMAGE_HEIGHT + PANEL_HEADER + PANEL_FOOTER
    panel = np.zeros((height, width, 3), np.uint8)
    panel[PANEL_HEADER:PANEL_HEADER + PANEL_IMAGE_HEIGHT] = image
    strip = np.full((height, STRIP_WIDTH, 3), 24, np.uint8)

    xy = getattr(data, row.xy_attr)[frame_idx]
    sd = getattr(data, row.sd_attr)[frame_idx]
    gate = gate_mask(data.likelihood[frame_idx], job.conf_threshold)
    origin_x, origin_y = geometry.crop[:2]
    entries = []
    for kp_idx, (column, landmark) in enumerate(zip(data.columns, data.landmarks)):
        if not gate[kp_idx] or not np.isfinite(xy[kp_idx]).all():
            continue
        color = keypoint_color(column)
        x = round((xy[kp_idx, 0] - origin_x) * geometry.scale_x)
        y = round((xy[kp_idx, 1] - origin_y) * geometry.scale_y) + PANEL_HEADER
        # cross arms: +-1 SD, magnified, in panel pixels
        arm_x = round(sd[kp_idx, 0] * geometry.scale_x * job.bar_scale)
        arm_y = round(sd[kp_idx, 1] * geometry.scale_y * job.bar_scale)
        cv2.line(panel, (x - arm_x, y), (x + arm_x, y), color, 1, cv2.LINE_AA)
        cv2.line(panel, (x, y - arm_y), (x, y + arm_y), color, 1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 2, color, -1, cv2.LINE_AA)
        put_text(panel, (x + 5, y - 5), landmark, 0.4, color)
        entries.append(
            (landmark, color, float(data.likelihood[frame_idx, kp_idx]),
             float(sd[kp_idx, 0]), float(sd[kp_idx, 1])),
        )
    _draw_strip(strip, row.sd_label, entries)

    header = f'{row.label} | {data.camera} | {len(entries)} gated keypoints'
    if geometry.is_cropped:
        x0, y0, x1, y1 = geometry.crop
        header += f' | crop {x0},{y0}-{x1},{y1}'
    cv2.rectangle(panel, (0, 0), (width, 24), (0, 0, 0), -1)
    put_text(panel, (5, 17), header, fit_font_scale(header, width - 10, 0.46))
    cv2.rectangle(panel, (0, height - PANEL_FOOTER), (width, height), (0, 0, 0), -1)
    video_frame = data.start_frame + frame_idx
    put_text(
        panel,
        (5, height - 8),
        f'frame {video_frame} | {frame_idx + 1}/{job.n_frames} | {video_frame / fps:.2f} s',
        0.42,
    )
    return panel, strip


def draw_banner(canvas: np.ndarray, job: VideoJob) -> None:
    """Draw the three banner lines (legend, what is shown, gate and bar scale) at the top.

    Args:
        canvas: full video frame, modified in place.
        job: the render settings.
    """
    width = canvas.shape[1]
    keypoints = [column for data in job.data.values() for column in data.columns]
    x = draw_legend_row(canvas, keypoints)
    put_text(
        canvas,
        (x + 8, 20),
        f'{" / ".join(job.group_names)} | session {job.session_id}',
        fit_font_scale('x' * 60, width - x - 16, 0.5),
    )
    line = 'TOP: ensemble median +/- ensemble SD | BOTTOM: single-camera Pose-EKS +/- posterior SD'
    if job.run_label:
        line += f' | {job.run_label}'
    put_text(canvas, (10, 46), line, fit_font_scale(line, width - 20, 0.5))
    line = (
        f'Mean ensemble likelihood > {job.conf_threshold:g} on BOTH rows | +/-1 SD bars '
        f'magnified x{job.bar_scale:g}; strips show TRUE pixel SD | no ground truth'
    )
    put_text(canvas, (10, 67), line, fit_font_scale(line, width - 20, 0.48))


def _open_encoder(
    out_path: Path,
    size: tuple[int, int],
    fps: float,
    log_path: Path,
) -> tuple[subprocess.Popen, object]:
    """Start an ffmpeg process that reads raw BGR frames from stdin and writes H264.

    Args:
        out_path: output mp4.
        size: frame `(width, height)`; both must be even for yuv420p.
        fps: output frame rate.
        log_path: file receiving ffmpeg's stderr.

    Returns:
        the process and the open log file (the caller closes it).

    Raises:
        ValueError: if a dimension is odd.
    """
    width, height = size
    if width % 2 or height % 2:
        raise ValueError(f'frame size {width}x{height} must be even to encode yuv420p')
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps:g}',
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
        '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path),
    ]
    log_file = log_path.open('w')
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log_file), log_file


def _open_captures(job: VideoJob) -> tuple[dict[str, cv2.VideoCapture], float]:
    """Open every camera's video, validate it against the job and seek to the first frame.

    Args:
        job: the render settings.

    Returns:
        the open captures and the shared frame rate.

    Raises:
        FileNotFoundError: if a video cannot be opened.
        ValueError: if the frame rates differ or a video is too short.
    """
    captures = {}
    for camera, path in job.video_paths.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f'cannot open video: {path}')
        captures[camera] = cap
    fps_values = {camera: cap.get(cv2.CAP_PROP_FPS) for camera, cap in captures.items()}
    fps = next(iter(fps_values.values()))
    if any(abs(value - fps) > MAX_FPS_MISMATCH for value in fps_values.values()):
        raise ValueError(f'cameras have different frame rates: {fps_values}')
    for camera, cap in captures.items():
        n_available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_available < job.start_frame + job.n_frames:
            raise ValueError(
                f'{job.video_paths[camera]} has {n_available} frames, '
                f'need {job.start_frame + job.n_frames}',
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, job.start_frame)
    return captures, fps


def _resolve_geometries(
    job: VideoJob,
    captures: dict[str, cv2.VideoCapture],
) -> dict[str, PanelGeometry]:
    """Choose the crop of every camera and build its panel geometry."""
    geometries = {}
    for camera, cap in captures.items():
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        explicit = job.crops.get(camera)
        if explicit is not None:
            crop = fit_crop_box(explicit, frame_size)
        elif job.auto_crop_pad is not None:
            crop = auto_crop_box(
                job.data[camera], job.conf_threshold, job.auto_crop_pad, frame_size,
            )
        else:
            crop = (0, 0, *frame_size)
        geometries[camera] = make_geometry(crop, frame_size)
        logger.info(f'  {camera}: frame {frame_size[0]}x{frame_size[1]}, crop {crop}')
    return geometries


def render_video(job: VideoJob) -> Path:
    """Render every frame of the comparison video.

    Args:
        job: the render settings.

    Returns:
        path of the written mp4.

    Raises:
        FileNotFoundError: if a source video cannot be opened.
        ValueError: if the videos do not match the job (frame rate, length) or the cameras'
            arrays have a different number of frames than `job.n_frames`.
        RuntimeError: if a source frame cannot be read, ffmpeg fails, or the written video does
            not have `job.n_frames` frames.
    """
    cv2.setNumThreads(1)
    cameras = list(job.data)
    for camera, data in job.data.items():
        if data.n_frames != job.n_frames or data.start_frame != job.start_frame:
            raise ValueError(f'{camera} data does not cover frames of the job')
    captures, fps = _open_captures(job)
    geometries = _resolve_geometries(job, captures)

    landmarks = job.data[cameras[0]].landmarks
    width = sum(geometries[camera].width + STRIP_WIDTH for camera in cameras)
    # the trace boxes of all cameras have the same size, whatever the panel widths above
    trace_widths = [width // len(cameras)] * len(cameras)
    trace_widths[-1] += width - sum(trace_widths)
    panel_height = PANEL_IMAGE_HEIGHT + PANEL_HEADER + PANEL_FOOTER
    overlay_height = BANNER_HEIGHT + len(ROWS) * panel_height
    height = overlay_height + trace_panel_height(len(landmarks))
    traces: list[TraceData] = [
        build_trace_data(
            camera, data.xy_median, data.xy_eks, data.likelihood, job.conf_threshold,
        )
        for camera, data in job.data.items()
    ]

    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    if job.save_diagnostics:
        pd.DataFrame(
            [row for trace in traces for row in compute_trace_diagnostics(trace, landmarks)],
        ).to_csv(job.out_path.with_suffix('.diagnostics.csv'), index=False)

    log_path = job.out_path.with_suffix('.ffmpeg.log')
    process, log_file = _open_encoder(job.out_path, (width, height), fps, log_path)
    logger.info(f'  rendering {job.n_frames} frames of {width}x{height} -> {job.out_path}')
    try:
        for frame_idx in range(job.n_frames):
            canvas = np.full((height, width, 3), CANVAS_BACKGROUND, np.uint8)
            draw_banner(canvas, job)
            col_x0 = 0
            for camera in cameras:
                ok, frame = captures[camera].read()
                if not ok:
                    raise RuntimeError(f'cannot read {camera} frame {job.start_frame + frame_idx}')
                geometry = geometries[camera]
                x0, y0, x1, y1 = geometry.crop
                image = cv2.resize(
                    frame[y0:y1, x0:x1],
                    (geometry.width, PANEL_IMAGE_HEIGHT),
                    interpolation=cv2.INTER_CUBIC,
                )
                for row_idx, row in enumerate(ROWS):
                    panel, strip = draw_row(
                        image, geometry, job.data[camera], row, frame_idx, job, fps,
                    )
                    y_row = BANNER_HEIGHT + row_idx * panel_height
                    canvas[y_row:y_row + panel_height, col_x0:col_x0 + geometry.width] = panel
                    canvas[
                        y_row:y_row + panel_height,
                        col_x0 + geometry.width:col_x0 + geometry.width + STRIP_WIDTH,
                    ] = strip
                col_x0 += geometry.width + STRIP_WIDTH
            canvas[overlay_height:] = render_trace_panel(
                traces, landmarks, frame_idx, width, trace_widths, job.trace_window, fps,
                job.start_frame,
            )
            process.stdin.write(canvas.tobytes())
            if frame_idx % LOG_EVERY_FRAMES == 0:
                logger.info(f'  rendered {frame_idx}/{job.n_frames}')
    finally:
        process.stdin.close()
        for cap in captures.values():
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
