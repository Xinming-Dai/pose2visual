"""Scrolling x/y coordinate traces drawn under the QC video.

One box per keypoint and axis, for the ensemble median and Pose-EKS. Both traces use the same
confidence gate and are never bridged across a gated-out gap. The vertical scale of every box is
fixed over the whole clip so that motion is comparable between frames.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pose2visual.qc.video.style import put_text

METHOD_NAMES = ('ensemble median', 'Pose-EKS')
# BGR; the Pose-EKS line is drawn thicker
METHOD_COLORS = ((0, 235, 245), (75, 245, 90))
BACKGROUND = 18
HEADER_HEIGHT = 100
ROW_HEIGHT = 152
FOOTER_HEIGHT = 16
MIN_PAD_PX = 0.4
MIN_RANGE_PX = 2.0


@dataclass
class TraceData:
    """Everything the trace panel needs for one camera.

    Attributes:
        camera: camera name ('left' or 'right').
        values: positions of both methods (ensemble median, Pose-EKS), shape
            (2, n_frames, n_keypoints, 2).
        gate: frames drawn per keypoint, shape (n_frames, n_keypoints).
        limits: fixed vertical axis limits `(low, high)`, shape (n_keypoints, 2 axes, 2).
    """

    camera: str
    values: np.ndarray
    gate: np.ndarray
    limits: np.ndarray


def trace_panel_height(n_keypoints: int) -> int:
    """Return the height in pixels of the trace panel for `n_keypoints` rows."""
    return HEADER_HEIGHT + n_keypoints * ROW_HEIGHT + FOOTER_HEIGHT


def gate_mask(likelihood: np.ndarray, threshold: float) -> np.ndarray:
    """Mark the frames whose likelihood exceeds `threshold`.

    Args:
        likelihood: mean ensemble likelihood, any shape.
        threshold: gate; frames at or below it (and NaN frames) are not drawn.

    Returns:
        boolean array of the same shape as `likelihood`.
    """
    with np.errstate(invalid='ignore'):
        return likelihood > threshold


def compute_trace_limits(values: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Compute fixed vertical limits for every keypoint and axis.

    The limits cover the gated values of both methods, padded by 12% of the range (at least
    `MIN_PAD_PX`); a range under `MIN_RANGE_PX` is widened to that size around its midpoint.
    A keypoint that is never gated falls back to its ungated range, and to (0, 1) if that is
    all NaN too.

    Args:
        values: positions of both methods, shape (n_methods, n_frames, n_keypoints, 2).
        gate: shape (n_frames, n_keypoints).

    Returns:
        array of shape (n_keypoints, 2, 2) holding `(low, high)` per keypoint and axis.
    """
    n_keypoints = values.shape[2]
    limits = np.zeros((n_keypoints, 2, 2))
    for kp_idx in range(n_keypoints):
        for axis in range(2):
            series = values[:, :, kp_idx, axis]
            valid = np.where(gate[None, :, kp_idx], series, np.nan)
            if np.isnan(valid).all():
                valid = series
            if np.isnan(valid).all():
                limits[kp_idx, axis] = (0.0, 1.0)
                continue
            low, high = np.nanmin(valid), np.nanmax(valid)
            pad = max(MIN_PAD_PX, (high - low) * 0.12)
            if high - low < MIN_RANGE_PX:
                mid = (high + low) / 2
                low, high = mid - MIN_RANGE_PX / 2, mid + MIN_RANGE_PX / 2
            limits[kp_idx, axis] = (low - pad, high + pad)
    return limits


def build_trace_data(
    camera: str,
    xy_median: np.ndarray,
    xy_eks: np.ndarray,
    likelihood: np.ndarray,
    conf_threshold: float,
) -> TraceData:
    """Assemble the trace inputs of one camera.

    Args:
        camera: camera name.
        xy_median: ensemble median, shape (n_frames, n_keypoints, 2).
        xy_eks: Pose-EKS position, same shape.
        likelihood: mean ensemble likelihood, shape (n_frames, n_keypoints).
        conf_threshold: likelihood gate.

    Returns:
        the `TraceData` with fixed limits computed over the whole clip.
    """
    values = np.stack([xy_median, xy_eks])
    gate = gate_mask(likelihood, conf_threshold)
    return TraceData(camera, values, gate, compute_trace_limits(values, gate))


def _polyline_groups(indices: np.ndarray) -> list[np.ndarray]:
    """Split sorted frame indices into runs of consecutive frames."""
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def render_trace_panel(
    traces: list[TraceData],
    landmarks: list[str],
    frame_idx: int,
    width: int,
    column_widths: list[int],
    window: int,
    fps: float,
    start_frame: int,
) -> np.ndarray:
    """Draw the trailing trace window ending at one frame.

    Args:
        traces: one entry per camera, drawn left to right.
        landmarks: label of each keypoint (shared by all cameras), one box row per label.
        frame_idx: current frame relative to the first frame of the clip.
        width: panel width in pixels.
        column_widths: pixel width of each camera's column; sums to `width`.
        window: number of trailing frames shown.
        fps: video frame rate, for the time labels.
        start_frame: video frame index of the first clip frame, so labels show video time.

    Returns:
        BGR image of shape (`trace_panel_height(len(landmarks))`, width, 3).
    """
    panel = np.full((trace_panel_height(len(landmarks)), width, 3), BACKGROUND, np.uint8)
    put_text(
        panel,
        (12, 21),
        f'COORDINATE TRACES | trailing {window / fps:.1f} seconds | raw image pixels | '
        'fixed vertical scales | white line = current frame',
        0.49,
    )
    x = 12
    for name, color in zip(METHOD_NAMES, METHOD_COLORS):
        cv2.line(panel, (x, 42), (x + 25, 42), color, 2, cv2.LINE_AA)
        put_text(panel, (x + 31, 47), name, 0.46, color)
        x += 205
    put_text(
        panel,
        (12, 67),
        'Both traces share the likelihood gate. Gaps are hidden. Smoothness is not accuracy.',
        0.43,
    )

    col_x0 = 0
    for trace, col_width in zip(traces, column_widths):
        put_text(panel, (col_x0 + 12, 91), f'{trace.camera.upper()} CAMERA', 0.5)
        box_width = col_width // 2
        first = max(0, frame_idx - window)
        frames = np.arange(first, frame_idx + 1)
        for kp_idx, landmark in enumerate(landmarks):
            for axis in range(2):
                x0 = col_x0 + axis * box_width
                y0 = HEADER_HEIGHT + kp_idx * ROW_HEIGHT
                left, right = x0 + 55, x0 + box_width - 13
                top, bottom = y0 + 22, y0 + ROW_HEIGHT - 29
                low, high = trace.limits[kp_idx, axis]
                put_text(panel, (left, y0 + 14), f'{landmark}: {"xy"[axis]} (px)', 0.44)
                for value in np.linspace(low, high, 3):
                    y_line = round(bottom - (value - low) / (high - low) * (bottom - top))
                    cv2.line(panel, (left, y_line), (right, y_line), (48, 48, 48), 1)
                    put_text(panel, (x0 + 3, y_line + 4), f'{value:.1f}', 0.34)

                x_pts = left + (frames - (frame_idx - window)) / window * (right - left)
                for method_idx, color in enumerate(METHOD_COLORS):
                    series = trace.values[method_idx, first:frame_idx + 1, kp_idx, axis]
                    y_pts = bottom - (series - low) / (high - low) * (bottom - top)
                    shown = trace.gate[first:frame_idx + 1, kp_idx] & np.isfinite(y_pts)
                    pts = np.rint(np.column_stack([x_pts, y_pts])).astype(np.int32, copy=False)
                    # draw each run of gated frames separately: never bridge a confidence gap
                    for run in _polyline_groups(np.flatnonzero(shown)):
                        if len(run) > 1:
                            thickness = 2 if method_idx == 1 else 1
                            cv2.polylines(panel, [pts[run]], False, color, thickness, cv2.LINE_AA)
                        elif len(run) == 1:
                            cv2.circle(panel, tuple(pts[run[0]]), 1, color, -1)
                cv2.line(panel, (right, top), (right, bottom), (225, 225, 225), 1)
                for frac in (0, 0.5, 1):
                    tick = left + round(frac * (right - left))
                    seconds = (start_frame + frame_idx - window + frac * window) / fps
                    put_text(panel, (tick - 15, bottom + 17), f'{seconds:.1f}s', 0.34)
        col_x0 += col_width
    return panel


def compute_trace_diagnostics(trace: TraceData, landmarks: list[str]) -> list[dict]:
    """Summarise smoothness and agreement of the two methods per keypoint and axis.

    Args:
        trace: one camera's trace data.
        landmarks: label of each keypoint.

    Returns:
        one record per keypoint, axis and method with `second_difference_rms_px` (roughness over
        gated runs of three consecutive frames), `median_abs_difference_from_ensemble_px` and the
        gated frame and triplet counts.
    """
    rows = []
    gate = trace.gate
    for kp_idx, landmark in enumerate(landmarks):
        common = gate[:, kp_idx]
        triplets = common[:-2] & common[1:-1] & common[2:]
        for axis in range(2):
            median = trace.values[0, :, kp_idx, axis]
            for method_idx, method in enumerate(METHOD_NAMES):
                series = trace.values[method_idx, :, kp_idx, axis]
                second_diff = np.diff(series, n=2)[triplets]
                rows.append({
                    'camera': trace.camera,
                    'keypoint': landmark,
                    'axis': 'xy'[axis],
                    'method': method,
                    'second_difference_rms_px': (
                        float(np.sqrt(np.mean(second_diff ** 2))) if len(second_diff) else np.nan
                    ),
                    'median_abs_difference_from_ensemble_px': (
                        float(np.median(np.abs(series[common] - median[common])))
                        if common.any() else np.nan
                    ),
                    'gated_frames': int(common.sum()),
                    'gated_triplets': int(triplets.sum()),
                })
    return rows
