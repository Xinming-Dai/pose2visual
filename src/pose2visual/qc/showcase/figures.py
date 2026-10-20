"""Matplotlib figures of the showcase: labeled frames, and frames with keypoint traces.

Every keypoint has one colour for the whole session, taken from a matplotlib colormap by its
position in the csv header (LightningPose labeled videos do the same), so it keeps its colour in
both cameras, in both figures and whichever other keypoints are filtered out. Only keypoints whose
likelihood exceeds the gate at the anchor frame are drawn, and in the traces the frames at or below
the gate are left blank (never bridged). The figures are built on `matplotlib.figure.Figure`
directly, so no pyplot backend is involved.
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pose2visual.qc.video.loading import CameraData
from pose2visual.qc.video.trace_panels import compute_trace_limits, gate_mask

FIGURE_WIDTH = 16.0
TRACE_ROW_HEIGHT = 0.85
ANCHOR_MARKER_SIZE = 9
DIGIT_TIP = re.compile(r'^d\d+_tip_(left|right)$')
TEXT_COLOR = '0.15'
GRID_COLOR = '0.85'


def keypoint_colors(n_keypoints: int, colormap: str = 'cool') -> np.ndarray:
    """Give every keypoint its own colour, evenly spaced along a colormap.

    Args:
        n_keypoints: number of keypoints of the csv.
        colormap: name of a matplotlib colormap.

    Returns:
        RGBA array of shape (n_keypoints, 4); row `k` is the colour of the `k`-th keypoint.

    Raises:
        ValueError: if `colormap` is not a matplotlib colormap.
    """
    if colormap not in matplotlib.colormaps:
        raise ValueError(f'unknown colormap {colormap!r}')
    return matplotlib.colormaps[colormap].resampled(max(n_keypoints, 1))(np.arange(n_keypoints))


def is_far_digit(name: str, camera: str) -> bool:
    """Tell whether a keypoint is a digit tip of the paw on the far side of a camera.

    A camera sees the paw on its own side closely, while the digit tips of the other paw are
    occluded yet still predicted (on the nearer paw), so they are not trustworthy.

    Args:
        name: keypoint name as it appears in the csv, e.g. `d2_tip_right`.
        camera: camera name ('left' or 'right').

    Returns:
        True for a digit tip (`d<n>_tip_left` or `d<n>_tip_right`) whose side is not `camera`.
    """
    match = DIGIT_TIP.match(name)
    return match is not None and match.group(1) != camera


def visible_keypoints(
    data: CameraData,
    frame_idx: int,
    conf_threshold: float,
    keep_far_digits: bool = False,
) -> np.ndarray:
    """Find the keypoints to show at one frame.

    Args:
        data: Pose-EKS arrays of one camera.
        frame_idx: row of `data` to look at.
        conf_threshold: a keypoint is visible when its likelihood exceeds this.
        keep_far_digits: keep the digit tips of the paw on the far side of the camera, which are
            dropped by default (see `is_far_digit`).

    Returns:
        indices of the visible keypoints with a finite position, in csv header order.
    """
    visible = gate_mask(data.likelihood[frame_idx], conf_threshold)
    visible &= np.isfinite(data.xy_eks[frame_idx]).all(axis=1)
    if not keep_far_digits:
        visible &= ~np.array([is_far_digit(name, data.camera) for name in data.columns])
    return np.flatnonzero(visible)


def plot_frame_overlay(
    ax: Axes,
    image: np.ndarray,
    data: CameraData,
    frame_idx: int,
    colors: np.ndarray,
    conf_threshold: float,
    dot_size: float,
    keep_far_digits: bool = False,
) -> np.ndarray:
    """Draw one camera frame with its keypoints above the likelihood gate.

    Args:
        ax: axes to draw on.
        image: RGB frame, shape (height, width, 3).
        data: Pose-EKS arrays of this camera.
        frame_idx: row of `data` that belongs to `image`.
        colors: colour of every keypoint, shape (n_keypoints, 4).
        conf_threshold: a keypoint is drawn when its likelihood exceeds this.
        dot_size: marker diameter in points.
        keep_far_digits: also draw the far-paw digit tips (see `visible_keypoints`).

    Returns:
        indices of the drawn keypoints.
    """
    xy = data.xy_eks[frame_idx]
    drawn = visible_keypoints(data, frame_idx, conf_threshold, keep_far_digits)
    ax.imshow(image)
    ax.scatter(
        xy[drawn, 0],
        xy[drawn, 1],
        s=dot_size ** 2,
        c=colors[drawn],
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
    )
    # scatter autoscales the axes; keep the frame extent when a keypoint lies outside it
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)
    ax.set_axis_off()
    video_frame = data.start_frame + frame_idx
    ax.set_title(
        f'{data.camera} camera | frame {video_frame} | {len(drawn)} keypoints '
        f'(likelihood > {conf_threshold:g})',
        fontsize=10,
        color=TEXT_COLOR,
    )
    return drawn


LEGEND_COLUMNS = 6
LEGEND_ROW_HEIGHT = 0.15


def _legend_height(n_names: int) -> float:
    """Height in inches of a legend of `n_names` entries in `LEGEND_COLUMNS` columns."""
    return LEGEND_ROW_HEIGHT * -(-max(n_names, 1) // LEGEND_COLUMNS) + 0.1


def _add_legend(ax: Axes, names: list[str], colors: np.ndarray) -> None:
    """Fill an otherwise empty axes with the colour legend of the drawn keypoints."""
    handles = [
        Line2D([], [], marker='o', linestyle='', markersize=5, color=color, mec='black', mew=0.3)
        for color in colors
    ]
    ax.set_axis_off()
    ax.legend(
        handles,
        names,
        loc='upper center',
        ncol=LEGEND_COLUMNS,
        fontsize=6,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0,
    )


def _image_height(images: dict[str, np.ndarray]) -> float:
    """Height in inches of one image drawn half the figure width wide."""
    height, width = next(iter(images.values())).shape[:2]
    return FIGURE_WIDTH / len(images) * height / width


def build_frame_figure(
    images: dict[str, np.ndarray],
    data: dict[str, CameraData],
    frame_idx: int,
    colors: np.ndarray,
    conf_threshold: float,
    dot_size: float,
    keep_far_digits: bool = False,
) -> Figure:
    """Build the labeled frames of all cameras side by side.

    Args:
        images: camera name -> RGB frame, drawn left to right in this order.
        data: camera name -> Pose-EKS arrays.
        frame_idx: row of the arrays that belongs to the frames.
        colors: colour of every keypoint, shape (n_keypoints, 4).
        conf_threshold: likelihood gate.
        dot_size: marker diameter in points.
        keep_far_digits: also draw the far-paw digit tips (see `visible_keypoints`).

    Returns:
        the figure, with a colour legend of the drawn keypoints under each frame.
    """
    cameras = list(images)
    drawn = {
        camera: visible_keypoints(data[camera], frame_idx, conf_threshold, keep_far_digits)
        for camera in cameras
    }
    image_height = _image_height(images)
    legend_height = _legend_height(max(len(idx) for idx in drawn.values()))
    fig = Figure(figsize=(FIGURE_WIDTH, image_height + legend_height + 0.5), layout='constrained')
    grid = fig.add_gridspec(2, len(cameras), height_ratios=[image_height, legend_height])
    for cam_idx, camera in enumerate(cameras):
        plot_frame_overlay(
            fig.add_subplot(grid[0, cam_idx]), images[camera], data[camera], frame_idx, colors,
            conf_threshold, dot_size, keep_far_digits,
        )
        names = [data[camera].columns[idx] for idx in drawn[camera]]
        _add_legend(fig.add_subplot(grid[1, cam_idx]), names, colors[drawn[camera]])
    return fig


def _plot_trace(
    ax: Axes,
    data: CameraData,
    kp_idx: int,
    axis: int,
    frame_idx: int,
    fps: float,
    color: np.ndarray,
    limits: np.ndarray,
    conf_threshold: float,
) -> None:
    """Draw the gated trace of one keypoint and axis, with the anchor as a large dot."""
    time = (np.arange(data.n_frames) - frame_idx) / fps
    series = data.xy_eks[:, kp_idx, axis]
    gate = gate_mask(data.likelihood[:, kp_idx], conf_threshold)
    # nan breaks the line, so a gated-out gap is never bridged
    shown = np.where(gate & np.isfinite(series), series, np.nan)
    ax.axvline(0, color='0.6', linewidth=0.6, zorder=1)
    ax.plot(time, shown, color=color, linewidth=1.2, marker='.', markersize=1.5, zorder=2)
    ax.plot(
        [0],
        [series[frame_idx]],
        marker='o',
        markersize=ANCHOR_MARKER_SIZE,
        color=color,
        markeredgecolor='black',
        markeredgewidth=1.0,
        linestyle='',
        zorder=4,
    )
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(*limits[kp_idx, axis])
    ax.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.text(
        0.01, 0.97, f'{data.columns[kp_idx]}: {"xy"[axis]} (px)', transform=ax.transAxes,
        fontsize=6, color=TEXT_COLOR, va='top', ha='left',
    )


def build_trace_figure(
    images: dict[str, np.ndarray],
    data: dict[str, CameraData],
    frame_idx: int,
    fps: float,
    colors: np.ndarray,
    conf_threshold: float,
    dot_size: float,
    keep_far_digits: bool = False,
) -> Figure:
    """Build the labeled frames with the x/y traces of their keypoints underneath.

    The top row holds the frames (first camera on the left half). Below, each camera has one row
    per keypoint drawn on its frame and two columns (x, then y): the trace over the whole window
    in the keypoint's colour, a large dot at the anchor and a vertical line at time 0. Vertical
    limits are fixed over the window.

    Args:
        images: camera name -> RGB frame, drawn left to right in this order.
        data: camera name -> Pose-EKS arrays covering the window; row `frame_idx` is the anchor.
        frame_idx: row of the arrays that belongs to the frames.
        fps: video frame rate, for the time axis.
        colors: colour of every keypoint, shape (n_keypoints, 4).
        conf_threshold: likelihood gate.
        dot_size: marker diameter of the frame overlays in points.
        keep_far_digits: also show the far-paw digit tips (see `visible_keypoints`).

    Returns:
        the figure.
    """
    cameras = list(images)
    rows = {
        camera: visible_keypoints(data[camera], frame_idx, conf_threshold, keep_far_digits)
        for camera in cameras
    }
    limits = {
        camera: compute_trace_limits(
            data[camera].xy_eks[None],
            gate_mask(data[camera].likelihood, conf_threshold),
        )
        for camera in cameras
    }
    n_rows = max(len(idx) for idx in rows.values())
    image_height = _image_height(images)
    legend_height = _legend_height(n_rows)
    fig = Figure(
        figsize=(FIGURE_WIDTH, image_height + legend_height + 0.6 + n_rows * TRACE_ROW_HEIGHT),
        layout='constrained',
    )
    grid = fig.add_gridspec(
        2 + n_rows,
        2 * len(cameras),
        height_ratios=[image_height, legend_height, *[TRACE_ROW_HEIGHT] * n_rows],
    )
    for cam_idx, camera in enumerate(cameras):
        ax_image = fig.add_subplot(grid[0, 2 * cam_idx:2 * cam_idx + 2])
        plot_frame_overlay(
            ax_image, images[camera], data[camera], frame_idx, colors, conf_threshold, dot_size,
            keep_far_digits,
        )
        _add_legend(
            fig.add_subplot(grid[1, 2 * cam_idx:2 * cam_idx + 2]),
            [data[camera].columns[idx] for idx in rows[camera]],
            colors[rows[camera]],
        )
        for row_idx, kp_idx in enumerate(rows[camera]):
            for axis in range(2):
                ax = fig.add_subplot(grid[2 + row_idx, 2 * cam_idx + axis])
                _plot_trace(
                    ax, data[camera], kp_idx, axis, frame_idx, fps, colors[kp_idx],
                    limits[camera], conf_threshold,
                )
                if row_idx == len(rows[camera]) - 1:
                    ax.set_xlabel('time from anchor (s)', fontsize=6, labelpad=1)
                else:
                    ax.tick_params(labelbottom=False)
    return fig


def save_figure(fig: Figure, stem: Path, dpi: int = 200) -> list[Path]:
    """Save a figure as png and pdf.

    Args:
        fig: the figure.
        stem: output path without suffix; its folder is created.
        dpi: resolution of the png.

    Returns:
        the written paths, `<stem>.png` then `<stem>.pdf`.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    # a dot in the stem (the session id is separated from other parts by dots) is not a suffix
    paths = [stem.parent / f'{stem.name}.png', stem.parent / f'{stem.name}.pdf']
    for path in paths:
        fig.savefig(path, dpi=dpi)
    return paths
