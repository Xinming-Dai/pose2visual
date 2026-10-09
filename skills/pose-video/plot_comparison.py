#!/usr/bin/env python3
"""Rainbow scatter plots of pose predictions across network seeds, by body-part group.

Visualizes prediction distributions by scattering all frames' predictions (5,000
by default) on a single static background frame. Each dot is one frame's (x, y)
for one landmark from one network. Compares: single-network seeds, their per-frame
ensemble median, and optionally Pose-EKS smoothed output. Renders both ungated (all frames) 
and gated (high-confidence frames per landmark).

Body parts are grouped per a `body_parts.json`-style file (default:
`scripts/delta/data/body_parts.json`'s `nonoverlapping_analysis_groups`); every group
is rendered as its own row of panels (one column per method: seeds, ensemble median,
optional Pose-EKS), stacked vertically into one combined figure per camera per
gated/ungated mode. Real keypoint column names (and whether each is bilateral or
midline) are discovered from the prediction csv header at runtime, so any body part
works without a hardcoded sub-part list.

Auto-discovers video from CSV filename (IBL-2view convention) and computes
output directory via sha256 hash of inputs for reproducibility.

`--model-dir`/`--seed-dirs` default to `DEFAULT_MODEL_DIR`/`DEFAULT_SEED_DIRS` near the
top of this file — edit those constants to your current working model instead of
retyping the flags every run (set either to `None` to make the flag required again).

`--session-group <name>` runs every session in that named group from `--sessions-file`
(default `scripts/delta/data/ibl_session_ids.json`, e.g. its `"sable"`/`"all"` groups
under `"neural_sessions"`) in one invocation, sharing one output run folder. Use
`--sessions <uuid> [<uuid> ...]` instead to name sessions explicitly.

Usage:
    python plot_comparison.py --session-group sable

    python plot_comparison.py \\
        --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
        --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
        --sessions 9b528ad0-4599-4a55-9148-96cc1d93fb24

    python plot_comparison.py \\
        --model-dir /path/to/models \\
        --seed-dirs seed0 seed1 seed2 \\
        --sessions <uuid1> <uuid2> \\
        --cameras left right \\
        --n-frames 5000 --conf-threshold 0.7 \\
        --eks-npz-dir /path/to/eks/<run> \\
        --group-set overlapping_analysis_groups \\
        --groups eye_related mouth_related \\
        --save-ensemble-median \\
        --tag my-qc

Default output: <model_dir>/rainbow_qc/<timestamp>[__<tag>]/{figures,csv,qc_manifest.json}
Folder name is `YYYYMMDD_HHMMSS`, suffixed with `__<tag>` when `--tag` is given. 
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

# Run-folder timestamps use this timezone, not the cluster's local time.
RUN_TIMEZONE = ZoneInfo('America/New_York')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_pose2visual_src = Path(__file__).parent.parent.parent / 'src'
if _pose2visual_src.exists() and str(_pose2visual_src) not in sys.path:
    sys.path.insert(0, str(_pose2visual_src))

from pose2visual.lightningpose.ensemble_median import (
    compute_ensemble_median,
    write_ensemble_median_csv,
)
from pose2visual.lightningpose.fit_eks_latent_model import (
    discover_keypoint_names,
    group_columns_for_camera,
    load_groups,
    load_latent_model,
    resolve_group_bases,
)
from pose2visual.preprocessing import resolve_video_path
from pose2visual.run_registry import append_entry

LANDMARK_COLOR_PALETTE = [
    '#00d9ff',  # cyan
    '#ff9900',  # orange
    '#ff3cdf',  # pink
    '#6aff46',  # green
    '#b388ff',  # purple
    '#ffe135',  # yellow
    '#ff5252',  # red-orange
    '#4fc3f7',  # blue
    '#d4ff00',  # chartreuse
    '#ff8a65',  # salmon
]

DEFAULT_GROUPS_FILE = _pose2visual_src.parent / 'scripts' / 'delta' / 'data' / 'body_parts.json'
DEFAULT_SESSIONS_FILE = _pose2visual_src.parent / 'scripts' / 'delta' / 'data' / 'ibl_session_ids.json'
DEFAULT_MODEL_DIR = (
    '/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/'
    'E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train'
)
DEFAULT_SEED_DIRS = ['vits_dinov3-seed0', 'vits_dinov3-seed1', 'vits_dinov3-seed2']
DEFAULT_ENSEMBLE_MEDIAN_DIR = None
# DEFAULT_ENSEMBLE_MEDIAN_DIR = "/work/hdd/bfsr/xdai3/pose2visual/mighty-mouse-models/E6c-LOO-ibl-headmlp256_3seeds/face+cheese+caz+kondo+hmv+c3d+kauf_train/rainbow_qc/20261009_154556/csv"


def colors_for_landmarks(landmarks: list[str]) -> list[str]:
    """Assign a saturated colour to each landmark, cycling the shared palette.

    Args:
        landmarks: landmark names, in display order.

    Returns:
        one hex colour per landmark, same length and order as `landmarks`.
    """
    return [LANDMARK_COLOR_PALETTE[i % len(LANDMARK_COLOR_PALETTE)] for i in range(len(landmarks))]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the dejavu-sans-bold font, falling back to PIL's default.

    Args:
        size: font size in pixels.

    Returns:
        a PIL font object.
    """
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_title_canvas(width: int, title_text: str, canvas_height: int = 120) -> Image.Image:
    """Create a black title canvas sized to stack above the plot image.

    Args:
        width: canvas width, matching the plot image's width.
        title_text: title text (can be multi-line).
        canvas_height: height of the canvas in pixels.

    Returns:
        PIL Image of the title banner alone.
    """
    title_img = Image.new('RGB', (width, canvas_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(title_img)
    font = _load_font(24)

    lines = title_text.split('\n')
    line_height = 30
    total_height = len(lines) * line_height
    y_start = (canvas_height - total_height) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(255, 255, 255), font=font)

    return title_img


def render_legend_canvas(
    rows_legend: list[dict[str, Any]],
    width: int,
    line_height: int = 36,
) -> Image.Image:
    """Create a black legend canvas, one line per group/row, to stack below the figure.

    Each line lists that row's landmarks with their colour and kept/total frame count,
    e.g. ``eye_related:  eye: 10/20   pupil: 15/20``, so the per-landmark legend no
    longer has to be drawn on top of the scatter image.

    Args:
        rows_legend: one entry per row, in the same top-to-bottom order the rows appear
            in the figure: `{'group_name', 'landmarks', 'colors', 'counts', 'n_frames'}`
            (see `render_combined_figure`).
        width: canvas width, matching the plot image's width.
        line_height: pixel height of one row's legend line.

    Returns:
        PIL Image of the legend banner alone.
    """
    padding = 10
    height = padding * 2 + line_height * len(rows_legend)
    canvas = Image.new('RGB', (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(16)
    entry_font = _load_font(14)

    for row_idx, row in enumerate(rows_legend):
        y = padding + row_idx * line_height + (line_height - 16) // 2
        x = 12

        label = f'{row["group_name"]}:  '
        draw.text((x, y), label, fill=(255, 255, 255), font=label_font)
        x += draw.textbbox((0, 0), label, font=label_font)[2]

        for landmark, color, count in zip(row['landmarks'], row['colors'], row['counts']):
            entry = f'{landmark}: {count:,}/{row["n_frames"]:,}   '
            draw.text((x, y + 1), entry, fill=color, font=entry_font)
            x += draw.textbbox((0, 0), entry, font=entry_font)[2]

    return canvas


def compose_figure_canvas(
    image_path: Path,
    title_text: str,
    rows_legend: list[dict[str, Any]],
    canvas_height: int = 120,
    legend_line_height: int = 36,
) -> Image.Image:
    """Stack a title banner above and a legend banner below the saved plot image.

    Args:
        image_path: path to the plot image (PNG).
        title_text: title text (can be multi-line), drawn in the top banner.
        rows_legend: per-row legend entries for the bottom banner (see
            `render_legend_canvas`); panels themselves carry no landmark legend/counts.
        canvas_height: height of the title banner in pixels.
        legend_line_height: pixel height of one row's legend line in the bottom banner.

    Returns:
        Combined PIL Image: title canvas, plot image, legend canvas, top to bottom.
    """
    plot_img = Image.open(image_path)
    width, plot_height = plot_img.size

    title_img = render_title_canvas(width, title_text, canvas_height)
    legend_img = render_legend_canvas(rows_legend, width, legend_line_height)

    combined_height = title_img.height + plot_height + legend_img.height
    combined_img = Image.new('RGB', (width, combined_height))
    combined_img.paste(title_img, (0, 0))
    combined_img.paste(plot_img, (0, title_img.height))
    combined_img.paste(legend_img, (0, title_img.height + plot_height))

    return combined_img


def find_video_preds_dir(seed_path: Path) -> Path:
    """Find the video_preds directory under a seed folder.

    Searches recursively for a `video_preds` directory. If the path already
    points to `video_preds`, returns it as is.

    Args:
        seed_path: path to seed folder or video_preds directory.

    Returns:
        Absolute path to video_preds directory.

    Raises:
        FileNotFoundError: if no video_preds directory is found.
        ValueError: if more than one video_preds directory is found.
    """
    seed_path = Path(seed_path).resolve()

    # If already pointing to video_preds, return it
    if seed_path.name == 'video_preds':
        return seed_path

    # Search recursively
    candidates = list(seed_path.rglob('video_preds'))
    candidates = [c for c in candidates if c.is_dir()]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f'No video_preds directory found under {seed_path}'
        )
    if len(candidates) > 1:
        raise ValueError(
            f'Multiple video_preds directories found under {seed_path}:\n'
            + '\n'.join(str(c) for c in candidates)
        )

    return candidates[0]


def find_seed_csv(
    preds_dir: Path,
    session_id: str,
    camera: str,
) -> Path:
    """Find the predictions CSV for a seed and camera.

    Args:
        preds_dir: path to video_preds directory.
        session_id: session UUID.
        camera: camera name ('left' or 'right').

    Returns:
        Path to the CSV.

    Raises:
        FileNotFoundError: if CSV does not exist.
    """
    csv_name = f'_iblrig_{camera}Camera.downsampled.{session_id}.csv'
    csv_path = preds_dir / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    return csv_path


def _count_csv_rows(csv_path: Path) -> int:
    """Number of data rows in a 3-level-header DLC-style csv.

    Args:
        csv_path: path to the csv.

    Returns:
        number of data rows (excludes the 3 header rows).
    """
    with open(csv_path) as f:
        return sum(1 for _ in f) - 3


def load_all_keypoints(
    csv_path: Path,
    keypoints: list[str],
    start_frame: int,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (x, y, likelihood) for every keypoint from one seed's prediction csv.

    Args:
        csv_path: path to predictions CSV.
        keypoints: keypoint names to load (as they appear in the csv header),
            discovered once via `discover_keypoint_names`.
        start_frame: starting frame index.
        n_frames: number of frames to load.

    Returns:
        Tuple of (xy_coords, likelihoods), shapes (n_frames, n_keypoints, 2) and
        (n_frames, n_keypoints).
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)

    xy_list = []
    conf_list = []
    for key in keypoints:
        xy_list.append(df[key][['x', 'y']].values)
        conf_list.append(df[key]['likelihood'].values)

    xy = np.stack(xy_list, axis=1)  # (n_total, n_keypoints, 2)
    conf = np.stack(conf_list, axis=1)  # (n_total, n_keypoints)

    end_frame = min(start_frame + n_frames, len(xy))
    xy = xy[start_frame:end_frame]
    conf = conf[start_frame:end_frame]

    # Pad if fewer frames than requested
    if len(xy) < n_frames:
        pad_width = ((0, n_frames - len(xy)), (0, 0), (0, 0))
        xy = np.pad(xy, pad_width, constant_values=np.nan)
        pad_width = ((0, n_frames - len(conf)), (0, 0))
        conf = np.pad(conf, pad_width, constant_values=np.nan)

    return xy, conf


def compute_statistics(
    xy: np.ndarray,
    keep: np.ndarray,
    landmarks: list[str],
    roi: Optional[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Compute per-keypoint statistics.

    Args:
        xy: prediction array (n_frames, n_landmarks, 2).
        keep: boolean mask (n_frames, n_landmarks) indicating valid frames.
        landmarks: landmark names.
        roi: optional ROI (x0, y0, x1, y1) for coverage analysis.

    Returns:
        Dictionary {landmark_name: stats_dict}.
    """
    stats_dict = {}

    for k, landmark in enumerate(landmarks):
        z = xy[:, k, :][keep[:, k]]  # (n_kept, 2)

        if len(z) == 0:
            continue

        center = np.median(z, axis=0)
        radius = np.linalg.norm(z - center, axis=1)

        stat_entry = {
            'n': len(z),
            'median_x': float(center[0]),
            'median_y': float(center[1]),
            'radius_p95_px': float(np.quantile(radius, 0.95)),
        }

        if roi is not None:
            x0, y0, x1, y1 = roi
            outside = (z[:, 0] < x0) | (z[:, 0] >= x1) | (z[:, 1] < y0) | (
                z[:, 1] >= y1
            )
            stat_entry['outside_roi_count'] = int(outside.sum())
            stat_entry['outside_roi_fraction'] = float(outside.mean())

        stats_dict[landmark] = stat_entry

    return stats_dict


def get_background_frame(
    csv_path: Path,
    video_path: Path,
    frame_idx: int = 2500,
) -> np.ndarray:
    """Extract a single background frame from video.

    Args:
        csv_path: predictions CSV path (for error messages).
        video_path: path to video file.
        frame_idx: frame number to extract.

    Returns:
        RGB image array (H, W, 3).
    """
    if not video_path.exists():
        raise FileNotFoundError(f'Video not found: {video_path}')

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(f'Failed to read frame {frame_idx} from {video_path}')

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def render_group_row(
    axes_row: list[plt.Axes],
    xy_list: list[np.ndarray],
    confidences_list: list[np.ndarray],
    background_img: np.ndarray,
    landmarks: list[str],
    colors: list[str],
    camera: str,
    group_name: str,
    gated: bool,
    conf_threshold: float,
    method_names: list[str],
    roi: Optional[tuple[int, int, int, int]],
) -> tuple[list[dict[str, dict[str, Any]]], np.ndarray]:
    """Render one body-part group's row of panels (one per method).

    Args:
        axes_row: pre-created axes to draw into, one per entry in `method_names`
            (callers with fewer methods than the figure's column count pass only
            that many axes; the rest are left blank by the caller).
        xy_list: list of (n_frames, n_landmarks, 2) arrays, one per method.
        confidences_list: list of (n_frames, n_landmarks) arrays, one per seed (used
            for gating; does not include the ensemble median or EKS columns).
        background_img: background frame (H, W, 3).
        landmarks: landmark display names, matching `xy_list`'s landmark axis.
        colors: one colour per landmark, same order as `landmarks`.
        camera: camera name ('left' or 'right').
        group_name: body-part group name, shown in each panel's title.
        gated: whether to apply confidence gating.
        conf_threshold: confidence threshold for gating.
        method_names: names for each method (e.g. ['Seed 0', ..., 'Ensemble median']).
        roi: optional ROI (x0, y0, x1, y1) for statistics.

    Returns:
        Tuple of (stats_list, counts): `stats_list` is one per-method stats dict
        (`{landmark: stats}`), same order as `xy_list`; `counts` is the kept-frame count
        per landmark (shape `(n_landmarks,)`), identical across every method in this row
        since gating doesn't depend on the method — the caller uses it to build a legend
        instead of drawing per-panel text on the image.
    """
    n_frames = xy_list[0].shape[0]
    n_landmarks = len(landmarks)

    seeds_conf = np.stack(confidences_list, axis=0)  # (n_seeds, n_frames, n_landmarks)
    if gated:
        keep = (seeds_conf > conf_threshold).all(axis=0)
    else:
        keep = np.ones((n_frames, n_landmarks), dtype=bool)

    h, w = background_img.shape[:2]
    stats_list = []

    for m, (xy, method_name, ax) in enumerate(zip(xy_list, method_names, axes_row)):
        ax.imshow(background_img)

        xy_flat = xy.reshape(-1, 2)
        keep_flat = keep.ravel()
        pts = xy_flat[keep_flat]

        colors_array = np.tile(colors, n_frames)[keep_flat]

        order = np.random.default_rng(19).permutation(len(pts))
        ax.scatter(
            pts[order, 0],
            pts[order, 1],
            c=colors_array[order],
            s=1.4,
            alpha=0.18,
            linewidths=0,
        )

        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis('off')
        ax.set_title(
            f'{camera.capitalize()} | {group_name} | {method_name}',
            color='white',
            fontsize=11,
        )

        ax.plot([12, 32], [h - 13, h - 13], color='white', lw=2)
        ax.text(12, h - 7, '20 px', color='white', fontsize=8)

        stats_list.append(compute_statistics(xy, keep, landmarks, roi))

    counts = keep.sum(axis=0)
    return stats_list, counts


def render_combined_figure(
    rows_data: list[dict[str, Any]],
    background_img: np.ndarray,
    session_id: str,
    camera: str,
    gated: bool,
    conf_threshold: float,
    roi: Optional[tuple[int, int, int, int]],
    n_frames: int,
) -> tuple[plt.Figure, dict[str, list[dict[str, dict[str, Any]]]], str, list[dict[str, Any]]]:
    """Render every body-part group's row, stacked vertically into one figure.

    Args:
        rows_data: one entry per group, each a dict with `group_name`, `landmarks`,
            `colors`, `xy_list`, `confidences_list`, `method_names` (see
            `render_group_row`).
        background_img: background frame (H, W, 3).
        session_id: session UUID.
        camera: camera name ('left' or 'right').
        gated: whether to apply confidence gating.
        conf_threshold: confidence threshold for gating.
        roi: optional ROI (x0, y0, x1, y1) for statistics.
        n_frames: number of frames plotted (for the title).

    Returns:
        Tuple of (figure, {group_name: stats_list}, title_text, rows_legend).
        `rows_legend` is one entry per row, in the same order as `rows_data`:
        `{'group_name', 'landmarks', 'colors', 'counts', 'n_frames'}`, for the caller to
        render as a legend canvas below the figure instead of drawing it on the image.
    """
    n_rows = len(rows_data)
    n_cols = max(len(row['method_names']) for row in rows_data)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        facecolor='#141414',
        squeeze=False,
    )

    stats_by_group = {}
    rows_legend = []
    for row_idx, row in enumerate(rows_data):
        n_methods = len(row['method_names'])
        axes_row = [axes[row_idx, col_idx] for col_idx in range(n_methods)]
        for col_idx in range(n_methods, n_cols):
            axes[row_idx, col_idx].axis('off')

        stats_list, counts = render_group_row(
            axes_row,
            row['xy_list'],
            row['confidences_list'],
            background_img,
            row['landmarks'],
            row['colors'],
            camera,
            row['group_name'],
            gated,
            conf_threshold,
            row['method_names'],
            roi,
        )
        stats_by_group[row['group_name']] = stats_list
        rows_legend.append({
            'group_name': row['group_name'],
            'landmarks': row['landmarks'],
            'colors': row['colors'],
            'counts': counts,
            'n_frames': n_frames,
        })

    gate_mode = (
        f'Matched frames per landmark: ALL seeds likelihood >{conf_threshold}' if gated
        else f'All {n_frames:,} frames; no confidence filtering'
    )
    group_names = ', '.join(row['group_name'] for row in rows_data)
    title_text = (
        f'Session {session_id[:8]} | {group_names}, frames 0–{n_frames - 1}\n{gate_mode}'
    )

    fig.subplots_adjust(top=0.96, bottom=0.03, left=0.015, right=0.985, hspace=0.22, wspace=0.055)

    return fig, stats_by_group, title_text, rows_legend


def load_eks(
    eks_npz_dir: Path,
    camera: str,
    n_frames: int,
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Load pre-computed Pose-EKS latent-model output for one group and camera.

    Args:
        eks_npz_dir: directory containing `{camera}_latent-model.npz` files for one
            body-part group and session (i.e. `<eks out_root>/<session>/<group>`).
        camera: camera name ('left' or 'right').
        n_frames: expected frame count (for validation).

    Returns:
        Tuple of (eks_xy, model_dict), or None if npz not found or multiple found.
    """
    if not eks_npz_dir.is_dir():
        return None

    npz_files = list(eks_npz_dir.glob(f'*_{camera}_latent-model.npz'))

    if len(npz_files) == 0:
        return None
    if len(npz_files) > 1:
        raise ValueError(
            f'Multiple EKS npz files for {camera} in {eks_npz_dir}:\n'
            + '\n'.join(str(f) for f in npz_files)
        )

    npz_path = npz_files[0]
    loaded = load_latent_model(npz_path)
    if loaded['xy'].shape[0] != n_frames:
        raise ValueError(
            f'EKS npz {npz_path} has {loaded["xy"].shape[0]} frames, expected {n_frames}',
        )

    return loaded['xy'], np.load(npz_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Rainbow scatter plots of pose predictions across seeds, by body-part group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_comparison.py --session-group sable

  python plot_comparison.py \\
    --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
    --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
    --sessions 9b528ad0-4599-4a55-9148-96cc1d93fb24

  python plot_comparison.py \\
    --model-dir /path/to/models \\
    --seed-dirs seed0 seed1 seed2 \\
    --sessions <uuid1> <uuid2> \\
    --n-frames 5000 --conf-threshold 0.7 \\
    --eks-npz-dir /path/to/eks/<run> \\
    --groups eye_related ear_related
        ''',
    )

    parser.add_argument(
        '--model-dir',
        default=DEFAULT_MODEL_DIR,
        required=DEFAULT_MODEL_DIR is None,
        help=f'Root directory holding seed folders (default: {DEFAULT_MODEL_DIR})',
    )
    parser.add_argument(
        '--seed-dirs',
        nargs='+',
        default=DEFAULT_SEED_DIRS,
        required=DEFAULT_SEED_DIRS is None,
        help=f'Seed folder names (relative to model-dir) or absolute paths '
        f'(default: {DEFAULT_SEED_DIRS})',
    )
    session_group_arg = parser.add_mutually_exclusive_group(required=True)
    session_group_arg.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='Session IDs (UUIDs) to process together, sharing one output run folder',
    )
    session_group_arg.add_argument(
        '--session-group',
        default=None,
        help='Name of a session group to load from --sessions-file (e.g. "sable", "all")',
    )
    parser.add_argument(
        '--sessions-file',
        default=str(DEFAULT_SESSIONS_FILE),
        help=(
            'Path to an ibl_session_ids.json-style file ({"neural_sessions": {<group>: '
            f'[<uuid>, ...]}}), only consulted with --session-group (default: {DEFAULT_SESSIONS_FILE})'
        ),
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=['left', 'right'],
        help='Camera names to process (default: left right)',
    )
    parser.add_argument(
        '--video-left',
        default=None,
        help='Override video path for left camera (auto-discovered by default)',
    )
    parser.add_argument(
        '--video-right',
        default=None,
        help='Override video path for right camera (auto-discovered by default)',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory (default: <model-dir>/rainbow_qc/<timestamp>[__<tag>])',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional tag for output filenames',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=None,
        help='Number of frames to process (default: every frame available, from start-frame)',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Starting frame index (default: 0)',
    )
    parser.add_argument(
        '--background-frame',
        type=int,
        default=None,
        help='Background frame index (default: start_frame + n_frames//2)',
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for gating (default: 0.7)',
    )
    parser.add_argument(
        '--groups-file',
        default=str(DEFAULT_GROUPS_FILE),
        help=(
            "Path to a body_parts.json-style file "
            f'(default: {DEFAULT_GROUPS_FILE})'
        ),
    )
    parser.add_argument(
        '--group-set',
        choices=['nonoverlapping_analysis_groups', 'overlapping_analysis_groups'],
        default='nonoverlapping_analysis_groups',
        help='Top-level key to load from --groups-file (default: nonoverlapping_analysis_groups)',
    )
    parser.add_argument(
        '--groups',
        nargs='+',
        default=None,
        help='Subset of group names to render (default: every group in --group-set), one row '
        'per group, stacked vertically',
    )
    parser.add_argument(
        '--save-ensemble-median',
        action='store_true',
        help='Also write the ensemble median (every discovered keypoint, not just the plotted '
        'groups) to csv/ensemble_median_<camera>.csv (default: off)',
    )
    parser.add_argument(
        '--ensemble-median-dir',
        default=DEFAULT_ENSEMBLE_MEDIAN_DIR,
        help='Directory of already-saved ensemble_median_<session[:8]>_<camera>.csv files '
        '(as written by --save-ensemble-median) to reuse instead of recomputing from the '
        f'seeds, for any (session, camera) found there (default: {DEFAULT_ENSEMBLE_MEDIAN_DIR})',
    )
    parser.add_argument(
        '--eks-npz-dir',
        default=None,
        help="Out-root directory containing <session>/<group>/<group>_<camera>_latent-model.npz "
        "subdirectories, as written by fit_eks_latent_model.py (optional)",
    )
    parser.add_argument(
        '--roi-left',
        type=int,
        nargs=4,
        default=None,
        help='Left camera ROI (x0 y0 x1 y1) for statistics',
    )
    parser.add_argument(
        '--roi-right',
        type=int,
        nargs=4,
        default=None,
        help='Right camera ROI (x0 y0 x1 y1) for statistics',
    )

    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    """Main execution logic."""
    model_dir = Path(args.model_dir).resolve()
    start_frame = args.start_frame
    conf_threshold = args.conf_threshold
    tag = args.tag

    groups = load_groups(args.groups_file, group_set=args.group_set)
    if args.groups is not None:
        missing = sorted(set(args.groups) - set(groups))
        if missing:
            print(f'Error: groups not found in {args.groups_file}: {missing}', file=sys.stderr)
            sys.exit(1)
        groups = {name: groups[name] for name in args.groups}

    if args.sessions is not None:
        sessions = args.sessions
    else:
        session_groups = load_groups(args.sessions_file, group_set='neural_sessions')
        if args.session_group not in session_groups:
            print(
                f'Error: session group {args.session_group!r} not found in '
                f'{args.sessions_file}: {sorted(session_groups)}',
                file=sys.stderr,
            )
            sys.exit(1)
        sessions = session_groups[args.session_group]
    print(f'Sessions: {sessions}')

    # Resolve seed directories
    seed_dirs = {}
    for seed_name in args.seed_dirs:
        seed_path = Path(seed_name)
        if not seed_path.is_absolute():
            seed_path = model_dir / seed_name
        seed_path = seed_path.resolve()
        if not seed_path.exists():
            print(f'Error: seed directory not found: {seed_path}', file=sys.stderr)
            sys.exit(1)
        seed_dirs[seed_name] = seed_path

    eks_npz_dir = Path(args.eks_npz_dir).resolve() if args.eks_npz_dir else None

    run_name = datetime.now(RUN_TIMEZONE).strftime('%Y%m%d_%H%M%S')
    if tag:
        run_name = f'{run_name}__{tag}'

    # Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = model_dir / 'rainbow_qc' / run_name

    fig_dir = out_dir / 'figures'
    csv_dir = out_dir / 'csv'
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f'Output: {out_dir}')

    all_stats = []
    audit = {}
    sessions_manifest = {}

    for session_id in sessions:
        print(f'=== Session {session_id} ===')

        # Find video_preds directories and CSVs
        print('Resolving seed predictions...')
        csv_paths = {}  # {camera: {seed_name: path}}
        for camera in args.cameras:
            csv_paths[camera] = {}
            for seed_name, seed_path in seed_dirs.items():
                preds_dir = find_video_preds_dir(seed_path)
                csv_path = find_seed_csv(preds_dir, session_id, camera)
                csv_paths[camera][seed_name] = csv_path
                print(f'  {camera}/{seed_name}: {csv_path}')

        if args.n_frames is not None:
            n_frames = args.n_frames
        else:
            # default: every frame available, from start_frame, across all resolved csvs
            available = min(
                _count_csv_rows(path)
                for paths in csv_paths.values()
                for path in paths.values()
            )
            n_frames = available - start_frame
            print(f'--n-frames not given; using all available frames: {n_frames}')
        background_frame = args.background_frame or (start_frame + n_frames // 2)

        # Auto-discover videos
        print('Resolving video paths...')
        video_paths = {}
        video_overrides = {'left': args.video_left, 'right': args.video_right}

        for camera in args.cameras:
            if video_overrides[camera]:
                video_paths[camera] = Path(video_overrides[camera]).resolve()
            else:
                # Resolve from seed 0's CSV
                seed_names_sorted = sorted(csv_paths[camera].keys())
                csv_for_discovery = csv_paths[camera][seed_names_sorted[0]]
                video_path = resolve_video_path(str(csv_for_discovery))
                if video_path:
                    video_paths[camera] = video_path
                else:
                    print(
                        f'Error: Could not auto-discover video for {camera} camera '
                        f'(session {session_id}).\n'
                        f'Provide --video-{camera} with the path to the video.',
                        file=sys.stderr,
                    )
                    sys.exit(1)

            print(f'  {camera}: {video_paths[camera]}')

        # Process per camera
        saved_median_csvs = {}
        median_sources = {}

        for camera in args.cameras:
            print(f'Processing {camera} camera...')

            seed_names_sorted = sorted(csv_paths[camera].keys())
            keypoint_names = discover_keypoint_names(csv_paths[camera][seed_names_sorted[0]])
            print(f'  discovered {len(keypoint_names)} keypoints')

            # Load every seed's full keypoint set once per camera.
            xy_seeds_all = []
            conf_seeds_all = []
            for seed_name in seed_names_sorted:
                xy, conf = load_all_keypoints(
                    csv_paths[camera][seed_name], keypoint_names, start_frame, n_frames,
                )
                xy_seeds_all.append(xy)
                conf_seeds_all.append(conf)

            existing_median_path = (
                Path(args.ensemble_median_dir) / f'ensemble_median_{session_id[:8]}_{camera}.csv'
                if args.ensemble_median_dir else None
            )
            if existing_median_path and existing_median_path.exists():
                print(f'  Reusing ensemble median: {existing_median_path}')
                median_keypoint_names = discover_keypoint_names(existing_median_path)
                if median_keypoint_names != keypoint_names:
                    raise ValueError(
                        f'{existing_median_path} keypoints {median_keypoint_names} do not match '
                        f'the seeds\' keypoints {keypoint_names} for {session_id}/{camera}',
                    )
                xy_median_all, conf_median_all = load_all_keypoints(
                    existing_median_path, keypoint_names, start_frame, n_frames,
                )
                median_sources[camera] = 'reused'
            else:
                xy_median_all, conf_median_all = compute_ensemble_median(
                    np.stack(xy_seeds_all, axis=0), np.stack(conf_seeds_all, axis=0),
                )
                median_sources[camera] = 'computed'

                if args.save_ensemble_median:
                    median_csv_path = csv_dir / f'ensemble_median_{session_id[:8]}_{camera}.csv'
                    write_ensemble_median_csv(
                        xy_median_all, conf_median_all, keypoint_names, median_csv_path,
                    )
                    saved_median_csvs[camera] = str(median_csv_path)
                    print(f'  Saved ensemble median: {median_csv_path}')

            # Resolve each group's real columns for this camera.
            groups_resolved = {}
            for group_name, tokens in groups.items():
                bases = resolve_group_bases(keypoint_names, tokens)
                columns = group_columns_for_camera(bases, camera)
                landmarks = [base for base, _ in bases]
                indices = [keypoint_names.index(c) for c in columns]
                groups_resolved[group_name] = {
                    'columns': columns,
                    'landmarks': landmarks,
                    'indices': indices,
                    'colors': colors_for_landmarks(landmarks),
                }
                print(f'  {group_name}: {len(landmarks)} landmarks ({landmarks})')

            # Get background frame
            bg_frame = get_background_frame(
                csv_paths[camera][seed_names_sorted[0]],
                video_paths[camera],
                frame_idx=background_frame,
            )

            roi = {
                'left': tuple(args.roi_left) if args.roi_left else None,
                'right': tuple(args.roi_right) if args.roi_right else None,
            }[camera]

            for gated in [False, True]:
                mode = 'matched-conf-gt07' if gated else 'ungated'
                print(f'  Rendering {mode}...')

                rows_data = []
                for group_name, resolved in groups_resolved.items():
                    idx = resolved['indices']
                    xy_list = [xy[:, idx, :] for xy in xy_seeds_all] + [xy_median_all[:, idx, :]]
                    confidences_list = [conf[:, idx] for conf in conf_seeds_all]
                    method_names = [
                        f'Single network: seed {i}' for i in range(len(seed_names_sorted))
                    ]
                    method_names.append('Ensemble median')

                    if eks_npz_dir:
                        group_eks_dir = eks_npz_dir / session_id / group_name
                        result = load_eks(group_eks_dir, camera, n_frames)
                        if result:
                            xy_eks, model = result

                            try:
                                np.testing.assert_allclose(
                                    xy_median_all[:, idx, :].reshape(n_frames, len(idx) * 2),
                                    model['observation_medians'],
                                    rtol=1e-5, atol=1e-3,
                                )
                            except AssertionError as e:
                                print(
                                    f'Warning: EKS observation_medians do not match seed median '
                                    f'for {session_id}/{camera}/{group_name}:\n{e}',
                                    file=sys.stderr,
                                )

                            xy_list.append(xy_eks)
                            method_names.append('Pose-EKS (rank 4)')
                            print(f'    EKS loaded for {session_id}/{camera}/{group_name}')
                        else:
                            print(
                                f'    EKS npz not found for {session_id}/{camera}/{group_name}; '
                                'skipping',
                            )

                    rows_data.append({
                        'group_name': group_name,
                        'landmarks': resolved['landmarks'],
                        'colors': resolved['colors'],
                        'xy_list': xy_list,
                        'confidences_list': confidences_list,
                        'method_names': method_names,
                    })

                fig, stats_by_group, title_text, rows_legend = render_combined_figure(
                    rows_data, bg_frame, session_id, camera, gated, conf_threshold, roi, n_frames,
                )

                # Save figure
                fig_name = (
                    f'rainbow-bodyparts__{args.group_set}-{mode}'
                    f'__{session_id[:8]}_{camera}_'
                    f'frames{start_frame:06d}-{start_frame + n_frames - 1:06d}.png'
                )
                fig_path = fig_dir / fig_name
                fig.savefig(fig_path, dpi=160, facecolor=fig.get_facecolor())
                plt.close(fig)

                # Add title canvas above and legend canvas below the plot
                combined_img = compose_figure_canvas(fig_path, title_text, rows_legend)
                combined_img.save(fig_path)

                print(f'  Saved: {fig_path}')

                # Record statistics
                for group_name, stats_list in stats_by_group.items():
                    method_names = rows_data[list(groups_resolved).index(group_name)]['method_names']
                    for method_name, stats_dict in zip(method_names, stats_list):
                        for landmark, stat in stats_dict.items():
                            record = {
                                'session': session_id,
                                'camera': camera,
                                'group': group_name,
                                'mode': mode,
                                'method': method_name,
                                'keypoint': landmark,
                            }
                            record.update(stat)
                            all_stats.append(record)

                    idx = groups_resolved[group_name]['indices']
                    seeds_conf = np.stack([conf[:, idx] for conf in conf_seeds_all], axis=0)
                    audit[f'{session_id}_{camera}_{group_name}_{mode}'] = {
                        'methods': method_names,
                        'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
                        'n_frames_kept': int((seeds_conf > conf_threshold).all(axis=0).sum())
                        if gated else n_frames,
                    }

        sessions_manifest[session_id] = {
            'csv_paths': {
                cam: {name: str(path) for name, path in paths.items()}
                for cam, paths in csv_paths.items()
            },
            'video_paths': {cam: str(path) for cam, path in video_paths.items()},
            'background_frame': background_frame,
            'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
            'n_frames': n_frames,
            'ensemble_median_csvs': saved_median_csvs,
            'ensemble_median_source': median_sources,
        }

    # Save statistics CSV
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_csv = csv_dir / 'distribution_diagnostics.csv'
        stats_df.to_csv(stats_csv, index=False)
        print(f'Saved: {stats_csv}')

    # Save manifest
    manifest = {
        'run_name': run_name,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'model_dir': str(model_dir),
        'seed_folders': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'sessions_requested': (
            {'sessions': args.sessions} if args.sessions is not None
            else {'sessions_file': str(args.sessions_file), 'session_group': args.session_group}
        ),
        'sessions': sessions_manifest,
        'cameras': args.cameras,
        'conf_threshold': conf_threshold,
        'groups_file': str(args.groups_file),
        'group_set': args.group_set,
        'groups': groups,
        'save_ensemble_median': args.save_ensemble_median,
        'eks_npz_dir': str(eks_npz_dir) if eks_npz_dir else None,
        'ensemble_median_dir': str(args.ensemble_median_dir) if args.ensemble_median_dir else None,
        'roi': {'left': args.roi_left, 'right': args.roi_right},
        'tag': tag,
        'audit': audit,
    }
    manifest_path = out_dir / 'qc_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved: {manifest_path}')

    registry_path = model_dir / 'rainbow_qc' / 'registry.json'
    append_entry(registry_path, run_name, manifest)
    print(f'Updated registry: {registry_path}')

    print(f'Done. Output in: {out_dir}')


def main() -> None:
    """CLI entry point."""
    try:
        args = parse_args()
        run(args)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
