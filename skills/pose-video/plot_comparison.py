#!/usr/bin/env python3
"""Rainbow scatter plots of ear landmark predictions across network seeds.

Visualizes prediction distributions by scattering all frames' predictions (5,000
by default) on a single static background frame. Each dot is one frame's (x, y)
for one landmark from one network. Compares: 3 single-network seeds, their
per-frame median, and optionally Pose-EKS smoothed output. Colour = landmark
(base/top/tip/bottom). Drawn in random order at low alpha, so clusters show
where predictions concentrate. Renders both ungated (all frames) and gated
(high-confidence frames per landmark).

Auto-discovers video from CSV filename (IBL-2view convention) and computes
output directory via sha256 hash of inputs for reproducibility.

Usage:
    python plot_comparison.py \\
        --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
        --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
        --session 9b528ad0-4599-4a55-9148-96cc1d93fb24

    python plot_comparison.py \\
        --model-dir /path/to/models \\
        --seed-dirs seed0 seed1 seed2 \\
        --session <uuid> \\
        --cameras left right \\
        --n-frames 5000 --conf-threshold 0.7 \\
        --eks-npz-dir /path/to/eks/npz \\
        --tag my-qc

Default output: <model_dir>/rainbow_qc/<hash8>/{figures,csv,qc_manifest.json}
Hash is based on inputs (CSVs, session, frame range, threshold, parts, EKS paths).
Same inputs → same hash folder (overwritten in place); different inputs → new folder.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

# Add src to path for development/skill usage (project-agnostic import)
_pose2visual_src = Path(__file__).parent.parent.parent / 'src'
if _pose2visual_src.exists() and str(_pose2visual_src) not in sys.path:
    sys.path.insert(0, str(_pose2visual_src))

from pose2visual.preprocessing import resolve_video_path

# Landmark colours: one saturated hue per part (base/top/tip/bottom).
LANDMARK_COLORS_RGB = {
    'base': '#00d9ff',
    'top': '#ff9900',
    'tip': '#ff3cdf',
    'bottom': '#6aff46',
}

_NAME_HASH_LEN = 8


def _short_hash(text: str) -> str:
    """First hex digits of the sha256 of text, for folder names.

    Args:
        text: text to hash.

    Returns:
        First `_NAME_HASH_LEN` hex characters.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:_NAME_HASH_LEN]


def add_title_canvas(
    image_path: Path,
    title_text: str,
    canvas_height: int = 120,
) -> Image.Image:
    """Create a black title canvas and stack it above the plot image.

    Args:
        image_path: path to the plot image (PNG).
        title_text: title text (can be multi-line).
        canvas_height: height of the black title canvas in pixels.

    Returns:
        Combined PIL Image (title canvas stacked above plot).
    """
    # Load the plot image
    plot_img = Image.open(image_path)
    plot_width, plot_height = plot_img.size

    # Create black title canvas
    title_img = Image.new('RGB', (plot_width, canvas_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(title_img)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=24)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text to center it
    lines = title_text.split('\n')
    line_height = 30
    total_height = len(lines) * line_height
    y_start = (canvas_height - total_height) // 2

    # Draw each line of text, centered
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (plot_width - text_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(255, 255, 255), font=font)

    # Vertically stack title canvas on top of plot
    combined_height = canvas_height + plot_height
    combined_img = Image.new('RGB', (plot_width, combined_height))
    combined_img.paste(title_img, (0, 0))
    combined_img.paste(plot_img, (0, canvas_height))

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


def load_seed_predictions(
    csv_path: Path,
    parts: list[str],
    start_frame: int,
    n_frames: int,
    keypoint_prefix: str = 'ear',
) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions CSV for a single seed.

    Args:
        csv_path: path to predictions CSV.
        parts: landmark part names (e.g. ['base', 'top', 'tip', 'bottom']).
        start_frame: starting frame index.
        n_frames: number of frames to load.
        keypoint_prefix: keypoint prefix (e.g. 'ear', 'nose', 'paw').

    Returns:
        Tuple of (xy_coords, likelihoods), each shape (n_frames, n_parts, 2/1).
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)

    # Extract keypoint names from camera in CSV path
    camera = 'left' if 'leftCamera' in csv_path.name else 'right'
    keys = [f'{keypoint_prefix}_{p}_{camera}' for p in parts]

    xy_list = []
    conf_list = []

    for key in keys:
        xy_list.append(df[key][['x', 'y']].values)
        conf_list.append(df[key]['likelihood'].values)

    # Slice to requested frame range
    xy = np.stack(xy_list, axis=1)  # (n_total, n_parts, 2)
    conf = np.stack(conf_list, axis=1)  # (n_total, n_parts)

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
    parts: list[str],
    roi: Optional[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Compute per-keypoint statistics.

    Args:
        xy: prediction array (n_frames, n_parts, 2).
        keep: boolean mask (n_frames, n_parts) indicating valid frames.
        parts: landmark part names.
        roi: optional ROI (x0, y0, x1, y1) for coverage analysis.

    Returns:
        Dictionary {part_name: stats_dict}.
    """
    stats_dict = {}

    for k, part in enumerate(parts):
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

        stats_dict[part] = stat_entry

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


def render_rainbow_figure(
    xy_list: list[np.ndarray],
    confidences_list: list[np.ndarray],
    background_img: np.ndarray,
    parts: list[str],
    session_id: str,
    camera: str,
    gated: bool,
    conf_threshold: float,
    method_names: list[str],
    roi: Optional[tuple[int, int, int, int]],
) -> tuple[plt.Figure, list[dict[str, dict[str, Any]]], str]:
    """Render rainbow scatter figure for one camera.

    Args:
        xy_list: list of (n_frames, n_parts, 2) arrays (one per method).
        confidences_list: list of (n_frames, n_parts) arrays (one per seed).
        background_img: background frame (H, W, 3).
        parts: landmark part names.
        session_id: session UUID.
        camera: camera name ('left' or 'right').
        gated: whether to apply confidence gating.
        conf_threshold: confidence threshold for gating.
        method_names: names for each method (e.g. ['Seed 0', 'Seed 1', ...]).
        roi: optional ROI (x0, y0, x1, y1) for statistics.

    Returns:
        Tuple of (figure, stats_list, title_text).
    """
    n_methods = len(xy_list)
    n_frames = xy_list[0].shape[0]
    n_parts = len(parts)

    # Stack confidences from seeds: (n_seeds, n_frames, n_parts)
    seeds_conf = np.stack(confidences_list, axis=0)

    # Compute gating mask: (n_frames, n_parts)
    if gated:
        keep = (seeds_conf > conf_threshold).all(axis=0)
    else:
        keep = np.ones((n_frames, n_parts), dtype=bool)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(5 * n_methods, 5),
        facecolor='#141414',
    )

    if n_methods == 1:
        axes = [axes]

    h, w = background_img.shape[:2]

    stats_list = []

    for m, (xy, method_name) in enumerate(zip(xy_list, method_names)):
        ax = axes[m]
        ax.imshow(background_img)

        # Flatten and filter by keep mask
        xy_flat = xy.reshape(-1, 2)  # (n_frames * n_parts, 2)
        keep_flat = keep.ravel()
        pts = xy_flat[keep_flat]

        # Tile colours by part and frame
        colors_array = np.tile(
            [LANDMARK_COLORS_RGB[p] for p in parts],
            n_frames,
        )[keep_flat]

        # Shuffle order for visibility
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
            f'{camera.capitalize()} | {method_name}',
            color='white',
            fontsize=13,
        )

        # Per-landmark counts
        counts = keep.sum(axis=0)
        for k, (part, color) in enumerate(zip(parts, [LANDMARK_COLORS_RGB[p] for p in parts])):
            ax.text(
                0.98, 0.97 - 0.05 * k,
                f'{part}: {counts[k]:,}/{n_frames:,}',
                transform=ax.transAxes,
                ha='right', va='top',
                color=color, fontsize=8,
                bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=1),
            )

        # Scale bar (20 px)
        ax.plot([12, 32], [h - 13, h - 13], color='white', lw=2)
        ax.text(12, h - 7, '20 px', color='white', fontsize=8)

        # Collect statistics
        stats_dict = compute_statistics(xy, keep, parts, roi)
        stats_list.append(stats_dict)

    gate_mode = 'Matched frames per landmark: ALL THREE members likelihood >0.7' if gated else (
        f'All {n_frames:,} frames; no confidence filtering'
    )
    title_text = (
        f'Session {session_id[:8]} | ear distributions, frames 0–{n_frames - 1}\n{gate_mode}'
    )

    handles = [
        Line2D(
            [], [], marker='o', linestyle='none',
            color=LANDMARK_COLORS_RGB[p], label=p.capitalize(),
        )
        for p in parts
    ]
    fig.legend(
        handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
        ncol=len(parts), labelcolor='white', facecolor='#222222', fontsize=13,
    )

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.015, right=0.985, hspace=0.17, wspace=0.055)

    return fig, stats_list, title_text


def load_eks(
    eks_npz_dir: Path,
    camera: str,
    n_frames: int,
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Load pre-computed Pose-EKS latent-model output.

    Args:
        eks_npz_dir: directory containing {camera}_latent-model.npz files.
        camera: camera name ('left' or 'right').
        n_frames: expected frame count (for validation).

    Returns:
        Tuple of (eks_xy, model_dict), or None if npz not found or multiple found.
    """
    npz_files = list(eks_npz_dir.glob(f'*_{camera}_latent-model.npz'))

    if len(npz_files) == 0:
        return None
    if len(npz_files) > 1:
        raise ValueError(
            f'Multiple EKS npz files for {camera} in {eks_npz_dir}:\n'
            + '\n'.join(str(f) for f in npz_files)
        )

    npz_path = npz_files[0]
    model = np.load(npz_path)

    eks = (model['mean'] + model['smoothed_means'] @ model['W'].T).reshape(
        n_frames, 4, 2
    )

    return eks, model


def compute_hash_key(
    csv_paths: dict[str, dict[str, Path]],
    session_id: str,
    start_frame: int,
    n_frames: int,
    conf_threshold: float,
    parts: list[str],
    background_frame: int,
    eks_npz_paths: dict[str, Optional[str]],
    tag: Optional[str],
    keypoint_prefix: str = 'ear',
) -> str:
    """Compute sha256 hash of input configuration for reproducibility.

    Args:
        csv_paths: {camera: {seed_name: path_to_csv}}.
        session_id: session UUID.
        start_frame: starting frame index.
        n_frames: number of frames.
        conf_threshold: confidence threshold.
        parts: landmark part names.
        background_frame: background frame index.
        eks_npz_paths: {camera: path_or_none}.
        tag: optional tag.
        keypoint_prefix: keypoint prefix (e.g. 'ear', 'nose').

    Returns:
        First 8 hex digits of sha256.
    """
    key_dict = {
        'csv_paths': {
            cam: {name: str(p) for name, p in paths.items()}
            for cam, paths in csv_paths.items()
        },
        'session': session_id,
        'start_frame': start_frame,
        'n_frames': n_frames,
        'conf_threshold': conf_threshold,
        'parts': parts,
        'background_frame': background_frame,
        'eks_npz_paths': eks_npz_paths,
        'tag': tag,
        'keypoint_prefix': keypoint_prefix,
    }
    key_json = json.dumps(key_dict, sort_keys=True)
    return _short_hash(key_json)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Rainbow scatter plots of pose predictions across seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_comparison.py \\
    --model-dir /path/to/mighty-mouse-models/E6c-LOO... \\
    --seed-dirs vits_dinov3-seed0 vits_dinov3-seed1 vits_dinov3-seed2 \\
    --session 9b528ad0-4599-4a55-9148-96cc1d93fb24

  python plot_comparison.py \\
    --model-dir /path/to/models \\
    --seed-dirs seed0 seed1 seed2 \\
    --session <uuid> \\
    --n-frames 5000 --conf-threshold 0.7 \\
    --eks-npz-dir /path/to/eks
        ''',
    )

    parser.add_argument(
        '--model-dir',
        required=True,
        help='Root directory holding seed folders',
    )
    parser.add_argument(
        '--seed-dirs',
        nargs='+',
        required=True,
        help='Seed folder names (relative to model-dir) or absolute paths',
    )
    parser.add_argument(
        '--session',
        required=True,
        help='Session ID (UUID)',
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
        help='Output directory (default: <model-dir>/rainbow_qc/<hash8>)',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional tag for output filenames',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=5000,
        help='Number of frames to process (default: 5000)',
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
        '--keypoint-prefix',
        default='ear',
        help='Keypoint prefix (default: ear). Looks for {prefix}_{part}_{camera} columns',
    )
    parser.add_argument(
        '--parts',
        nargs='+',
        default=['base', 'top', 'tip', 'bottom'],
        help='Landmark parts (default: base top tip bottom)',
    )
    parser.add_argument(
        '--eks-npz-dir',
        default=None,
        help='Directory containing EKS latent-model .npz files (optional)',
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
    session_id = args.session
    n_frames = args.n_frames
    start_frame = args.start_frame
    background_frame = args.background_frame or (start_frame + n_frames // 2)
    conf_threshold = args.conf_threshold
    tag = args.tag
    parts = args.parts
    keypoint_prefix = args.keypoint_prefix

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
                    f'Error: Could not auto-discover video for {camera} camera.\n'
                    f'Provide --video-{camera} with the path to the video.',
                    file=sys.stderr,
                )
                sys.exit(1)

        print(f'  {camera}: {video_paths[camera]}')

    # Compute output directory hash
    eks_npz_dir = Path(args.eks_npz_dir).resolve() if args.eks_npz_dir else None
    eks_npz_paths = {camera: None for camera in args.cameras}
    if eks_npz_dir:
        for camera in args.cameras:
            npz_files = list(eks_npz_dir.glob(f'*_{camera}_latent-model.npz'))
            if len(npz_files) == 1:
                eks_npz_paths[camera] = str(npz_files[0])

    hash_key = compute_hash_key(
        csv_paths, session_id, start_frame, n_frames, conf_threshold, parts,
        background_frame, eks_npz_paths, tag, keypoint_prefix,
    )

    # Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = model_dir / 'rainbow_qc' / hash_key

    fig_dir = out_dir / 'figures'
    csv_dir = out_dir / 'csv'
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f'Output: {out_dir}')

    # Process per camera
    all_stats = []
    audit = {}

    for camera in args.cameras:
        print(f'Processing {camera} camera...')

        # Load predictions for each seed
        xy_seeds = []
        confidences_seeds = []
        seed_names_sorted = sorted(csv_paths[camera].keys())

        for seed_name in seed_names_sorted:
            xy, conf = load_seed_predictions(
                csv_paths[camera][seed_name],
                parts,
                start_frame,
                n_frames,
                keypoint_prefix=keypoint_prefix,
            )
            xy_seeds.append(xy)
            confidences_seeds.append(conf)

        # Compute median
        xy_median = np.median(np.stack(xy_seeds, axis=0), axis=0)

        # Build list for rendering
        xy_list = xy_seeds + [xy_median]
        method_names = [f'Single network: seed {i}' for i in range(len(seed_names_sorted))] + ['Ensemble median']

        # Optionally load EKS
        if eks_npz_dir and eks_npz_paths[camera]:
            result = load_eks(Path(eks_npz_paths[camera]).parent, camera, n_frames)
            if result:
                xy_eks, model = result

                # Verify observation_medians match seed median
                computed_median = np.median(
                    np.stack(confidences_seeds, axis=0), axis=0
                )
                try:
                    np.testing.assert_allclose(
                        xy_median.reshape(n_frames, len(parts) * 2),
                        model['observation_medians'],
                        rtol=1e-5, atol=1e-3,
                    )
                except AssertionError as e:
                    print(
                        f'Warning: EKS observation_medians do not match seed median for {camera}:\n{e}',
                        file=sys.stderr,
                    )

                xy_list.append(xy_eks)
                method_names.append('Pose-EKS (rank 4)')
                print(f'  EKS loaded for {camera}')
            else:
                print(f'  EKS npz not found for {camera}; skipping')

        # Get background frame
        bg_frame = get_background_frame(
            csv_paths[camera][seed_names_sorted[0]],
            video_paths[camera],
            frame_idx=background_frame,
        )

        # Render both gated and ungated
        roi = {
            'left': tuple(args.roi_left) if args.roi_left else None,
            'right': tuple(args.roi_right) if args.roi_right else None,
        }[camera]

        for gated in [False, True]:
            mode = 'matched-conf-gt07' if gated else 'ungated'
            print(f'  Rendering {mode}...')

            fig, stats_list, title_text = render_rainbow_figure(
                xy_list, confidences_seeds, bg_frame, parts, session_id,
                camera, gated, conf_threshold, method_names, roi,
            )

            # Save figure
            fig_name = (
                f'ear-rainbow-single-networks-vs-pose-eks-{mode}'
                f'__{session_id[:8]}_'
                f'frames{start_frame:06d}-{start_frame + n_frames - 1:06d}.png'
            )
            fig_path = fig_dir / fig_name
            fig.savefig(fig_path, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)

            # Add title canvas on top of the plot
            combined_img = add_title_canvas(fig_path, title_text, canvas_height=120)
            combined_img.save(fig_path)

            print(f'  Saved: {fig_path}')

            # Record statistics
            for method_idx, (method_name, stats_dict) in enumerate(
                zip(method_names, stats_list)
            ):
                for part, stat in stats_dict.items():
                    record = {
                        'camera': camera,
                        'mode': mode,
                        'method': method_name,
                        'keypoint': part,
                    }
                    record.update(stat)
                    all_stats.append(record)

            audit[f'{camera}_{mode}'] = {
                'methods': method_names,
                'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
                'n_frames_kept': int((np.stack(confidences_seeds, axis=0) > conf_threshold).all(axis=0).sum())
                if gated else n_frames,
            }

    # Save statistics CSV
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_csv = csv_dir / 'distribution_diagnostics.csv'
        stats_df.to_csv(stats_csv, index=False)
        print(f'Saved: {stats_csv}')

    # Save manifest
    manifest = {
        'hash_key': hash_key,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'model_dir': str(model_dir),
        'seed_folders': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'session': session_id,
        'cameras': args.cameras,
        'csv_paths': {
            cam: {name: str(path) for name, path in paths.items()}
            for cam, paths in csv_paths.items()
        },
        'video_paths': {cam: str(path) for cam, path in video_paths.items()},
        'background_frame': background_frame,
        'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
        'n_frames': n_frames,
        'conf_threshold': conf_threshold,
        'keypoint_prefix': keypoint_prefix,
        'parts': parts,
        'eks_npz_dir': str(eks_npz_dir) if eks_npz_dir else None,
        'eks_npz_paths': eks_npz_paths,
        'roi': {'left': args.roi_left, 'right': args.roi_right},
        'tag': tag,
        'audit': audit,
    }
    manifest_path = out_dir / 'qc_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved: {manifest_path}')

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
