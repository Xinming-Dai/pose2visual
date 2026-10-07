#!/usr/bin/env python3
"""Render keypoint overlays on video from DLC-style predictions CSV.

This script visualizes pose predictions by drawing circles and labels on video frames.
Supports both 3-level MultiIndex headers (scorer, keypoint, coord) and 2-level headers.

For IBL-2view dataset: video is auto-discovered from CSV filename (camera + UUID).
Pass --video only if auto-discovery fails or you want a different video.

Usage:
    python render_overlay.py --csv predictions.csv
    python render_overlay.py --csv predictions.csv --duration 10 --tag 10s
    python render_overlay.py --csv predictions.csv --video /custom/video.mp4
    python render_overlay.py --csv predictions.csv --output /path/to/output.mp4

Default output: <csv_dir>/overlay_videos/<csv_stem>.mp4
With --tag: <csv_dir>/overlay_videos/<csv_stem>__<tag>.mp4
Auto-versioning: if file exists, use _v2.mp4, _v3.mp4, etc.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import numpy as np

# Add src to path for development/skill usage (project-agnostic import)
_pose2visual_src = Path(__file__).parent.parent.parent / 'src'
if _pose2visual_src.exists() and str(_pose2visual_src) not in sys.path:
    sys.path.insert(0, str(_pose2visual_src))

from pose2visual.preprocessing import resolve_video_path


def detect_csv_header_levels(csv_path: str) -> int:
    """Detect number of header levels in DLC-style CSV.

    Args:
        csv_path: Path to predictions CSV

    Returns:
        Number of header levels (2 or 3)
    """
    with open(csv_path) as f:
        f.readline()  # row 0
        f.readline()  # row 1
        third_row = f.readline().strip()  # row 2

    if third_row and ('x' in third_row or 'y' in third_row or 'likelihood' in third_row):
        return 3
    return 2


def resolve_output_path(
    csv_path: str,
    output_path: Optional[str] = None,
    tag: Optional[str] = None,
) -> Path:
    """Resolve output video path with auto-versioning.

    Args:
        csv_path: Path to predictions CSV
        output_path: Explicit output path (used verbatim, no versioning)
        tag: Optional tag to append to filename (e.g. '10s', 'conf0.9')

    Returns:
        Resolved output path
    """
    if output_path:
        return Path(output_path)

    csv_path = Path(csv_path)
    csv_stem = csv_path.stem
    overlay_dir = csv_path.parent / 'overlay_videos'
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if tag:
        base_name = f'{csv_stem}__{tag}.mp4'
    else:
        base_name = f'{csv_stem}.mp4'

    output_path = overlay_dir / base_name

    if output_path.exists():
        idx = 2
        while (overlay_dir / f'{csv_stem}{"__" + tag if tag else ""}_v{idx}.mp4').exists():
            idx += 1
        output_path = overlay_dir / f'{csv_stem}{"__" + tag if tag else ""}_v{idx}.mp4'
        print(f'Output file exists; using {output_path.name}')

    return output_path


def render_keypoints_overlay(
    video_path: str,
    csv_path: str,
    output_path: str,
    duration_sec: Optional[float] = None,
    conf_threshold: float = 0.7,
    marker_size: int = 4,
    text_scale: float = 0.3,
) -> None:
    """Render keypoint overlays on video.

    Args:
        video_path: Path to input video
        csv_path: Path to predictions CSV (DLC format)
        output_path: Path to output video
        duration_sec: Duration in seconds (None = full video)
        conf_threshold: Confidence threshold for solid markers (below = hollow)
        marker_size: Radius of keypoint circles in pixels
        text_scale: Font scale for keypoint labels
    """
    # Detect header levels
    header_levels = detect_csv_header_levels(csv_path)
    print(f'Detected {header_levels}-level CSV header')

    # Read CSV
    df = pd.read_csv(csv_path, header=list(range(header_levels)))

    # Get scorer name dynamically
    scorer = df.columns.get_level_values(0)[1]

    # Extract keypoint names
    if header_levels == 3:
        all_kps = df.columns.get_level_values(1).unique()
    else:
        # For 2-level headers, filter out coordinate names
        all_kps = df.columns.get_level_values(1).unique()

    # Filter out pseudo-keypoints and coordinate names
    keypoints = [k for k in all_kps if k not in ['bodyparts', 'x', 'y', 'likelihood']]
    print(f'Found {len(keypoints)} keypoints (scorer: {scorer})')

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Failed to open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Video: {width}x{height} @ {fps} fps, {total_frames} frames')

    # Calculate frame limit
    max_frames = total_frames
    if duration_sec is not None:
        max_frames = min(max_frames, int(duration_sec * fps))

    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise ValueError(f'Failed to create video writer: {output_path}')

    # Generate colors for keypoints
    np.random.seed(42)
    colors = {kp: tuple(np.random.randint(50, 255, 3).tolist()) for kp in keypoints}

    # Render frames
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw keypoints for this frame
        if frame_idx < len(df):
            row = df.iloc[frame_idx]

            for kp in keypoints:
                try:
                    if header_levels == 3:
                        x = row[(scorer, kp, 'x')]
                        y = row[(scorer, kp, 'y')]
                        likelihood = row[(scorer, kp, 'likelihood')]
                    else:
                        x = row[(kp, 'x')]
                        y = row[(kp, 'y')]
                        likelihood = row[(kp, 'likelihood')]

                    # Skip invalid values
                    if pd.isna(x) or pd.isna(y) or pd.isna(likelihood):
                        continue

                    x, y = int(x), int(y)
                    color = colors[kp]

                    # Draw marker
                    is_confident = likelihood > conf_threshold
                    thickness = -1 if is_confident else 1
                    cv2.circle(frame, (x, y), marker_size, color, thickness)

                    # Draw label (abbreviated)
                    label = kp[:8]
                    cv2.putText(
                        frame,
                        label,
                        (x + marker_size + 2, y - marker_size),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        color,
                        1,
                    )
                except (KeyError, ValueError, TypeError):
                    pass

        # Add frame info
        info = f'Frame {frame_idx}/{max_frames} | {len(keypoints)} kpts | conf>{conf_threshold:.1f}'
        cv2.putText(frame, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f'  {frame_idx}/{max_frames} frames')

    cap.release()
    out.release()

    print(f'Rendered {frame_idx} frames to {output_path}')

    # Re-encode to h264
    print('Re-encoding to h264 (browser-compatible)...')
    import subprocess
    import shutil

    output_dir = Path(output_path).parent
    tmp_path = output_dir / f'.tmp_{Path(output_path).name}'

    try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i', str(output_path),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
        )
        # Replace original with re-encoded
        Path(output_path).unlink()
        shutil.move(str(tmp_path), str(output_path))
        print(f'Final video: {output_path}')
    except subprocess.CalledProcessError as e:
        print(f'Warning: ffmpeg re-encoding failed: {e}')
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Render keypoint overlays on video from DLC-style predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python render_overlay.py --csv predictions.csv
  python render_overlay.py --csv predictions.csv --duration 10 --tag 10s
  python render_overlay.py --csv predictions.csv --video /custom/video.mp4
  python render_overlay.py --csv predictions.csv --output /custom/output.mp4

For IBL-2view dataset: video is auto-discovered from CSV filename (camera + UUID).
Pass --video only if the auto-discovery fails or you want a different video.

Default output: <csv_dir>/overlay_videos/<csv_stem>.mp4
With --tag: <csv_dir>/overlay_videos/<csv_stem>__<tag>.mp4
Auto-versioning: if file exists, use _v2.mp4, _v3.mp4, etc.
        ''',
    )
    parser.add_argument(
        '--video',
        default=None,
        help='Path to input video (auto-discovered from CSV filename for IBL-2view)',
    )
    parser.add_argument('--csv', required=True, help='Path to predictions CSV (DLC format)')
    parser.add_argument(
        '--output',
        default=None,
        help='Path to output video (default: <csv_dir>/overlay_videos/<csv_stem>.mp4)',
    )
    parser.add_argument(
        '--tag',
        default=None,
        help='Tag to append to filename for variant videos (e.g., 10s, conf0.9)',
    )
    parser.add_argument('--duration', type=float, help='Duration in seconds (default: full video)')
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for solid markers (default: 0.7)',
    )
    parser.add_argument('--marker-size', type=int, default=4, help='Marker radius in pixels (default: 4)')
    parser.add_argument('--text-scale', type=float, default=0.3, help='Font scale for labels (default: 0.3)')

    args = parser.parse_args()

    try:
        # Resolve video path
        video_path = args.video
        if not video_path:
            video_path = resolve_video_path(args.csv)
            if video_path:
                print(f'Auto-discovered video: {video_path}')
            else:
                print(
                    'Error: Could not auto-discover video from CSV filename.\n'
                    'Provide --video with the path to the input video.',
                    file=sys.stderr,
                )
                sys.exit(1)

        output_path = resolve_output_path(args.csv, args.output, args.tag)
        render_keypoints_overlay(
            video_path,
            args.csv,
            str(output_path),
            duration_sec=args.duration,
            conf_threshold=args.conf_threshold,
            marker_size=args.marker_size,
            text_scale=args.text_scale,
        )
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
