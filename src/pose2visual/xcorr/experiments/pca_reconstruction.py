"""What the rank-k PCA of the face keeps and what it throws away (part of Experiment B).

Experiment B (``pca.py``) uses the face PCA only through cross-correlations. This module shows the
decomposition itself, for each camera: the x/y pixel positions of the face keypoints (the same
keypoints, loader and PCA as Experiment B: eye, pupil, mouth, upperlip, lowerlip, nose, pad, ear,
both sides pooled; no tongue, forelimb or digits) form ``X`` of shape
``[n_frames, 2 * n_keypoints]``, fit once per camera on every frame of the session, and are
split as::

    X_centered = X - mean
    X_rank     = X_centered @ B @ B.T      (top ``n_components`` components, default 2)
    X_residual = X_centered - X_rank

Outputs per camera:

* a three-panel heatmap of ``X_centered``, ``X_rank`` and ``X_residual`` over one representative
  window, and one over the whole recording (columns binned) with the window marked; identical rows
  (the column order of ``X``) and one colour scale per figure;
* a synchronized video (``pose2visual.qc.video.pca_panels``) of the window: the tracked keypoints,
  ``mean + X_rank`` and ``mean + X_residual``, all on the video frame in the same fixed image
  coordinates;
* ``decomposition.<eid>.md``, which explains the PCA, the three terms, the legend of the video and
  how to read the heatmaps.

The window is picked on the left camera (highest rank-k motion energy among windows with reliable
keypoints) and reused by time for the right camera, unless a start is given. ``pca.run_session``
calls `write_decomposition` with the face PCA it already fit, so the figures describe exactly the
PCA behind the xcorrs (``--no-decomposition`` skips it). This module is also a command line that
recomputes only the decomposition (another window, rank or crop) without any neural data; it
needs only the keypoint json of ``list_litpose_keypoints`` and the camera timestamps.

Files: ``<session>/results/experiment_B_pca/<run>/pca_decomposition.<eid>[.<camera>].{npz,json}``
and ``<session>/plot_video/experiment_B_pca/<run>_decomposition/`` (heatmaps, videos, the md).

Usage::

    PYTHONPATH=src python -m pose2visual.xcorr.experiments.pca_reconstruction \\
      --config configs/encoding/litpose_encoding.yaml --eids <eid> \\
      --video-timestamps-dir <dir> --exclude 'd*'
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.encoding.litpose_keypoints import load_encoding_yaml, load_filter_settings
from pose2visual.encoding.litpose_pca import decompose_group
from pose2visual.preprocessing import resolve_video_path
from pose2visual.xcorr.plots import timestamp_subdir
from pose2visual.xcorr.run_xcorr import CAMERAS
from pose2visual.xcorr.settings import (
    EXPERIMENT_PCA,
    freeze_config,
    load_xcorr_settings,
    new_run_dirs,
    resolve_eids,
    xcorr_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS = 2
DEFAULT_DURATION_SEC = 10.0
DEFAULT_MARGIN_SEC = 30.0
# windows with a smaller fraction of frames with every keypoint reliable are avoided
MIN_RELIABLE_FRACTION = 0.9
HEATMAP_PERCENTILE = 99.0
HEATMAP_STEP = 0.1
# the whole-recording heatmap has at most this many time bins
MAX_OVERVIEW_BINS = 4000
REFERENCE_CAMERA = 'left'
DECOMPOSITION_SUFFIX = 'decomposition'


@dataclass
class DecompositionOptions:
    """Settings of the decomposition figures and video.

    Attributes:
        start_sec: window start in session seconds; `None` picks it on the left camera.
        duration_sec: window length in seconds.
        margin_sec: seconds skipped at both ends when the window is picked.
        vmax: half-range in pixels of the window heatmap's colour scale; `None` = from the data.
        heatmap: draw the heatmaps.
        video: render the videos.
        video_root: folder with the `<left|right>Camera.video` folders; `None` = the default root.
        crop_pad: zoom the video to the keypoints plus this many pixels; `None` = whole frame.
    """

    start_sec: float | None = None
    duration_sec: float = DEFAULT_DURATION_SEC
    margin_sec: float = DEFAULT_MARGIN_SEC
    vmax: float | None = None
    heatmap: bool = True
    video: bool = True
    video_root: str | Path | None = None
    crop_pad: float | None = None


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the command-line options of `DecompositionOptions` to a parser.

    Args:
        parser: the parser to extend (shared by Experiment B and this module's command line).
    """
    parser.add_argument(
        '--window-start-sec', type=float, default=None,
        help='decomposition window start in session seconds; default: picked on the left camera.',
    )
    parser.add_argument(
        '--window-duration-sec', type=float, default=DEFAULT_DURATION_SEC,
        help=f'decomposition window length in seconds (default: {DEFAULT_DURATION_SEC:g}).',
    )
    parser.add_argument(
        '--window-margin-sec', type=float, default=DEFAULT_MARGIN_SEC,
        help='seconds skipped at both ends when picking the window '
        f'(default: {DEFAULT_MARGIN_SEC:g}).',
    )
    parser.add_argument(
        '--window-vmax', type=float, default=None,
        help='half-range in pixels of the window heatmap colour scale shared by its three '
        f'panels; default: the {HEATMAP_PERCENTILE:g}th percentile of |X_centered|.',
    )
    parser.add_argument(
        '--video-root', type=str, default=None,
        help='folder with the <left|right>Camera.video folders (default: $IBL_2VIEW_ROOT, then '
        'the IBL-2view root).',
    )
    parser.add_argument(
        '--crop-pad', type=float, default=None,
        help='zoom the video to the keypoints plus this many pixels; default: the whole frame.',
    )
    parser.add_argument('--no-video', action='store_true', help='do not render the videos.')
    parser.add_argument(
        '--no-decomposition-heatmap', action='store_true',
        help='do not draw the decomposition heatmaps.',
    )


def options_from_args(args: argparse.Namespace) -> DecompositionOptions:
    """Build `DecompositionOptions` from arguments added by `add_window_arguments`.

    Args:
        args: parsed command-line arguments.

    Returns:
        The options.
    """
    return DecompositionOptions(
        start_sec=args.window_start_sec, duration_sec=args.window_duration_sec,
        margin_sec=args.window_margin_sec, vmax=args.window_vmax,
        heatmap=not args.no_decomposition_heatmap, video=not args.no_video,
        video_root=args.video_root, crop_pad=args.crop_pad,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Rank-k PCA decomposition of the face (Experiment B): heatmaps, video, md.',
    )
    parser.add_argument('--config', type=str, required=True, help='encoding yaml.')
    parser.add_argument('--eids', type=str, nargs='+', default=None, help='session ids.')
    parser.add_argument('--eid-group', type=str, default=None, help='named group of eids.')
    parser.add_argument('--eid-groups-json', type=str, default=None)
    parser.add_argument(
        '--video-timestamps-dir', type=str, required=True,
        help='directory containing _ibl_{left,right}Camera.times.<eid>.npy.',
    )
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=None,
        help='cameras to analyse (default: left right).',
    )
    parser.add_argument(
        '--exclude', type=str, nargs='*', default=[], metavar='PATTERN',
        help="glob patterns of keypoints to leave out, e.g. 'd*' (the digits). Quote them.",
    )
    parser.add_argument(
        '--n-components', type=int, default=DEFAULT_N_COMPONENTS,
        help=f'rank of the reconstruction (default: {DEFAULT_N_COMPONENTS}).',
    )
    parser.add_argument(
        '--plots-subdir', type=str, default=None,
        help='run name: results/experiment_B_pca/<name> and plot_video/experiment_B_pca/'
        '<name>_decomposition (default: New York time).',
    )
    add_window_arguments(parser)
    return parser.parse_args(argv)


def decompose_positions(
    positions: dict[str, tuple[np.ndarray, np.ndarray]],
    keypoints: list[str],
    n_components: int,
) -> dict[str, Any]:
    """Split the positions of the keypoints into mean, rank-k part and residual.

    `X` stacks `(x, y)` of each keypoint in `keypoints` order, as `pca.camera_pca` does.

    Args:
        positions: keypoint -> finite `(x, y)` arrays `[n_frames]`.
        keypoints: keypoints to decompose, in column order.
        n_components: requested rank, clamped to the number of x/y features.

    Returns:
        Dict with `mean` `[K, 2]`, `centered`, `rank` and `residual` (each `[n_frames, K, 2]`),
        `explained_variance_ratio` `[n_components]` and `n_components` (effective).
    """
    xy = np.stack([v for name in keypoints for v in positions[name]], axis=1)
    effective = min(n_components, xy.shape[1])
    if effective < n_components:
        logger.warning(
            f'{len(keypoints)} keypoints give {xy.shape[1]} x/y features but {n_components} '
            f'components were requested; using {effective}',
        )
    mean, centered, low_rank, residual, ratio = decompose_group(xy, effective)
    shape = (xy.shape[0], len(keypoints), 2)
    return {
        'mean': mean.reshape(len(keypoints), 2),
        'centered': centered.reshape(shape),
        'rank': low_rank.reshape(shape),
        'residual': residual.reshape(shape),
        'explained_variance_ratio': ratio,
        'n_components': effective,
    }


def explained_by_keypoint(parts: dict[str, Any]) -> np.ndarray:
    """Fraction of each keypoint's position variance that the rank-k part explains.

    Args:
        parts: output of `decompose_positions`.

    Returns:
        `[K]`, `1 - var(residual) / var(centered)` with x and y variances added; NaN for a
        keypoint that does not move.
    """
    total = np.var(parts['centered'], axis=0).sum(axis=1)
    left_over = np.var(parts['residual'], axis=0).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, 1 - left_over / total, np.nan)


def select_window(
    times: np.ndarray,
    energy: np.ndarray,
    reliable_all: np.ndarray,
    duration_sec: float,
    margin_sec: float,
) -> float:
    """Start of the window with the highest mean energy among reliably tracked ones.

    Candidate starts lie every second between `margin_sec` after the first frame and
    `margin_sec + duration_sec` before the last (the margin shrinks for a short recording). Windows
    where fewer than `MIN_RELIABLE_FRACTION` of the frames have every keypoint reliable are skipped
    unless every window is like that.

    Args:
        times: frame timestamps `[F]`, increasing.
        energy: per-frame energy `[F]`, e.g. the squared norm of the rank-k part.
        reliable_all: boolean `[F]`, True where every keypoint is reliable.
        duration_sec: window length.
        margin_sec: seconds skipped at both ends.

    Returns:
        Start time of the window in seconds.

    Raises:
        ValueError: if the recording is shorter than `duration_sec`.
    """
    span = float(times[-1] - times[0])
    if span < duration_sec:
        raise ValueError(
            f'recording of {span:.1f} s is shorter than the {duration_sec:g} s window',
        )
    margin = max(0.0, min(margin_sec, (span - duration_sec) / 2))
    starts = np.arange(times[0] + margin, times[-1] - margin - duration_sec + 1e-9, 1.0)
    idx_a = np.searchsorted(times, starts)
    idx_b = np.searchsorted(times, starts + duration_sec)
    n_frames = np.maximum(idx_b - idx_a, 1)
    cum_energy = np.concatenate([[0.0], np.cumsum(energy)])
    cum_reliable = np.concatenate([[0.0], np.cumsum(reliable_all.astype(np.float64))])
    score = (cum_energy[idx_b] - cum_energy[idx_a]) / n_frames
    fraction = (cum_reliable[idx_b] - cum_reliable[idx_a]) / n_frames
    eligible = (fraction >= MIN_RELIABLE_FRACTION) & (idx_b > idx_a)
    if not eligible.any():
        logger.warning('no window has reliable keypoints in most frames; picking among all')
        eligible = idx_b > idx_a
    return float(starts[np.argmax(np.where(eligible, score, -np.inf))])


def window_frames(
    times: np.ndarray,
    start_sec: float,
    duration_sec: float,
) -> tuple[int, int]:
    """Frames of one camera inside a time window.

    Args:
        times: frame timestamps `[F]` of that camera.
        start_sec: window start.
        duration_sec: window length.

    Returns:
        `(idx_start, n_frames)`.

    Raises:
        ValueError: if fewer than two frames fall in the window.
    """
    idx_start = int(np.searchsorted(times, start_sec))
    idx_end = int(np.searchsorted(times, start_sec + duration_sec))
    if idx_end - idx_start < 2:
        raise ValueError(
            f'window {start_sec:g}-{start_sec + duration_sec:g} s holds {idx_end - idx_start} '
            f'frames of this camera (recording {times[0]:g}-{times[-1]:g} s)',
        )
    return idx_start, idx_end - idx_start


def bin_frames(
    times: np.ndarray,
    values: list[np.ndarray],
    max_bins: int = MAX_OVERVIEW_BINS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Average consecutive frames into at most `max_bins` time bins.

    Args:
        times: frame timestamps `[F]`.
        values: arrays `[F, ...]` binned along the first axis.
        max_bins: largest number of bins.

    Returns:
        `(bin_times, binned)`: the mean time of each bin `[B]` and the mean of each array per bin
        `[B, ...]`.
    """
    n_frames = len(times)
    n_bins = min(max_bins, n_frames)
    edges = np.linspace(0, n_frames, n_bins + 1).astype(int)[:-1]
    counts = np.diff(np.append(edges, n_frames)).reshape(-1, *([1] * (values[0].ndim - 1)))
    binned = [np.add.reduceat(v, edges, axis=0) / counts for v in values]
    bin_times = np.add.reduceat(times, edges) / counts.reshape(-1)
    return bin_times, binned


def plot_decomposition_heatmaps(
    eid: str,
    camera: str,
    keypoints: list[str],
    times: np.ndarray,
    x_centered: np.ndarray,
    x_rank: np.ndarray,
    x_residual: np.ndarray,
    explained_variance: float,
    rank: int,
    vmax: float | None = None,
    mark: tuple[float, float] | None = None,
    note: str = '',
):
    """Three stacked heatmaps of `X_centered`, `X_rank` and `X_residual`.

    Rows are the columns of `X` (x then y of each keypoint, keypoints in order) in all three
    panels and the colour scale is one symmetric range in pixels shared by the panels.

    Args:
        eid: session id.
        camera: camera name.
        keypoints: keypoint names, in column order.
        times: time of each column `[n]` (frames, or bin centers).
        x_centered: `[n, K, 2]`, pixels.
        x_rank: `[n, K, 2]`.
        x_residual: `[n, K, 2]`.
        explained_variance: fraction of the variance kept by the rank-k part.
        rank: number of components kept.
        vmax: half-range of the colour scale; `None` = a percentile of `|x_centered|`.
        mark: `(start, end)` seconds shaded in every panel (the video window), or `None`.
        note: extra words for the title, e.g. how the columns were binned.

    Returns:
        The matplotlib figure; save it with `figures.save_figure`.
    """
    # lazy: matplotlib is only needed to draw
    from matplotlib.figure import Figure

    from pose2visual.xcorr.figures import COLORMAP, TEXT_COLOR, heatmap_vmax

    n_frames = x_centered.shape[0]
    panels = [
        ('X_centered', x_centered),
        (f'X_rank{rank}  ({100 * explained_variance:.1f} % of the variance)', x_rank),
        ('X_residual', x_residual),
    ]
    scale = vmax if vmax is not None else heatmap_vmax(
        x_centered, HEATMAP_PERCENTILE, HEATMAP_STEP,
    )
    labels = [f'{name} {axis}' for name in keypoints for axis in ('x', 'y')]
    height = 2.0 + 3 * max(2.4, 0.115 * len(labels))
    fig = Figure(figsize=(11, height), layout='constrained')
    axes = fig.subplots(len(panels), 1, sharex=True)
    extent = (float(times[0]), float(times[-1]), len(labels), 0)
    image = None
    for ax, (title, data) in zip(axes, panels):
        image = ax.imshow(
            data.reshape(n_frames, -1).T, aspect='auto', cmap=COLORMAP, vmin=-scale, vmax=scale,
            interpolation='nearest', extent=extent,
        )
        ax.set_title(title, fontsize=9, color=TEXT_COLOR, loc='left')
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels, fontsize=5)
        for boundary in range(2, len(labels), 2):
            ax.axhline(boundary, color='0.5', linewidth=0.2)
        if mark is not None:
            ax.axvspan(mark[0], mark[1], facecolor='none', edgecolor='k', linewidth=1.0)
    axes[-1].set_xlabel('time (s)', fontsize=8)
    colorbar = fig.colorbar(image, ax=list(axes), shrink=0.6, pad=0.01)
    colorbar.set_label('displacement from the mean position (px)', fontsize=8)
    fig.suptitle(
        f'{eid} {camera} camera: rank-{rank} PCA of {len(keypoints)} keypoints; colour scale '
        f'+-{scale:.2f} px, same rows and scale in all panels{note}',
        fontsize=9, color=TEXT_COLOR,
    )
    return fig


def build_video_path(
    out_dir: Path,
    eid: str,
    camera: str,
    start_sec: float,
    duration_sec: float,
) -> Path:
    """Path of the video of one camera; the session id is separated from the rest by `.`.

    Args:
        out_dir: output folder.
        eid: session id.
        camera: camera name.
        start_sec: window start.
        duration_sec: window length.

    Returns:
        `<out_dir>/video.pca_decomposition.<eid>.<camera>.start<S>s_dur<D>s.mp4`.
    """
    window = f'start{round(start_sec)}s_dur{round(duration_sec)}s'
    return out_dir / f'video.pca_decomposition.{eid}.{camera}.{window}.mp4'


def decomposition_text(report: dict[str, Any]) -> str:
    """Markdown explanation of the decomposition figures and video of one run.

    Args:
        report: the report built by `write_decomposition`: `eid`, `start_sec`, `duration_sec`,
            `likelihood_threshold`, `n_components`, `vmax_window`, `vmax_full` (camera -> value)
            and `cameras` (camera -> `keypoints`, `explained_variance_ratio`,
            `explained_by_keypoint`, `start_frame`, `n_frames`, `n_overview_bins`).

    Returns:
        Markdown text.
    """
    rank = report['n_components']
    end_sec = report['start_sec'] + report['duration_sec']
    scale_window = ', '.join(f'{c} +-{v:.1f} px' for c, v in report['vmax_window'].items())
    scale_full = ', '.join(f'{c} +-{v:.1f} px' for c, v in report['vmax_full'].items())
    n_bins = max(info['n_overview_bins'] for info in report['cameras'].values())
    lines = [
        f'# PCA decomposition of the face ({report["eid"]})',
        '',
        f'Part of Experiment B. Window shown: **{report["start_sec"]:.1f} - {end_sec:.1f} s** '
        f'of the session ({report["duration_sec"]:g} s, picked on the left camera as the stretch '
        f'with the most rank-{rank} motion among reliably tracked ones, unless set by hand; the '
        'right camera uses the same times).',
        '',
        '## What the PCA is fit on',
        '',
        'For each camera separately, `X` holds the x and y **pixel positions (of the video '
        'frame)** of the face keypoints, one row per video frame and `2K` columns (x then y of '
        'each keypoint, in the order of the heatmap rows): eye, pupil, upperlip, mouth, lowerlip, '
        'nose, pad and ear keypoints, left and right side pooled. The tongue, wrist and digits '
        'are not in it. Frames where a keypoint has a likelihood below '
        f'{report["likelihood_threshold"]:g} are masked and linearly interpolated (the same '
        'loader as the cross-correlations), and the PCA uses **every frame of the session**. The '
        'positions are centered but not scaled, so x and y stay in pixels.',
        '',
        '## The three terms',
        '',
        '- `X_centered = X - mean`: the position of every keypoint minus its own average position '
        'over all frames of the session (the raw movement, nothing removed).',
        f'- PCA: eigendecomposition of the covariance of `X_centered`; the top {rank} '
        f'eigenvectors form `B` (`2K x {rank}`), the main shared movement patterns of the face '
        '(PC1, PC2, ...).',
        f'- `X_rank{rank} = X_centered @ B @ B.T`: the part of the movement that these {rank} '
        'patterns explain, back in pixels.',
        f'- `X_residual = X_centered - X_rank{rank}`: what is left over, i.e. the projection on '
        f'the other `2K - {rank}` components (movement that is not shared between keypoints, '
        'small independent jitter, tracking noise).',
        f'- They add up exactly, cell by cell: `X_centered = X_rank{rank} + X_residual`.',
        '',
        '| camera | keypoints (K) | variance kept by each component | kept in total |',
        '|---|---|---|---|',
    ]
    for camera, info in report['cameras'].items():
        ratio = info['explained_variance_ratio']
        lines.append(
            f'| {camera} | {len(info["keypoints"])} | '
            + ', '.join(f'{100 * v:.1f} %' for v in ratio) + f' | {100 * sum(ratio):.1f} % |',
        )
    lines += [
        '',
        f"Keypoints best and least explained by the rank-{rank} part (share of the keypoint's "
        'position variance over the session):',
        '',
    ]
    for camera, info in report['cameras'].items():
        share = {k: v for k, v in info['explained_by_keypoint'].items() if v == v}
        ranked = sorted(share, key=share.get, reverse=True)
        best = ', '.join(f'{k} {100 * share[k]:.0f} %' for k in ranked[:3])
        worst = ', '.join(f'{k} {100 * share[k]:.0f} %' for k in ranked[-3:])
        lines.append(f'- {camera}: best {best}; least {worst}')
    lines += [
        '',
        '## Reading the heatmaps (`heatmap.pca_decomposition.*`)',
        '',
        '- Each **row** is the x or y coordinate of one face keypoint; the rows and their order '
        'are the same in all three panels.',
        '- Each **column** is one video frame (the x axis is session time in seconds).',
        '- **Red**: the coordinate is above its average position (larger value); **blue**: below '
        'it; **near-white**: close to it. In image coordinates a larger `y` is *lower* in the '
        'picture and a larger `x` is further right, so red in a `y` row means the point is lower '
        'than usual. The average is the keypoint\'s mean over the whole session, not over the '
        'window.',
        '- The values are position displacements in **video pixels**, not speed and not Pearson '
        'correlations. One symmetric colour scale (+-vmax pixels, printed in the title) is shared '
        'by the three panels and saturates beyond it, so a small residual looks pale by design.',
        f'- Top panel `X_centered`: where each keypoint is relative to its average. Middle '
        f'`X_rank{rank}`: the part the {rank} shared patterns explain. Bottom `X_residual`: the '
        'rest. How to compare them: (1) pick a row in `X_centered`; (2) if the same row in '
        f'`X_rank{rank}` has the same colours at the same times, the {rank} patterns explain that '
        'keypoint; (3) the same row in `X_residual` is pale when it is fully explained and '
        'coloured when it is not; (4) a vertical stripe shared by many rows in '
        f'`X_rank{rank}` is one coordinated movement of the whole face, and rows that switch '
        'colour together or in opposite directions are parts moving together or against each '
        'other.',
        '- `heatmap.pca_decomposition.<eid>.<camera>`: the window above, one column per frame '
        f'(colour scale from the window: {scale_window}). '
        '`heatmap.pca_decomposition.full.<eid>.<camera>`: the whole recording, first to last '
        'frame; the window is outlined in black. To fit the screen each column averages '
        f'neighbouring frames (up to {n_bins} columns), which shows slow changes and hides fast '
        f'oscillations; use the window figure for those (colour scale {scale_full}).',
        '',
        '## The video (`video.pca_decomposition.*`)',
        '',
        'One video per camera, three panels side by side, the **same video frame behind all '
        'three**, the same frame index and the same fixed image coordinates in every frame:',
        '',
        '| panel | shows |',
        '|---|---|',
        '| left, *tracked keypoints* | the tracked position of each keypoint, `mean + X_centered` '
        '(= `X`) |',
        f'| middle, *rank-{rank} reconstruction* | `mean + X_rank{rank}`: where the keypoints '
        f'would be if only the {rank} shared patterns moved them |',
        '| right, *residual* | `mean + X_residual`: the mean position plus only what the '
        f'{rank} patterns do not explain |',
        '',
        'Legend of the marks:',
        '',
        '- **ring**: the keypoint\'s mean position (the same in every frame; the mean is added '
        'back so the motion is shown in image coordinates). Drawn in the middle and right panels.',
        '- **filled dot**: the keypoint\'s current position in that panel; the thin line from the '
        'ring to the dot is its displacement from the mean.',
        '- **hollow circle** (left panel): the position of that frame was imputed because the '
        'keypoint\'s likelihood was low, it is an interpolated value.',
        '- Colours are those of the QC videos (one hue per body part). The motion is shown at '
        'true pixel scale, not magnified; the footer of the middle and right panels gives the RMS '
        'displacement from the mean over the keypoints.',
        '',
    ]
    return '\n'.join(lines)


def write_decomposition(
    cameras: dict[str, dict[str, Any]],
    eid: str,
    config_path: str | Path,
    results_run_dir: Path,
    plots_dir: Path,
    options: DecompositionOptions,
) -> dict[str, Any]:
    """Pick the window, save the arrays, draw the heatmaps, render the videos and write the md.

    Args:
        cameras: camera -> dict with `keypoints` (names, column order), `times` (frame times
            `[F]`), `reliable` (boolean `[F, K]`), `parts` (`decompose_positions` output) and
            `csv_path` (the keypoint csv, from which the video is located).
        eid: session id.
        config_path: encoding yaml (filter settings for the text).
        results_run_dir: folder of the npz/json (`results/experiment_B_pca/<run>`).
        plots_dir: folder of the figures, videos and md (`<run>_decomposition`).
        options: window, colour scale and output choices.

    Returns:
        Dict with `start_sec`, `arrays` (camera -> npz), `report` (json), `md`, `heatmaps` and
        `heatmaps_full` (camera -> png) and `videos` (camera -> mp4), the last three only for what
        was made.

    Raises:
        ValueError: if the window does not fit a recording.
        FileNotFoundError: if a video cannot be found.
    """
    # lazy: matplotlib is only needed to draw
    from pose2visual.xcorr.figures import heatmap_vmax, save_figure

    plots_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(cameras, key=lambda camera: camera != REFERENCE_CAMERA)
    start_sec = options.start_sec
    if start_sec is None:
        data = cameras[order[0]]
        energy = np.sum(data['parts']['rank'] ** 2, axis=(1, 2))
        start_sec = select_window(
            data['times'], energy, data['reliable'].all(axis=1), options.duration_sec,
            options.margin_sec,
        )
        logger.info(f'{order[0]}: window starts at {start_sec:.1f} s (auto-picked)')
    duration = options.duration_sec
    threshold, _ = load_filter_settings(config_path)
    result: dict[str, Any] = {
        'start_sec': start_sec, 'arrays': {}, 'heatmaps': {}, 'heatmaps_full': {}, 'videos': {},
    }
    report: dict[str, Any] = {
        'eid': eid, 'start_sec': start_sec, 'duration_sec': duration,
        'likelihood_threshold': threshold, 'n_components': 0, 'vmax_window': {},
        'vmax_full': {}, 'cameras': {},
    }
    for camera in order:
        data = cameras[camera]
        parts, keypoints, times = data['parts'], data['keypoints'], data['times']
        idx_start, n_frames = window_frames(times, start_sec, duration)
        window = slice(idx_start, idx_start + n_frames)
        ratio = parts['explained_variance_ratio']
        rank = parts['n_components']
        explained = float(np.sum(ratio))
        per_keypoint = explained_by_keypoint(parts)
        bin_times, (full_c, full_r, full_x) = bin_frames(
            times, [parts['centered'], parts['rank'], parts['residual']],
        )
        logger.info(
            f'{camera}: {len(keypoints)} keypoints, rank {rank} keeps {100 * explained:.1f} % of '
            f'the variance; window frames {idx_start}-{idx_start + n_frames - 1}',
        )
        arrays_path = results_run_dir / f'pca_decomposition.{eid}.{camera}.npz'
        np.savez_compressed(
            arrays_path,
            mean=parts['mean'], x_centered=parts['centered'][window].astype(np.float32),
            x_rank=parts['rank'][window].astype(np.float32),
            x_residual=parts['residual'][window].astype(np.float32),
            reliable=data['reliable'][window], times=times[window],
            keypoints=np.asarray(keypoints), explained_variance_ratio=ratio,
            explained_by_keypoint=per_keypoint, start_frame=idx_start,
            full_times=bin_times, full_centered=full_c.astype(np.float32),
            full_rank=full_r.astype(np.float32), full_residual=full_x.astype(np.float32),
        )
        result['arrays'][camera] = arrays_path
        report['n_components'] = rank
        report['cameras'][camera] = {
            'keypoints': keypoints, 'explained_variance_ratio': [float(v) for v in ratio],
            'explained_by_keypoint': {k: float(v) for k, v in zip(keypoints, per_keypoint)},
            'start_frame': idx_start, 'n_frames': n_frames, 'n_overview_bins': len(bin_times),
        }
        report['vmax_window'][camera] = (
            options.vmax if options.vmax is not None else heatmap_vmax(
                parts['centered'][window], HEATMAP_PERCENTILE, HEATMAP_STEP,
            )
        )
        report['vmax_full'][camera] = heatmap_vmax(full_c, HEATMAP_PERCENTILE, HEATMAP_STEP)

        if options.heatmap:
            fig = plot_decomposition_heatmaps(
                eid, camera, keypoints, times[window], parts['centered'][window],
                parts['rank'][window], parts['residual'][window], explained, rank,
                report['vmax_window'][camera],
            )
            result['heatmaps'][camera] = save_figure(
                fig, plots_dir / f'heatmap.pca_decomposition.{eid}.{camera}',
            )[0]
            mark = (float(times[idx_start]), float(times[idx_start + n_frames - 1]))
            fig = plot_decomposition_heatmaps(
                eid, camera, keypoints, bin_times, full_c, full_r, full_x, explained, rank,
                report['vmax_full'][camera], mark=mark,
                note=f'; whole recording, {len(bin_times)} time bins averaged',
            )
            result['heatmaps_full'][camera] = save_figure(
                fig, plots_dir / f'heatmap.pca_decomposition.full.{eid}.{camera}',
            )[0]
        if options.video:
            # lazy: cv2 is only needed to render
            from pose2visual.qc.video.pca_panels import PcaVideoJob, render_pca_video

            video_path = resolve_video_path(data['csv_path'], video_root=options.video_root)
            if video_path is None:
                raise FileNotFoundError(
                    f'could not find the {camera} video of session {eid}; pass --video-root',
                )
            job = PcaVideoJob(
                eid=eid, camera=camera, keypoints=keypoints, video_path=video_path,
                out_path=build_video_path(plots_dir, eid, camera, start_sec, duration),
                start_frame=idx_start, mean=parts['mean'],
                xy_centered=parts['centered'][window], xy_rank=parts['rank'][window],
                xy_residual=parts['residual'][window], reliable=data['reliable'][window],
                explained_variance=explained, rank=rank, group='face',
                crop_pad=options.crop_pad,
            )
            result['videos'][camera] = render_pca_video(job)

    report_path = results_run_dir / f'pca_decomposition.{eid}.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    md_path = plots_dir / f'decomposition.{eid}.md'
    md_path.write_text(decomposition_text(report), encoding='utf-8')
    result['report'] = report_path
    result['md'] = md_path
    return result


def run_session(
    config_path: str | Path,
    eid: str,
    video_timestamps_dir: str | Path,
    cameras: list[str] | None = None,
    exclude: list[str] | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    options: DecompositionOptions | None = None,
    run: str | None = None,
) -> dict[str, Any]:
    """Recompute only the decomposition of one session (no neural data, no xcorr).

    The keypoints are Experiment B's face group (`pca.assign_face_paw_groups`), so the PCA is the
    one behind B's cross-correlations.

    Args:
        config_path: path of the encoding yaml.
        eid: session id.
        video_timestamps_dir: directory with the camera timestamp files.
        cameras: cameras to analyse; `None` = all of `run_xcorr.CAMERAS`.
        exclude: glob patterns of keypoints to leave out.
        n_components: rank of the reconstruction.
        options: window and output choices; `None` = the defaults.
        run: run name; `None` = New York time.

    Returns:
        The dict of `write_decomposition` plus `results_dir` and `plots_dir`.

    Raises:
        ValueError: if `cameras` has unknown names or a camera has no face keypoint.
        FileNotFoundError: if a video cannot be found.
    """
    # lazy: pca imports this module
    from pose2visual.xcorr.experiments.pca import assign_face_paw_groups, load_face_tokens
    from pose2visual.xcorr.experiments.relative_motion import load_camera_positions

    cameras = list(dict.fromkeys(cameras or CAMERAS))
    unknown = sorted(set(cameras) - set(CAMERAS))
    if unknown:
        raise ValueError(f'unknown cameras {unknown}; use {list(CAMERAS)}')
    options = options or DecompositionOptions()
    face_tokens = load_face_tokens(load_encoding_yaml(config_path)['body_parts_json'])
    settings = load_xcorr_settings(config_path)
    session_dir = xcorr_dir(config_path, eid, settings['output_name'])
    run = run or timestamp_subdir()
    results_run_dir, plots_root = new_run_dirs(session_dir, EXPERIMENT_PCA, run)
    per_camera: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        data = load_camera_positions(config_path, eid, camera, video_timestamps_dir, exclude)
        face = assign_face_paw_groups(data['keypoints'], face_tokens)[0]['face']
        if not face:
            raise ValueError(f'the {camera} camera has no face keypoint')
        per_camera[camera] = {
            'keypoints': face, 'times': data['times'], 'csv_path': data['csv_path'],
            'reliable': np.stack([np.asarray(data['reliable'][k], bool) for k in face], axis=1),
            'parts': decompose_positions(data['positions'], face, n_components),
        }
    result = write_decomposition(
        per_camera, eid, config_path, results_run_dir,
        plots_root / f'{run}_{DECOMPOSITION_SUFFIX}', options,
    )
    result['results_dir'] = results_run_dir
    result['plots_dir'] = plots_root / f'{run}_{DECOMPOSITION_SUFFIX}'
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the decomposition for every selected session; failures are listed at the end.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        SystemExit: with status 1 if any session failed.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    failures: dict[str, str] = {}
    run = args.plots_subdir or timestamp_subdir()
    options = options_from_args(args)
    with freeze_config(args.config) as config:
        eids = resolve_eids(config, args.eids, args.eid_group, args.eid_groups_json)
        for idx, eid in enumerate(eids, start=1):
            logger.info(f'=== [{idx}/{len(eids)}] {eid} ===')
            try:
                result = run_session(
                    config, eid, args.video_timestamps_dir, args.cameras, args.exclude,
                    args.n_components, options, run,
                )
                logger.info(f'wrote {result["plots_dir"]} and {result["results_dir"]}')
            except Exception as error:  # noqa: BLE001 - one session must not stop the others
                logger.error(f'{eid} failed: {error}')
                failures[eid] = str(error)
    if failures:
        logger.error(f'{len(failures)} session(s) failed: {sorted(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
