"""One (session, camera) unit of work: load seeds, ensemble, render and save."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pose2visual.lightningpose.ensemble_median import (
    compute_ensemble_median,
    write_ensemble_median_csv,
)
from pose2visual.lightningpose.fit_eks_latent_model import (
    discover_keypoint_names,
    group_columns_for_camera,
    resolve_group_bases,
)
from pose2visual.qc.rainbow.canvas import colors_for_landmarks, compose_figure_canvas
from pose2visual.qc.rainbow.loading import get_background_frame, load_all_keypoints, load_eks
from pose2visual.qc.rainbow.render import render_combined_figure

logger = logging.getLogger(__name__)


@dataclass
class CameraTask:
    """Inputs for processing one (session, camera) pair in a worker process.

    Attributes:
        session_id: session UUID.
        camera: camera name ('left' or 'right').
        csv_paths: seed name -> prediction csv path for this session and camera.
        video_path: video used for the background frame.
        n_frames: number of frames to process.
        start_frame: starting frame index.
        background_frame: frame index used as the scatter background.
        groups: body-part group name -> token list.
        group_set: name of the group set (used in figure filenames).
        conf_threshold: confidence threshold for gating.
        roi: optional ROI (x0, y0, x1, y1) for statistics.
        eks_dir: optional directory of single-camera Pose-EKS csvs (see `load_eks`).
        ensemble_median_dir: optional directory of already-saved median csvs to reuse.
        save_ensemble_median: whether to write the computed ensemble median csv.
        fig_dir: directory the figures are written to.
        csv_dir: directory the ensemble median csvs are written to.
    """

    session_id: str
    camera: str
    csv_paths: dict[str, Path]
    video_path: Path
    n_frames: int
    start_frame: int
    background_frame: int
    groups: dict[str, list[str]]
    group_set: str
    conf_threshold: float
    roi: tuple[int, int, int, int] | None
    eks_dir: Path | None
    ensemble_median_dir: Path | None
    save_ensemble_median: bool
    fig_dir: Path
    csv_dir: Path


def process_camera(task: CameraTask) -> dict[str, Any]:
    """Load, ensemble, render and save everything for one (session, camera) pair.

    Runs in a worker process (or in the main process when a single worker is used), so it
    only touches files unique to its (session, camera) and returns its records to the caller
    instead of mutating shared state.

    Args:
        task: inputs for this (session, camera) pair.

    Returns:
        Dict with `stats` (list of per-keypoint statistics records), `audit` (dict keyed
        `{session}_{camera}_{group}_{mode}`), `saved_median_csv` (path string or None) and
        `median_source` ('reused' or 'computed').
    """
    session_id = task.session_id
    camera = task.camera
    start_frame = task.start_frame
    n_frames = task.n_frames
    conf_threshold = task.conf_threshold
    prefix = f'[{session_id[:8]}/{camera}]'

    seed_names_sorted = sorted(task.csv_paths.keys())
    csv_stem = task.csv_paths[seed_names_sorted[0]].stem
    keypoint_names = discover_keypoint_names(task.csv_paths[seed_names_sorted[0]])
    logger.info(f'{prefix} discovered {len(keypoint_names)} keypoints')

    xy_seeds_all = []
    conf_seeds_all = []
    for seed_name in seed_names_sorted:
        xy, conf = load_all_keypoints(
            task.csv_paths[seed_name], keypoint_names, start_frame, n_frames,
        )
        xy_seeds_all.append(xy)
        conf_seeds_all.append(conf)

    saved_median_csv = None
    existing_median_path = (
        task.ensemble_median_dir / f'{csv_stem}.csv' if task.ensemble_median_dir else None
    )
    if existing_median_path and existing_median_path.exists():
        logger.info(f'{prefix} reusing ensemble median: {existing_median_path}')
        median_keypoint_names = discover_keypoint_names(existing_median_path)
        if median_keypoint_names != keypoint_names:
            raise ValueError(
                f'{existing_median_path} keypoints {median_keypoint_names} do not match '
                f'the seeds\' keypoints {keypoint_names} for {session_id}/{camera}',
            )
        xy_median_all, conf_median_all = load_all_keypoints(
            existing_median_path, keypoint_names, start_frame, n_frames,
        )
        median_source = 'reused'
    else:
        xy_median_all, conf_median_all = compute_ensemble_median(
            np.stack(xy_seeds_all, axis=0), np.stack(conf_seeds_all, axis=0),
        )
        median_source = 'computed'

        if task.save_ensemble_median:
            median_csv_path = task.csv_dir / f'{csv_stem}.csv'
            write_ensemble_median_csv(
                xy_median_all, conf_median_all, keypoint_names, median_csv_path,
            )
            saved_median_csv = str(median_csv_path)
            logger.info(f'{prefix} saved ensemble median: {median_csv_path}')

    # resolve each group's real columns for this camera
    groups_resolved = {}
    for group_name, tokens in task.groups.items():
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
        logger.info(f'{prefix} {group_name}: {len(landmarks)} landmarks ({landmarks})')

    eks = None
    if task.eks_dir:
        eks = load_eks(task.eks_dir, session_id, camera, keypoint_names, n_frames)
        if eks is None:
            logger.info(f'{prefix} EKS csv not found in {task.eks_dir}; skipping')
        else:
            logger.info(f'{prefix} EKS loaded')

    bg_frame = get_background_frame(
        task.csv_paths[seed_names_sorted[0]],
        task.video_path,
        frame_idx=task.background_frame,
    )

    stats_records = []
    audit = {}
    for gated in [False, True]:
        mode = 'matched-conf-gt07' if gated else 'ungated'
        logger.info(f'{prefix} rendering {mode}...')

        rows_data = []
        for group_name, resolved in groups_resolved.items():
            idx = resolved['indices']
            xy_list = [xy[:, idx, :] for xy in xy_seeds_all] + [xy_median_all[:, idx, :]]
            confidences_list = [conf[:, idx] for conf in conf_seeds_all]
            method_names = [f'Single network: seed {i}' for i in range(len(seed_names_sorted))]
            method_names.append('Ensemble median')

            if eks is not None:
                xy_eks, xy_eks_median = eks
                try:
                    np.testing.assert_allclose(
                        xy_median_all[:, idx, :], xy_eks_median[:, idx, :],
                        rtol=1e-5, atol=1e-3,
                    )
                except AssertionError as e:
                    logger.warning(
                        f'{prefix} EKS ensemble median does not match seed median '
                        f'for {group_name}: {e}',
                    )

                xy_list.append(xy_eks[:, idx, :])
                method_names.append('Pose-EKS')

            rows_data.append({
                'group_name': group_name,
                'landmarks': resolved['landmarks'],
                'colors': resolved['colors'],
                'xy_list': xy_list,
                'confidences_list': confidences_list,
                'method_names': method_names,
            })

        fig, stats_by_group, title_text, rows_legend = render_combined_figure(
            rows_data, bg_frame, session_id, camera, gated, conf_threshold, task.roi, n_frames,
        )

        # save figure, named after the source csv (- is reserved for the UUID)
        mode_safe = mode.replace('-', '_')
        fig_name = (
            f'{csv_stem}.bodyparts_{task.group_set}_{mode_safe}_{camera}_'
            f'frames{start_frame:06d}_{start_frame + n_frames - 1:06d}.png'
        )
        fig_path = task.fig_dir / fig_name
        fig.savefig(fig_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)

        combined_img = compose_figure_canvas(fig_path, title_text, rows_legend)
        combined_img.save(fig_path)

        logger.info(f'{prefix} saved: {fig_path}')

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
                    stats_records.append(record)

            idx = groups_resolved[group_name]['indices']
            seeds_conf = np.stack([conf[:, idx] for conf in conf_seeds_all], axis=0)
            audit[f'{session_id}_{camera}_{group_name}_{mode}'] = {
                'methods': method_names,
                'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
                'n_frames_kept': int((seeds_conf > conf_threshold).all(axis=0).sum())
                if gated else n_frames,
            }

    return {
        'stats': stats_records,
        'audit': audit,
        'saved_median_csv': saved_median_csv,
        'median_source': median_source,
    }
