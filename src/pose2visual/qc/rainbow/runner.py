"""Run the rainbow QC over sessions and cameras: task fan-out, manifest and registry."""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import pandas as pd

from pose2visual.lightningpose.fit_eks_latent_model import load_groups
from pose2visual.preprocessing import resolve_video_path
from pose2visual.qc.rainbow.camera_task import CameraTask, process_camera
from pose2visual.qc.rainbow.loading import count_csv_rows, find_seed_csv, find_video_preds_dir
from pose2visual.run_registry import append_entry

logger = logging.getLogger(__name__)

# run-folder timestamps use this timezone, not the cluster's local time
RUN_TIMEZONE = ZoneInfo('America/New_York')


def _init_worker() -> None:
    """Configure logging and cv2 threading in a freshly spawned worker process."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cv2.setNumThreads(1)


def resolve_num_workers(requested: int | None, n_tasks: int) -> int:
    """Pick the number of worker processes.

    Args:
        requested: value of `--num-workers`; when None, falls back to the SLURM allocation
            (`SLURM_CPUS_PER_TASK`) and finally to 1, so a bare login-node run stays serial.
        n_tasks: number of (session, camera) tasks; more workers than this are useless.

    Returns:
        worker count in `[1, max(n_tasks, 1)]`.
    """
    if requested is None:
        requested = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    return max(1, min(requested, n_tasks))


def run_tasks(tasks: list[CameraTask], num_workers: int) -> list[dict[str, Any]]:
    """Run every task, in-process for one worker or in a process pool otherwise.

    Args:
        tasks: (session, camera) tasks to process.
        num_workers: number of worker processes (1 runs serially in this process).

    Returns:
        one result dict per task (see `process_camera`), in the same order as `tasks`.
    """
    n_tasks = len(tasks)
    results: list[dict[str, Any] | None] = [None] * n_tasks
    run_start = time.time()

    if num_workers == 1:
        for task_idx, task in enumerate(tasks):
            results[task_idx] = process_camera(task)
            logger.info(
                f'[done {task_idx + 1}/{n_tasks}] {task.session_id}/{task.camera} '
                f'(total elapsed {time.time() - run_start:.1f}s)',
            )
        return results

    # keep numpy/BLAS from spawning num_cpus threads in each worker; spawned children
    # inherit this environment before they import numpy
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[var] = '1'

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_worker,
    )
    try:
        future_to_idx = {
            executor.submit(process_camera, task): task_idx
            for task_idx, task in enumerate(tasks)
        }
        for n_done, future in enumerate(as_completed(future_to_idx), start=1):
            task_idx = future_to_idx[future]
            results[task_idx] = future.result()  # re-raises any worker exception
            task = tasks[task_idx]
            logger.info(
                f'[done {n_done}/{n_tasks}] {task.session_id}/{task.camera} '
                f'(total elapsed {time.time() - run_start:.1f}s)',
            )
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()

    return results


def run(args: argparse.Namespace) -> None:
    """Resolve inputs, fan the (session, camera) tasks out to workers and write the manifest.

    Args:
        args: parsed command-line arguments (see `pose2visual.qc.rainbow.cli.parse_args`).
    """
    model_dir = Path(args.model_dir).resolve()
    start_frame = args.start_frame
    conf_threshold = args.conf_threshold
    tag = args.tag

    groups = load_groups(args.groups_file, group_set=args.group_set)
    if args.groups is not None:
        missing = sorted(set(args.groups) - set(groups))
        if missing:
            logger.error(f'groups not found in {args.groups_file}: {missing}')
            sys.exit(1)
        groups = {name: groups[name] for name in args.groups}

    if args.sessions is not None:
        sessions = args.sessions
    else:
        session_groups = load_groups(args.sessions_file, group_set='neural_sessions')
        if args.session_group not in session_groups:
            logger.error(
                f'session group {args.session_group!r} not found in '
                f'{args.sessions_file}: {sorted(session_groups)}',
            )
            sys.exit(1)
        sessions = session_groups[args.session_group]
    logger.info(f'Sessions: {sessions}')

    # resolve seed directories
    seed_dirs = {}
    for seed_name in args.seed_dirs:
        seed_path = Path(seed_name)
        if not seed_path.is_absolute():
            seed_path = model_dir / seed_name
        seed_path = seed_path.resolve()
        if not seed_path.exists():
            logger.error(f'seed directory not found: {seed_path}')
            sys.exit(1)
        seed_dirs[seed_name] = seed_path

    eks_npz_dir = Path(args.eks_npz_dir).resolve() if args.eks_npz_dir else None

    run_name = datetime.now(RUN_TIMEZONE).strftime('%Y%m%d_%H%M%S')
    if tag:
        run_name = f'{run_name}__{tag}'

    # resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = model_dir / 'rainbow_qc' / run_name

    fig_dir = out_dir / 'rainbow_qc'
    csv_dir = out_dir / 'ensemble_median'
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Output: {out_dir}')

    base_manifest = {
        'run_name': run_name,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'model_dir': str(model_dir),
        'seed_folders': args.seed_dirs,
        'resolved_seed_dirs': {name: str(path) for name, path in seed_dirs.items()},
        'sessions_requested': (
            {'sessions': args.sessions} if args.sessions is not None
            else {'sessions_file': str(args.sessions_file), 'session_group': args.session_group}
        ),
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
    }
    registry_path = model_dir / 'rainbow_qc' / 'registry.json'
    append_entry(registry_path, run_name, {**base_manifest, 'status': 'running'})
    logger.info(f'Registered run (status=running): {registry_path}')

    ensemble_median_dir = Path(args.ensemble_median_dir) if args.ensemble_median_dir else None
    rois = {
        'left': tuple(args.roi_left) if args.roi_left else None,
        'right': tuple(args.roi_right) if args.roi_right else None,
    }
    video_overrides = {'left': args.video_left, 'right': args.video_right}

    # prep (serial): resolve csvs, frame counts and videos for every session, then build one
    # task per (session, camera)
    tasks = []
    sessions_manifest = {}
    n_sessions = len(sessions)
    for session_idx, session_id in enumerate(sessions, start=1):
        logger.info(f'=== [{session_idx}/{n_sessions}] Session {session_id} ===')

        # find video_preds directories and csvs
        csv_paths = {}  # {camera: {seed_name: path}}
        for camera in args.cameras:
            csv_paths[camera] = {}
            for seed_name, seed_path in seed_dirs.items():
                preds_dir = find_video_preds_dir(seed_path)
                csv_path = find_seed_csv(preds_dir, session_id, camera)
                csv_paths[camera][seed_name] = csv_path
                logger.info(f'  {camera}/{seed_name}: {csv_path}')

        if args.n_frames is not None:
            n_frames = args.n_frames
        else:
            # default: every frame available, from start_frame, across all resolved csvs
            available = min(
                count_csv_rows(path)
                for paths in csv_paths.values()
                for path in paths.values()
            )
            n_frames = available - start_frame
            logger.info(f'--n-frames not given; using all available frames: {n_frames}')
        background_frame = args.background_frame or (start_frame + n_frames // 2)

        # auto-discover videos
        video_paths = {}
        for camera in args.cameras:
            if video_overrides[camera]:
                video_paths[camera] = Path(video_overrides[camera]).resolve()
            else:
                # resolve from seed 0's csv
                seed_names_sorted = sorted(csv_paths[camera].keys())
                csv_for_discovery = csv_paths[camera][seed_names_sorted[0]]
                video_path = resolve_video_path(str(csv_for_discovery), video_root=args.video_root)
                if video_path:
                    video_paths[camera] = video_path
                else:
                    logger.error(
                        f'Could not auto-discover video for {camera} camera '
                        f'(session {session_id}). Provide --video-{camera} with the path to '
                        'the video.',
                    )
                    sys.exit(1)

            logger.info(f'  {camera}: {video_paths[camera]}')

            tasks.append(CameraTask(
                session_id=session_id,
                camera=camera,
                csv_paths=csv_paths[camera],
                video_path=video_paths[camera],
                n_frames=n_frames,
                start_frame=start_frame,
                background_frame=background_frame,
                groups=groups,
                group_set=args.group_set,
                conf_threshold=conf_threshold,
                roi=rois[camera],
                eks_npz_dir=eks_npz_dir,
                ensemble_median_dir=ensemble_median_dir,
                save_ensemble_median=args.save_ensemble_median,
                fig_dir=fig_dir,
                csv_dir=csv_dir,
            ))

        sessions_manifest[session_id] = {
            'csv_paths': {
                cam: {name: str(path) for name, path in paths.items()}
                for cam, paths in csv_paths.items()
            },
            'video_paths': {cam: str(path) for cam, path in video_paths.items()},
            'background_frame': background_frame,
            'frame_range': f'{start_frame}-{start_frame + n_frames - 1}',
            'n_frames': n_frames,
            'ensemble_median_csvs': {},
            'ensemble_median_source': {},
        }

    # execute (parallel): one worker per (session, camera)
    num_workers = resolve_num_workers(args.num_workers, len(tasks))
    logger.info(f'Processing {len(tasks)} (session, camera) tasks with {num_workers} worker(s)')
    results = run_tasks(tasks, num_workers)

    # aggregate in task order so outputs match a serial run
    all_stats = []
    audit = {}
    for task, result in zip(tasks, results):
        all_stats.extend(result['stats'])
        audit.update(result['audit'])
        session_manifest = sessions_manifest[task.session_id]
        session_manifest['ensemble_median_source'][task.camera] = result['median_source']
        if result['saved_median_csv'] is not None:
            session_manifest['ensemble_median_csvs'][task.camera] = result['saved_median_csv']

    # save statistics csv
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_csv = csv_dir / 'distribution_diagnostics.csv'
        stats_df.to_csv(stats_csv, index=False)
        logger.info(f'Saved: {stats_csv}')

    # save manifest
    manifest = {
        **base_manifest,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'sessions': sessions_manifest,
        'audit': audit,
        'status': 'done',
    }
    manifest_path = out_dir / 'qc_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    logger.info(f'Saved: {manifest_path}')

    append_entry(registry_path, run_name, manifest)
    logger.info(f'Updated registry (status=done): {registry_path}')

    logger.info(f'Done. Output in: {out_dir}')
