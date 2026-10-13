"""Locate and load seed predictions, background frames and Pose-EKS output."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose2visual.lightningpose.fit_eks_latent_model import load_latent_model


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

    # if already pointing to video_preds, return it
    if seed_path.name == 'video_preds':
        return seed_path

    # search recursively
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


def count_csv_rows(csv_path: Path) -> int:
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

    # pad if fewer frames than requested
    if len(xy) < n_frames:
        pad_width = ((0, n_frames - len(xy)), (0, 0), (0, 0))
        xy = np.pad(xy, pad_width, constant_values=np.nan)
        pad_width = ((0, n_frames - len(conf)), (0, 0))
        conf = np.pad(conf, pad_width, constant_values=np.nan)

    return xy, conf


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


def load_eks(
    eks_npz_dir: Path,
    camera: str,
    n_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
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
