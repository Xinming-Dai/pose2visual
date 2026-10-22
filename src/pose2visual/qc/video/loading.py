"""Load the Pose-EKS csv of one session and camera for the QC video.

The csvs are written by `pose2visual.lightningpose.fit_eks_latent_model`: a 3-level header
`(scorer, bodyparts, coords)` with, per keypoint, the coords `x`, `y`, `likelihood`,
`x_ens_median`, `y_ens_median`, `x_ens_var`, `y_ens_var`, `x_posterior_var` and
`y_posterior_var`. A csv holds every frame of the session (100k) for every keypoint, so only the
requested frames of the requested keypoints are read.

Encoding step 0 (`pose2visual.encoding.list_litpose_keypoints`) additionally saves a per-group PCA
reprojection of the usable keypoints as a csv (`pose2visual.encoding.save_litpose_pca`);
`load_pca_positions` reads it back for the red trace of the video.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    csv_filename,
    get_keypoint_coords,
    get_keypoint_names,
    load_keypoint_csv,
)

REQUIRED_COORDS = (
    'x',
    'y',
    'likelihood',
    'x_ens_median',
    'y_ens_median',
    'x_ens_var',
    'y_ens_var',
    'x_posterior_var',
    'y_posterior_var',
)
# the csv index is the first column; the header rows are the first three lines
N_HEADER_ROWS = 3


@dataclass
class CameraData:
    """Pose-EKS output of one camera for a frame range and a set of keypoints.

    Attributes:
        camera: camera name ('left' or 'right').
        columns: keypoint names as they appear in the csv, e.g. `ear_top_left`.
        landmarks: label of each keypoint (`columns` without the `_left`/`_right` side suffix of
            bilateral landmarks), e.g. `ear_top`.
        start_frame: video frame index of the first row.
        xy_median: ensemble median, shape (n_frames, n_keypoints, 2).
        xy_eks: Pose-EKS smoothed position, shape (n_frames, n_keypoints, 2).
        likelihood: mean ensemble likelihood, shape (n_frames, n_keypoints).
        ens_sd: ensemble SD, `sqrt(x_ens_var)`: seed spread divided by the mean likelihood
            (the observation noise of the smoother); shape (n_frames, n_keypoints, 2).
        post_sd: posterior SD, `sqrt(x_posterior_var)`: the smoother's uncertainty about the
            position; shape (n_frames, n_keypoints, 2).
        pca_xy: PCA reprojection of the position, shape (n_frames, n_keypoints, 2), NaN for a
            keypoint without PCA; None when no PCA file was loaded.
    """

    camera: str
    columns: list[str]
    landmarks: list[str]
    start_frame: int
    xy_median: np.ndarray
    xy_eks: np.ndarray
    likelihood: np.ndarray
    ens_sd: np.ndarray
    post_sd: np.ndarray
    pca_xy: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        """Number of frames loaded."""
        return self.xy_median.shape[0]


def resolve_eks_csv(eks_dir: str | Path, session_id: str, camera: str) -> Path:
    """Find the Pose-EKS csv of one session and camera.

    Args:
        eks_dir: directory holding the csvs written by `fit_eks_latent_model.py`.
        session_id: session UUID.
        camera: camera name ('left' or 'right').

    Returns:
        path of `_iblrig_<camera>Camera.downsampled.<session_id>.csv` in `eks_dir`.

    Raises:
        FileNotFoundError: if the csv does not exist.
    """
    csv_path = Path(eks_dir) / f'_iblrig_{camera}Camera.downsampled.{session_id}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f'Pose-EKS csv not found: {csv_path}')
    return csv_path


def _find_column_positions(
    csv_path: Path,
    columns: list[str],
) -> dict[tuple[str, str], int]:
    """Map `(keypoint, coord)` to the 0-based file column of each requested keypoint.

    Args:
        csv_path: Pose-EKS csv.
        columns: keypoint names to locate.

    Returns:
        mapping covering every keypoint in `columns` and every coord in `REQUIRED_COORDS`.

    Raises:
        ValueError: if a keypoint or one of its coords is missing from the header.
    """
    header = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0, nrows=0)
    # +1 because the index is file column 0 but not part of `header.columns`
    positions = {
        (keypoint, coord): pos + 1
        for pos, (_, keypoint, coord) in enumerate(header.columns)
    }
    for keypoint in columns:
        missing = [coord for coord in REQUIRED_COORDS if (keypoint, coord) not in positions]
        if missing:
            raise ValueError(
                f'{csv_path}: keypoint {keypoint!r} is missing columns {missing}; is this a '
                'Pose-EKS csv written by fit_eks_latent_model.py?',
            )
    return positions


def load_camera_data(
    csv_path: str | Path,
    columns: list[str],
    landmarks: list[str],
    camera: str,
    start_frame: int,
    n_frames: int,
) -> CameraData:
    """Read a frame range of some keypoints from a Pose-EKS csv.

    Args:
        csv_path: Pose-EKS csv of one session and camera.
        columns: keypoint names to load, as they appear in the csv header.
        landmarks: label of each keypoint in `columns`.
        camera: camera name ('left' or 'right').
        start_frame: first frame to load.
        n_frames: number of frames to load.

    Returns:
        the loaded arrays; SDs are square roots of the csv variance columns.

    Raises:
        ValueError: if a keypoint or coord is missing, `columns` and `landmarks` differ in length,
            or the csv has fewer than `n_frames` frames from `start_frame`.
    """
    csv_path = Path(csv_path)
    if len(columns) != len(landmarks):
        raise ValueError(f'got {len(columns)} columns but {len(landmarks)} landmarks')
    positions = _find_column_positions(csv_path, columns)

    wanted = sorted(
        {positions[keypoint, coord] for keypoint in columns for coord in REQUIRED_COORDS},
    )
    df = pd.read_csv(
        csv_path,
        header=None,
        skiprows=N_HEADER_ROWS + start_frame,
        usecols=wanted,
        nrows=n_frames,
    )
    if len(df) < n_frames:
        raise ValueError(
            f'{csv_path} has only {len(df)} frames from frame {start_frame}, '
            f'but {n_frames} were requested',
        )

    def _field(coord: str) -> np.ndarray:
        """Return the (n_frames, n_keypoints) array of one coord."""
        series = [df[positions[keypoint, coord]].to_numpy(float) for keypoint in columns]
        return np.stack(series, axis=1)

    def _sd(coord_x: str, coord_y: str) -> np.ndarray:
        """Return the (n_frames, n_keypoints, 2) SD of a pair of variance columns."""
        variance = np.stack([_field(coord_x), _field(coord_y)], axis=-1)
        return np.sqrt(np.clip(variance, 0, None))

    return CameraData(
        camera=camera,
        columns=list(columns),
        landmarks=list(landmarks),
        start_frame=start_frame,
        xy_median=np.stack([_field('x_ens_median'), _field('y_ens_median')], axis=-1),
        xy_eks=np.stack([_field('x'), _field('y')], axis=-1),
        likelihood=_field('likelihood'),
        ens_sd=_sd('x_ens_var', 'y_ens_var'),
        post_sd=_sd('x_posterior_var', 'y_posterior_var'),
    )


def pca_positions_path(pca_dir: str | Path, session_id: str, camera: str) -> Path:
    """Name the PCA csv of one session and camera, as written by encoding step 0.

    Args:
        pca_dir: directory of the PCA files (`<video_preds_dir>/encoding/pca/<group set>`).
        session_id: session UUID.
        camera: camera name ('left' or 'right').

    Returns:
        path of `_iblrig_<camera>Camera.downsampled.<session_id>.csv` in `pca_dir`.
    """
    return Path(pca_dir) / csv_filename(camera, session_id)


def load_pca_positions(
    pca_dir: str | Path,
    session_id: str,
    camera: str,
    columns: list[str],
    start_frame: int,
    n_frames: int,
) -> np.ndarray | None:
    """Read a frame range of the saved PCA reprojection for some keypoints.

    Args:
        pca_dir: directory of the PCA files (`<video_preds_dir>/encoding/pca/<group set>`).
        session_id: session UUID.
        camera: camera name ('left' or 'right').
        columns: keypoint names to return, as in the csv header.
        start_frame: first frame.
        n_frames: number of frames.

    Returns:
        array of shape (n_frames, len(columns), 2), NaN for a keypoint the PCA does not cover (e.g.
        dropped for being masked too often); None if the csv does not exist.

    Raises:
        ValueError: if the file has fewer than `n_frames` frames from `start_frame`.
    """
    path = pca_positions_path(pca_dir, session_id, camera)
    if not path.is_file():
        return None
    df = load_keypoint_csv(path)
    if len(df) - start_frame < n_frames:
        raise ValueError(
            f'{path} has only {max(len(df) - start_frame, 0)} frames from frame {start_frame}, '
            f'but {n_frames} were requested',
        )
    names = get_keypoint_names(df)
    out = np.full((n_frames, len(columns), 2), np.nan)
    for idx_kp, column in enumerate(columns):
        if column in names:
            x, y, _ = get_keypoint_coords(df, column)
            out[:, idx_kp, 0] = x[start_frame:start_frame + n_frames]
            out[:, idx_kp, 1] = y[start_frame:start_frame + n_frames]
    return out


def read_run_label(eks_dir: str | Path) -> str | None:
    """Describe the model behind an EKS run, for the video banner.

    Args:
        eks_dir: directory of the Pose-EKS csvs; the run's `manifest.json` is expected in its
            parent.

    Returns:
        e.g. `<model folder name> | seeds: <seed folders>`, or None if there is no manifest.
    """
    manifest_path = Path(eks_dir).resolve().parent / 'manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    model_name = Path(manifest.get('model_dir', '')).name
    seeds = ', '.join(manifest.get('seed_dirs', []))
    return f'{model_name} | seeds: {seeds}'
