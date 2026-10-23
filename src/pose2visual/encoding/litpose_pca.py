"""PCA reprojection of Lightning Pose keypoint positions, one fit per body-part group.

Used by step B (``extract_litpose_dynamic_vars_trials``) when the encoding yaml's
``step_order`` pipeline contains the ``2d_pose_pca`` step.
After the low-likelihood frames are masked and interpolated, the x/y positions of all usable
keypoints of a body-part group (a group of a ``body_parts.json``-style file, e.g.
``eye_related``) are stacked into a ``[n_frames, 2 * n_keypoints]`` matrix, projected onto its top
``n_components`` principal components and mapped back to pixel coordinates. The speed is then
taken on the reprojected positions, so the model inputs are still one speed per keypoint, under
the original keypoint names, and ``base_keypoints`` / ``additional_keypoints`` need no change.

The PCA is fit per group on every frame of the session and camera (unsupervised, no spikes), on
the centered but unscaled pixel positions, because x and y share their unit and the reprojection
has to return pixels. ``n_components`` of each group is ``PCA_n_components`` ->
``<group set>`` -> ``<group>`` in the same json file. It counts x/y features, so it is clamped to
``2 * n_keypoints`` (where the reprojection is the identity); a group without any usable
keypoint is skipped. Both cases are logged as warnings.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pose2visual.lightningpose.fit_eks_latent_model import load_groups

logger = logging.getLogger(__name__)

_PCA_KEY = 'PCA_n_components'


def load_pca_n_components(
    body_parts_json: str | Path,
    group_set: str,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Load the body-part groups and their number of principal components.

    Args:
        body_parts_json: path of a `body_parts.json`-style file.
        group_set: top-level key holding the groups, e.g. `'nonoverlapping_analysis_groups'`.

    Returns:
        Tuple `(groups, n_components)`: group name -> keypoint name tokens, and group name ->
        requested number of components.

    Raises:
        ValueError: if the groups or the `PCA_n_components` entry of `group_set` are missing,
            a group has no count or a count has no group, or a count is not a positive int.
    """
    groups = load_groups(body_parts_json, group_set)
    raw = json.loads(Path(body_parts_json).read_text(encoding='utf-8'))
    counts = raw.get(_PCA_KEY, {}).get(group_set)
    if not isinstance(counts, dict) or not counts:
        raise ValueError(f"'{_PCA_KEY}' has no entry for '{group_set}' in {body_parts_json}")
    missing = sorted(set(groups) - set(counts))
    unknown = sorted(set(counts) - set(groups))
    if missing or unknown:
        raise ValueError(
            f"'{_PCA_KEY}' of '{group_set}' in {body_parts_json} does not match its groups "
            f'(groups without a count: {missing}, counts without a group: {unknown})',
        )
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f'{_PCA_KEY} of {name!r} must be a positive int, got {count!r}')
    return groups, {name: counts[name] for name in groups}


def assign_keypoints_to_groups(
    keypoints: list[str],
    groups: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Assign each keypoint to the body-part group whose token starts its name.

    A keypoint belongs to a group when the first `_`-separated segment of its name is one of the
    group's tokens (`ear_top_left` -> `ear`), the rule `fit_eks_latent_model` uses.

    Args:
        keypoints: keypoint names, e.g. the usable keypoints of a camera.
        groups: group name -> tokens, as from `load_pca_n_components`.

    Returns:
        Tuple `(members, unassigned)`: group name -> its keypoints in input order (empty list for
        a group without keypoint), and the keypoints that belong to no group.

    Raises:
        ValueError: if a keypoint belongs to several groups (use non-overlapping groups).
    """
    members: dict[str, list[str]] = {name: [] for name in groups}
    unassigned: list[str] = []
    for keypoint in keypoints:
        token = keypoint.split('_')[0]
        matching = [name for name, tokens in groups.items() if token in tokens]
        if len(matching) > 1:
            raise ValueError(
                f'keypoint {keypoint!r} belongs to several groups {matching}; PCA needs '
                'non-overlapping groups',
            )
        if matching:
            members[matching[0]].append(keypoint)
        else:
            unassigned.append(keypoint)
    return members, unassigned


def _fit_group_basis(
    xy: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared eigendecomposition behind `reproject_group` and `project_group`.

    Args:
        xy: array of shape `[n_frames, n_features]`.
        n_components: number of components to keep, in `[1, n_features]`.

    Returns:
        Tuple `(mean, centered, basis, explained_variance_ratio)`: the feature mean
        `[n_features]`, the centered input `[n_frames, n_features]` (float64), the top components
        `[n_features, n_components]` (unit-norm, orthogonal) and their variance ratio
        `[n_components]`.

    Raises:
        ValueError: if `n_components` is not in `[1, n_features]`.
    """
    n_features = xy.shape[1]
    if not 1 <= n_components <= n_features:
        raise ValueError(f'n_components must be in [1, {n_features}], got {n_components}')
    xy = np.asarray(xy, dtype=np.float64)
    mean = xy.mean(axis=0)
    centered = xy - mean
    # eigh of the small feature covariance is much cheaper than an svd of all frames
    eigvals, eigvecs = np.linalg.eigh(centered.T @ centered)
    idx_top = np.argsort(eigvals)[::-1][:n_components]
    basis = eigvecs[:, idx_top]
    total = float(eigvals.clip(min=0).sum())
    ratio = eigvals[idx_top].clip(min=0) / total if total > 0 else np.zeros(n_components)
    return mean, centered, basis, ratio


def reproject_group(xy: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Project features onto their top principal components and map them back.

    Args:
        xy: array of shape `[n_frames, n_features]`.
        n_components: number of components to keep, in `[1, n_features]`.

    Returns:
        Tuple `(xy_hat, explained_variance_ratio)`: the reprojected features `[n_frames,
        n_features]` (float64) and the variance ratio of the kept components `[n_components]`.

    Raises:
        ValueError: if `n_components` is not in `[1, n_features]`.
    """
    mean, centered, basis, ratio = _fit_group_basis(xy, n_components)
    return centered @ basis @ basis.T + mean, ratio


def project_group(xy: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Project features onto their top principal components, without mapping them back.

    Unlike `reproject_group`, which returns denoised pixel-space positions, this returns the
    component scores themselves (the centered input projected onto the basis) to use as behaviors
    in their own right, e.g. for a cross-correlation of the leading components directly.

    Args:
        xy: array of shape `[n_frames, n_features]`.
        n_components: number of components to keep, in `[1, n_features]`.

    Returns:
        Tuple `(scores, explained_variance_ratio)`: the component scores `[n_frames,
        n_components]` (float64, zero mean) and the variance ratio of the kept components
        `[n_components]`.

    Raises:
        ValueError: if `n_components` is not in `[1, n_features]`.
    """
    mean, centered, basis, ratio = _fit_group_basis(xy, n_components)
    return centered @ basis, ratio


def reproject_keypoints(
    positions: dict[str, tuple[np.ndarray, np.ndarray]],
    groups: dict[str, list[str]],
    n_components: dict[str, int],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Reproject the x/y positions of every body-part group onto its top components.

    Args:
        positions: keypoint name -> `(x, y)` finite arrays `[n_frames]` (interpolated).
        groups: group name -> tokens, from `load_pca_n_components`.
        n_components: group name -> requested number of components.

    Returns:
        Tuple `(positions_hat, info)`. `positions_hat` has the keypoints of `positions`, with
        the reprojected `(x, y)` for the keypoints of a group and the input for those that
        belong to no group or to a skipped group. `info` is json-serializable: `groups` maps a
        group name to `keypoints`, `n_components_requested`, `n_components` (effective; `0` when
        skipped) and `explained_variance_ratio`, and `unassigned_keypoints` lists the keypoints
        of no group.

    Raises:
        ValueError: if a keypoint belongs to several groups.
    """
    members, unassigned = assign_keypoints_to_groups(list(positions), groups)
    if unassigned:
        logger.info(f'PCA: keypoints without a group are left unchanged: {unassigned}')
    out = dict(positions)
    info_groups: dict[str, Any] = {}
    for group, keypoints in members.items():
        requested = n_components[group]
        entry: dict[str, Any] = {
            'keypoints': keypoints,
            'n_components_requested': requested,
            'n_components': 0,
            'explained_variance_ratio': [],
        }
        info_groups[group] = entry
        if not keypoints:
            logger.warning(f'PCA: group {group!r} has no usable keypoint; skipped')
            continue
        n_features = 2 * len(keypoints)
        effective = min(requested, n_features)
        if effective < requested:
            logger.warning(
                f'PCA: group {group!r} has {len(keypoints)} usable keypoints ({n_features} '
                f'x/y features) but {requested} components were requested; using {effective}',
            )
        xy = np.stack([values for name in keypoints for values in positions[name]], axis=1)
        xy_hat, ratio = reproject_group(xy, effective)
        for idx_kp, name in enumerate(keypoints):
            out[name] = (xy_hat[:, 2 * idx_kp], xy_hat[:, 2 * idx_kp + 1])
        entry['n_components'] = effective
        entry['explained_variance_ratio'] = [float(v) for v in ratio]
        logger.info(
            f'PCA: group {group!r}, {len(keypoints)} keypoints -> {effective} components, '
            f'explained variance {float(np.sum(ratio)):.4f}',
        )
    return out, {'groups': info_groups, 'unassigned_keypoints': unassigned}
