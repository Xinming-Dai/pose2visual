"""Per-group PCA reprojection of the Pose-EKS keypoints of a session, saved with its settings.

Takes the csvs written by ``fit_eks_latent_model`` (``scripts/delta/lightningpose/fit_eks.sh``) and
applies, per session and camera, the same preprocessing and PCA as step 1 of the encoding pipeline
(``pose2visual.encoding.extract_litpose_dynamic_vars_trials``), but on every frame and without
binning or taking speeds: the result is the reprojected x/y of each keypoint, in pixels.

Steps, all settings from ``--config`` (the encoding yaml) and ``paths.body_parts_json``:

1. keypoints masked in more than ``filter.max_masked_frac`` of the frames are dropped (those of
   ``filter.group_set.unfiltered_groups`` ignore the likelihood, as in step 0);
2. frames below ``filter.likelihood_threshold`` or with NaN x/y are masked and linearly
   interpolated over the frame index;
3. the x/y of the keypoints of each body-part group (the groups of ``--group-set``, default
   ``pca.group_set``) are projected onto the top ``PCA_n_components`` principal components of the
   group and mapped back to pixels (see ``pose2visual.encoding.litpose_pca``).

The position that goes in is the csv columns ``position.columns`` selects (``ens_post_mean`` = the
EKS posterior mean ``x``/``y``, or ``ens_median``).

Output, in ``--out-dir`` (default ``<eks-dir>/../pca/<group-set>``), per session and camera:

* ``pca_positions.<eid>.<camera>.npz``: ``keypoints``, ``group`` and ``n_components`` (of the
  group, per keypoint) and ``xy_pca`` (``[n_frames, n_keypoints, 2]`` float32, pixels). The QC
  video (``pose2visual.qc.video``) draws it as a red trace.
* ``pca_config.<eid>.<camera>.json``: how the file was made (command, yaml text, filter and PCA
  settings, groups and component counts, masked fractions, dropped keypoints, explained variance).

A failing session/camera does not stop the others; all failures are listed at the end and make the
command exit non-zero.

Usage::

    PYTHONPATH=src python -m pose2visual.lightningpose.pca_eks \\
      --eks-dir <run>/eks --sessions <eid1> <eid2> \\
      --config configs/encoding/litpose_encoding.yaml \\
      --group-set nonoverlapping_analysis_groups
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    DEFAULT_PCA_GROUP_SET,
    DEFAULT_POSITION,
    csv_filename,
    get_keypoint_names,
    impute_keypoint_positions,
    load_encoding_yaml,
    load_filter_settings,
    load_keypoint_csv,
    load_position_columns,
    load_unfiltered_keypoints,
    masked_fraction,
    resolve_session_eids,
)
from pose2visual.encoding.litpose_pca import (
    assign_keypoints_to_groups,
    load_pca_n_components,
    reproject_keypoints,
)

logger = logging.getLogger(__name__)

PIPELINE = (
    'drop keypoints masked in more than max_masked_frac of the frames',
    'mask frames with likelihood < likelihood_threshold or NaN x/y (exempt keypoints: NaN x/y '
    'only)',
    'linearly interpolate the masked frames over the frame index',
    'per body-part group: center the x/y pixel positions, project onto the top n_components '
    'principal components (fit on all frames), map back to pixels',
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Per-group PCA reprojection of the Pose-EKS keypoints, saved with its config.',
    )
    parser.add_argument(
        '--eks-dir',
        required=True,
        help='directory with _iblrig_<camera>Camera.downsampled.<session>.csv, as written by '
        'fit_eks_latent_model.py',
    )
    parser.add_argument(
        '--config',
        required=True,
        help='encoding yaml: filter, position.columns, paths.body_parts_json and pca.group_set',
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='session ids; mutually exclusive with --session-group '
        '(default: session.eid_group of --config)',
    )
    parser.add_argument(
        '--session-group',
        default=None,
        help='name of a group in --sessions-file\'s `neural_sessions`',
    )
    parser.add_argument(
        '--sessions-file',
        default=None,
        help='ibl_session_ids.json-style file for --session-group '
        '(default: paths.eid_groups_json of --config)',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        choices=CAMERAS,
        default=list(CAMERAS),
        help='cameras to process (default: left right)',
    )
    parser.add_argument(
        '--group-set',
        default=None,
        help=f'group set of paths.body_parts_json to reduce, with its PCA_n_components '
        f'(default: pca.group_set of --config, else {DEFAULT_PCA_GROUP_SET})',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='output directory (default: <eks-dir>/../pca/<group-set>)',
    )
    return parser.parse_args(argv)


def pca_output_paths(out_dir: str | Path, eid: str, camera: str) -> tuple[Path, Path]:
    """Name the two output files of one session and camera.

    The session id is the only part containing `-`; it is separated from the other parts by `.`.

    Args:
        out_dir: output directory.
        eid: session id.
        camera: `'left'` or `'right'`.

    Returns:
        Tuple `(npz_path, config_path)`.
    """
    out_dir = Path(out_dir)
    return (
        out_dir / f'pca_positions.{eid}.{camera}.npz',
        out_dir / f'pca_config.{eid}.{camera}.json',
    )


def select_usable_keypoints(
    df: pd.DataFrame,
    keypoints: list[str],
    likelihood_threshold: float,
    max_masked_frac: float,
    no_filter_keypoints: list[str],
    position_columns: tuple[str, str],
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Keep the keypoints that are masked in at most `max_masked_frac` of the frames.

    Args:
        df: dataframe from `load_keypoint_csv`.
        keypoints: keypoint names to judge, in header order.
        likelihood_threshold: frames below this likelihood are masked.
        max_masked_frac: keypoints with a larger masked fraction are dropped.
        no_filter_keypoints: keypoints whose likelihood is ignored (only NaN x/y is masked).
        position_columns: `(x_column, y_column)` holding the position.

    Returns:
        Tuple `(usable, masked_fractions, dropped)`: the kept names in input order, name -> masked
        fraction for every keypoint, and name -> reason for the dropped ones.
    """
    masked = {
        name: masked_fraction(
            df, name, None if name in no_filter_keypoints else likelihood_threshold,
            position_columns,
        )
        for name in keypoints
    }
    usable = [name for name in keypoints if masked[name] <= max_masked_frac]
    dropped = {
        name: f'masked fraction {masked[name]:.2f} > {max_masked_frac}'
        for name in keypoints if name not in usable
    }
    return usable, masked, dropped


def pca_session(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    n_components: dict[str, int],
    likelihood_threshold: float,
    max_masked_frac: float,
    no_filter_keypoints: list[str],
    position_columns: tuple[str, str],
) -> dict[str, Any]:
    """Filter, interpolate and PCA-reproject the keypoints of one session and camera.

    Args:
        df: dataframe from `load_keypoint_csv`.
        groups: group name -> keypoint name tokens.
        n_components: group name -> requested number of components.
        likelihood_threshold: frames below this likelihood are masked.
        max_masked_frac: keypoints with a larger masked fraction are dropped.
        no_filter_keypoints: keypoints whose likelihood is ignored.
        position_columns: `(x_column, y_column)` holding the position.

    Returns:
        Dict with `keypoints` (those of a reprojected group, in group then input order), `group`
        and `n_components` (effective, per keypoint), `xy_pca` (`[n_frames, n_keypoints, 2]`
        float32), `masked_fraction`, `dropped`, `unassigned` (usable keypoints in no group) and
        `info` (the `reproject_keypoints` info).

    Raises:
        ValueError: if a keypoint belongs to several groups, or no keypoint ends up in a
            reprojected group.
    """
    usable, masked, dropped = select_usable_keypoints(
        df, get_keypoint_names(df), likelihood_threshold, max_masked_frac, no_filter_keypoints,
        position_columns,
    )
    members, unassigned = assign_keypoints_to_groups(usable, groups)
    assigned = [name for names in members.values() for name in names]
    # the frame index is the time axis: the interpolation does not need the video timestamps
    times = np.arange(len(df), dtype=np.float64)
    positions, _ = impute_keypoint_positions(
        df, assigned, times, likelihood_threshold, position_columns, no_filter_keypoints,
    )
    positions_hat, info = reproject_keypoints(positions, groups, n_components)
    keypoints: list[str] = []
    keypoint_group: list[str] = []
    keypoint_n_components: list[int] = []
    for group, entry in info['groups'].items():
        if entry['n_components'] == 0:
            continue
        keypoints += entry['keypoints']
        keypoint_group += [group] * len(entry['keypoints'])
        keypoint_n_components += [entry['n_components']] * len(entry['keypoints'])
    if not keypoints:
        raise ValueError(f'no usable keypoint in any group {sorted(groups)}; dropped: {dropped}')
    xy_pca = np.stack(
        [np.column_stack(positions_hat[name]) for name in keypoints], axis=1,
    ).astype(np.float32)
    return {
        'keypoints': keypoints,
        'group': keypoint_group,
        'n_components': keypoint_n_components,
        'xy_pca': xy_pca,
        'masked_fraction': masked,
        'dropped': dropped,
        'unassigned': unassigned,
        'info': info,
    }


def resolve_settings(
    config_path: str | Path,
    group_set: str | None,
) -> dict[str, Any]:
    """Read everything the PCA depends on from the encoding yaml and `body_parts.json`.

    Args:
        config_path: path of the encoding yaml.
        group_set: group set to reduce; `None` uses `pca.group_set` of the yaml.

    Returns:
        Dict with `likelihood_threshold`, `max_masked_frac`, `position_name`, `position_columns`,
        `body_parts_json`, `group_set`, `groups` and `n_components`.

    Raises:
        ValueError: if the yaml is invalid, `paths.body_parts_json` is missing, or the groups or
            their `PCA_n_components` are invalid (see `load_pca_n_components`).
    """
    cfg = load_encoding_yaml(config_path)
    body_parts_json = cfg.get('body_parts_json')
    if body_parts_json is None:
        raise ValueError(f'{config_path}: paths.body_parts_json is missing')
    group_set = group_set or str(cfg.get('pca_group_set', DEFAULT_PCA_GROUP_SET))
    likelihood_threshold, max_masked_frac = load_filter_settings(config_path)
    groups, n_components = load_pca_n_components(body_parts_json, group_set)
    return {
        'likelihood_threshold': likelihood_threshold,
        'max_masked_frac': max_masked_frac,
        'position_name': str(cfg.get('position_columns', DEFAULT_POSITION)),
        'position_columns': load_position_columns(config_path),
        'body_parts_json': str(body_parts_json),
        'group_set': group_set,
        'groups': groups,
        'n_components': n_components,
    }


def save_session(
    eid: str,
    camera: str,
    eks_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path,
    settings: dict[str, Any],
) -> Path:
    """Run the PCA of one session and camera and write its npz and config json.

    Args:
        eid: session id.
        camera: `'left'` or `'right'`.
        eks_dir: directory of the Pose-EKS csvs.
        out_dir: output directory.
        config_path: encoding yaml the settings came from; its text is stored in the json.
        settings: output of `resolve_settings`.

    Returns:
        Path of the written npz.

    Raises:
        FileNotFoundError: if the csv is missing.
        ValueError: see `pca_session`.
    """
    eks_dir = Path(eks_dir).resolve()
    csv_path = eks_dir / csv_filename(camera, eid)
    if not csv_path.is_file():
        raise FileNotFoundError(f'Pose-EKS csv not found: {csv_path}')
    df = load_keypoint_csv(csv_path)
    no_filter_keypoints = load_unfiltered_keypoints(config_path, get_keypoint_names(df))
    result = pca_session(
        df, settings['groups'], settings['n_components'], settings['likelihood_threshold'],
        settings['max_masked_frac'], no_filter_keypoints, settings['position_columns'],
    )
    if result['dropped']:
        logger.info(f'  dropped keypoints: {result["dropped"]}')

    npz_path, json_path = pca_output_paths(out_dir, eid, camera)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        keypoints=np.asarray(result['keypoints']),
        group=np.asarray(result['group']),
        n_components=np.asarray(result['n_components'], dtype=np.int64),
        xy_pca=result['xy_pca'],
        camera=np.asarray(camera),
        eid=np.asarray(eid),
    )
    manifest_path = eks_dir.parent / 'manifest.json'
    config = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'eid': eid,
        'camera': camera,
        'csv_path': str(csv_path),
        'n_frames': int(len(df)),
        'eks_run_manifest': (
            json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest_path.is_file() else None
        ),
        'config_path': str(Path(config_path).resolve()),
        'config_yaml': Path(config_path).read_text(encoding='utf-8'),
        'pipeline': list(PIPELINE),
        'settings': {
            'likelihood_threshold': settings['likelihood_threshold'],
            'max_masked_frac': settings['max_masked_frac'],
            'no_filter_keypoints': no_filter_keypoints,
            'position_columns_name': settings['position_name'],
            'position_columns': list(settings['position_columns']),
            'body_parts_json': settings['body_parts_json'],
            'group_set': settings['group_set'],
            'groups': settings['groups'],
            'pca_n_components_requested': settings['n_components'],
        },
        'masked_fraction': result['masked_fraction'],
        'dropped_keypoints': result['dropped'],
        'unassigned_keypoints': result['unassigned'],
        'pca': result['info']['groups'],
        'npz_path': str(npz_path),
        'npz_contents': {
            'keypoints': 'keypoint names, axis 1 of xy_pca',
            'group': 'body-part group of each keypoint',
            'n_components': 'effective number of components of the keypoint\'s group',
            'xy_pca': '[n_frames, n_keypoints, 2] reprojected x/y in pixels, float32',
        },
    }
    json_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    logger.info(f'  wrote {npz_path} and {json_path.name}')
    return npz_path


def main(argv: list[str] | None = None) -> None:
    """Run the PCA for every requested session and camera.

    A failing session/camera is logged and skipped; the others still run.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        ValueError: if no sessions are selected, or the yaml / groups are invalid.
        RuntimeError: after all runs, if any session/camera failed; the message lists them.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    eids = resolve_session_eids(args.config, args.sessions, args.session_group, args.sessions_file)
    if eids is None:
        raise ValueError(
            f'no sessions: pass --sessions or --session-group, or set session.eid_group in '
            f'{args.config}',
        )
    settings = resolve_settings(args.config, args.group_set)
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(args.eks_dir).resolve().parent / 'pca' / settings['group_set']
    )
    logger.info(
        f'group set {settings["group_set"]}, position {settings["position_name"]}, '
        f'likelihood >= {settings["likelihood_threshold"]}, masked <= '
        f'{settings["max_masked_frac"]}; output {out_dir}',
    )

    failures: list[str] = []
    n_total = len(eids) * len(args.cameras)
    for eid in eids:
        for camera in dict.fromkeys(args.cameras):
            logger.info(f'[{eid} {camera}] PCA')
            try:
                save_session(eid, camera, args.eks_dir, out_dir, args.config, settings)
            except Exception as exc:  # keep going; failures are reported below
                logger.exception(f'[{eid} {camera}] failed')
                failures.append(f'{eid}/{camera}: {type(exc).__name__}: {exc}')
    logger.info(f'{n_total - len(failures)}/{n_total} runs succeeded')
    if failures:
        raise RuntimeError(f'{len(failures)} of {n_total} runs failed:\n' + '\n'.join(failures))


if __name__ == '__main__':
    main()
