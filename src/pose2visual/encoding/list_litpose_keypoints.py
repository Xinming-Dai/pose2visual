"""Step A: list the keypoint names of every Lightning Pose prediction csv.

For each ``_iblrig_<camera>Camera.downsampled.<eid>.csv`` in ``--video-preds-dir`` (default: the
``video_preds_dir`` of ``--config``) this writes
``<video-preds-dir>/encoding/keypoint_names/<camera>_<eid>.json`` with the keypoint names (second
csv header row), the keypoints that are usable, and the keypoints skipped because too many of
their frames are masked (NaN ``x``/``y`` or ``likelihood`` below the likelihood threshold;
keypoints on the far side of the animal have likelihood near 0 and are skipped this way). The
sessions are ``--eids``, else ``--eid-group`` (a named group of an ``ibl_session_ids.json``-style
file, ``--eid-groups-json``), else the ``session.eid_group`` of ``--config`` (file:
``paths.eid_groups_json``); with none of them every csv in the directory is listed. Jsons of
sessions that are not selected are left untouched. The json files tell
which keypoints can go in the ``base_keypoints`` / ``additional_keypoints`` lists of the
encoding config, and are read by ``extract_litpose_dynamic_vars_trials``.

``likelihood_threshold`` and ``max_masked_frac`` come from ``--config`` (the encoding yaml), the
single source shared by all steps; ``--likelihood-threshold`` / ``--max-masked-frac`` override them
for this run only. Steps B and C fail when their config disagrees with the json written here, so
rerun this step after changing either number in the yaml.

When the yaml has a ``merge`` section, this step first builds the csvs of ``paths.video_preds_dir``
from the models listed in ``merge.sources`` (see ``merge_litpose_keypoints``) for the selected
sessions (required) and ``--cameras``, then lists them. ``--video-preds-dir`` must then be the
yaml's ``paths.video_preds_dir`` (or unset). Rerun this step after changing ``merge``.

Keypoints of the groups in the yaml's ``filter.group_set.unfiltered_groups`` (looked up in
``filter.group_set.name`` of ``paths.body_parts_json``) skip confidence filtering: their
likelihood is ignored, so only NaN ``x``/``y`` frames count as masked and ``max_masked_frac``
applies to those alone. They are listed in the json as ``no_filter_keypoints``; steps B and C fail when the yaml's
list differs from it.

Groups of the yaml's ``visibility`` section (e.g. the tongue, which is often hidden) do not become
speed features: their keypoints are neither usable nor unusable here, whatever their masked
fraction. Step B turns each group into one binary ``<group>_visible`` feature (1 when any of its
keypoints has a likelihood above ``visibility.likelihood_threshold``). The json lists them as
``visibility`` (threshold and feature -> keypoints) with the share of visible frames as
``visible_fraction``; steps B and C fail when the yaml's ``visibility`` differs from it.

Usage::

    PYTHONPATH=src python -m pose2visual.encoding.list_litpose_keypoints \\
      --config configs/encoding/litpose_encoding.yaml \\
      [--video-preds-dir <model_dir>/video_preds] [--likelihood-threshold 0.9] \\
      [--max-masked-frac 0.3] [--eids <eid1> <eid2> | --eid-group sable] [--cameras left right]
"""

import argparse
import json
import logging
from pathlib import Path

from pose2visual.encoding.litpose_keypoints import (
    CAMERAS,
    DEFAULT_POSITION,
    POSITION_COLUMNS,
    frame_fractions,
    get_keypoint_names,
    keypoint_names_json_path,
    list_keypoint_csvs,
    load_filter_settings,
    load_keypoint_csv,
    load_merge_settings,
    load_position_columns,
    load_unfiltered_keypoints,
    load_video_preds_dir,
    load_visibility_features,
    resolve_session_eids,
    validate_filter_settings,
    visibility_features_from_df,
)
from pose2visual.encoding.merge_litpose_keypoints import merge_sessions

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='List the keypoint names of every Lightning Pose prediction csv.',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='encoding yaml holding likelihood_threshold and max_masked_frac',
    )
    parser.add_argument(
        '--video-preds-dir',
        type=str,
        default=None,
        help='directory holding _iblrig_<camera>Camera.downsampled.<eid>.csv files (default: '
        'video_preds_dir of --config)',
    )
    parser.add_argument(
        '--eids',
        type=str,
        nargs='+',
        default=None,
        help='session ids; mutually exclusive with --eid-group '
        '(default: session.eid_group of --config, else every csv in the directory)',
    )
    parser.add_argument(
        '--eid-group',
        type=str,
        default=None,
        help='name of a group in --eid-groups-json\'s `neural_sessions` (e.g. `sable`, `all`, '
        '`ear_both`, `ear_left`); mutually exclusive with --eids',
    )
    parser.add_argument(
        '--eid-groups-json',
        type=str,
        default=None,
        help='path to an `ibl_session_ids.json`-style file; '
        'default: paths.eid_groups_json of --config',
    )
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        choices=CAMERAS,
        default=list(CAMERAS),
        help='cameras to merge when --config has a merge section (default: left right); '
        'ignored otherwise, where every csv of the selected sessions is listed',
    )
    parser.add_argument(
        '--likelihood-threshold',
        type=float,
        default=None,
        help='overrides likelihood_threshold of --config for this run; frames below this '
        'likelihood count as masked',
    )
    parser.add_argument(
        '--max-masked-frac',
        type=float,
        default=None,
        help='overrides max_masked_frac of --config for this run; skip a keypoint when the '
        'fraction of masked frames (NaN x/y or low likelihood) exceeds this',
    )
    return parser.parse_args(argv)


def build_keypoint_info(
    csv_path: Path,
    camera: str,
    eid: str,
    likelihood_threshold: float,
    max_masked_frac: float,
    config_path: str | Path | None = None,
) -> dict:
    """Summarize the keypoints of one prediction csv.

    Args:
        csv_path: path of the csv.
        camera: `'left'` or `'right'`.
        eid: session id.
        likelihood_threshold: frames below this likelihood count as masked.
        max_masked_frac: keypoints with a larger masked fraction are not usable.
        config_path: encoding yaml whose `filter.group_set.unfiltered_groups` names the
            groups exempt from confidence filtering and whose `visibility` section names the
            groups turned into binary features; `None` filters every keypoint and has no binary
            feature. For an exempt keypoint the likelihood is ignored, so `masked_fraction`
            counts NaN x/y frames only. Its `position.columns` picks the x/y columns that are
            checked for NaN (`None` = `x`, `y`).

    Returns:
        Json-serializable dict with `csv_path`, `video_preds_dir`, `model_dir`, `camera`, `eid`,
        `n_frames`, `likelihood_threshold`, `max_masked_frac`, `no_filter_keypoints`,
        `position_columns` (the x/y csv columns the yaml's `position.columns` selects),
        `keypoints`, per-keypoint `masked_fraction`, `nan_fraction` and
        `low_likelihood_fraction`, `usable`, `unusable` (names, header order),
        `unusable_reasons` (name -> text) and `skipped_masked` (name -> masked fraction, same
        keys as `unusable`). `visibility` (output of `load_visibility_features`, `None` without a
        `visibility` section) and `visible_fraction` (feature name -> share of visible frames,
        empty without the section) describe the binary features; the keypoints behind them are
        in neither `usable` nor `unusable`.
    """
    df = load_keypoint_csv(csv_path)
    keypoints = get_keypoint_names(df)
    no_filter_keypoints = (
        [] if config_path is None
        else load_unfiltered_keypoints(config_path, keypoints)
    )
    visibility = None if config_path is None else load_visibility_features(config_path, keypoints)
    position_columns = (
        POSITION_COLUMNS[DEFAULT_POSITION] if config_path is None
        else load_position_columns(config_path)
    )
    visibility_keypoints = {
        name for members in (visibility or {'features': {}})['features'].values()
        for name in members
    }
    fractions = {
        name: frame_fractions(
            df, name, None if name in no_filter_keypoints else likelihood_threshold,
            position_columns,
        )
        for name in keypoints
    }
    masked_fractions = {name: frac['masked'] for name, frac in fractions.items()}
    # keypoints behind a binary feature are judged by their likelihood alone, not by this test
    skipped_masked = {
        name: frac for name, frac in masked_fractions.items()
        if frac > max_masked_frac and name not in visibility_keypoints
    }
    unusable_reasons = {
        name: (
            f'masked fraction {masked_fractions[name]:.2f} > {max_masked_frac} '
            f'(NaN x/y {fractions[name]["nan"]:.2f}, likelihood ignored)'
            if name in no_filter_keypoints else
            f'masked fraction {masked_fractions[name]:.2f} > {max_masked_frac} '
            f'(NaN x/y/likelihood {fractions[name]["nan"]:.2f}, '
            f'likelihood < {likelihood_threshold}: {fractions[name]["low_likelihood"]:.2f})'
        )
        for name in skipped_masked
    }
    video_preds_dir = csv_path.resolve().parent
    return {
        'csv_path': str(csv_path.resolve()),
        'video_preds_dir': str(video_preds_dir),
        # the lightning pose checkpoint directory that produced these predictions
        'model_dir': str(video_preds_dir.parent),
        'camera': camera,
        'eid': eid,
        'n_frames': int(len(df)),
        'likelihood_threshold': likelihood_threshold,
        'max_masked_frac': max_masked_frac,
        'no_filter_keypoints': no_filter_keypoints,
        'position_columns': list(position_columns),
        'keypoints': keypoints,
        'masked_fraction': masked_fractions,
        'nan_fraction': {name: frac['nan'] for name, frac in fractions.items()},
        'low_likelihood_fraction': {
            name: frac['low_likelihood'] for name, frac in fractions.items()
        },
        'visibility': visibility,
        'visible_fraction': (
            {}
            if visibility is None
            else {
                feature: float(values.astype(float).mean())
                for feature, values in visibility_features_from_df(df, visibility).items()
            }
        ),
        'usable': [
            name for name in keypoints
            if name not in skipped_masked and name not in visibility_keypoints
        ],
        'unusable': [name for name in keypoints if name in skipped_masked],
        'unusable_reasons': unusable_reasons,
        'skipped_masked': skipped_masked,
    }


def main(argv: list[str] | None = None) -> None:
    """Write one keypoint-names json per selected prediction csv in a `video_preds` directory.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Raises:
        FileNotFoundError: if the directory contains no prediction csv, or a selected session has
            none.
        ValueError: if the config lacks a valid `likelihood_threshold` / `max_masked_frac`, it has
            a `merge` section but no sessions are selected or `--video-preds-dir` is another
            directory than its `paths.video_preds_dir`, its
            `filter.group_set.unfiltered_groups` is invalid or not in `filter.group_set.name`, its
            `visibility` section is invalid or names a group without keypoint in a csv,
            neither `--video-preds-dir` nor the config's `video_preds_dir` is given, both
            `--eids` and `--eid-group` are given, `--eid-group` is given without
            `--eid-groups-json` or `paths.eid_groups_json`, or the group is unknown.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args(argv)
    video_preds_dir_text = args.video_preds_dir or load_video_preds_dir(args.config)
    if video_preds_dir_text is None:
        raise ValueError(
            f'no video_preds_dir: pass --video-preds-dir or set video_preds_dir in {args.config}',
        )
    video_preds_dir = Path(video_preds_dir_text)
    logger.info(f'video_preds_dir={video_preds_dir}')
    likelihood_threshold, max_masked_frac = load_filter_settings(args.config)
    if args.likelihood_threshold is not None or args.max_masked_frac is not None:
        # an override applies to this run only; the config stays the shared default
        if args.likelihood_threshold is not None:
            likelihood_threshold = args.likelihood_threshold
        if args.max_masked_frac is not None:
            max_masked_frac = args.max_masked_frac
        likelihood_threshold, max_masked_frac = validate_filter_settings(
            likelihood_threshold, max_masked_frac, 'command line',
        )
        logger.warning(
            f'overriding {args.config}: likelihood_threshold={likelihood_threshold}, '
            f'max_masked_frac={max_masked_frac}; steps B and C will reject this json unless '
            'their config holds the same values',
        )
    logger.info(
        f'likelihood_threshold={likelihood_threshold}, max_masked_frac={max_masked_frac}',
    )

    selected_eids = resolve_session_eids(
        args.config, args.eids, args.eid_group, args.eid_groups_json,
    )
    if load_merge_settings(args.config) is not None:
        # the csvs of video_preds_dir are built from several models first, then listed below
        if args.video_preds_dir is not None and (
            Path(args.video_preds_dir).resolve()
            != Path(str(load_video_preds_dir(args.config))).resolve()
        ):
            raise ValueError(
                f'--video-preds-dir {args.video_preds_dir} differs from paths.video_preds_dir '
                f'of {args.config}, where the merge section writes the merged csvs',
            )
        if selected_eids is None:
            raise ValueError(
                f'{args.config} has a merge section, which needs sessions: pass --eids or '
                '--eid-group, or set session.eid_group',
            )
        merge_sessions(args.config, selected_eids, list(dict.fromkeys(args.cameras)))

    csvs = list_keypoint_csvs(video_preds_dir)
    if not csvs:
        raise FileNotFoundError(
            f'no _iblrig_<camera>Camera.downsampled.<eid>.csv found in {video_preds_dir}',
        )
    if selected_eids is None:
        logger.info(f'no session selection: listing all {len(csvs)} csvs in {video_preds_dir}')
    else:
        eids_found = {eid for _, eid, _ in csvs}
        eids_missing = [eid for eid in selected_eids if eid not in eids_found]
        if eids_missing:
            raise FileNotFoundError(
                f'no _iblrig_<camera>Camera.downsampled.<eid>.csv in {video_preds_dir} for '
                f'{len(eids_missing)} selected session(s): {eids_missing}',
            )
        csvs = [item for item in csvs if item[1] in selected_eids]
        logger.info(f'selected {len(selected_eids)} session(s), {len(csvs)} csvs')
    for camera, eid, csv_path in csvs:
        info = build_keypoint_info(
            csv_path, camera, eid, likelihood_threshold, max_masked_frac, args.config,
        )
        json_path = keypoint_names_json_path(video_preds_dir, camera, eid)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        logger.info(
            f'{camera} {eid}: {len(info["usable"])}/{len(info["keypoints"])} usable keypoints, '
            f'unusable: {info["unusable"]}, without confidence filtering: '
            f'{info["no_filter_keypoints"]}, visible fraction: {info["visible_fraction"]} '
            f'-> {json_path}',
        )


if __name__ == '__main__':
    main()
