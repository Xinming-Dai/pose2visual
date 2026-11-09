"""Settings of the cross-correlation pipeline, read from the ``xcorr`` section of the yaml.

The section lives in the shared encoding yaml (``configs/encoding/litpose_encoding.yaml``) so
the likelihood filter, position columns and visibility features are defined once for the
encoding and the cross-correlation pipelines. The cameras are not read from the yaml: they are a
command-line choice (``--cameras``).
"""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pose2visual.encoding.litpose_keypoints import (
    load_encoding_yaml,
    load_video_preds_dir,
    resolve_session_eids,
)

DEFAULTS: dict[str, Any] = {
    'output_name': 'xcorr',
    'max_lag_sec': 1.0,
    'smooth_sigma_bins': 2.0,
    'min_fr_hz': 5.0,
    'n_permutations': 0,
    'min_shift_sec': 10.0,
    'alpha': 0.05,
    'correction': 'fdr_bh',
    'seed': 42,
    'peth_window_sec': [-1.0, 1.0],
    'peth_baseline_sec': [-1.0, -0.5],
    'peth_min_trials': 20,
}
CORRECTIONS = ('fdr_bh', 'bonferroni')
_PREFIX = 'xcorr_'

# one subfolder per experiment inside `plot_video/` (figures, videos and their guides: the folder
# to download and share) and inside `results/` (npz, csv, json, pkl); step-0 inputs stay in the
# session folder
EXPERIMENT_BASELINE = 'experiment_A_baseline'
EXPERIMENT_PCA = 'experiment_B_pca'
EXPERIMENT_RELMOTION = 'experiment_C_relative_motion'
EXPERIMENT_PCA_GROUPS = 'experiment_D_pca_groups'
PLOT_VIDEO_DIR = 'plot_video'
RESULTS_DIR = 'results'


def _check_number(
    name: str,
    value: Any,
    source: str | Path,
    minimum: float,
    maximum: float | None = None,
    integer: bool = False,
) -> None:
    """Raise when a setting is not a number in the allowed range.

    Args:
        name: setting name, used in the error message.
        value: value read from the yaml.
        source: yaml path, used in the error message.
        minimum: smallest allowed value (inclusive).
        maximum: largest allowed value (inclusive); `None` = no upper bound.
        integer: require an `int`.

    Raises:
        ValueError: if the value is not a number (booleans included), not an `int` when
            `integer` is set, or outside `[minimum, maximum]`.
    """
    kind = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ValueError(f'{source}: xcorr.{name} must be a number, got {value!r}')
    if value < minimum or (maximum is not None and value > maximum):
        upper = '' if maximum is None else f', {maximum}'
        raise ValueError(f'{source}: xcorr.{name} must be in [{minimum}{upper}], got {value!r}')


def _check_interval(
    name: str,
    value: Any,
    source: str | Path,
) -> tuple[float, float]:
    """Validate a `[start, end]` pair of seconds from the yaml.

    Args:
        name: setting name, used in the error message.
        value: value read from the yaml.
        source: yaml path, used in the error message.

    Returns:
        `(start, end)` as floats.

    Raises:
        ValueError: if the value is not two numbers (booleans excluded) with `start < end`.
    """
    ok = (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )
    if not ok or not value[0] < value[1]:
        raise ValueError(
            f'{source}: xcorr.{name} must be [start, end] in seconds with start < end, '
            f'got {value!r}',
        )
    return float(value[0]), float(value[1])


def load_xcorr_settings(config_path: str | Path) -> dict[str, Any]:
    """Read and validate the optional ``xcorr`` section of the encoding yaml.

    Args:
        config_path: path of the encoding yaml.

    Returns:
        Dict with the keys of `DEFAULTS` (`output_name`, `max_lag_sec`, `smooth_sigma_bins`,
        `min_fr_hz`, `n_permutations`, `min_shift_sec`, `alpha`, `correction`, `seed`,
        `peth_window_sec`, `peth_baseline_sec`, `peth_min_trials`); keys the yaml does not set
        take their default.

    Raises:
        ValueError: if the yaml is invalid (see `load_encoding_yaml`), a value is not a number in
            its range, `output_name` is not a non-empty string or `correction` is unknown.
    """
    cfg = load_encoding_yaml(config_path)
    settings = dict(DEFAULTS)
    for key in DEFAULTS:
        if f'{_PREFIX}{key}' in cfg:
            settings[key] = cfg[f'{_PREFIX}{key}']
    name = settings['output_name']
    if not isinstance(name, str) or not name:
        raise ValueError(f'{config_path}: xcorr.output_name must be a non-empty string')
    _check_number('max_lag_sec', settings['max_lag_sec'], config_path, 0.0)
    _check_number('smooth_sigma_bins', settings['smooth_sigma_bins'], config_path, 0.0)
    _check_number('min_fr_hz', settings['min_fr_hz'], config_path, 0.0)
    _check_number('n_permutations', settings['n_permutations'], config_path, 0, integer=True)
    _check_number('min_shift_sec', settings['min_shift_sec'], config_path, 0.0)
    _check_number('alpha', settings['alpha'], config_path, 0.0, 1.0)
    _check_number('seed', settings['seed'], config_path, 0, integer=True)
    _check_number('peth_min_trials', settings['peth_min_trials'], config_path, 1, integer=True)
    window = _check_interval('peth_window_sec', settings['peth_window_sec'], config_path)
    baseline = _check_interval('peth_baseline_sec', settings['peth_baseline_sec'], config_path)
    if not window[0] < 0 < window[1]:
        raise ValueError(
            f'{config_path}: xcorr.peth_window_sec must contain 0 (the event), got {list(window)}',
        )
    if baseline[0] < window[0] or baseline[1] > 0:
        raise ValueError(
            f'{config_path}: xcorr.peth_baseline_sec {list(baseline)} must lie inside the window '
            f'{list(window)} and end at or before the event (0)',
        )
    settings['peth_window_sec'] = list(window)
    settings['peth_baseline_sec'] = list(baseline)
    if settings['correction'] not in CORRECTIONS:
        raise ValueError(
            f'{config_path}: xcorr.correction must be one of {list(CORRECTIONS)}, '
            f'got {settings["correction"]!r}',
        )
    return settings


def resolve_eids(
    config_path: str | Path,
    eids: list[str] | None = None,
    eid_group: str | None = None,
    eid_groups_json: str | Path | None = None,
) -> list[str]:
    """Pick the sessions of a run: explicit eids, else a group, else the yaml's `session.eid`.

    Args:
        config_path: path of the encoding yaml.
        eids: session ids given on the command line, or `None`.
        eid_group: group name given on the command line, or `None`.
        eid_groups_json: `ibl_session_ids.json`-style file for `eid_group`; `None` uses
            `paths.eid_groups_json` of the yaml.

    Returns:
        The session ids without duplicates, in the given order.

    Raises:
        ValueError: if no session is selected (see `resolve_session_eids`) or neither the
            command line nor the yaml names one.
    """
    selected = resolve_session_eids(config_path, eids, eid_group, eid_groups_json)
    if eids is not None or eid_group is not None:
        return selected or []
    eid = load_encoding_yaml(config_path).get('eid')
    if eid is None:
        raise ValueError(
            f'{config_path}: no session; pass --eids / --eid-group or set session.eid',
        )
    return [str(eid)]


def xcorr_dir(config_path: str | Path, eid: str, output_name: str) -> Path:
    """Directory holding the cross-correlation inputs and outputs of one session.

    The folder is camera-free: the neural bundle is the same for every camera and the results of
    all cameras are written side by side.

    Args:
        config_path: path of the encoding yaml (`paths.video_preds_dir` is read from it).
        eid: session id.
        output_name: folder name, `xcorr.output_name`.

    Returns:
        `<video_preds_dir>/<output_name>/<eid>`.

    Raises:
        ValueError: if the yaml has no `paths.video_preds_dir`.
    """
    video_preds_dir = load_video_preds_dir(config_path)
    if video_preds_dir is None:
        raise ValueError(f'{config_path}: paths.video_preds_dir is required')
    return Path(video_preds_dir) / output_name / eid


def plot_video_dir(session_dir: str | Path, experiment: str) -> Path:
    """Folder of the figures and videos of one experiment (the folder to download and share).

    Args:
        session_dir: session folder (`xcorr_dir`).
        experiment: `EXPERIMENT_BASELINE`, `EXPERIMENT_PCA`, `EXPERIMENT_RELMOTION` or
            `EXPERIMENT_PCA_GROUPS`.

    Returns:
        `<session_dir>/plot_video/<experiment>`; not created. Each run draws into subfolders
        named `<run>_<order>` (or `<run>_decomposition`).
    """
    return Path(session_dir) / PLOT_VIDEO_DIR / experiment


def results_dir(session_dir: str | Path, experiment: str) -> Path:
    """Folder of the saved arrays, tables and reports of one experiment.

    Args:
        session_dir: session folder (`xcorr_dir`).
        experiment: `EXPERIMENT_BASELINE`, `EXPERIMENT_PCA`, `EXPERIMENT_RELMOTION` or
            `EXPERIMENT_PCA_GROUPS`.

    Returns:
        `<session_dir>/results/<experiment>`; not created. Each compute run writes into its own
        `<run>` subfolder.
    """
    return Path(session_dir) / RESULTS_DIR / experiment


def new_run_dirs(
    session_dir: str | Path,
    experiment: str,
    run: str,
) -> tuple[Path, Path]:
    """Create the results folder of one compute run and the figure folder of its experiment.

    Args:
        session_dir: session folder (`xcorr_dir`).
        experiment: experiment folder name.
        run: run name, e.g. `plots.timestamp_subdir()` or `--plots-subdir`.

    Returns:
        `(results_run_dir, plot_video_experiment_dir)`: `results/<experiment>/<run>` and
        `plot_video/<experiment>`, both created. Figures go into `<run>_<order>` subfolders of
        the second.
    """
    run_dir = results_dir(session_dir, experiment) / run
    plots_root = plot_video_dir(session_dir, experiment)
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    return run_dir, plots_root


def latest_results_dir(session_dir: str | Path, experiment: str, marker: str) -> Path:
    """Newest results run of an experiment that holds a given file.

    The newest run is the one whose `marker` file was written last (run names are New York
    timestamps, but `--plots-subdir` can name a run anything). Results written before the
    `results/` layout lie directly in `<session_dir>/<experiment>/`; that folder is used when no
    run folder has the file.

    Args:
        session_dir: session folder (`xcorr_dir`).
        experiment: experiment folder name.
        marker: a file the run must contain, e.g. `xcorr_arrays.<eid>.npz`.

    Returns:
        The folder holding `marker`.

    Raises:
        FileNotFoundError: if neither a run folder nor the legacy folder has `marker`.
    """
    root = results_dir(session_dir, experiment)
    files = [p / marker for p in root.iterdir() if (p / marker).is_file()] if root.is_dir() else []
    if files:
        return max(files, key=lambda path: (path.stat().st_mtime_ns, path.parent.name)).parent
    legacy = Path(session_dir) / experiment
    if (legacy / marker).is_file():
        return legacy
    raise FileNotFoundError(f'{marker} not found in {root}/<run>/ or in {legacy}')


@contextmanager
def freeze_config(config_path: str | Path) -> Iterator[Path]:
    """Work on a private copy of the yaml, so editing the original mid-run changes nothing.

    Every loader reads the yaml again, so without a copy an edit during a run could be applied to
    half of the work. The copy is made once, when the context is entered, and deleted on exit.

    Args:
        config_path: path of the encoding yaml.

    Yields:
        Path of the frozen copy; pass it wherever the config path is expected.

    Raises:
        FileNotFoundError: if the yaml does not exist.
    """
    source = Path(config_path)
    if not source.is_file():
        raise FileNotFoundError(f'config {source} not found')
    with tempfile.TemporaryDirectory() as tmp_dir:
        frozen = Path(tmp_dir) / source.name
        shutil.copyfile(source, frozen)
        yield frozen
