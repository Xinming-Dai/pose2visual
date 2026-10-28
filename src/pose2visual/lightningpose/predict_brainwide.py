"""Prepare and run Lightning Pose prediction on the Brainwide (ONE cache layout) IBL sessions.

The Brainwide videos live in an alyx / ONE cache layout::

    <root>/<lab>/Subjects/<subject>/<date>/<number>/raw_video_data/_iblrig_<cam>Camera.raw.mp4
    <root>/<lab>/Subjects/<subject>/<date>/<number>/alf/_ibl_<cam>Camera.times.npy
    <root>/.rest/<hash>                                  ONE REST cache (holds the session ids)

whereas the rest of pose2visual (``predict_ibl``, EKS, merge, PCA) works on the IBL-2view layout
and its file names, which contain the session id. The session id is not in the Brainwide paths;
it is read offline from the ONE REST cache. This module bridges the two layouts without
touching the real model directories:

* ``prepare`` symlinks every video and timestamp file into ``<out-root>/inputs`` under the
  IBL-2view names (``_iblrig_<cam>Camera.downsampled.<eid>.mp4`` and
  ``_ibl_<cam>Camera.times.<eid>.npy``; ``downsampled`` is kept only because the downstream
  parsers require it, the videos are the raw full-resolution ones). It also builds a *shadow*
  model directory per model and draw under ``<out-root>/models/<model>/<draw>``: it symlinks
  everything of the real draw directory except the folders ``litpose`` writes to, so predictions
  land in ``<out-root>`` and never in the real ``video_preds``.
* ``predict`` runs ``litpose predict`` (through ``predict_ibl``) with one model draw on the
  staged videos of the selected sessions.

Usage::

    PYTHONPATH=src python -m pose2visual.lightningpose.predict_brainwide prepare \\
      --root /work/hdd/bfsx/yzhang39/brainwide/video/raw --out-root <out-root> [--dry-run]

    conda activate lp
    PYTHONPATH=src python -m pose2visual.lightningpose.predict_brainwide predict \\
      --out-root <out-root> --model digits --draw tfall-draw0 --sessionids <eid> \\
      [--cameras right] [--dry-run] [-- <extra litpose predict args>]
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pose2visual.lightningpose import predict_ibl

logger = logging.getLogger(__name__)

CAMERAS = ('left', 'right')
DEFAULT_ROOT = Path('/work/hdd/bfsx/yzhang39/brainwide/video/raw')
DEFAULT_OUT_ROOT = Path('/work/hdd/bfsr/xdai3/brainwide-litpose')
DEFAULT_MODEL_DIRS = {
    'digits': Path(
        '/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/'
        'ibl-digits-alora-v11/ibl-digits/mm-anchored-lora',
    ),
    'rest': Path(
        '/work/nvme/bfsr/xdai3/pose2visual/mighty-mouse-models/'
        'ibl-alora-v11/mm-anchored-lora',
    ),
}
DEFAULT_DRAWS = ('tfall-draw0', 'tfall-draw1', 'tfall-draw2')
DEFAULT_PREDICT_CONFIG = (
    Path(__file__).resolve().parents[3] / 'configs' / 'lightningpose' / 'extraction_keypoints.yaml'
)

# folders litpose (or eks) writes into a model directory; they stay out of the shadow directory
# so that every output of a shadow model lands under the out root
_MODEL_DIR_OUTPUTS = (
    'video_preds',
    'image_preds',
    'cropped_images',
    'cropped_videos',
    'exports_onnx',
    'eks',
)
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


@dataclass(frozen=True)
class BrainwideSession:
    """One Brainwide session found under the ONE cache root.

    Attributes:
        eid: session id (uuid).
        lab: lab folder name, e.g. ``'steinmetzlab'``.
        subject: subject name.
        date: session date, ``YYYY-MM-DD``.
        number: session number folder, e.g. ``'001'``.
        session_dir: directory holding ``raw_video_data`` and ``alf``.
    """

    eid: str
    lab: str
    subject: str
    date: str
    number: str
    session_dir: Path

    def video_path(self, camera: str) -> Path:
        """Return the raw mp4 path of a camera."""
        return self.session_dir / 'raw_video_data' / f'_iblrig_{camera}Camera.raw.mp4'

    def times_path(self, camera: str) -> Path:
        """Return the frame timestamp npy path of a camera."""
        return self.session_dir / 'alf' / f'_ibl_{camera}Camera.times.npy'


def _dataset_records(payload: object) -> list[dict]:
    """Extract the dataset records of one ONE REST cache file.

    Cache files hold ``[body, timestamp]``; for a dataset query ``body`` is either a paginated
    ``{'results': [...]}``, a single dataset record or a list of records. Everything else (session
    details, schema, repository listings) yields no records.

    Args:
        payload: parsed json of a cache file.

    Returns:
        dataset records, each having ``session`` and ``file_records``; empty if there are none.
    """
    body = payload[0] if isinstance(payload, list) and payload else payload
    if isinstance(body, dict) and isinstance(body.get('results'), list):
        records = body['results']
    elif isinstance(body, dict):
        records = [body]
    elif isinstance(body, list):
        records = body
    else:
        return []
    return [
        record for record in records
        if isinstance(record, dict) and record.get('session') and record.get('file_records')
    ]


def load_eid_map(root: str | Path) -> dict[tuple[str, str, str], str]:
    """Map ``(subject, date, number)`` to the session id using the ONE REST cache.

    Args:
        root: Brainwide video root holding the ``.rest`` cache folder.

    Returns:
        dict from ``(subject, date, number)`` to the session id.

    Raises:
        FileNotFoundError: if ``<root>/.rest`` does not exist.
        ValueError: if two different session ids are cached for the same session folder.
    """
    rest_dir = Path(root) / '.rest'
    if not rest_dir.is_dir():
        raise FileNotFoundError(f'ONE REST cache not found: {rest_dir}')

    eid_map: dict[tuple[str, str, str], str] = {}
    for cache_file in sorted(rest_dir.iterdir()):
        try:
            payload = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for record in _dataset_records(payload):
            eid = str(record['session']).rstrip('/').rsplit('/', 1)[-1]
            parts = str(record['file_records'][0].get('relative_path', '')).split('/')
            if not _UUID_RE.match(eid) or len(parts) < 4:
                continue
            key = (parts[0], parts[1], parts[2])
            if eid_map.setdefault(key, eid) != eid:
                raise ValueError(
                    f'conflicting session ids for {key} in {rest_dir}: {eid_map[key]} and {eid}'
                )
    return eid_map


def discover_sessions(
    root: str | Path,
    sessionids: list[str] | None = None,
) -> list[BrainwideSession]:
    """Find the sessions that have a video and timestamps for every camera.

    Args:
        root: Brainwide video root.
        sessionids: optional session ids to restrict to.

    Returns:
        sessions sorted by session id.

    Raises:
        FileNotFoundError: if ``root`` or its ``.rest`` cache does not exist.
        ValueError: if a session folder has no cached session id, or a requested session id was
            not found.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'video root not found: {root}')
    eid_map = load_eid_map(root)

    sessions = []
    # `*` does not match the hidden `.rest` folder
    for raw_dir in sorted(root.glob('*/Subjects/*/*/*/raw_video_data')):
        session_dir = raw_dir.parent
        number, date, subject = (
            session_dir.name, session_dir.parent.name, session_dir.parent.parent.name,
        )
        lab = session_dir.parent.parent.parent.parent.name
        eid = eid_map.get((subject, date, number))
        if eid is None:
            raise ValueError(f'no session id in the ONE REST cache for {session_dir}')
        session = BrainwideSession(eid, lab, subject, date, number, session_dir)
        missing = [
            path for cam in CAMERAS for path in (session.video_path(cam), session.times_path(cam))
            if not path.is_file()
        ]
        if missing:
            logger.warning(f'skipping {eid} ({session_dir}): missing {[p.name for p in missing]}')
            continue
        sessions.append(session)

    sessions.sort(key=lambda session: session.eid)
    if sessionids:
        missing_ids = sorted(set(sessionids) - {session.eid for session in sessions})
        if missing_ids:
            raise ValueError(f'session ids not found under {root}: {missing_ids}')
        sessions = [session for session in sessions if session.eid in set(sessionids)]
    return sessions


def _symlink(link: Path, target: Path, dry_run: bool = False) -> bool:
    """Create a symlink, doing nothing if it already points to the target.

    Args:
        link: symlink path to create.
        target: path the symlink points to.
        dry_run: if True, only report what would be created.

    Returns:
        True if the symlink was (or, with ``dry_run``, would be) created.

    Raises:
        FileExistsError: if ``link`` exists and is not a symlink to ``target``.
    """
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return False
        raise FileExistsError(f'{link} exists and does not point to {target}')
    if dry_run:
        print(f'{link} -> {target}')
        return True
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return True


def stage_inputs(
    sessions: list[BrainwideSession],
    out_root: str | Path,
    dry_run: bool = False,
) -> int:
    """Symlink the videos and frame timestamps into the IBL-2view layout and names.

    Creates ``<out_root>/inputs/<cam>Camera.video/_iblrig_<cam>Camera.downsampled.<eid>.mp4``,
    ``<out_root>/inputs/timestamps/_ibl_<cam>Camera.times.<eid>.npy`` and
    ``<out_root>/inputs/sessions.json`` (eid -> lab, subject, date, number, session_dir).

    Args:
        sessions: sessions from ``discover_sessions``.
        out_root: output root.
        dry_run: if True, only print the symlinks that would be created.

    Returns:
        number of symlinks created.

    Raises:
        FileExistsError: if a link exists that points elsewhere.
    """
    inputs = Path(out_root) / 'inputs'
    naming = predict_ibl.VideoNaming()
    n_created = 0
    for session in sessions:
        for cam in CAMERAS:
            video_link = (
                inputs
                / naming.camera_video_subdir.format(cam=cam)
                / naming.video_filename.format(cam=cam, session_id=session.eid)
            )
            times_link = inputs / 'timestamps' / f'_ibl_{cam}Camera.times.{session.eid}.npy'
            n_created += _symlink(video_link, session.video_path(cam), dry_run)
            n_created += _symlink(times_link, session.times_path(cam), dry_run)

    if not dry_run:
        info = {
            session.eid: {
                'lab': session.lab,
                'subject': session.subject,
                'date': session.date,
                'number': session.number,
                'session_dir': str(session.session_dir),
            }
            for session in sessions
        }
        (inputs / 'sessions.json').write_text(json.dumps(info, indent=2) + '\n')
    return n_created


def make_shadow_model_dir(
    real_dir: str | Path,
    shadow_dir: str | Path,
    dry_run: bool = False,
) -> int:
    """Symlink a model directory except the folders that predictions are written to.

    ``litpose predict`` writes to ``<model_dir>/video_preds``; running it on the shadow directory
    keeps the predictions out of the real model directory.

    Args:
        real_dir: real Lightning Pose model (draw) directory holding ``config.yaml``.
        shadow_dir: shadow directory to create.
        dry_run: if True, only print what would be created.

    Returns:
        number of symlinks created.

    Raises:
        FileNotFoundError: if ``real_dir`` has no ``config.yaml``.
        FileExistsError: if the shadow directory already links elsewhere.
    """
    real_dir, shadow_dir = Path(real_dir), Path(shadow_dir)
    if not (real_dir / 'config.yaml').is_file():
        raise FileNotFoundError(f'not a model directory (no config.yaml): {real_dir}')
    n_created = 0
    for entry in sorted(real_dir.iterdir()):
        if entry.name in _MODEL_DIR_OUTPUTS:
            continue
        n_created += _symlink(shadow_dir / entry.name, entry, dry_run)
    return n_created


def shadow_model_dir(out_root: str | Path, model: str, draw: str) -> Path:
    """Return the shadow model directory of one model draw.

    Args:
        out_root: output root.
        model: model name, e.g. ``'digits'`` or ``'rest'``.
        draw: draw folder name, e.g. ``'tfall-draw0'``.

    Returns:
        ``<out_root>/models/<model>/<draw>``.
    """
    return Path(out_root) / 'models' / model / draw


def _parse_model_dirs(items: list[str] | None) -> dict[str, Path]:
    """Parse ``NAME=PATH`` items, falling back to the default model directories.

    Args:
        items: ``NAME=PATH`` strings, or None for the defaults.

    Returns:
        dict from model name to real model directory.

    Raises:
        ValueError: if an item is not of the form ``NAME=PATH``.
    """
    if not items:
        return dict(DEFAULT_MODEL_DIRS)
    model_dirs = {}
    for item in items:
        name, sep, path = item.partition('=')
        if not sep or not name or not path:
            raise ValueError(f'expected NAME=PATH, got {item!r}')
        model_dirs[name] = Path(path)
    return model_dirs


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``prepare`` and ``predict`` subcommands.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    prepare = sub.add_parser('prepare', help='stage videos/timestamps and shadow model dirs')
    prepare.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='Brainwide video root')
    prepare.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT, help='output root')
    prepare.add_argument(
        '--model-dir',
        nargs='+',
        default=None,
        metavar='NAME=PATH',
        help='real model directories holding the draw folders '
        f'(default: {" ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_DIRS.items())})',
    )
    prepare.add_argument('--draws', nargs='+', default=list(DEFAULT_DRAWS), help='draw folders')
    prepare.add_argument('--sessionids', nargs='+', default=None, help='restrict to these ids')
    prepare.add_argument('--dry-run', action='store_true', help='print, do not create')

    predict = sub.add_parser('predict', help='run litpose predict for one model draw')
    predict.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT, help='output root')
    predict.add_argument('--model', required=True, help='model name, e.g. digits or rest')
    predict.add_argument('--draw', required=True, help='draw folder, e.g. tfall-draw0')
    predict.add_argument('--sessionids', nargs='+', required=True, help='session ids to predict')
    predict.add_argument(
        '--cameras',
        nargs='+',
        default=None,
        choices=CAMERAS,
        help='cameras to predict (default: all); e.g. right to finish after left is done',
    )
    predict.add_argument('--config', type=Path, default=DEFAULT_PREDICT_CONFIG)
    predict.add_argument('--dry-run', action='store_true', help='print litpose commands only')
    predict.add_argument(
        'litpose_argv',
        nargs=argparse.REMAINDER,
        help='extra args forwarded to litpose predict (use: -- --precision fp16 ...)',
    )
    return parser


def _run_prepare(args: argparse.Namespace) -> None:
    """Run the ``prepare`` subcommand.

    Args:
        args: parsed arguments.
    """
    model_dirs = _parse_model_dirs(args.model_dir)
    sessions = discover_sessions(args.root, args.sessionids)
    if not sessions:
        raise RuntimeError(f'no complete sessions under {args.root}')
    for session in sessions:
        logger.info(
            f'{session.eid} {session.lab}/{session.subject}/{session.date}/{session.number}'
        )

    n_inputs = stage_inputs(sessions, args.out_root, args.dry_run)
    n_models = 0
    for model, model_dir in model_dirs.items():
        for draw in args.draws:
            n_models += make_shadow_model_dir(
                model_dir / draw,
                shadow_model_dir(args.out_root, model, draw),
                args.dry_run,
            )
    verb = 'would create' if args.dry_run else 'created'
    logger.info(f'{verb} {n_inputs} input symlinks and {n_models} shadow model symlinks')


def _run_predict(args: argparse.Namespace) -> None:
    """Run the ``predict`` subcommand through ``predict_ibl``.

    Args:
        args: parsed arguments.

    Raises:
        FileNotFoundError: if the shadow model directory was not prepared.
    """
    shadow = shadow_model_dir(args.out_root, args.model, args.draw)
    if not (shadow / 'config.yaml').is_file():
        raise FileNotFoundError(f'shadow model dir not prepared (run `prepare`): {shadow}')

    extra = list(args.litpose_argv)
    if extra and extra[0] == '--':
        extra = extra[1:]
    argv = [
        '--root', str(Path(args.out_root) / 'inputs'),
        '--model-dir', str(shadow),
        '--config', str(args.config),
        '--sessionids', *args.sessionids,
        '--skip-existing',
    ]
    if args.cameras:
        argv += ['--cameras', *args.cameras]
    if args.dry_run:
        argv.append('--dry-run')
    # skip overlay videos; the shadow dir doubles as the config's data_dir
    argv += ['--', '--skip-viz', '--overrides', f'data.data_dir={shadow}', *extra]
    predict_ibl.main(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    Args:
        argv: command line arguments; defaults to ``sys.argv[1:]``.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = build_parser().parse_args(argv)
    if args.command == 'prepare':
        _run_prepare(args)
    else:
        _run_predict(args)


if __name__ == '__main__':
    main()
