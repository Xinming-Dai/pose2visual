"""Draw a zoom box per body-part group and camera with the mouse; write the QC crop boxes json.

Standalone: needs only matplotlib, numpy and Pillow (not the `pose2visual` package, nor a GPU or
the cluster), so copy this file, `body_parts.json` and the pngs of `extract_crop_frames.py` to a
machine with a display. The boxes go into the crop boxes file (`paths.crop_boxes_json`,
`scripts/delta/data/qc_crop_boxes.json`), which has the top-level keys

    "default":  [group_set][group][camera] = [x0, y0, x1, y1]   (used for every session)
    "sessions": [eid][group_set][group][camera] = [x0, y0, x1, y1]   (where a session differs)

and the QC video renderer zooms each group to its box (per camera the session's box wins over the
default). With `--session <eid>` the boxes are stored under that session, without it as default.
With `--out FILE` they are merged into FILE if it exists: only the picked groups of that session
(or of the default) and group set are replaced, everything else in the file stays.

Per group and camera: drag a box, press Enter to accept (the next image opens), `s` to skip (no
box for that group and camera, an earlier one in the file is kept). Boxes are in source-pixel
coordinates; the renderer widens them to the frame's aspect ratio.

Example (a new session):
    python scripts/delta/qc/extract_crop_frames.py --session <eid> --out-dir crop_frames
    python scripts/delta/qc/select_crops.py --frames-dir crop_frames \\
      --groups-file scripts/delta/data/body_parts.json --session <eid> \\
      --out scripts/delta/data/qc_crop_boxes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

CAMERAS = ('left', 'right')


def find_frame(frames_dir: Path, camera: str, session: str | None = None) -> Path:
    """Find the single `crop_frame.<session>.<camera>.png` in a folder.

    Args:
        frames_dir: folder written by `extract_crop_frames.py`.
        camera: `'left'` or `'right'`.
        session: session id in the file name; None matches any session.

    Returns:
        the png path.

    Raises:
        FileNotFoundError: if there is no such png, or more than one.
    """
    pattern = f'crop_frame.{session or "*"}.{camera}.png'
    paths = sorted(frames_dir.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(f'expected one {pattern} in {frames_dir}: {paths}')
    return paths[0]


def pick_box(image: np.ndarray, title: str) -> list[int] | None:
    """Show an image and let the user drag one box.

    Args:
        image: frame to show.
        title: window title.

    Returns:
        `[x0, y0, x1, y1]` in image pixels, or None if skipped or nothing was drawn.
    """
    # lazy: only the picker needs a display and matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f'{title}\ndrag a box; Enter = accept, s = skip')
    state = {'box': None, 'skip': False}

    def on_select(click, release) -> None:
        x0, x1 = sorted((click.xdata, release.xdata))
        y0, y1 = sorted((click.ydata, release.ydata))
        state['box'] = [round(x0), round(y0), round(x1), round(y1)]

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], interactive=True, minspanx=3, minspany=3,
    )

    def on_key(event) -> None:
        if event.key == 's':
            state['box'], state['skip'] = None, True
        if event.key in ('enter', 's'):
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    del selector
    return None if state['skip'] else state['box']


def merge_boxes(
    existing: dict,
    group_set: str,
    session: str | None,
    boxes: dict[str, dict[str, list[int]]],
) -> dict:
    """Put picked boxes into the content of a crop boxes file, leaving everything else alone.

    Args:
        existing: content of the file (an empty dict for a new file).
        group_set: group set the groups belong to.
        session: session id the boxes are for; None for the `default` entry.
        boxes: group -> camera -> `[x0, y0, x1, y1]`, only the boxes that were picked.

    Returns:
        a new dict with `default` and `sessions` keys; per group in `boxes` the cameras given
        replace the earlier ones of that session (or default) and group set.
    """
    merged = json.loads(json.dumps(existing))
    merged.setdefault('default', {})
    merged.setdefault('sessions', {})
    target = merged['default'] if session is None else merged['sessions'].setdefault(session, {})
    groups = target.setdefault(group_set, {})
    for group, cameras in boxes.items():
        groups.setdefault(group, {}).update(cameras)
    return merged


def format_boxes_json(content: dict | list | str | float, indent: int = 0) -> str:
    """Dump a crop boxes file with each `[x0, y0, x1, y1]` box on one line.

    Args:
        content: dict as returned by `merge_boxes` (or any json value inside it).
        indent: nesting level, two spaces each; leave at 0 for a whole file.

    Returns:
        the json text; for a whole file (`indent=0` and a dict) it ends with a newline.
    """
    pad = '  ' * indent
    if isinstance(content, list):
        return '[' + ', '.join(json.dumps(value) for value in content) + ']'
    if isinstance(content, dict):
        if not content:
            return '{}'
        items = [
            f'{pad}  {json.dumps(key)}: {format_boxes_json(value, indent + 1)}'
            for key, value in content.items()
        ]
        text = '{\n' + ',\n'.join(items) + f'\n{pad}}}'
        return text + '\n' if indent == 0 else text
    return json.dumps(content)


def main() -> None:
    """Loop over groups and cameras and write the json."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--frames-dir', required=True, help='folder with the crop_frame pngs')
    parser.add_argument('--groups-file', required=True, help='body_parts.json')
    parser.add_argument(
        '--group-set', default='nonoverlapping_analysis_groups', help='group set to cover',
    )
    parser.add_argument('--cameras', nargs='+', choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument(
        '--session', default=None, help='session id the boxes are for (default: all sessions)',
    )
    parser.add_argument(
        '--out', default=None,
        help='crop boxes json to write; merged into it if it exists (default: only print)',
    )
    args = parser.parse_args()
    from PIL import Image  # lazy: not needed by merge_boxes

    groups = json.loads(Path(args.groups_file).read_text())[args.group_set]
    images = {
        camera: np.asarray(Image.open(find_frame(Path(args.frames_dir), camera, args.session)).convert('RGB'))
        for camera in args.cameras
    }
    boxes: dict[str, dict[str, list[int]]] = {}
    for group in groups:
        for camera in args.cameras:
            box = pick_box(images[camera], f'{group} ({camera})')
            if box is not None:
                boxes.setdefault(group, {})[camera] = box
    existing = {}
    if args.out and Path(args.out).is_file():
        existing = json.loads(Path(args.out).read_text())
    text = format_boxes_json(merge_boxes(existing, args.group_set, args.session, boxes))
    if args.out:
        Path(args.out).write_text(text)
        print(f'wrote {args.out}')
    else:
        print(f'\ncrop boxes json (pass --out to write it):\n{text}')


if __name__ == '__main__':
    main()
