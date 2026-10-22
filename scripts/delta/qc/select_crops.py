"""Draw a zoom box per body-part group and camera with the mouse; print a `crop_boxes` json block.

Standalone: needs only matplotlib, numpy and Pillow (not the `pose2visual` package, nor a GPU or
the cluster), so copy this file, `body_parts.json` and the pngs of `extract_crop_frames.py` to a
machine with a display. The printed block is pasted into the top level of `body_parts.json`; the QC
video renderer then zooms each group to its box.

Per group and camera: drag a box, press Enter to accept (the next image opens), `s` to skip (no
zoom for that group and camera). Boxes are in source-pixel coordinates; the renderer widens them to
the frame's aspect ratio.

Example:
    python select_crops.py --frames-dir crop_frames --groups-file body_parts.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image

CAMERAS = ('left', 'right')


def find_frame(frames_dir: Path, camera: str) -> Path:
    """Find the single `crop_frame.*.<camera>.png` in a folder.

    Args:
        frames_dir: folder written by `extract_crop_frames.py`.
        camera: `'left'` or `'right'`.

    Returns:
        the png path.

    Raises:
        FileNotFoundError: if there is no such png, or more than one.
    """
    paths = sorted(frames_dir.glob(f'crop_frame.*.{camera}.png'))
    if len(paths) != 1:
        raise FileNotFoundError(f'expected one crop_frame.*.{camera}.png in {frames_dir}: {paths}')
    return paths[0]


def pick_box(image: np.ndarray, title: str) -> list[int] | None:
    """Show an image and let the user drag one box.

    Args:
        image: frame to show.
        title: window title.

    Returns:
        `[x0, y0, x1, y1]` in image pixels, or None if skipped or nothing was drawn.
    """
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


def main() -> None:
    """Loop over groups and cameras and print the json block."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--frames-dir', required=True, help='folder with the crop_frame pngs')
    parser.add_argument('--groups-file', required=True, help='body_parts.json')
    parser.add_argument(
        '--group-set', default='nonoverlapping_analysis_groups', help='group set to cover',
    )
    parser.add_argument('--cameras', nargs='+', choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument('--out', default=None, help='also write the block to this json file')
    args = parser.parse_args()

    groups = json.loads(Path(args.groups_file).read_text())[args.group_set]
    images = {
        camera: np.asarray(Image.open(find_frame(Path(args.frames_dir), camera)).convert('RGB'))
        for camera in args.cameras
    }
    boxes: dict[str, dict[str, list[int]]] = {}
    for group in groups:
        for camera in args.cameras:
            box = pick_box(images[camera], f'{group} ({camera})')
            if box is not None:
                boxes.setdefault(group, {})[camera] = box
    block = {'crop_boxes': {args.group_set: boxes}}
    text = json.dumps(block, indent=2)
    print(f'\npaste into body_parts.json (top level):\n{text}')
    if args.out:
        Path(args.out).write_text(text + '\n')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
