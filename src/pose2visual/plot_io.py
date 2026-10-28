"""Shared figure saving: every figure is written as png and as vector pdf, in separate folders.

A figure with stem ``<dir>/<name>`` is saved as ``<dir>/png/<name>.png`` and
``<dir>/pdf/<name>.pdf``, so a figure folder can be shared as a whole in either format. Callers
can also ask for ``<dir>/svg/<name>.svg``.
"""

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

PNG_DIR = 'png'
PDF_DIR = 'pdf'
SVG_DIR = 'svg'
FIGURE_SUFFIXES = ('.png', '.pdf', '.svg')


def strip_figure_suffix(path: str | Path) -> Path:
    """Drop a trailing ``.png``, ``.pdf`` or ``.svg`` from a path, leaving other dots alone.

    Args:
        path: output path, with or without a figure suffix.

    Returns:
        The path as a stem. A dot inside the name (the session id is separated from other parts
        by dots) is not a suffix, so `Path.with_suffix` is not used.
    """
    path = Path(path)
    if path.suffix.lower() in FIGURE_SUFFIXES:
        return path.with_name(path.name[:-4])
    return path


def save_figure(
    fig: Figure,
    stem: str | Path,
    dpi: int = 200,
    pdf: bool = True,
    svg: bool = False,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save a figure as png in `png/`, as pdf in `pdf/` (unless `pdf` is `False`) and as svg.

    The svg is written to `svg/` only when `svg` is `True`.

    Args:
        fig: the figure.
        stem: output path without suffix; the `png/`, `pdf/` and `svg/` folders are created next
            to it.
        dpi: resolution of the png (the pdf and svg are vector, apart from embedded images).
        pdf: also write the pdf.
        svg: also write the svg.
        **savefig_kwargs: passed to `Figure.savefig`, e.g. `bbox_inches='tight'`.

    Returns:
        The written paths: the png, then (if `pdf`) the pdf, then (if `svg`) the svg.
    """
    stem = Path(stem)
    paths = [stem.parent / PNG_DIR / f'{stem.name}.png']
    if pdf:
        paths.append(stem.parent / PDF_DIR / f'{stem.name}.pdf')
    if svg:
        paths.append(stem.parent / SVG_DIR / f'{stem.name}.svg')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, **savefig_kwargs)
    return paths
