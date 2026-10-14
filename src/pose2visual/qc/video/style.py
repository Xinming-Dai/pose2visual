"""Colours and text helpers shared by the QC video renderers.

The keypoint palette follows the house style of the `pose-video` skill: one saturated BGR hue per
keypoint group, nothing white or grey. Colours are BGR because they are drawn with cv2.
"""

import cv2
import numpy as np

# (group label, BGR colour, keypoint-name prefixes); digit tips are split by side
KEYPOINT_GROUPS: list[tuple[str, tuple[int, int, int], list[str]]] = [
    ('eye', (255, 255, 0), ['eye_']),
    ('pupil', (0, 255, 255), ['pupil_']),
    ('ear', (255, 0, 255), ['ear_']),
    ('nose', (0, 140, 255), ['nose_']),
    ('whisker pad', (150, 190, 0), ['pad_']),
    ('mouth/lips', (255, 130, 60), ['mouth', 'lowerlip', 'upperlip_']),
    ('tongue', (170, 110, 255), ['tongue_']),
    ('wrist', (255, 0, 127), ['wrist_']),
    ('digit tips R', (0, 255, 120), [f'd{idx}_tip_right' for idx in range(1, 5)]),
    ('digit tips L', (0, 0, 255), [f'd{idx}_tip_left' for idx in range(1, 5)]),
]
UNKNOWN_COLOR = (200, 200, 200)
TEXT_COLOR = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def keypoint_color(keypoint: str) -> tuple[int, int, int]:
    """Return the BGR colour of a keypoint's group.

    Args:
        keypoint: keypoint name as it appears in the csv header, e.g. `ear_top_left`.

    Returns:
        the group colour, or light grey for a name outside every group.
    """
    for _, color, prefixes in KEYPOINT_GROUPS:
        if keypoint.startswith(tuple(prefixes)):
            return color
    return UNKNOWN_COLOR


def put_text(
    img: np.ndarray,
    xy: tuple[int, int],
    text: str,
    font_scale: float = 0.5,
    color: tuple[int, int, int] = TEXT_COLOR,
) -> int:
    """Draw anti-aliased text in place.

    Args:
        img: BGR image, modified in place.
        xy: bottom-left corner of the text.
        text: string to draw.
        font_scale: cv2 font scale.
        color: BGR colour.

    Returns:
        the x coordinate just after the text.
    """
    cv2.putText(img, text, xy, FONT, font_scale, color, 1, cv2.LINE_AA)
    return xy[0] + cv2.getTextSize(text, FONT, font_scale, 1)[0][0]


def fit_font_scale(
    text: str,
    width: int,
    font_scale: float = 0.5,
    floor: float = 0.28,
) -> float:
    """Return the largest font scale at most `font_scale` at which `text` fits in `width`.

    Args:
        text: string to measure.
        width: available width in pixels.
        font_scale: starting (maximum) font scale.
        floor: smallest scale to fall back to; the text may still overflow at this scale.

    Returns:
        the font scale.
    """
    while cv2.getTextSize(text, FONT, font_scale, 1)[0][0] > width and font_scale > floor:
        font_scale -= 0.02
    return font_scale


def draw_legend_row(
    canvas: np.ndarray,
    keypoints: list[str],
    xy: tuple[int, int] = (8, 20),
    radius: int = 5,
) -> int:
    """Draw one row of group swatches, only for groups present in `keypoints`.

    Args:
        canvas: BGR image, modified in place.
        keypoints: keypoint names being drawn.
        xy: left end of the row (x) and text baseline (y).
        radius: swatch radius in pixels.

    Returns:
        the x coordinate just after the row.
    """
    x, y = xy
    for name, color, prefixes in KEYPOINT_GROUPS:
        if not any(keypoint.startswith(tuple(prefixes)) for keypoint in keypoints):
            continue
        cv2.circle(canvas, (x + radius, y - 4), radius, color, -1, cv2.LINE_AA)
        x = put_text(canvas, (x + 2 * radius + 3, y), name, fit_font_scale(name, 200, 0.42)) + 12
    return x
