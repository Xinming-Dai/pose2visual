"""Palette and PIL canvases (title banner, legend banner) around the rainbow figures."""

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

LANDMARK_COLOR_PALETTE = [
    '#00d9ff',  # cyan
    '#ff9900',  # orange
    '#ff3cdf',  # pink
    '#6aff46',  # green
    '#b388ff',  # purple
    '#ffe135',  # yellow
    '#ff5252',  # red-orange
    '#4fc3f7',  # blue
    '#d4ff00',  # chartreuse
    '#ff8a65',  # salmon
]


def colors_for_landmarks(landmarks: list[str]) -> list[str]:
    """Assign a saturated colour to each landmark, cycling the shared palette.

    Args:
        landmarks: landmark names, in display order.

    Returns:
        one hex colour per landmark, same length and order as `landmarks`.
    """
    return [LANDMARK_COLOR_PALETTE[i % len(LANDMARK_COLOR_PALETTE)] for i in range(len(landmarks))]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the dejavu-sans-bold font, falling back to PIL's default.

    Args:
        size: font size in pixels.

    Returns:
        a PIL font object.
    """
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_title_canvas(width: int, title_text: str, canvas_height: int = 120) -> Image.Image:
    """Create a black title canvas sized to stack above the plot image.

    Args:
        width: canvas width, matching the plot image's width.
        title_text: title text (can be multi-line).
        canvas_height: height of the canvas in pixels.

    Returns:
        PIL Image of the title banner alone.
    """
    title_img = Image.new('RGB', (width, canvas_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(title_img)
    font = _load_font(24)

    lines = title_text.split('\n')
    line_height = 30
    total_height = len(lines) * line_height
    y_start = (canvas_height - total_height) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(255, 255, 255), font=font)

    return title_img


def render_legend_canvas(
    rows_legend: list[dict[str, Any]],
    width: int,
    line_height: int = 36,
) -> Image.Image:
    """Create a black legend canvas, one line per group/row, to stack below the figure.

    Each line lists that row's landmarks with their colour and kept/total frame count,
    e.g. ``eye_related:  eye: 10/20   pupil: 15/20``, so the per-landmark legend no
    longer has to be drawn on top of the scatter image.

    Args:
        rows_legend: one entry per row, in the same top-to-bottom order the rows appear
            in the figure: `{'group_name', 'landmarks', 'colors', 'counts', 'n_frames'}`
            (see `render_combined_figure`).
        width: canvas width, matching the plot image's width.
        line_height: pixel height of one row's legend line.

    Returns:
        PIL Image of the legend banner alone.
    """
    padding = 10
    height = padding * 2 + line_height * len(rows_legend)
    canvas = Image.new('RGB', (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(16)
    entry_font = _load_font(14)

    for row_idx, row in enumerate(rows_legend):
        y = padding + row_idx * line_height + (line_height - 16) // 2
        x = 12

        label = f'{row["group_name"]}:  '
        draw.text((x, y), label, fill=(255, 255, 255), font=label_font)
        x += draw.textbbox((0, 0), label, font=label_font)[2]

        for landmark, color, count in zip(row['landmarks'], row['colors'], row['counts']):
            entry = f'{landmark}: {count:,}/{row["n_frames"]:,}   '
            draw.text((x, y + 1), entry, fill=color, font=entry_font)
            x += draw.textbbox((0, 0), entry, font=entry_font)[2]

    return canvas


def compose_figure_canvas(
    image_path: Path,
    title_text: str,
    rows_legend: list[dict[str, Any]],
    canvas_height: int = 120,
    legend_line_height: int = 36,
) -> Image.Image:
    """Stack a title banner above and a legend banner below the saved plot image.

    Args:
        image_path: path to the plot image (PNG).
        title_text: title text (can be multi-line), drawn in the top banner.
        rows_legend: per-row legend entries for the bottom banner (see
            `render_legend_canvas`); panels themselves carry no landmark legend/counts.
        canvas_height: height of the title banner in pixels.
        legend_line_height: pixel height of one row's legend line in the bottom banner.

    Returns:
        Combined PIL Image: title canvas, plot image, legend canvas, top to bottom.
    """
    plot_img = Image.open(image_path)
    width, plot_height = plot_img.size

    title_img = render_title_canvas(width, title_text, canvas_height)
    legend_img = render_legend_canvas(rows_legend, width, legend_line_height)

    combined_height = title_img.height + plot_height + legend_img.height
    combined_img = Image.new('RGB', (width, combined_height))
    combined_img.paste(title_img, (0, 0))
    combined_img.paste(plot_img, (0, title_img.height))
    combined_img.paste(legend_img, (0, title_img.height + plot_height))

    return combined_img
