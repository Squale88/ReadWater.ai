"""Grid overlay drawing for satellite imagery analysis."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_grid_overlay(
    image_path: str, sections: int = 4, output_path: str | None = None,
) -> str:
    """Draw a numbered grid overlay on a satellite image.

    Creates a new image with grid lines and numbered cells (1-N, left to right,
    top to bottom). The original image is not modified.

    Args:
        image_path: Path to the source satellite image.
        sections: Grid divisions per side (4 = 4x4 = 16 cells).
        output_path: Where to save the grid image. If None, saves alongside
            the original with a _grid suffix.

    Returns:
        Path to the grid overlay image.
    """
    src = Path(image_path)
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_grid{src.suffix}"))

    img = Image.open(src).copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cell_w = w / sections
    cell_h = h / sections

    # --- Grid lines: 2px white with 1px black shadow ---
    for i in range(1, sections):
        x = int(i * cell_w)
        y = int(i * cell_h)
        # Black shadow (offset by 1px)
        draw.line([(x + 1, 0), (x + 1, h)], fill="black", width=1)
        draw.line([(0, y + 1), (w, y + 1)], fill="black", width=1)
        # White line on top
        draw.line([(x, 0), (x, h)], fill="white", width=2)
        draw.line([(0, y), (w, y)], fill="white", width=2)

    # --- Cell numbers ---
    font_size = max(12, int(h * 0.05))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    cell_num = 1
    for row in range(sections):
        for col in range(sections):
            cx = int((col + 0.5) * cell_w)
            cy = int((row + 0.5) * cell_h)
            text = str(cell_num)

            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw // 2
            ty = cy - th // 2

            # Black outline
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), text, fill="black", font=font)
            # White text
            draw.text((tx, ty), text, fill="white", font=font)

            cell_num += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path
