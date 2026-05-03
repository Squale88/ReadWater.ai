"""Ground-truth evidence builder for Claude's discovery prompt.

Converts the stack of per-tile binary masks produced by our data-source
pipelines (water, charted channel, surveyed oyster reef, surveyed seagrass)
into per-grid-cell coverage fractions and formats them as a prompt section
Claude can reason over.

Typical flow from the agent side:

    evidence = build_evidence_table(
        {
            "water":    "path/to/root-10-8_water_mask.png",
            "channel":  "path/to/root-10-8_channel_mask.png",
            "oyster":   "path/to/root-10-8_oyster_mask.png",
            "seagrass": "path/to/root-10-8_seagrass_mask.png",
        },
        grid_rows=8, grid_cols=8,
    )
    section = format_evidence_for_prompt(evidence)
    # -> inject `section` into the discovery prompt template

Masks are read as grayscale PNGs: any pixel > 127 counts as the feature.
All four layer names are optional — pass only the ones you have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Layer display names, in the order they appear in the evidence table.
LAYER_ORDER = ("water", "channel", "oyster", "seagrass")

# Header labels shown in the table rendered for Claude.
_HEADER = {
    "water": "water",
    "channel": "channel",
    "oyster": "oyster",
    "seagrass": "seagrass",
}

# Cells with no "notable" layer above this fraction are summarized as a group
# rather than listed individually, to keep the prompt compact.
_DEFAULT_NOTABLE_FRACTION = 0.05

# Layer interpretation notes injected into the prompt below the table so
# Claude knows how to weight each signal.
_LAYER_NOTES = (
    "Layer definitions (read these before interpreting the table):\n"
    "  water%    — NDWI from NAIP 4-band imagery. High precision open-water\n"
    "              vs land/vegetation signal.\n"
    "  channel%  — NOAA ENC charted navigation channels (fairways, dredged\n"
    "              areas, deep-water depth polygons). ANY non-zero value means\n"
    "              this cell contains a maintained BOATING CHANNEL and must\n"
    "              NOT be labeled as a drain, creek mouth, or cut.\n"
    "  oyster%   — FWC surveyed wild oyster reef polygons (NOT aquaculture\n"
    "              leases). >0 is strong positive evidence of oyster structure.\n"
    "              0% means 'not in the surveys', NOT necessarily 'no oysters\n"
    "              present' — unmapped wild reefs can exist.\n"
    "  seagrass% — FWC surveyed SAV (compilation spanning 2007-2021). Marco\n"
    "              and Naples have seen seagrass die-off since the surveys, so\n"
    "              these values can overstate current live extent. Treat as an\n"
    "              upper bound, not a guarantee.\n"
)


def _row_label(idx: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    if idx < 26:
        return chr(ord("A") + idx)
    return chr(ord("A") + (idx // 26) - 1) + chr(ord("A") + (idx % 26))


def _read_mask(mask_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load a grayscale mask PNG and return a boolean array of `image_size`.

    Masks produced by our pipeline are 8-bit PNGs (255 = feature). We threshold
    at 127 so either thresholded binary or smoothed grayscale masks work.
    """
    img = Image.open(mask_path).convert("L")
    if img.size != image_size:
        img = img.resize(image_size, Image.NEAREST)
    return np.array(img) > 127


def compute_cell_coverage(
    mask_path: str | Path,
    grid_rows: int,
    grid_cols: int,
    image_size: tuple[int, int] = (1280, 1280),
) -> dict[str, float]:
    """For each grid cell, return the fraction of pixels that are True."""
    mask = _read_mask(mask_path, image_size)
    w, h = image_size
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    out: dict[str, float] = {}
    for row in range(grid_rows):
        y0 = int(round(row * cell_h))
        y1 = int(round((row + 1) * cell_h))
        for col in range(grid_cols):
            x0 = int(round(col * cell_w))
            x1 = int(round((col + 1) * cell_w))
            cell_slice = mask[y0:y1, x0:x1]
            frac = float(cell_slice.mean()) if cell_slice.size else 0.0
            out[f"{_row_label(row)}{col + 1}"] = frac
    return out


def build_evidence_table(
    mask_paths: dict[str, str | Path],
    grid_rows: int,
    grid_cols: int,
    image_size: tuple[int, int] = (1280, 1280),
) -> dict[str, dict[str, float]]:
    """Compute per-cell coverage for each provided layer.

    Args:
        mask_paths: layer name -> path to mask PNG. Keys should come from
            LAYER_ORDER but unknown keys are allowed and passed through.
        grid_rows, grid_cols: the discovery grid dimensions.
        image_size: expected image dimensions in pixels (defaults to 1280x1280,
            matching Google Static scale=2 at size=640 and a typical z16 cell).

    Returns:
        Nested dict: {cell_label: {layer: fraction}}. Cell labels span the
        full grid even when some layers were not provided.
    """
    coverages: dict[str, dict[str, float]] = {}
    cell_labels: list[str] | None = None
    for layer, path in mask_paths.items():
        cov = compute_cell_coverage(path, grid_rows, grid_cols, image_size)
        coverages[layer] = cov
        if cell_labels is None:
            cell_labels = list(cov.keys())

    if cell_labels is None:
        # No layers given; emit empty table with all cells at zero.
        cell_labels = [
            f"{_row_label(r)}{c + 1}"
            for r in range(grid_rows)
            for c in range(grid_cols)
        ]

    return {
        cell: {layer: coverages.get(layer, {}).get(cell, 0.0) for layer in mask_paths}
        for cell in cell_labels
    }


def format_evidence_for_prompt(
    evidence: dict[str, dict[str, float]],
    notable_layers: tuple[str, ...] | None = None,
    min_notable_fraction: float = _DEFAULT_NOTABLE_FRACTION,
) -> str:
    """Render the evidence table as a prompt section for Claude.

    Cells with any `notable_layer` >= `min_notable_fraction` are listed
    individually with all layer percentages. The remaining cells are
    summarized compactly by the water-only range so the prompt doesn't
    balloon to 64 rows when most cells are uninteresting.
    """
    if not evidence:
        return ""

    layers_present = _layers_in_order(evidence)
    if notable_layers is None:
        notable_layers = tuple(lyr for lyr in layers_present if lyr != "water")

    notable_cells: list[tuple[str, dict[str, float]]] = []
    boring_cells: list[tuple[str, dict[str, float]]] = []
    for cell, cov in evidence.items():
        triggered = any(
            cov.get(layer, 0.0) >= min_notable_fraction for layer in notable_layers
        )
        (notable_cells if triggered else boring_cells).append((cell, cov))

    lines: list[str] = []
    lines.append(
        "EVIDENCE FROM SURVEYED GROUND-TRUTH SOURCES (per grid cell, as %):"
    )
    lines.append("")
    header_cols = ["cell"] + [_HEADER[lyr] for lyr in layers_present]
    lines.append("  " + "  ".join(f"{c:>8}" for c in header_cols))

    if notable_cells:
        for cell, cov in notable_cells:
            row = [f"{cell:>8}"]
            for lyr in layers_present:
                row.append(f"{_pct(cov.get(lyr, 0.0)):>8}")
            lines.append("  " + "  ".join(row))
    else:
        lines.append("  (no cells have notable non-water signal)")

    if boring_cells:
        waters = [cov.get("water", 0.0) for _, cov in boring_cells]
        if "water" in layers_present:
            if waters:
                lines.append("")
                lines.append(
                    f"  (other {len(boring_cells)} cells: "
                    f"water {min(waters):.0%}-{max(waters):.0%}; no charted "
                    f"channel, no surveyed oyster reef, no surveyed seagrass)"
                )
        else:
            lines.append("")
            lines.append(
                f"  (other {len(boring_cells)} cells: no notable signal on any layer)"
            )

    lines.append("")
    lines.append(_LAYER_NOTES)
    return "\n".join(lines)


def _layers_in_order(evidence: dict[str, dict[str, float]]) -> list[str]:
    """Return the layer names present in the evidence, using LAYER_ORDER first."""
    any_cell = next(iter(evidence.values()))
    present = set(any_cell.keys())
    ordered = [lyr for lyr in LAYER_ORDER if lyr in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _pct(frac: float) -> str:
    """Render a fraction in the range [0, 1] as a clean percentage string."""
    if frac <= 0:
        return "0%"
    pct = int(round(frac * 100))
    if pct == 0:  # fractional but rounds to 0
        return "<1%"
    return f"{pct}%"


# --- Per-cell convenience entry point used by anchor-discovery / harnesses ---


def _default_mask_paths(cell_id: str, area_root: Path) -> dict[str, Path]:
    """Standard per-cell mask layout under data/areas/<area>_<layer>/.

    Convention from the existing data:
      data/areas/<area>_google_water/<cell_id>_water_mask.png  (preferred — Google-derived)
      data/areas/<area>_naip/<cell_id>_water_mask.png          (fallback — NAIP NDWI)
      data/areas/<area>_channels/<cell_id>_channel_mask.png
      data/areas/<area>_habitats/<cell_id>_oyster_mask.png
      data/areas/<area>_habitats/<cell_id>_seagrass_mask.png
    """
    base = area_root.parent
    area_name = area_root.name
    google_water = base / f"{area_name}_google_water" / f"{cell_id}_water_mask.png"
    naip_water = base / f"{area_name}_naip" / f"{cell_id}_water_mask.png"
    return {
        "water": google_water if google_water.exists() else naip_water,
        "channel": base / f"{area_name}_channels" / f"{cell_id}_channel_mask.png",
        "oyster": base / f"{area_name}_habitats" / f"{cell_id}_oyster_mask.png",
        "seagrass": base / f"{area_name}_habitats" / f"{cell_id}_seagrass_mask.png",
    }


def build_cell_evidence_section(
    cell_id: str,
    area_root: Path,
    grid_rows: int = 8,
    grid_cols: int = 8,
    image_size: tuple[int, int] = (1280, 1280),
    mask_paths: dict[str, Path] | None = None,
) -> str:
    """Find this cell's habitat masks on disk and produce a prompt-ready section.

    Returns a placeholder string if no mask files were found, so the caller
    can always pass the result into a prompt template that has an
    `{evidence_table}` placeholder.

    Pass `mask_paths` explicitly to override the default layout (useful for
    tests / non-standard layouts).
    """
    if mask_paths is None:
        candidates = _default_mask_paths(cell_id, area_root)
    else:
        candidates = mask_paths
    found = {layer: str(p) for layer, p in candidates.items() if Path(p).exists()}
    if not found:
        return f"(no habitat evidence masks found for {cell_id})"
    table = build_evidence_table(found, grid_rows=grid_rows,
                                 grid_cols=grid_cols, image_size=image_size)
    return format_evidence_for_prompt(table)


# --- Habitat composite overlay (image-side evidence, complement to the table) ---


# Layer color scheme for the composite. Layers paint in dict-iteration
# order, so EARLIER layers sit UNDER later ones (water under oyster +
# seagrass). Per-layer alpha lets large-area layers (water) stay subtle
# while spot layers (oyster/seagrass) pop. Picked to be distinct from
# the satellite image's natural greens/blues.
_COMPOSITE_LAYERS: dict[str, tuple[int, int, int, int]] = {
    # (R, G, B, alpha)  — water FIRST so oyster + seagrass paint over it
    "water": (80, 170, 220, 70),     # soft cyan — Google-derived water mask
    "oyster": (220, 40, 40, 110),    # red — FWC surveyed oyster reefs
    "seagrass": (60, 220, 60, 110),  # bright lime green — FWC surveyed SAV
}
# `channel` intentionally excluded from the composite per the
# evaluation-pass direction. Can be added back as another entry.


def build_habitat_composite_overlay(
    cell_id: str,
    z16_image_path: Path,
    area_root: Path,
    output_path: Path,
    mask_paths: dict[str, Path] | None = None,
    image_size: tuple[int, int] = (1280, 1280),
    with_grid: bool = False,
) -> Path | None:
    """Render z16 base + water (cyan) + oyster (red) + seagrass (lime) overlays
    as a single composite PNG.

    The composite gives a downstream model SPATIAL information about the habitat
    data — where water boundaries and FWC survey polygons sit on the image —
    rather than just per-grid-cell percentages.

    `with_grid=True` draws an 8x8 A1-H8 overlay on top of the habitat layers
    but under the legend. Style matches `structure/grid_overlay.draw_label_grid`
    so the rendered image lines up with prompts that reference grid cells.

    Returns the output_path on success, or None if the source z16 is missing
    or no habitat masks are found (composite would just be the z16, no value).

    Cache validity: the caller is responsible for deciding when to
    regenerate; this function always re-renders. Cheap (~hundreds of ms).
    """
    if not z16_image_path.exists():
        return None
    if mask_paths is None:
        mask_paths = _default_mask_paths(cell_id, area_root)

    available = {
        layer: Path(p) for layer, p in mask_paths.items()
        if layer in _COMPOSITE_LAYERS and Path(p).exists()
    }
    if not available:
        return None  # no habitat data — composite would equal the base z16

    base = Image.open(z16_image_path).convert("RGBA")
    if base.size != image_size:
        # Refuse silently — caller should ensure dims match.
        return None

    # Iterate in _COMPOSITE_LAYERS order (water first, then oyster, then
    # seagrass) so later layers paint over earlier ones — water sits as
    # a base tint with the FWC layers on top.
    overlay = Image.new("RGBA", image_size, (0, 0, 0, 0))
    rendered_layers: list[str] = []
    for layer, (r, g, b, alpha) in _COMPOSITE_LAYERS.items():
        if layer not in available:
            continue
        mask_path = available[layer]
        with Image.open(mask_path) as mask_im:
            mask_arr = np.array(mask_im.convert("L"))
        if mask_arr.shape != (image_size[1], image_size[0]):
            with Image.open(mask_path) as mask_im:
                mask_arr = np.array(
                    mask_im.convert("L").resize(image_size, Image.NEAREST)
                )
        binary = mask_arr > 127
        if not binary.any():
            continue  # layer present but empty for this cell

        rgba = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
        rgba[binary] = (r, g, b, alpha)
        layer_im = Image.fromarray(rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, layer_im)
        rendered_layers.append(layer)

    # Optional 8x8 A1-H8 grid drawn on top of habitat layers, under the
    # legend. Matches the style used by structure/grid_overlay so a paired
    # grid-mode image stays consistent.
    from PIL import ImageDraw, ImageFont
    if with_grid:
        rows, cols = 8, 8
        gw, gh = image_size
        cell_w = gw / cols
        cell_h = gh / rows
        gridlayer = Image.new("RGBA", image_size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(gridlayer, "RGBA")
        for i in range(1, cols):
            x = int(i * cell_w)
            gd.line([(x + 1, 0), (x + 1, gh)], fill=(0, 0, 0, 200), width=1)
            gd.line([(x, 0), (x, gh)], fill=(255, 255, 255, 230), width=2)
        for j in range(1, rows):
            y = int(j * cell_h)
            gd.line([(0, y + 1), (gw, y + 1)], fill=(0, 0, 0, 200), width=1)
            gd.line([(0, y), (gw, y)], fill=(255, 255, 255, 230), width=2)
        try:
            grid_font = ImageFont.truetype("arial.ttf", max(10, int(min(cell_w, cell_h) * 0.30)))
        except (OSError, IOError):
            grid_font = ImageFont.load_default()
        for r in range(rows):
            for c in range(cols):
                label = f"{chr(ord('A') + r)}{c + 1}"
                cx = int((c + 0.5) * cell_w)
                cy = int((r + 0.5) * cell_h)
                bbox_t = grid_font.getbbox(label)
                tw = bbox_t[2] - bbox_t[0]
                th = bbox_t[3] - bbox_t[1]
                tx = cx - tw // 2
                ty = cy - th // 2
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx or dy:
                            gd.text((tx + dx, ty + dy), label, fill="black", font=grid_font)
                gd.text((tx, ty), label, fill="white", font=grid_font)
        overlay = Image.alpha_composite(overlay, gridlayer)

    # Legend in bottom-right corner so a viewer (model or human) knows what
    # each color is.
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
    if rendered_layers:
        line_h = 24
        box_w = 240
        box_h = line_h * len(rendered_layers) + 16
        box_x0 = image_size[0] - box_w - 8
        box_y0 = image_size[1] - box_h - 8
        draw.rectangle(
            [box_x0, box_y0, box_x0 + box_w, box_y0 + box_h],
            fill=(0, 0, 0, 200), outline=(255, 255, 255, 255), width=2,
        )
        ty = box_y0 + 8
        layer_label = {
            "water": "Water (Google)",
            "oyster": "Oyster (FWC)",
            "seagrass": "Seagrass (FWC)",
        }
        for layer in rendered_layers:
            r, g, b, _ = _COMPOSITE_LAYERS[layer]
            sx = box_x0 + 8
            draw.rectangle([sx, ty + 2, sx + 16, ty + 18], fill=(r, g, b, 255))
            label = layer_label.get(layer, layer)
            draw.text((sx + 24, ty), label, fill=(255, 255, 255, 255), font=font)
            ty += line_h

    composed = Image.alpha_composite(base, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output_path)
    return output_path
