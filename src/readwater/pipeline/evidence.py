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
