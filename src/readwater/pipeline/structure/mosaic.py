"""Zoom-18 tile selection, mosaic stitching, and annotated rendering.

This module provides the geometry and rendering plumbing for the structure
phase. Per the plan:

- `select_z18_centers` is the deterministic PLAN_CAPTURE step. Given an
  anchor's zoom-16 pixel bbox and center lat/lon, it returns the set of
  zoom-18 tile centers needed to cover the anchor (plus a margin).
- `Mosaic` stitches those tiles into a single PIL image and tracks each
  tile's pixel origin so pixel polygons on the mosaic can be mapped back to
  lat/lon.
- `render_annotated` draws all four zone levels on the mosaic with a legend.

No LLM involvement here. Pure geometry + Pillow.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from readwater.api.providers.base import ImageProvider
from readwater.models.structure import Z18FetchPlan
from readwater.pipeline.structure.geo import (
    METERS_PER_DEG_LAT,
    deg_lat_per_pixel,
    deg_lon_per_pixel,
    pixel_to_latlon,
)

# Google Static Maps at size=640 + scale=2 returns 1280x1280 px.
TILE_PX = 1280
Z16_CELL_PX = 1280  # the source zoom-16 image the anchor bboxes are defined on
INITIAL_GRID_CAP = 4
MAX_GRID_CAP = 5
DEFAULT_MARGIN = 0.15

# Anthropic base64 images are capped at 5 MB decoded. Claude also performs best
# at around 1.15 MP. Cap the stitched mosaic canvas at this longest dimension
# and the per-axis ratio stays intact; we scale coordinates uniformly.
MAX_MOSAIC_DIM = 1600


@dataclass
class TilePlacement:
    """A single zoom-18 tile and where it sits in the mosaic."""

    row: int
    col: int
    origin_px: tuple[int, int]  # top-left in mosaic coords
    center_latlon: tuple[float, float]
    zoom: int = 18
    image_path: str | None = None


@dataclass
class TilePlan:
    """Result of select_z18_centers — the grid of tile centers to fetch."""

    rows: int
    cols: int
    centers: list[list[tuple[float, float]]]  # [row][col] -> (lat, lon)

    def flatten(self) -> list[tuple[int, int, tuple[float, float]]]:
        out = []
        for r in range(self.rows):
            for c in range(self.cols):
                out.append((r, c, self.centers[r][c]))
        return out


def z18_tile_span(lat: float) -> tuple[float, float]:
    """Ground span (deg_lat, deg_lon) of one zoom-18 scale=2 1280px tile."""
    return (
        TILE_PX * deg_lat_per_pixel(18, lat),
        TILE_PX * deg_lon_per_pixel(18, lat),
    )


def z18_tile_meters(lat: float) -> float:
    """Ground coverage in meters of one zoom-18 scale=2 1280px tile, at `lat`.

    At Rookery Bay latitudes (~26°N) this is ~343 m per tile, not the ~150 m
    estimate that appears in `docs/PHASE_C_TASKS.md` TASK-5 (which was based
    on standard z18 raster tiles, 256 px). The 1280-px Static-Maps tile this
    codebase actually fetches is roughly twice as wide. Tests below use the
    real number.
    """
    deg_lat, _deg_lon = z18_tile_span(lat)
    return deg_lat * METERS_PER_DEG_LAT


def z18_tile_plan_from_latlon(
    anchor_center_latlon: tuple[float, float],
    rough_extent_meters: float,
    tile_budget: int = 25,
) -> Z18FetchPlan:
    """lat/lon-native PLAN_CAPTURE for Phase C v1 (TASK-5).

    Replaces the grid-cell-based path in `select_z18_centers()` for callers
    that have an anchor lat/lon and a rough extent in meters (TASK-6 wires
    it into `run_structure_phase`). The grid-cell version is still used by
    legacy IDENTIFY paths.

    Algorithm:
      1. Pad `rough_extent_meters` by 25% per the addendum / spec.
      2. Compute meters-per-tile at this latitude (lat-dependent).
      3. Pick `n` per axis = ceil(target / tile_meters); bump to odd so the
         anchor lands on a tile center, not on a tile seam.
      4. Cap at the largest odd `n` whose `n*n <= tile_budget`.
      5. Lay out an `n × n` grid in row-major order: rows N→S, cols W→E.

    Returns a `Z18FetchPlan` whose `tile_centers` are (lat, lon) pairs and
    whose `extent_meters` is the actual covered side length (which may be
    smaller than `rough_extent_meters * 1.25` when `tile_budget` clamps).
    """
    if tile_budget < 1:
        raise ValueError(f"tile_budget must be >= 1, got {tile_budget}")
    lat, lon = anchor_center_latlon

    deg_lat_tile, deg_lon_tile = z18_tile_span(lat)
    tile_m = deg_lat_tile * METERS_PER_DEG_LAT  # meters per tile (N-S)

    # Step 1: padded target extent
    target = max(rough_extent_meters, 0.0) * 1.25

    # Step 2-3: tiles per axis to cover target, centered on anchor (odd count)
    n = max(1, math.ceil(target / tile_m)) if tile_m > 0 else 1
    if n % 2 == 0:
        n += 1

    # Step 4: clamp to budget. Largest odd a with a*a <= tile_budget.
    max_axis = int(math.isqrt(tile_budget))
    if max_axis % 2 == 0:
        max_axis -= 1
    if max_axis < 1:
        max_axis = 1
    if n > max_axis:
        n = max_axis

    # Step 5: row-major centers
    half = (n - 1) / 2.0
    centers: list[tuple[float, float]] = []
    for r in range(n):
        # row 0 is northernmost (highest lat)
        row_lat = lat + (half - r) * deg_lat_tile
        for c in range(n):
            # col 0 is westernmost (lowest lon)
            col_lon = lon + (c - half) * deg_lon_tile
            centers.append((row_lat, col_lon))

    return Z18FetchPlan(
        tile_centers=centers,
        tile_budget=tile_budget,
        extent_meters=n * tile_m,
    )


def _anchor_bbox_to_latlon(
    bbox_px_z16: tuple[int, int, int, int],
    z16_center: tuple[float, float],
    img_size_px: int = Z16_CELL_PX,
    zoom_z16: int = 16,
) -> tuple[float, float, float, float]:
    """Convert a zoom-16 pixel bbox [x, y, w, h] to (north, south, east, west) latlon."""
    x, y, w, h = bbox_px_z16
    nw_lat, nw_lon = pixel_to_latlon(
        x, y, img_size_px, z16_center[0], z16_center[1], zoom_z16,
    )
    se_lat, se_lon = pixel_to_latlon(
        x + w, y + h, img_size_px, z16_center[0], z16_center[1], zoom_z16,
    )
    return (nw_lat, se_lat, max(nw_lon, se_lon), min(nw_lon, se_lon))


def select_z18_centers(
    anchor_bbox_px_z16: tuple[int, int, int, int],
    z16_center: tuple[float, float],
    continuation_edges: dict[str, bool] | None = None,
    margin: float = DEFAULT_MARGIN,
    initial_cap: int = INITIAL_GRID_CAP,
) -> TilePlan:
    """Deterministic PLAN_CAPTURE — choose the zoom-18 tile grid for an anchor.

    1. Convert anchor bbox to lat/lon.
    2. Pad by `margin` on each side so tile edges don't bisect the feature.
    3. Compute rows/cols from bbox span / tile span, clamp to [1, initial_cap].
    4. Pre-extend by one row/col on sides flagged in `continuation_edges`.
    5. Lay out centers on a regular grid centered on the anchor center.
    """
    ce = continuation_edges or {}
    north, south, east, west = _anchor_bbox_to_latlon(anchor_bbox_px_z16, z16_center)
    lat_span = abs(north - south)
    lon_span = abs(east - west)

    # Margin
    lat_span_padded = lat_span * (1.0 + 2 * margin)
    lon_span_padded = lon_span * (1.0 + 2 * margin)

    anchor_center_lat = (north + south) / 2.0
    anchor_center_lon = (east + west) / 2.0

    tile_lat_span, tile_lon_span = z18_tile_span(anchor_center_lat)

    import math

    rows = max(1, min(initial_cap, math.ceil(lat_span_padded / tile_lat_span)))
    cols = max(1, min(initial_cap, math.ceil(lon_span_padded / tile_lon_span)))

    # Pre-extend from discovery's continuation_edges
    if ce.get("north") and rows < MAX_GRID_CAP:
        rows += 1
    if ce.get("south") and rows < MAX_GRID_CAP:
        rows += 1
    if ce.get("east") and cols < MAX_GRID_CAP:
        cols += 1
    if ce.get("west") and cols < MAX_GRID_CAP:
        cols += 1

    rows = min(rows, MAX_GRID_CAP)
    cols = min(cols, MAX_GRID_CAP)

    # Center the grid on the anchor center. For odd counts the center tile
    # lands directly on the anchor; for even counts the center falls between
    # two tiles.
    def _offsets(n: int) -> list[float]:
        # Offsets in units of one tile, relative to anchor center.
        return [(i - (n - 1) / 2.0) for i in range(n)]

    row_offsets = _offsets(rows)
    col_offsets = _offsets(cols)

    centers: list[list[tuple[float, float]]] = []
    for r in row_offsets:
        row_list = []
        # row offsets: +r moves south (lower lat)
        tile_lat = anchor_center_lat - r * tile_lat_span
        for c in col_offsets:
            # col offsets: +c moves east (higher lon)
            tile_lon = anchor_center_lon + c * tile_lon_span
            row_list.append((tile_lat, tile_lon))
        centers.append(row_list)

    return TilePlan(rows=rows, cols=cols, centers=centers)


def expand_plan(
    plan: TilePlan,
    anchor_center: tuple[float, float],
    extends: dict[str, bool],
) -> TilePlan:
    """Add a row/col on flagged sides, up to MAX_GRID_CAP on each axis."""
    rows = plan.rows
    cols = plan.cols

    tile_lat_span, tile_lon_span = z18_tile_span(anchor_center[0])

    # Figure out the NW corner of the existing grid.
    first_lat = plan.centers[0][0][0]
    first_lon = plan.centers[0][0][1]

    add_north = extends.get("north", False) and rows < MAX_GRID_CAP
    add_south = extends.get("south", False) and rows < MAX_GRID_CAP
    add_west = extends.get("west", False) and cols < MAX_GRID_CAP
    add_east = extends.get("east", False) and cols < MAX_GRID_CAP

    new_rows = rows + (1 if add_north else 0) + (1 if add_south else 0)
    new_cols = cols + (1 if add_west else 0) + (1 if add_east else 0)
    new_rows = min(new_rows, MAX_GRID_CAP)
    new_cols = min(new_cols, MAX_GRID_CAP)

    # New NW anchor: one tile north/west of existing NW if expanding that way.
    new_first_lat = first_lat + tile_lat_span if add_north else first_lat
    new_first_lon = first_lon - tile_lon_span if add_west else first_lon

    centers: list[list[tuple[float, float]]] = []
    for r in range(new_rows):
        row_list = []
        tile_lat = new_first_lat - r * tile_lat_span
        for c in range(new_cols):
            tile_lon = new_first_lon + c * tile_lon_span
            row_list.append((tile_lat, tile_lon))
        centers.append(row_list)

    return TilePlan(rows=new_rows, cols=new_cols, centers=centers)


class Mosaic:
    """A stitched zoom-18 mosaic with per-tile provenance.

    Build with Mosaic.build(...). Use `pixel_to_latlon(x, y)` to convert a
    pixel in the mosaic back to a geographic coordinate.

    If the nominal stitched canvas (rows*cols * TILE_PX) exceeds MAX_MOSAIC_DIM
    on its longest axis, it is downscaled to fit. `self.scale` captures
    rendered-canvas-px per nominal-tile-px, so coordinates the LLM returns on
    the rendered image can be mapped correctly.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tiles: list[TilePlacement],
        image: Image.Image,
        scale: float = 1.0,
    ):
        self.rows = rows
        self.cols = cols
        self.tiles = tiles
        self.image = image
        self.scale = scale
        self.width, self.height = image.size

    @classmethod
    async def build(
        cls,
        plan: TilePlan,
        provider: ImageProvider,
        out_dir: Path,
        zoom: int = 18,
        throttle_s: float = 0.5,
        tile_cache: dict[tuple[float, float, int], str] | None = None,
        on_fetch: callable | None = None,
    ) -> Mosaic:
        """Fetch each tile (respecting an optional cache) and stitch them together.

        `tile_cache` is a cross-call cache: key = (round(lat,6), round(lon,6), zoom)
        -> image_path. It is mutated in place.
        `on_fetch` is called once per actual provider fetch (for budget tracking).
        """
        if tile_cache is None:
            tile_cache = {}

        out_dir.mkdir(parents=True, exist_ok=True)
        nominal_w = plan.cols * TILE_PX
        nominal_h = plan.rows * TILE_PX
        canvas = Image.new("RGB", (nominal_w, nominal_h), color=(0, 0, 0))
        placements: list[TilePlacement] = []

        first = True
        for r in range(plan.rows):
            for c in range(plan.cols):
                lat, lon = plan.centers[r][c]
                key = (round(lat, 6), round(lon, 6), zoom)
                fname = f"z{zoom}_r{r}_c{c}.png"
                target = out_dir / fname
                if key in tile_cache:
                    image_path = tile_cache[key]
                else:
                    if not first and throttle_s > 0:
                        await asyncio.sleep(throttle_s)
                    first = False
                    await provider.fetch((lat, lon), zoom, str(target))
                    image_path = str(target)
                    tile_cache[key] = image_path
                    if on_fetch is not None:
                        on_fetch()

                origin = (c * TILE_PX, r * TILE_PX)
                tile_img = Image.open(image_path).convert("RGB")
                if tile_img.size != (TILE_PX, TILE_PX):
                    tile_img = tile_img.resize((TILE_PX, TILE_PX))
                canvas.paste(tile_img, origin)
                placements.append(TilePlacement(
                    row=r, col=c, origin_px=origin, center_latlon=(lat, lon),
                    zoom=zoom, image_path=image_path,
                ))

        # Downscale canvas if it exceeds the cap. Tile origins stay in nominal
        # (pre-scale) coordinates — Mosaic methods scale incoming pixel coords.
        longest = max(nominal_w, nominal_h)
        scale = 1.0
        if longest > MAX_MOSAIC_DIM:
            scale = MAX_MOSAIC_DIM / longest
            new_size = (int(round(nominal_w * scale)), int(round(nominal_h * scale)))
            canvas = canvas.resize(new_size, Image.LANCZOS)

        return cls(plan.rows, plan.cols, placements, canvas, scale=scale)

    def save(self, path: str | Path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.image.save(str(path))
        return str(path)

    def _tile_for_pixel(self, x: float, y: float) -> TilePlacement:
        # x, y are in rendered-canvas pixel space. Convert to nominal tile
        # space to find the owning tile.
        nx = x / self.scale
        ny = y / self.scale
        col = int(nx // TILE_PX)
        row = int(ny // TILE_PX)
        col = max(0, min(self.cols - 1, col))
        row = max(0, min(self.rows - 1, row))
        for t in self.tiles:
            if t.row == row and t.col == col:
                return t
        # Fallback — shouldn't happen for a fully-built mosaic.
        return self.tiles[0]

    def pixel_to_latlon(self, x: float, y: float, zoom: int = 18) -> tuple[float, float]:
        """Convert a mosaic pixel (in rendered canvas space) to (lat, lon)."""
        tile = self._tile_for_pixel(x, y)
        nx = x / self.scale
        ny = y / self.scale
        local_x = nx - tile.origin_px[0]
        local_y = ny - tile.origin_px[1]
        return pixel_to_latlon(
            local_x, local_y, TILE_PX,
            tile.center_latlon[0], tile.center_latlon[1], zoom,
        )

    def polygon_px_to_latlon(
        self, polygon_px: list[tuple[int, int]], zoom: int = 18,
    ) -> list[tuple[float, float]]:
        return [self.pixel_to_latlon(x, y, zoom) for (x, y) in polygon_px]


# --- Rendering ---

_ZONE_COLORS = {
    "anchor": (255, 215, 0),      # gold
    "complex": (0, 191, 255),     # deep sky blue
    "influence": (50, 205, 50),   # lime green
    "subzone": (255, 69, 0),      # orange-red
}


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon_px: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int = 3,
    fill_alpha: int | None = None,
    label: str | None = None,
    font: ImageFont.ImageFont | None = None,
    dashed: bool = False,
) -> None:
    if len(polygon_px) < 3:
        return
    pts = [(int(x), int(y)) for (x, y) in polygon_px]
    if fill_alpha is not None:
        draw.polygon(pts, fill=(*color, fill_alpha))
    # Outline as a closed line loop so we can control width.
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        if dashed:
            _draw_dashed_line(draw, a, b, color, width=width, dash=14, gap=10)
        else:
            draw.line([a, b], fill=color, width=width)
    if label and font is not None:
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    draw.text((cx + dx, cy + dy), label, fill="black", font=font)
        draw.text((cx, cy), label, fill="white", font=font)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[int, int],
    b: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 3,
    dash: int = 14,
    gap: int = 10,
) -> None:
    import math as _m
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0
    length = _m.hypot(dx, dy)
    if length < 1e-6:
        return
    ux = dx / length
    uy = dy / length
    step = dash + gap
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash, length)
        p0 = (int(round(x0 + ux * pos)), int(round(y0 + uy * pos)))
        p1 = (int(round(x0 + ux * seg_end)), int(round(y0 + uy * seg_end)))
        draw.line([p0, p1], fill=color, width=width)
        pos += step


def _draw_seed_points(
    draw: ImageDraw.ImageDraw,
    label: str,
    positives: list[tuple[int, int]],
    negatives: list[tuple[int, int]],
    font: ImageFont.ImageFont,
) -> None:
    radius = 7
    for (x, y) in positives:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 220, 0),      # green
            outline=(0, 0, 0),
            width=2,
        )
    for (x, y) in negatives:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 30, 30),    # red
            outline=(0, 0, 0),
            width=2,
        )
    # Label near the first positive, slightly offset so it doesn't cover the dot.
    if positives:
        x, y = positives[0]
        tx, ty = x + radius + 2, y - radius - 14
        for dxo in (-1, 0, 1):
            for dyo in (-1, 0, 1):
                if dxo or dyo:
                    draw.text((tx + dxo, ty + dyo), label, fill="black", font=font)
        draw.text((tx, ty), label, fill="white", font=font)


def render_annotated(
    base_image: Image.Image,
    out_path: str | Path,
    anchor_polygons_px: list[tuple[str, list[tuple[int, int]]]],
    complex_polygons_px: list[tuple[str, list[tuple[int, int]]]],
    influence_polygons_px: list[tuple[str, list[tuple[int, int]]]],
    subzone_polygons_px: list[tuple[str, list[tuple[int, int]]]],
    seed_points_px: list[tuple[str, list[tuple[int, int]], list[tuple[int, int]]]] | None = None,
) -> str:
    """Render all four zone levels on top of a base image and save.

    Anchors, local-complex members, and subzones are OBSERVED geometry and
    are drawn with SOLID outlines. InfluenceZone is INTERPRETED geometry
    and is drawn with a DASHED outline plus a translucent fill.

    `seed_points_px` is a list of (label, positives, negatives). When given,
    green/red dots are drawn on top of everything else showing where the
    LLM clicked for each feature.
    """
    canvas = base_image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(20)
    small = _load_font(14)
    tiny = _load_font(12)

    # Paint order: influence (translucent + dashed) < complex < anchor < subzones < seeds.
    for label, poly in influence_polygons_px:
        _draw_polygon(
            draw, poly, _ZONE_COLORS["influence"],
            width=3, fill_alpha=60, label=label, font=small, dashed=True,
        )
    for label, poly in complex_polygons_px:
        _draw_polygon(
            draw, poly, _ZONE_COLORS["complex"],
            width=3, label=label, font=small,
        )
    for label, poly in anchor_polygons_px:
        _draw_polygon(
            draw, poly, _ZONE_COLORS["anchor"],
            width=5, label=label, font=font,
        )
    for label, poly in subzone_polygons_px:
        _draw_polygon(
            draw, poly, _ZONE_COLORS["subzone"],
            width=3, label=label, font=small,
        )

    # Seed points on top.
    if seed_points_px:
        for label, positives, negatives in seed_points_px:
            _draw_seed_points(draw, label, positives, negatives, tiny)

    # Legend
    legend_rows = [
        ("Anchor (observed)", _ZONE_COLORS["anchor"], "solid"),
        ("Member (observed)", _ZONE_COLORS["complex"], "solid"),
        ("Influence (interpreted)", _ZONE_COLORS["influence"], "dashed"),
        ("Subzone (observed)", _ZONE_COLORS["subzone"], "solid"),
        ("Seed + / -", (0, 220, 0), "dots"),
    ]
    pad = 12
    sw = 24
    row_h = 26
    legend_w = 260
    legend_h = pad * 2 + row_h * len(legend_rows)
    lx = pad
    ly = canvas.size[1] - legend_h - pad
    draw.rectangle([lx, ly, lx + legend_w, ly + legend_h], fill=(0, 0, 0, 180))
    for i, (name, color, style) in enumerate(legend_rows):
        y = ly + pad + i * row_h
        swatch_x0 = lx + pad
        swatch_y0 = y + 3
        swatch_x1 = swatch_x0 + sw
        swatch_y1 = swatch_y0 + 16
        if style == "solid":
            draw.rectangle([swatch_x0, swatch_y0, swatch_x1, swatch_y1], fill=color)
        elif style == "dashed":
            # Show a dashed line as the swatch
            _draw_dashed_line(
                draw, (swatch_x0, (swatch_y0 + swatch_y1) // 2),
                (swatch_x1, (swatch_y0 + swatch_y1) // 2),
                color, width=3, dash=6, gap=3,
            )
        elif style == "dots":
            # Green dot + red dot side by side
            cy = (swatch_y0 + swatch_y1) // 2
            draw.ellipse([swatch_x0, cy - 4, swatch_x0 + 8, cy + 4],
                         fill=(0, 220, 0), outline=(0, 0, 0), width=1)
            draw.ellipse([swatch_x0 + 14, cy - 4, swatch_x0 + 22, cy + 4],
                         fill=(255, 30, 30), outline=(0, 0, 0), width=1)
        draw.text((lx + pad + sw + 10, y), name, fill="white", font=small)

    out = Image.alpha_composite(canvas, overlay).convert("RGB")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_path))
    return str(out_path)


def polygon_iou(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
    canvas_wh: tuple[int, int],
) -> float:
    """Cheap IoU via rasterization on a small canvas — good enough for overlap audit."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    w, h = canvas_wh
    mask_a = Image.new("L", (w, h), 0)
    mask_b = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_a).polygon(a, fill=1)
    ImageDraw.Draw(mask_b).polygon(b, fill=1)
    pa = list(mask_a.getdata())
    pb = list(mask_b.getdata())
    inter = sum(1 for x, y in zip(pa, pb) if x and y)
    union = sum(1 for x, y in zip(pa, pb) if x or y)
    if union == 0:
        return 0.0
    return inter / union


def convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Andrew's monotone chain — small convex hull helper."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def expand_polygon(
    polygon: list[tuple[int, int]], expand_px: int,
) -> list[tuple[int, int]]:
    """Dilate a polygon outward by `expand_px` via centroid-radial expansion.

    Simple approximation sufficient for the `convex_hull_of_anchor` influence
    escape hatch. Not a proper Minkowski offset.
    """
    if not polygon or expand_px <= 0:
        return polygon
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    out: list[tuple[int, int]] = []
    for x, y in polygon:
        dx = x - cx
        dy = y - cy
        import math
        d = math.hypot(dx, dy)
        if d < 1e-9:
            out.append((int(x), int(y)))
            continue
        nx = x + dx / d * expand_px
        ny = y + dy / d * expand_px
        out.append((int(round(nx)), int(round(ny))))
    return out


# Expose a small helper that wraps `field` usage, purely to keep this module importable.
@dataclass
class MosaicState:
    tile_cache: dict[tuple[float, float, int], str] = field(default_factory=dict)
    tiles_fetched: int = 0

    def on_fetch(self) -> None:
        self.tiles_fetched += 1
