"""Per-cell water mask combining Google Static + NAIP + connectivity filters.

The mask is built in three stacked passes; later passes refine the result
of earlier ones. Each pass can be disabled with a CLI flag.

  1. Google styled tile (always on).
     A heavily-styled Google Static Map covering the cell renders water
     as solid pure blue (#0000FF) and everything else (land, roads,
     POIs, labels) as white. We threshold by color into a binary mask.
     This sidesteps the spectral failure modes of NAIP NDWI (bright
     roofs and fresh asphalt confused for water) by relying on Google's
     pre-classified water boundaries.

  2. NAIP NDWI cleanup (--no-naip-cleanup to skip).
     Google's water polygons are cartographer-smoothed: small sandbars,
     exposed shoals, and mid-bay islets often get rendered over with
     the surrounding water. We intersect Google's mask with an
     aggressive NAIP NDWI > 0 mask. NAIP false-positives on roofs are
     fine because Google says not-water there anyway, so the
     intersection drops them. What survives is Google's curated water
     minus the bright land features Google smoothed over.

  3. Two-pass connectivity filter (--no-connectivity-filter to skip).
     To exclude inland fresh-water bodies (retention ponds, residential
     ponds, golf-course lakes) we fetch two extra wider styled tiles:
       - Detail pass at --wide-zoom (default 14, ~2.5 km bbox): finds
         water pixels reachable from the wide perimeter via 4-connected
         paths. Resolution is high enough that narrow residential
         canals stay continuous through their off-cell connections.
       - Isolation pass at --isolation-zoom (default 13, ~5 km bbox):
         same algorithm at a coarser zoom whose larger bbox fully
         encloses pond-sized water bodies. Anything classified as water
         here but NOT perimeter-connected is flagged as isolated.
     Final filter = (detail perim-connected) AND NOT (isolation
     isolated). This combination keeps narrow canals visible at z14
     while still subtracting near-edge ponds that z13 confirms are
     enclosed.

Per cell, we write under ``data/areas/rookery_bay_v2_google_water/``:

  <cell>_styled.png            cell-level styled tile (z16, debug)
  <cell>_wide_z14_styled.png   detail-pass styled tile (debug)
  <cell>_wide_z13_styled.png   isolation-pass styled tile (debug)
  <cell>_water_mask.png        final binary mask (white = water)
  <cell>_water_overlay.png     final mask tinted on the satellite tile

Usage:
  python scripts/google_water_mask.py --cell root-2-9
  python scripts/google_water_mask.py --cell all
  python scripts/google_water_mask.py --cell root-2-2 --no-naip-cleanup
  python scripts/google_water_mask.py --cell root-2-2 --isolation-zoom=-1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

REPO_ROOT = Path("D:/dropbox_root/Dropbox/CascadeProjects/ReadWater.ai")
_env_path = REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ[_k.strip()] = _v.strip()

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "scripts"))

from _cells import CELLS  # noqa: E402
from readwater.api.data_sources.naip_4band import bbox_from_center  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_google_water"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
NAIP_DIR = REPO_ROOT / "data" / "areas" / "rookery_bay_v2_naip"

GOOGLE_MAPS_STATIC_URL = "https://maps.googleapis.com/maps/api/staticmap"

# Style chain: every visible feature is forced white, then water is
# painted pure blue, and every label is hidden so text doesn't leak
# into either color region. The order matters — later rules override
# earlier ones for the same feature.
WATER_MASK_STYLES = [
    # Step 1: all geometry white, all labels off (default everything to
    # the "non-water" color so we only have to override water).
    "feature:all|element:labels|visibility:off",
    "feature:all|element:geometry|color:0xffffff",
    "feature:all|element:geometry.stroke|visibility:off",
    # Step 2: kill everything we don't care about.
    "feature:administrative|visibility:off",
    "feature:poi|visibility:off",
    "feature:transit|visibility:off",
    "feature:road|visibility:off",
    "feature:landscape|element:geometry|color:0xffffff",
    # Step 3: paint water pure blue.
    "feature:water|element:geometry|color:0x0000ff",
    "feature:water|element:geometry.fill|color:0x0000ff",
    "feature:water|element:geometry.stroke|color:0x0000ff",
]


def _api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_MAPS_API_KEY not set (expected in .env)")
    return key


def fetch_styled_water(
    center: tuple[float, float],
    zoom: int,
    out_path: Path,
    image_size: int = 640,
) -> Path:
    """Fetch a Google Static map styled to render water blue and land white."""
    lat, lon = center
    params: list[tuple[str, str]] = [
        ("center", f"{lat},{lon}"),
        ("zoom", str(zoom)),
        ("size", f"{image_size}x{image_size}"),
        ("scale", "2"),
        ("maptype", "roadmap"),
        ("key", _api_key()),
    ]
    for s in WATER_MASK_STYLES:
        params.append(("style", s))

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(GOOGLE_MAPS_STATIC_URL, params=params)
        resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    return out_path


def water_mask_from_styled(styled_png: Path) -> np.ndarray:
    """Threshold the styled tile into a boolean water mask.

    A pixel is water iff its blue channel is much higher than red+green
    (i.e., it landed in the pure-blue region rather than the white
    background or any leftover color near boundaries).
    """
    img = Image.open(styled_png).convert("RGB")
    arr = np.array(img)  # (H, W, 3) uint8
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    # Pure blue: B high, R+G low. Use generous tolerance to capture
    # antialiased pixels along boundaries without bleeding into white.
    return (b > 128) & (r < 96) & (g < 96)


def _dilate_4conn(mask: np.ndarray, iters: int) -> np.ndarray:
    """4-connectivity binary dilation, iters rounds. Pure numpy."""
    out = mask.copy()
    for _ in range(iters):
        up = np.zeros_like(out)
        up[:-1, :] = out[1:, :]
        dn = np.zeros_like(out)
        dn[1:, :] = out[:-1, :]
        lt = np.zeros_like(out)
        lt[:, :-1] = out[:, 1:]
        rt = np.zeros_like(out)
        rt[:, 1:] = out[:, :-1]
        out = out | up | dn | lt | rt
    return out


def perimeter_connected_mask(
    water_mask: np.ndarray,
    bridge_dilate_iters: int = 1,
) -> np.ndarray:
    """Return a mask of water pixels reachable from the image perimeter
    via 4-connected water-only paths.

    Used as an "is this connected to off-image water?" filter. When the
    input ``water_mask`` covers a wide area (zoom 13–14 around a target
    cell), components fully enclosed inside the area are isolated bodies
    of water — retention ponds, residential ponds, golf-course lakes —
    and get rejected. Components that reach any edge are presumed to
    extend into a larger network (Gulf, river, intracoastal waterway)
    and get kept.

    ``bridge_dilate_iters`` dilates the water mask before flood-fill to
    bridge thin rendering gaps where a narrow canal (1–2 pixels wide at
    zoom 13) is briefly broken by a single non-blue pixel. After
    flood-fill, the result is intersected with the original (un-dilated)
    water mask so the output never claims pixels that weren't water in
    the source.

    Implementation: pure-numpy iterative dilation seeded from the
    perimeter water pixels, intersected with the (possibly bridged)
    water mask each step. Converges when no new pixels are reached.
    """
    if not water_mask.any():
        return np.zeros_like(water_mask, dtype=bool)

    # Bridge thin rendering gaps so a 1-pixel break in a narrow canal
    # doesn't disconnect it from its parent network.
    if bridge_dilate_iters > 0:
        bridged = _dilate_4conn(water_mask, bridge_dilate_iters)
    else:
        bridged = water_mask

    # Seed from bridged-water pixels on the four edges.
    reachable = np.zeros_like(bridged, dtype=bool)
    reachable[0, :] = bridged[0, :]
    reachable[-1, :] = bridged[-1, :]
    reachable[:, 0] = bridged[:, 0]
    reachable[:, -1] = bridged[:, -1]

    if not reachable.any():
        return np.zeros_like(water_mask, dtype=bool)

    prev_count = -1
    while reachable.sum() != prev_count:
        prev_count = int(reachable.sum())
        up = np.zeros_like(reachable)
        up[:-1, :] = reachable[1:, :]
        dn = np.zeros_like(reachable)
        dn[1:, :] = reachable[:-1, :]
        lt = np.zeros_like(reachable)
        lt[:, :-1] = reachable[:, 1:]
        rt = np.zeros_like(reachable)
        rt[:, 1:] = reachable[:, :-1]
        reachable = (reachable | up | dn | lt | rt) & bridged

    # Restrict back to original water pixels — never claim pixels that
    # weren't water before bridging.
    return reachable & water_mask


def crop_wide_mask_to_cell(
    wide_mask: np.ndarray,
    wide_bbox: tuple[float, float, float, float],
    cell_bbox: tuple[float, float, float, float],
    cell_shape: tuple[int, int],
) -> np.ndarray:
    """Crop a wide-area boolean mask to the cell's WGS84 bbox and
    nearest-neighbor resize to ``cell_shape`` (h, w).

    Both bboxes are (xmin, ymin, xmax, ymax) in WGS84. The wide bbox
    must fully contain the cell bbox.
    """
    h, w = wide_mask.shape
    wxmin, wymin, wxmax, wymax = wide_bbox
    cxmin, cymin, cxmax, cymax = cell_bbox

    # x grows east; image y grows south (y=0 at top = highest lat).
    px_x0 = (cxmin - wxmin) / (wxmax - wxmin) * w
    px_x1 = (cxmax - wxmin) / (wxmax - wxmin) * w
    px_y0 = (wymax - cymax) / (wymax - wymin) * h
    px_y1 = (wymax - cymin) / (wymax - wymin) * h

    px_x0 = max(0, int(round(px_x0)))
    px_x1 = min(w, int(round(px_x1)))
    px_y0 = max(0, int(round(px_y0)))
    px_y1 = min(h, int(round(px_y1)))

    if px_x1 <= px_x0 or px_y1 <= px_y0:
        return np.zeros(cell_shape, dtype=bool)

    cropped = wide_mask[px_y0:px_y1, px_x0:px_x1]
    if cropped.shape == cell_shape:
        return cropped

    img = Image.fromarray(cropped.astype(np.uint8) * 255, mode="L")
    img = img.resize((cell_shape[1], cell_shape[0]), Image.NEAREST)
    return np.array(img) > 0


def _perim_connected_at_zoom(
    center: tuple[float, float],
    zoom: int,
    out_dir: Path,
    cell_id: str,
    bridge_dilate_iters: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Fetch a wider styled tile at ``zoom`` + return
    (water_mask, perim_connected_mask, bbox)."""
    wide_styled = out_dir / f"{cell_id}_wide_z{zoom}_styled.png"
    fetch_styled_water(center, zoom, wide_styled)
    wide_mask = water_mask_from_styled(wide_styled)
    perim_mask = perimeter_connected_mask(
        wide_mask, bridge_dilate_iters=bridge_dilate_iters,
    )
    bbox = bbox_from_center(center, zoom, image_size=640)
    return wide_mask, perim_mask, bbox


def gulf_connected_mask_for_cell(
    center: tuple[float, float],
    cell_bbox: tuple[float, float, float, float],
    cell_shape: tuple[int, int],
    out_dir: Path,
    cell_id: str,
    wide_zoom: int = 14,
    isolation_zoom: int | None = 13,
    bridge_dilate_iters: int = 1,
) -> np.ndarray:
    """Build a Gulf-connected water mask for the cell using a two-pass
    test against wider styled Google tiles.

    Pass 1 (detail) at ``wide_zoom`` (~2.5 km at z14):
        Identifies water pixels that reach the wide-tile perimeter via
        4-connected paths. At z14 narrow canals stay visible (~1–2 px
        wide), so the detail pass keeps thin residential canals that
        funnel out of the cell to a Gulf-connected sister channel.

    Pass 2 (isolation) at ``isolation_zoom`` (~5 km at z13):
        Same algorithm at a coarser zoom whose larger bbox fully
        encloses pond-sized water bodies (a 200–300 m residential pond
        touches the z14 perimeter but is well inside the z13 perimeter).
        Anything classified as water by Google but NOT perimeter-
        connected at this coarser zoom is flagged as isolated.

    Final mask = detail_perim_connected AND (NOT isolation_isolated).
    Pass through ``isolation_zoom=None`` to skip the second pass and
    fall back to single-zoom behavior.
    """
    detail_water, detail_perim, detail_bbox = _perim_connected_at_zoom(
        center, wide_zoom, out_dir, cell_id, bridge_dilate_iters,
    )
    detail_cell = crop_wide_mask_to_cell(
        detail_perim, detail_bbox, cell_bbox, cell_shape,
    )

    if isolation_zoom is None or isolation_zoom == wide_zoom:
        return detail_cell

    iso_water, iso_perim, iso_bbox = _perim_connected_at_zoom(
        center, isolation_zoom, out_dir, cell_id, bridge_dilate_iters,
    )
    # "Isolated" = water at the coarser zoom that does NOT reach its
    # perimeter. Reproject to the cell grid.
    iso_isolated = iso_water & ~iso_perim
    iso_isolated_cell = crop_wide_mask_to_cell(
        iso_isolated, iso_bbox, cell_bbox, cell_shape,
    )

    return detail_cell & ~iso_isolated_cell


def naip_aggressive_water_mask_aligned(
    cell_id: str,
    bbox_4326: tuple[float, float, float, float],
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Aggressive (NDWI > 0, no NIR cap) NAIP water mask, reprojected
    to ``target_shape`` covering ``bbox_4326`` in WGS84.

    Returns None if the NAIP TIF isn't cached for the cell.

    The mask is intentionally over-inclusive — it false-positives on
    roofs/asphalt — because we use it only to *carve* land features
    out of Google's water polygons. Anywhere Google says "not water"
    is going to be ANDed with this mask anyway, so the over-inclusive
    parts on actual land never make it into the final mask.
    """
    tif = NAIP_DIR / f"{cell_id}_naip_4band.tif"
    if not tif.exists():
        return None

    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    from readwater.pipeline.water_mask import compute_ndwi, load_4band_tif

    bands = load_4band_tif(tif)
    ndwi = compute_ndwi(bands.green, bands.nir)
    src_mask = (ndwi > 0).astype(np.uint8) * 255  # uint8 source for reproject

    bh, bw = target_shape
    dst_transform = from_bounds(*bbox_4326, bw, bh)
    dst = np.zeros((bh, bw), dtype=np.uint8)

    with rasterio.open(tif) as src:
        reproject(
            source=src_mask,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )

    # Where NAIP coverage is absent (NAIP is partial-area for some tiles),
    # we should NOT carve away Google water — there's no information there.
    # Reproject a "ones" source the same way to find covered pixels.
    src_ones = np.ones((bands.green.shape[0], bands.green.shape[1]), dtype=np.uint8) * 255
    covered = np.zeros((bh, bw), dtype=np.uint8)
    with rasterio.open(tif) as src:
        reproject(
            source=src_ones,
            destination=covered,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )

    # Pixels with no NAIP coverage are treated as "water" so the
    # subsequent AND with Google's mask doesn't carve them out.
    aggressive_water = (dst > 0) | (covered == 0)
    return aggressive_water


def make_overlay(
    base_image: Path,
    mask: np.ndarray,
    out_path: Path,
    rgba: tuple[int, int, int, int] = (0, 150, 255, 70),
) -> Path:
    """Paint the water mask over the existing satellite image.

    The base image is the Google Static *satellite* tile we've already
    been using elsewhere, so the overlay aligns pixel-for-pixel.
    """
    base = Image.open(base_image).convert("RGBA")
    bw, bh = base.size

    if mask.shape != (bh, bw):
        # The styled tile from Google should match the satellite tile
        # because we use the same center/zoom/size. Resample if not.
        m_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        m_img = m_img.resize((bw, bh), Image.NEAREST)
        mask = np.array(m_img) > 0

    overlay_arr = np.zeros((bh, bw, 4), dtype=np.uint8)
    overlay_arr[mask] = rgba
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def run_one(
    cell_id: str,
    zoom: int = 16,
    naip_cleanup: bool = True,
    connectivity_filter: bool = True,
    wide_zoom: int = 14,
    isolation_zoom: int | None = 13,
) -> dict:
    spec = CELLS[cell_id]
    center = spec["cell_center"]
    print(f"\n=== {cell_id}  center={center}  zoom={zoom} ===")

    styled_png = OUT_ROOT / f"{cell_id}_styled.png"
    print(f"  fetching styled water tile -> {styled_png.name}")
    fetch_styled_water(center, zoom, styled_png)

    google_mask = water_mask_from_styled(styled_png)
    google_pct = google_mask.mean() * 100

    cell_bbox = bbox_from_center(center, zoom, image_size=640)
    final_mask = google_mask

    naip_pct = None
    if naip_cleanup:
        naip_aggressive = naip_aggressive_water_mask_aligned(
            cell_id, cell_bbox, google_mask.shape,
        )
        if naip_aggressive is None:
            print("  no cached NAIP TIF — skipping NAIP cleanup pass")
        else:
            final_mask = final_mask & naip_aggressive
            naip_pct = final_mask.mean() * 100

    connected_pct = None
    if connectivity_filter:
        gulf_connected = gulf_connected_mask_for_cell(
            center, cell_bbox, google_mask.shape,
            out_dir=OUT_ROOT, cell_id=cell_id,
            wide_zoom=wide_zoom, isolation_zoom=isolation_zoom,
        )
        before = final_mask.mean() * 100
        final_mask = final_mask & gulf_connected
        connected_pct = final_mask.mean() * 100
        iso_label = (
            f"z{wide_zoom} detail + z{isolation_zoom} isolation"
            if isolation_zoom is not None
            else f"z{wide_zoom} only"
        )
        print(
            f"  connectivity ({iso_label}): "
            f"{before:.1f}% -> {connected_pct:.1f}%   "
            f"isolated bodies dropped: {before - connected_pct:.1f}%",
        )

    if naip_pct is not None and connected_pct is not None:
        print(
            f"  google: {google_pct:.1f}%   "
            f"+ NAIP carve: {naip_pct:.1f}%   "
            f"+ connectivity: {connected_pct:.1f}%",
        )
    elif naip_pct is not None:
        print(
            f"  google: {google_pct:.1f}%   "
            f"hybrid (google AND NAIP NDWI>0): {naip_pct:.1f}%",
        )
    else:
        print(f"  water fraction: {google_pct:.1f}%")

    mask_png = OUT_ROOT / f"{cell_id}_water_mask.png"
    Image.fromarray((final_mask.astype(np.uint8) * 255), mode="L").save(mask_png)

    # Use the existing satellite tile as the overlay base so the user
    # can put it side-by-side with the NAIP overlay.
    base_satellite = (
        REPO_ROOT
        / "data" / "areas" / "rookery_bay_v2" / "images"
        / f"z0_{cell_id.removeprefix('root-').replace('-', '_')}.png"
    )
    overlay_png = OUT_ROOT / f"{cell_id}_water_overlay.png"
    if base_satellite.exists():
        make_overlay(base_satellite, final_mask, overlay_png)
        print(f"  overlay -> {overlay_png}")
    else:
        print(f"  base satellite tile {base_satellite.name} not found, skipping overlay")

    return {
        "cell_id": cell_id,
        "google_pct": google_pct,
        "hybrid_pct": naip_pct,
        "connected_pct": connected_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        choices=list(CELLS.keys()) + ["all"],
        default="root-2-9",
    )
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument(
        "--no-naip-cleanup",
        action="store_true",
        help="Disable the NAIP-intersection cleanup; output the raw "
             "Google water mask only.",
    )
    parser.add_argument(
        "--no-connectivity-filter",
        action="store_true",
        help="Disable the wide-area Gulf-connectivity filter that drops "
             "isolated water bodies (retention ponds, residential ponds).",
    )
    parser.add_argument(
        "--wide-zoom",
        type=int,
        default=14,
        help="Zoom level of the detail wider styled tile (default 14, "
             "~2.5 km on a side at lat 26°). High enough to keep narrow "
             "canals visible and connected.",
    )
    parser.add_argument(
        "--isolation-zoom",
        type=int,
        default=13,
        help="Zoom level of the isolation-detection tile (default 13, "
             "~5 km on a side). Coarser than --wide-zoom so even larger "
             "residential ponds become fully enclosed; the algorithm "
             "subtracts whatever this tile classifies as isolated water "
             "from the detail-zoom result. Pass --isolation-zoom=-1 to "
             "disable the second pass.",
    )
    args = parser.parse_args()

    isolation_zoom = args.isolation_zoom if args.isolation_zoom >= 0 else None

    cells = list(CELLS.keys()) if args.cell == "all" else [args.cell]
    results = []
    for cid in cells:
        results.append(run_one(
            cid, args.zoom,
            naip_cleanup=not args.no_naip_cleanup,
            connectivity_filter=not args.no_connectivity_filter,
            wide_zoom=args.wide_zoom,
            isolation_zoom=isolation_zoom,
        ))

    print("\n=== Summary ===")
    print(f"{'cell':12s}  google %  + NAIP %  + conn %")
    for r in results:
        g = r["google_pct"]
        h = r["hybrid_pct"]
        c = r["connected_pct"]
        h_str = f"{h:6.1f}%" if h is not None else "  --  "
        c_str = f"{c:6.1f}%" if c is not None else "  --  "
        print(f"{r['cell_id']:12s}  {g:6.1f}%   {h_str}   {c_str}")
    print(f"\nArtifacts in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
