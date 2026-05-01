"""Google Maps Static water-mask prototype with optional NAIP cleanup.

Fetches a heavily-styled Google Static Map per cell where:
  - water is rendered as solid pure blue (#0000FF)
  - everything else (land, roads, POIs, labels) is rendered white
This sidesteps the spectral failure modes of NAIP NDWI (bright roofs and
fresh asphalt confused for water) by relying on Google's pre-classified
water boundaries.

Optional NAIP-cleanup pass (default on when a cached 4-band TIF exists
for the cell):
  Google's water polygons are smoothed by their cartographers — small
  sandbars, exposed shoals, and mid-bay islets often get rendered over
  with the surrounding water. We compute an aggressive NAIP NDWI mask
  (NDWI > 0, no NIR cap) and INTERSECT it with the Google mask. The
  aggressive NDWI is overly inclusive (it false-positives on roofs and
  asphalt), but those false positives are on land where Google said
  not-water anyway, so the intersection drops them. What survives is
  Google's curated water minus anything NAIP detected as bright land
  inside a water polygon — i.e. the small islands Google smoothed over.

Per cell, we produce artifacts under
``data/areas/rookery_bay_v2_google_water/``:

  <cell>_styled.png        the raw styled tile (debug / sanity check)
  <cell>_water_mask.png    binary mask (white = water) — hybrid by
                           default, Google-only with --no-naip-cleanup
  <cell>_water_overlay.png satellite image with water tinted blue,
                           directly comparable to the NAIP overlay

Usage:
  python scripts/google_water_mask.py --cell root-2-9
  python scripts/google_water_mask.py --cell all
  python scripts/google_water_mask.py --cell root-2-9 --no-naip-cleanup
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


def run_one(cell_id: str, zoom: int = 16, naip_cleanup: bool = True) -> dict:
    spec = CELLS[cell_id]
    center = spec["cell_center"]
    print(f"\n=== {cell_id}  center={center}  zoom={zoom} ===")

    styled_png = OUT_ROOT / f"{cell_id}_styled.png"
    print(f"  fetching styled water tile -> {styled_png.name}")
    fetch_styled_water(center, zoom, styled_png)

    google_mask = water_mask_from_styled(styled_png)
    google_pct = google_mask.mean() * 100

    final_mask = google_mask
    naip_pct = None
    if naip_cleanup:
        bbox = bbox_from_center(center, zoom, image_size=640)
        naip_aggressive = naip_aggressive_water_mask_aligned(
            cell_id, bbox, google_mask.shape,
        )
        if naip_aggressive is None:
            print("  no cached NAIP TIF — skipping cleanup pass")
        else:
            final_mask = google_mask & naip_aggressive
            naip_pct = final_mask.mean() * 100
            print(
                f"  google: {google_pct:.1f}%   "
                f"hybrid (google AND NAIP NDWI>0): {naip_pct:.1f}%   "
                f"carved out: {google_pct - naip_pct:.1f}%",
            )
    if naip_pct is None:
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
    args = parser.parse_args()

    cells = list(CELLS.keys()) if args.cell == "all" else [args.cell]
    results = []
    for cid in cells:
        results.append(run_one(cid, args.zoom, naip_cleanup=not args.no_naip_cleanup))

    print("\n=== Summary ===")
    if args.no_naip_cleanup:
        print(f"{'cell':12s}  google %")
        for r in results:
            print(f"{r['cell_id']:12s}  {r['google_pct']:6.1f}%")
    else:
        print(f"{'cell':12s}  google %  hybrid %  carved")
        for r in results:
            g = r["google_pct"]
            h = r["hybrid_pct"]
            if h is None:
                print(f"{r['cell_id']:12s}  {g:6.1f}%   (no NAIP)")
            else:
                print(f"{r['cell_id']:12s}  {g:6.1f}%   {h:6.1f}%   {g - h:5.1f}%")
    print(f"\nArtifacts in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
