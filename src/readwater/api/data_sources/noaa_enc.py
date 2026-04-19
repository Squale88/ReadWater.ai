"""NOAA ENC (Electronic Navigational Chart) fetcher + parser.

Downloads S-57 vector charts from NOAA, extracts navigation-channel features
(FAIRWY, DRGARE, deep DEPARE), and writes them as GeoJSON for downstream
rasterization.

Primary use: distinguish charted boating channels from narrow tidal drains.
A drain is a small tidal cut between land features; a boating channel is a
maintained / marked navigation route. From satellite imagery alone they
can look identical. From ENC charts, they're trivially separable.

Feature classes we extract:
  FAIRWY  — designated fairway (navigation route polygon)
  DRGARE  — dredged area (maintained channel polygon)
  DEPARE  — depth area polygon; we keep those with min-depth above a
            threshold (default 6 ft / ~1.83 m) as a proxy for deep
            navigable water. Shallower DEPARE polygons are ignored
            because they overlap too broadly with non-channel water.

Everything here is optional-deps; fiona + shapely are gated behind the 'cv'
extras, same as NAIP 4-band + rasterio.

Public functions
  download_enc(chart_id, cache_dir)  -> Path to extracted .000 file
  extract_channels(enc_path, out_geojson, bbox_4326, min_depth_ft)  -> metadata dict
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

# NOAA ENC download URL pattern. Each chart ZIP contains an .000 S-57 file
# plus supporting metadata / update files.
ENC_ZIP_URL = "https://www.charts.noaa.gov/ENCs/{chart_id}.zip"

# NOAA's master ENC product catalog. Small (10 MB) XML listing every chart,
# its coverage polygon, and its compilation scale. Parsed once per run to
# discover which charts cover a given bbox.
ENC_CATALOG_URL = "https://www.charts.noaa.gov/ENCs/ENCProdCat.xml"


@dataclass
class ENCDownloadResult:
    chart_id: str
    zip_path: str
    enc_file: str  # path to the .000 file
    extracted_dir: str


@dataclass
class ENCCatalogEntry:
    name: str
    long_name: str
    compilation_scale: int   # 1:N, smaller = more detailed
    bbox_4326: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


@dataclass
class ChannelExtractResult:
    path: str                  # output GeoJSON path
    feature_count: int
    counts_by_class: dict      # {"FAIRWY": n, "DRGARE": n, ...}
    clipped_out: int           # features that fell fully outside the bbox
    bbox_4326: tuple[float, float, float, float] | None


def find_charts_covering_point(
    lat: float,
    lon: float,
    min_scale: int | None = None,
    max_scale: int | None = None,
    timeout_s: float = 60.0,
) -> list[ENCCatalogEntry]:
    """Query NOAA's ENC product catalog for charts whose polygon contains (lat, lon).

    Returns entries sorted ascending by compilation scale (most detailed first).

    Args:
        lat, lon: decimal degrees, WGS84.
        min_scale: optional lower bound on 1:N scale (e.g. 10000 for harbor).
        max_scale: optional upper bound (e.g. 100000 to exclude overview charts).
        timeout_s: HTTP timeout for the ~10 MB catalog download.
    """
    import xml.etree.ElementTree as ET

    with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
        r = client.get(ENC_CATALOG_URL)
        r.raise_for_status()
        root = ET.fromstring(r.content)

    hits: list[ENCCatalogEntry] = []
    for cell in root.iter("cell"):
        name = cell.findtext("name", "")
        lname = cell.findtext("lname", "")
        cscale_text = cell.findtext("cscale", "")
        try:
            cscale = int(cscale_text)
        except ValueError:
            continue
        if min_scale is not None and cscale < min_scale:
            continue
        if max_scale is not None and cscale > max_scale:
            continue

        cov = cell.find("cov")
        if cov is None:
            continue
        for panel in cov.findall("panel"):
            vertices = []
            for v in panel.findall("vertex"):
                try:
                    vlat = float(v.findtext("lat", "nan"))
                    vlon = float(v.findtext("long", "nan"))
                except ValueError:
                    continue
                vertices.append((vlat, vlon))
            if not vertices:
                continue
            lats = [v[0] for v in vertices]
            lons = [v[1] for v in vertices]
            if min(lats) <= lat <= max(lats) and min(lons) <= lon <= max(lons):
                hits.append(ENCCatalogEntry(
                    name=name,
                    long_name=lname,
                    compilation_scale=cscale,
                    bbox_4326=(min(lons), min(lats), max(lons), max(lats)),
                ))
                break  # one panel match is enough

    hits.sort(key=lambda e: e.compilation_scale)
    return hits


def _require_cv_deps():
    try:
        import fiona  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "NOAA ENC parsing requires the 'cv' extras. "
            "Install with: pip install -e '.[cv]'"
        ) from e


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------


def download_enc(
    chart_id: str,
    cache_dir: str | Path,
    timeout_s: float = 60.0,
    force: bool = False,
) -> ENCDownloadResult:
    """Download and extract an ENC chart ZIP.

    ENC ZIPs contain:
      <chart_id>/<chart_id>.000   — the main S-57 dataset
      <chart_id>/...              — updates, metadata, catalog

    We return the path to the .000 file. Subsequent update files (.001, .002,
    …) are parsed automatically by the S-57 driver when LNAM_REFS / UPDATES
    options are set.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / f"{chart_id}.zip"
    extracted_dir = cache / chart_id

    if not force and zip_path.exists() and extracted_dir.exists():
        enc_file = _find_enc_file(extracted_dir, chart_id)
        if enc_file:
            return ENCDownloadResult(
                chart_id=chart_id,
                zip_path=str(zip_path),
                enc_file=str(enc_file),
                extracted_dir=str(extracted_dir),
            )

    url = ENC_ZIP_URL.format(chart_id=chart_id)
    with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
        r = client.get(url)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted_dir)

    enc_file = _find_enc_file(extracted_dir, chart_id)
    if enc_file is None:
        raise RuntimeError(
            f"ENC ZIP for {chart_id} did not contain a .000 file after extract",
        )

    return ENCDownloadResult(
        chart_id=chart_id,
        zip_path=str(zip_path),
        enc_file=str(enc_file),
        extracted_dir=str(extracted_dir),
    )


def _find_enc_file(extracted_dir: Path, chart_id: str) -> Path | None:
    """Locate the .000 file. NOAA ZIPs put it at either
    <extracted_dir>/<chart_id>/<chart_id>.000 or
    <extracted_dir>/ENC_ROOT/<chart_id>/<chart_id>.000.
    """
    candidates = [
        extracted_dir / chart_id / f"{chart_id}.000",
        extracted_dir / "ENC_ROOT" / chart_id / f"{chart_id}.000",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to a recursive search.
    for f in extracted_dir.rglob(f"{chart_id}.000"):
        return f
    return None


# ------------------------------------------------------------------
# Parse + extract channels
# ------------------------------------------------------------------


# S-57 feature classes we care about. Keep this list tight — expanding it
# bloats the mask and lets non-channel features into the "it's a boating
# channel" category, defeating the purpose.
CHANNEL_LAYERS = ("FAIRWY", "DRGARE")

# Depth areas (DEPARE) with minimum depth above this threshold are treated
# as deep/navigable water. 6 ft ~= 1.83 m is a common threshold for
# inshore-boat drafts; a drain typically has <3 ft at MLW.
DEFAULT_MIN_DEEPWATER_FT = 6.0
FEET_PER_METER = 3.28084


def extract_channels(
    enc_path: str | Path,
    out_geojson: str | Path,
    bbox_4326: tuple[float, float, float, float] | None = None,
    min_deepwater_ft: float = DEFAULT_MIN_DEEPWATER_FT,
    include_deep_depare: bool = True,
) -> ChannelExtractResult:
    """Extract navigation-channel polygons from an ENC file as GeoJSON.

    Args:
        enc_path: path to the .000 file.
        out_geojson: output path for the GeoJSON FeatureCollection.
        bbox_4326: optional (xmin, ymin, xmax, ymax) in WGS84 to clip to.
        min_deepwater_ft: DEPARE polygons with DRVAL1 above this (in meters
            converted from feet) are included.
        include_deep_depare: whether to include DEPARE polygons that meet
            the depth threshold. Turn off for a stricter "marked channel
            only" mask.
    """
    _require_cv_deps()
    import fiona
    from shapely.geometry import box, mapping, shape

    clip_poly = box(*bbox_4326) if bbox_4326 else None
    counts: dict[str, int] = {"FAIRWY": 0, "DRGARE": 0, "DEPARE": 0}
    clipped_out = 0
    out_features: list[dict] = []

    # The S-57 driver exposes each feature class as its own layer.
    try:
        available_layers = set(fiona.listlayers(str(enc_path)))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't read ENC {enc_path}: {e}") from e

    # Set driver options so metadata is consistent. OGR is happy with the
    # defaults for read-only extraction.
    target_layers = [layer for layer in CHANNEL_LAYERS if layer in available_layers]
    if include_deep_depare and "DEPARE" in available_layers:
        target_layers.append("DEPARE")

    min_depth_m = min_deepwater_ft / FEET_PER_METER

    for layer_name in target_layers:
        with fiona.open(str(enc_path), layer=layer_name) as src:
            for feat in src:
                geom_dict = feat["geometry"]
                if not geom_dict:
                    continue
                if geom_dict["type"] not in ("Polygon", "MultiPolygon"):
                    continue
                try:
                    shp = shape(geom_dict)
                except Exception:  # noqa: BLE001
                    continue

                props = dict(feat["properties"] or {})

                # DEPARE filter: need minimum charted depth above threshold.
                if layer_name == "DEPARE":
                    drval1 = props.get("DRVAL1")
                    if drval1 is None or drval1 < min_depth_m:
                        continue

                # Clip to bbox.
                if clip_poly is not None:
                    shp = shp.intersection(clip_poly)
                    if shp.is_empty:
                        clipped_out += 1
                        continue

                out_features.append({
                    "type": "Feature",
                    "geometry": mapping(shp),
                    "properties": {
                        "feature_class": layer_name,
                        **{k: _coerce_value(v) for k, v in props.items()},
                    },
                })
                counts[layer_name] = counts.get(layer_name, 0) + 1

    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": out_features},
            f,
        )

    return ChannelExtractResult(
        path=str(out_geojson),
        feature_count=len(out_features),
        counts_by_class=counts,
        clipped_out=clipped_out,
        bbox_4326=bbox_4326,
    )


def _coerce_value(v):
    """Normalize fiona property values to JSON-safe types."""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    # Dates, bytes, and exotic types: stringify.
    try:
        return str(v)
    except Exception:  # noqa: BLE001
        return None
