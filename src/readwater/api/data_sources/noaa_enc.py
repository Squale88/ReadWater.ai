"""NOAA ENC (Electronic Navigational Chart) fetcher + parser.

Downloads S-57 vector charts from NOAA, extracts navigation-channel
features, and writes them as GeoJSON for downstream rasterization.

Primary use: distinguish charted boating channels from narrow tidal drains.
A drain is a small tidal cut between land features; a boating channel is a
maintained / marked navigation route. From satellite imagery alone they
can look identical. From ENC charts, they're trivially separable.

The "channel" signal is built from three sources (defaults in parens):

  FAIRWY  (on)  — designated fairway polygon. Always included.
  DRGARE  (on)  — dredged area polygon. Always included.
  Lateral markers (on) — BCNLAT + BOYLAT points. Each marker has an
                  OBJNAM like "Gordon Pass Channel Daybeacon 5" that
                  names the channel and gives its sequence number, plus
                  a CATLAM value indicating the port (green) or
                  starboard (red) side. We parse these, group by
                  channel, sort each side by number, and build a
                  channel-corridor polygon from the green line going
                  forward plus the red line going backward. This
                  mirrors how a mariner actually reads the channel:
                  follow the numbered markers in order with greens on
                  one side and reds on the other.

  SEAARE  (off) — named water bodies. Useful when OBJNAM matches
                  channel keywords (Pass/Channel/Cut/etc.) but at coarse
                  scale it tends to pull in whole bays. Opt-in only.
  DEPARE  (off) — bathymetric depth zones. Same problem — a single
                  polygon can cover the whole navigable seabed. Opt-in
                  if you specifically want depth overlays.

CANALS and RIVERS are intentionally out — they conflate inland waterways
with boating channels for fishing-spot discovery.

Everything here is optional-deps; fiona + shapely are gated behind the 'cv'
extras, same as NAIP 4-band + rasterio.

Public functions
  download_enc(chart_id, cache_dir)  -> Path to extracted .000 file
  extract_channels(enc_path, out_geojson, bbox_4326, min_depth_ft)  -> metadata dict
"""

from __future__ import annotations

import json
import re
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


# --- Feature-class selection for a "channel" mask ---

CHANNEL_LAYERS = ("FAIRWY", "DRGARE")

# Layers containing lateral channel markers (the numbered red/green
# daybeacons and buoys that define a boating channel). Each marker's
# OBJNAM starts with the channel name and ends with a sequence number;
# CATLAM indicates which side of the channel the marker sits on.
MARKER_LAYERS = ("BCNLAT", "BOYLAT")

# OBJNAM shape examples:
#   "Gordon Pass Channel Daybeacon 5"
#   "Gordon Pass Channel Light 6"
#   "Gordon Pass Channel Lighted Buoy 2"
#   "Big Marco Pass - Gordon Pass Daybeacon 51"
#   "Capri Pass Daybeacon 8"
# The optional suffix letter ("Buoy 3A") is captured but not used for
# ordering — we order by the numeric part only.
_MARKER_OBJNAM_RE = re.compile(
    r"^(?P<channel>.+?)\s+"
    r"(?:Lighted\s+Buoy|Daybeacon|Beacon|Buoy|Light)\s+"
    r"(?P<num>\d+)(?P<suffix>[A-Za-z]?)\s*$",
    re.IGNORECASE,
)

# CATLAM values in S-57:
#   1 = port-hand lateral (green in IALA region B / US)
#   2 = starboard-hand lateral (red in IALA region B / US)
#   3 = preferred channel to starboard
#   4 = preferred channel to port
_CATLAM_PORT = 1
_CATLAM_STARBOARD = 2

# Default buffer applied to marker-derived channel geometry. 5 m gives a
# ~10 m-wide indicator ribbon — the goal is to mark *where* the channel
# runs through a tile, not to fill the full navigable width.
DEFAULT_MARKER_BUFFER_M = 5.0

# Word-boundary keyword regex matched against SEAARE.OBJNAM.
# Includes: "Gordon Pass", "Rookery Channel", "Big Marco Pass", "Flotilla
# Passage", "Southwest Gate", etc.
# Excludes: "Gulf of America", "Dollar Bay", "Bear Point Cove", "Sanctuary
# Sound", "Clam Factory Shoal", "Sunfish Flat".
DEFAULT_CHANNEL_NAME_KEYWORDS: tuple[str, ...] = (
    "pass", "passage", "channel", "cut", "inlet", "run", "gate",
)


def _build_channel_name_regex(keywords: tuple[str, ...]) -> re.Pattern:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE,
    )


# DEPARE filter defaults — inshore-boat "navigable but not ocean" range.
# DRVAL1 is the charted minimum depth (meters) at the polygon.
#   < 0.9 m (~3 ft)     — mud flats / intertidal; a drain depth not a channel.
#   0.9 m to 5.0 m      — inshore navigable water: channel territory.
#   > 5.0 m (~16 ft)    — deeper offshore water; typical on open-ocean DEPAREs.
# Combined with the area cap, this keeps channel-shaped polygons in bays
# while dropping both background ocean zones (hundreds of km²) and large
# bay interiors whose depth happens to fall in range.
DEFAULT_DEPARE_MAX_AREA_KM2 = 5.0
DEFAULT_DEPARE_MIN_DEPTH_M = 0.9
DEFAULT_DEPARE_MAX_DEPTH_M = 5.0

# Approximate area conversion for a rough-and-ready filter.
# 1 degree latitude ~= 111 km; longitude shrinks with cos(lat) but for a
# 26° N chart area the error is ~10% — fine for a coarse size filter whose
# threshold is already a round number.
_KM_PER_DEG_LAT = 111.0


def _polygon_area_km2(shp) -> float:
    """Rough projection-free area of a WGS84 polygon in km²."""
    # shp.area is in square degrees. A 0.001 sq-deg polygon is ~12 km²
    # near the equator, shrinking with latitude. Using the flat
    # approximation is fine here — the filter is a coarse size gate.
    return float(shp.area) * _KM_PER_DEG_LAT * _KM_PER_DEG_LAT


def _parse_marker(objnam: str) -> tuple[str, int, str] | None:
    """Parse a lateral-marker OBJNAM into (channel_name, sequence_number, suffix).

    The suffix (e.g. "A" in "Daybeacon 27A") is returned separately so
    callers can keep plain markers and their auxiliary-suffix siblings
    in a stable order — sorting by ``(num, suffix)`` with empty suffix
    sorting first puts "27" before "27A".

    Returns None if the name doesn't match the expected shape.
    """
    m = _MARKER_OBJNAM_RE.match(objnam or "")
    if m is None:
        return None
    try:
        num = int(m.group("num"))
    except ValueError:
        return None
    suffix = (m.group("suffix") or "").upper()
    return m.group("channel").strip(), num, suffix


def _buffer_deg_for_meters(meters: float, lat_deg: float) -> float:
    """Convert a metric buffer to degrees at the given latitude.

    Uses the smaller of the two axes (longitude, which shrinks with
    latitude) so the resulting buffer is at least ``meters`` wide in
    both directions — minor over-buffering in the latitude direction
    is harmless for a channel-corridor mask.
    """
    import math
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat_deg))
    return meters / max(meters_per_deg_lon, 1.0)


def _build_channel_polygon(
    green_pts: list[tuple[float, float, int, str]],
    red_pts: list[tuple[float, float, int, str]],
    buffer_deg: float,
    max_num_gap: int = 6,
    max_segment_dist_deg: float = 0.007,  # ~775 m at 26° N
):
    """Build a thin channel-lane indicator from lateral markers.

    Nautical lateral markers aren't installed in pairs — they're placed
    wherever a mariner needs one. Each color defines a single *side*
    of the channel (in US IALA Region B: even=red=starboard-when-returning,
    odd=green=port-when-returning; 100% consistent in every chart we
    tested). The channel is the space *between* the two sides, not
    centered on any marker.

    Method:

    1. Sort all markers (both colors) by ``(num, suffix)`` so plain
       markers and their auxiliary siblings (e.g. "27" then "27A")
       stay in a stable, predictable order.
    2. For each consecutive pair, compute the midpoint. Cross-color
       pairs give a midpoint on the channel centerline; same-color
       pairs give a midpoint on that side — both are within the channel
       corridor, and including both keeps the ribbon continuous across
       stretches where the chart only marks one side (e.g. "3 reds in
       a row with shore on the other side") or runs of auxiliary
       markers that don't alternate.
    3. Connect midpoints in number order with line segments and buffer
       thinly (default ~5 m → 10 m-wide lane) to give a visible
       indication of the channel path.

    Gaps in numbering (``max_num_gap``) or distance
    (``max_segment_dist_deg``) are skipped so straight-line connectors
    never cut across land.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    if not green_pts and not red_pts:
        return None

    def _dist(a, b) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # Tag each marker with its color and sort by (num, suffix) so
    # "27" precedes "27A", then both precede "28".
    tagged = (
        [(p[0], p[1], p[2], p[3], "G") for p in green_pts]
        + [(p[0], p[1], p[2], p[3], "R") for p in red_pts]
    )
    tagged.sort(key=lambda x: (x[2], x[3]))

    # Walk consecutive pairs and collect their midpoints.
    midpoints: list[tuple[float, float, float]] = []  # (sort_key, lon, lat)
    for i in range(len(tagged) - 1):
        a = tagged[i]
        b = tagged[i + 1]
        if b[2] - a[2] > max_num_gap:
            continue
        if _dist(a, b) > max_segment_dist_deg:
            continue
        midpoints.append((
            (a[2] + b[2]) / 2.0,
            (a[0] + b[0]) / 2.0,
            (a[1] + b[1]) / 2.0,
        ))

    shapes: list = []

    if len(midpoints) >= 2:
        # Normal case: buffer line segments between consecutive midpoints.
        for i in range(len(midpoints) - 1):
            a = midpoints[i]
            b = midpoints[i + 1]
            if ((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5 > max_segment_dist_deg:
                continue
            shapes.append(
                LineString([(a[1], a[2]), (b[1], b[2])]).buffer(buffer_deg),
            )
    elif len(midpoints) == 1:
        # Two markers only — nothing to connect, but buffer the single
        # midpoint so the channel still has a visible indicator.
        m = midpoints[0]
        shapes.append(Point(m[1], m[2]).buffer(buffer_deg))
    else:
        # No valid midpoints (single isolated marker, or all pairs filtered
        # out by gap/distance limits). Fall back to buffering each marker.
        for lon, lat, _, _, _ in tagged:
            shapes.append(Point(lon, lat).buffer(buffer_deg))

    return unary_union(shapes) if shapes else None


def _extract_marker_channels(
    enc_path: str | Path,
    marker_buffer_m: float = DEFAULT_MARKER_BUFFER_M,
) -> tuple[list[dict], int]:
    """Scan BCNLAT + BOYLAT, build one polygon per named channel.

    Returns a list of GeoJSON-shaped Feature dicts with
    feature_class='MARKERS' and channel metadata in properties, plus
    the number of unparseable OBJNAM strings encountered (for debug /
    telemetry only).
    """
    import fiona
    from shapely.geometry import mapping

    by_channel: dict[str, dict] = {}  # channel -> {greens:[], reds:[], lats:[]}
    unparsed = 0

    try:
        available = set(fiona.listlayers(str(enc_path)))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't read ENC {enc_path}: {e}") from e

    for layer in MARKER_LAYERS:
        if layer not in available:
            continue
        with fiona.open(str(enc_path), layer=layer) as src:
            for feat in src:
                geom = feat.get("geometry")
                if not geom or geom["type"] != "Point":
                    continue
                props = feat.get("properties") or {}
                parsed = _parse_marker(props.get("OBJNAM") or "")
                if parsed is None:
                    if props.get("OBJNAM"):
                        unparsed += 1
                    continue
                channel_name, num, suffix = parsed
                catlam = props.get("CATLAM")
                lon, lat = geom["coordinates"]
                entry = by_channel.setdefault(
                    channel_name, {"greens": [], "reds": [], "lats": []},
                )
                marker_tuple = (lon, lat, num, suffix)
                if catlam == _CATLAM_PORT:
                    entry["greens"].append(marker_tuple)
                elif catlam == _CATLAM_STARBOARD:
                    entry["reds"].append(marker_tuple)
                else:
                    # Unusual CATLAM (preferred-channel markers, unknown, etc.)
                    # — treat as starboard so the polygon still closes, but
                    # downstream consumers can tell from properties.
                    entry["reds"].append(marker_tuple)
                entry["lats"].append(lat)

    features: list[dict] = []
    for cname, data in by_channel.items():
        if not data["lats"]:
            continue
        mean_lat = sum(data["lats"]) / len(data["lats"])
        buffer_deg = _buffer_deg_for_meters(marker_buffer_m, mean_lat)
        poly = _build_channel_polygon(data["greens"], data["reds"], buffer_deg)
        if poly is None or poly.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "feature_class": "MARKERS",
                "channel_name": cname,
                "n_greens": len(data["greens"]),
                "n_reds": len(data["reds"]),
                "green_nums": sorted(f"{n}{s}" for _, _, n, s in data["greens"]),
                "red_nums": sorted(f"{n}{s}" for _, _, n, s in data["reds"]),
            },
        })
    return features, unparsed


def extract_channels(
    enc_path: str | Path,
    out_geojson: str | Path,
    bbox_4326: tuple[float, float, float, float] | None = None,
    include_marker_channels: bool = True,
    marker_buffer_m: float = DEFAULT_MARKER_BUFFER_M,
    include_named_channels: bool = False,
    named_channel_keywords: tuple[str, ...] = DEFAULT_CHANNEL_NAME_KEYWORDS,
    include_depare: bool = False,
    depare_max_area_km2: float = DEFAULT_DEPARE_MAX_AREA_KM2,
    depare_min_depth_m: float = DEFAULT_DEPARE_MIN_DEPTH_M,
    depare_max_depth_m: float = DEFAULT_DEPARE_MAX_DEPTH_M,
) -> ChannelExtractResult:
    """Extract navigation-channel polygons from an ENC file as GeoJSON.

    Args:
        enc_path: path to the .000 file.
        out_geojson: output path for the GeoJSON FeatureCollection.
        bbox_4326: optional (xmin, ymin, xmax, ymax) in WGS84 to clip to.
        include_marker_channels: when True (default), parse BCNLAT and
            BOYLAT lateral markers, group them by channel name, and
            build a corridor polygon per channel by joining the port
            (green) line forward with the starboard (red) line reversed.
            This matches how mariners actually navigate — marker-to-marker
            along the numbered sequence — and is the most accurate
            channel signal in an ENC for inshore fishing work.
        marker_buffer_m: half-width in meters used when buffering marker
            geometry. 25 m covers the typical positional tolerance
            between opposite-side markers on narrow channels without
            bleeding into adjacent flats.
        include_named_channels: off by default. When True, also include
            SEAARE polygons whose OBJNAM matches a channel keyword.
            Noisy at coarse chart scales — prefer marker channels.
        named_channel_keywords: lowercase keyword list for SEAARE name
            filtering (when ``include_named_channels`` is set).
        include_depare: off by default. When True, include DEPARE
            polygons whose area and DRVAL1 fall within the inshore
            navigable range defined by the ``depare_*`` params. DEPARE
            covers the whole charted seabed and is almost never what a
            channel mask wants — opt-in only for depth-overlay output.
        depare_max_area_km2, depare_min_depth_m, depare_max_depth_m:
            filter bounds for ``include_depare=True``. Defaults are
            tuned for inshore SW-Florida charts.
    """
    _require_cv_deps()
    import fiona
    from shapely.geometry import box, mapping, shape

    name_re = _build_channel_name_regex(named_channel_keywords)
    clip_poly = box(*bbox_4326) if bbox_4326 else None
    counts: dict[str, int] = {
        "FAIRWY": 0, "DRGARE": 0, "MARKERS": 0, "SEAARE": 0, "DEPARE": 0,
    }
    clipped_out = 0
    out_features: list[dict] = []

    try:
        available_layers = set(fiona.listlayers(str(enc_path)))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"can't read ENC {enc_path}: {e}") from e

    # Polygon layers (pipeline #1)
    target_layers: list[str] = [
        layer for layer in CHANNEL_LAYERS if layer in available_layers
    ]
    if include_named_channels and "SEAARE" in available_layers:
        target_layers.append("SEAARE")
    if include_depare and "DEPARE" in available_layers:
        target_layers.append("DEPARE")

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

                # Per-layer inclusion filters.
                if layer_name == "SEAARE":
                    objnam = props.get("OBJNAM") or ""
                    if not name_re.search(objnam):
                        continue
                elif layer_name == "DEPARE":
                    drval1 = props.get("DRVAL1")
                    if drval1 is None:
                        continue
                    if drval1 < depare_min_depth_m or drval1 > depare_max_depth_m:
                        continue
                    if _polygon_area_km2(shp) > depare_max_area_km2:
                        continue

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

    # Lateral-marker channels (pipeline #2)
    if include_marker_channels:
        marker_feats, _unparsed = _extract_marker_channels(
            enc_path, marker_buffer_m=marker_buffer_m,
        )
        for feat in marker_feats:
            shp = shape(feat["geometry"])
            if clip_poly is not None:
                shp = shp.intersection(clip_poly)
                if shp.is_empty:
                    clipped_out += 1
                    continue
            out_features.append({
                "type": "Feature",
                "geometry": mapping(shp),
                "properties": feat["properties"],
            })
            counts["MARKERS"] += 1

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
