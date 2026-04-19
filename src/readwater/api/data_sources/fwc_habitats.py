"""Fetch wild oyster reef and seagrass polygons from FWC ArcGIS REST services.

FWC (Florida Fish and Wildlife Conservation Commission) publishes two
statewide habitat compilations relevant to the fishing pipeline:

  - "Oyster Beds in Florida" — SURVEYED WILD oyster reefs (NOT aquaculture
    leases). Provenance in the Marco/Naples area: majority from the 2014
    Rookery Bay NERR benthic habitat map, supplemented by a 2006 Naples
    survey and 1999 NWI polygons. Dead-oyster polygons were removed by
    FWC in January 2022, so every live polygon represents present-day
    reef extent.

  - "Seagrass Habitat in Florida" — surveyed SAV beds with binary density
    classification (Continuous / Discontinuous). Provenance in our area:
    2014 Rookery Bay, 2021 SFWMD west coast, 2007 Naples, with older
    supplementary surveys.

Both services are public ArcGIS REST /query endpoints, no auth. Response
comes back as a standard GeoJSON FeatureCollection when queried with
`f=geojson`. Pagination is handled via resultOffset + resultRecordCount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

OYSTER_SERVICE_URL = (
    "https://gis.myfwc.com/hosting/rest/services/"
    "Open_Data/Oyster_Beds_Statewide/MapServer/17/query"
)
SEAGRASS_SERVICE_URL = (
    "https://gis.myfwc.com/hosting/rest/services/"
    "Open_Data/Seagrass_Statewide/MapServer/15/query"
)

# FWC ArcGIS services cap at 2000 features per response by default.
DEFAULT_PAGE_SIZE = 2000


@dataclass
class FWCFetchResult:
    path: str
    feature_count: int
    bbox_4326: tuple[float, float, float, float]
    source_url: str


def fetch_oyster_beds(
    bbox_4326: tuple[float, float, float, float],
    out_geojson_path: str | Path,
    timeout_s: float = 60.0,
) -> FWCFetchResult:
    """Fetch surveyed wild oyster reef polygons in `bbox_4326` to GeoJSON."""
    return query_arcgis_feature_service(
        OYSTER_SERVICE_URL, bbox_4326, out_geojson_path, timeout_s,
    )


def fetch_seagrass(
    bbox_4326: tuple[float, float, float, float],
    out_geojson_path: str | Path,
    timeout_s: float = 60.0,
) -> FWCFetchResult:
    """Fetch surveyed seagrass polygons in `bbox_4326` to GeoJSON."""
    return query_arcgis_feature_service(
        SEAGRASS_SERVICE_URL, bbox_4326, out_geojson_path, timeout_s,
    )


def query_arcgis_feature_service(
    service_url: str,
    bbox_4326: tuple[float, float, float, float],
    out_geojson_path: str | Path,
    timeout_s: float = 60.0,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> FWCFetchResult:
    """Paginated query of an ArcGIS REST /query endpoint into a merged GeoJSON.

    Uses `inSR=4326` for the bbox envelope and `outSR=4326` for returned
    geometries so everything downstream stays in WGS84. Returns an
    FWCFetchResult regardless of source (the struct is generic enough).
    """
    xmin, ymin, xmax, ymax = bbox_4326
    all_features: list[dict] = []
    offset = 0
    with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
        while True:
            params = {
                "where": "1=1",
                "geometry": f"{xmin},{ymin},{xmax},{ymax}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "outSR": "4326",
                "outFields": "*",
                "f": "geojson",
                "resultOffset": str(offset),
                "resultRecordCount": str(page_size),
            }
            r = client.get(service_url, params=params)
            r.raise_for_status()
            page = r.json()
            feats = page.get("features", []) or []
            all_features.extend(feats)
            if len(feats) < page_size:
                break
            offset += page_size

    geo = {"type": "FeatureCollection", "features": all_features}
    Path(out_geojson_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_geojson_path, "w", encoding="utf-8") as f:
        json.dump(geo, f)

    return FWCFetchResult(
        path=str(out_geojson_path),
        feature_count=len(all_features),
        bbox_4326=bbox_4326,
        source_url=service_url,
    )
