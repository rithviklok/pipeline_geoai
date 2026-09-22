"""
ee_service.py  —  Google Earth Engine processing for Punjab Change Detection.

Translates the GIS Analyst's AlphaEarth + NDBI + NDVI change detection
script from JavaScript to Python. All computation happens server-side on
Google's Earth Engine infrastructure.

Authentication
--------------
Cloud Run : Application Default Credentials (ADC) via the attached service account.
Local dev : cached user credentials from `earthengine authenticate`.
No service-account key file is required or used.
"""

from __future__ import annotations

import os
import ee
import google.auth


# ── Constants ──────────────────────────────────────────────────────────────────

# Default bounding boxes per city  [W, S, E, N]
CITY_BOUNDS: dict[str, list[float]] = {
    "Mohali":    [76.6500, 30.6500, 76.8000, 30.8000],
    "Barnala":   [75.5000, 30.3500, 75.5700, 30.4100],
    "Amritsar":  [74.8100, 31.5800, 74.9200, 31.6700],
}

# Baseline year options: the year the GIS survey was conducted per city
GIS_SURVEY_YEARS: dict[str, int] = {
    "Mohali":   2014,
    "Barnala":  2017,
    "Amritsar": 2017,
}

EE_SCOPES = ["https://www.googleapis.com/auth/earthengine"]


# ── Initialization ─────────────────────────────────────────────────────────────

def initialize_ee() -> None:
    """
    Initialize Google Earth Engine using Application Default Credentials.

    Cloud Run : The attached service account is picked up automatically via ADC.
    Local dev : Falls back to cached user credentials from `earthengine authenticate`.
    """
    project = os.environ.get("GEE_PROJECT_ID", "change-detection-494607")

    try:
        credentials, detected_project = google.auth.default(scopes=EE_SCOPES)
        ee.Initialize(credentials, project=project or detected_project)
        print(f"[OK] EE initialized (ADC) — project: {project or detected_project}")
    except Exception:
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            print(f"[OK] EE initialized (user auth)")
        except Exception as exc:
            raise RuntimeError(f"EE Initialization failed: {exc}") from exc


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_roi(bounds: list[float]) -> ee.Geometry:
    """Create an Earth Engine Rectangle from [W, S, E, N] bounds."""
    return ee.Geometry.Rectangle(bounds)


# ── Public computation functions ───────────────────────────────────────────────

def compute_change_map(year1: int, year2: int, bounds: list[float]) -> str:
    """
    AlphaEarth + NDBI + NDVI urban growth detection.

    Faithfully translates the GIS Analyst's script:
    - AlphaEarth satellite embeddings for semantic change
    - Landsat 8 NDBI for baseline year built-up index
    - Sentinel-2 NDBI/NDVI for current year built-up and vegetation

    Returns a GEE tile URL that Leaflet/Mapbox can render directly.
    """
    aoi = _get_roi(bounds)

    # ── 1. Landsat 8 — Baseline Year NDBI (Starts 2013) ───────────────────
    l8_baseline = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                   .filterBounds(aoi)
                   .filterDate(f"{year1}-01-01", f"{year1}-12-31")
                   .filter(ee.Filter.lt("CLOUD_COVER", 20))
                   .median()
                   .clip(aoi))

    ndbi_baseline = l8_baseline.normalizedDifference(["SR_B6", "SR_B5"])

    # ── 2. Sentinel-2 — Current Year NDBI & NDVI ─────────────────────────
    s2_current = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(f"{year2}-01-01", f"{year2}-12-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .median()
                  .clip(aoi))

    ndbi_current = s2_current.normalizedDifference(["B11", "B8"])
    ndvi_current = s2_current.normalizedDifference(["B8", "B4"])

    # ── 3. Built-up change ────────────────────────────────────────────────
    built_change = ndbi_current.subtract(ndbi_baseline)

    # ── 4. Final urban growth mask ────────────────────────────────────────
    # AlphaEarth embeddings are only available from 2017 onwards.
    if year1 >= 2017 and year2 >= 2017:
        alpha = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        alpha_y1 = alpha.filterDate(f"{year1}-01-01", f"{year1 + 1}-01-01").filterBounds(aoi).first()
        alpha_y2 = alpha.filterDate(f"{year2}-01-01", f"{year2 + 1}-01-01").filterBounds(aoi).first()
        similarity = alpha_y1.multiply(alpha_y2).reduce(ee.Reducer.sum()).clip(aoi)
        alpha_change = ee.Image(1).subtract(similarity)
        
        urban_growth = (alpha_change.gt(0.05)
                        .And(built_change.gt(0.05))
                        .And(ndvi_current.lt(0.5)))
    else:
        # Pre-2017 fallback (Mohali 2014): Rely on strict NDBI/NDVI
        urban_growth = (built_change.gt(0.1)
                        .And(ndvi_current.lt(0.4)))

    # selfMask: pixels where urban_growth == 0 become transparent
    urban_growth_masked = urban_growth.selfMask()

    # ── 6. Generate tile URL ──────────────────────────────────────────────
    map_id = urban_growth_masked.getMapId({"palette": ["red"]})
    return map_id["tile_fetcher"].url_format


def compute_area_stats(year1: int, year2: int, bounds: list[float]) -> dict:
    """
    Compute total new built-up area (sq km) detected between two years.
    """
    aoi = _get_roi(bounds)

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(aoi).filterDate(f"{year1}-01-01", f"{year1}-12-31")
          .filter(ee.Filter.lt("CLOUD_COVER", 20)).median().clip(aoi))
    ndbi_b = l8.normalizedDifference(["SR_B6", "SR_B5"])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(f"{year2}-01-01", f"{year2}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).median().clip(aoi))
    ndbi_c = s2.normalizedDifference(["B11", "B8"])
    ndvi_c = s2.normalizedDifference(["B8", "B4"])

    built_change = ndbi_c.subtract(ndbi_b)
    
    if year1 >= 2017 and year2 >= 2017:
        alpha = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        alpha_y1 = alpha.filterDate(f"{year1}-01-01", f"{year1 + 1}-01-01").filterBounds(aoi).first()
        alpha_y2 = alpha.filterDate(f"{year2}-01-01", f"{year2 + 1}-01-01").filterBounds(aoi).first()
        similarity = alpha_y1.multiply(alpha_y2).reduce(ee.Reducer.sum()).clip(aoi)
        alpha_change = ee.Image(1).subtract(similarity)
        urban_growth = (alpha_change.gt(0.05)
                        .And(built_change.gt(0.05))
                        .And(ndvi_c.lt(0.5)))
    else:
        urban_growth = (built_change.gt(0.1)
                        .And(ndvi_c.lt(0.4)))

    area_image = urban_growth.multiply(ee.Image.pixelArea()).divide(1e6)  # sq km
    result = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=20,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()

    area_sqkm = round(list(result.values())[0] or 0, 2)

    return {
        "year1": year1,
        "year2": year2,
        "new_builtup_sqkm": area_sqkm,
    }


def compute_change_shapefile(year1: int, year2: int, bounds: list[float], city_name: str) -> str:
    """
    Computes urban growth, converts it to polygon clusters, filters out
    small clusters (< 0.5 ha), and returns a URL to download the result as a Shapefile.
    """
    aoi = _get_roi(bounds)

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(aoi).filterDate(f"{year1}-01-01", f"{year1}-12-31")
          .filter(ee.Filter.lt("CLOUD_COVER", 20)).median().clip(aoi))
    ndbi_b = l8.normalizedDifference(["SR_B6", "SR_B5"])

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(f"{year2}-01-01", f"{year2}-12-31")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).median().clip(aoi))
    ndbi_c = s2.normalizedDifference(["B11", "B8"])
    ndvi_c = s2.normalizedDifference(["B8", "B4"])

    built_change = ndbi_c.subtract(ndbi_b)
    
    if year1 >= 2017 and year2 >= 2017:
        alpha = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        alpha_y1 = alpha.filterDate(f"{year1}-01-01", f"{year1 + 1}-01-01").filterBounds(aoi).first()
        alpha_y2 = alpha.filterDate(f"{year2}-01-01", f"{year2 + 1}-01-01").filterBounds(aoi).first()
        similarity = alpha_y1.multiply(alpha_y2).reduce(ee.Reducer.sum()).clip(aoi)
        alpha_change = ee.Image(1).subtract(similarity)
        urban_growth = (alpha_change.gt(0.05)
                        .And(built_change.gt(0.05))
                        .And(ndvi_c.lt(0.5)))
    else:
        urban_growth = (built_change.gt(0.1)
                        .And(ndvi_c.lt(0.4)))

    # Convert raster to polygons
    builtup_clusters = urban_growth.selfMask().reduceToVectors(
        geometry=aoi,
        scale=20,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="builtup",
        reducer=ee.Reducer.countEvery()
    )

    # Map function to calculate area
    def calc_area(feature):
        geom = feature.geometry().transform("EPSG:4326", 1)
        area = geom.area(maxError=1)
        return feature.set({
            "area_sqm": area,
            "area_ha": ee.Number(area).divide(10000)
        })

    builtup_clusters = builtup_clusters.map(calc_area)

    # Filter out clusters < 0.5 ha
    large_clusters = builtup_clusters.filter(ee.Filter.gt("area_ha", 0.5))

    # Generate a download URL for the shapefile (.zip containing .shp)
    filename = f"{city_name}_Change_Detection_{year1}_{year2}"
    download_url = large_clusters.getDownloadURL(
        filetype="SHP",
        filename=filename
    )
    
    return download_url
