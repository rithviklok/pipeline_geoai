"""
main.py  —  FastAPI application for Punjab Change Detection.

Deployment : Google Cloud Run (any project with EE-enabled service account)
Auth       : Application Default Credentials via attached service account.
CORS       : Vercel frontend domain + localhost for local dev.
Credentials: never returned to the browser; all GEE calls happen server-side.
"""

from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import ee_service
import database
from models import ChangeRequest, TileResponse, AreaStatsResponse, ShapefileResponse


# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Punjab Change Detection API",
    description="On-the-fly AlphaEarth + NDBI/NDVI urban growth detection for Punjab cities.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict to Vercel domain in production via CORS_ORIGIN env var.
_vercel_origin = os.environ.get("CORS_ORIGIN", "")
_cors_origins = (
    [_vercel_origin, "http://localhost:5500", "http://localhost:3000",
     "http://127.0.0.1:5500", "http://127.0.0.1:3000"]
    if _vercel_origin
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialise GEE once at startup
ee_service.initialize_ee()


# ── Utility ────────────────────────────────────────────────────────────────────

def _cached_tile(cache_key: str, compute_fn, *args) -> str:
    """Return a cached tile URL or compute + cache a new one."""
    cached = database.get_cached(cache_key)
    if cached:
        return cached
    tile_url = compute_fn(*args)
    database.save_cached(cache_key, tile_url)
    return tile_url


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "punjab-change-detection",
        "supported_cities": list(ee_service.CITY_BOUNDS.keys()),
        "gis_survey_years": ee_service.GIS_SURVEY_YEARS,
    }


@app.get("/cities")
def list_cities():
    """Return supported cities with their bounding boxes and GIS survey years."""
    cities = []
    for name, bounds in ee_service.CITY_BOUNDS.items():
        cities.append({
            "name": name,
            "bounds": bounds,
            "gis_survey_year": ee_service.GIS_SURVEY_YEARS.get(name),
        })
    return {"cities": cities}


# ── Change detection tiles ─────────────────────────────────────────────────────

@app.post("/change-map", response_model=TileResponse)
def change_map(req: ChangeRequest):
    """
    Compute AlphaEarth + NDBI urban growth detection and return a map tile URL.

    The dashboard calls this with e.g. { "city": "Mohali", "year1": 2014, "year2": 2025 }
    and renders the returned tile_url directly in Leaflet/Mapbox.
    """
    bounds = req.bounds or ee_service.CITY_BOUNDS.get(req.city)
    if not bounds:
        raise HTTPException(status_code=400, detail=f"Unknown city: {req.city}")

    cache_key = f"{req.city}_change_{req.year1}_{req.year2}"
    try:
        tile_url = _cached_tile(
            cache_key,
            ee_service.compute_change_map,
            req.year1, req.year2, bounds,
        )
        return TileResponse(
            tile_url=tile_url,
            city=req.city,
            year1=req.year1,
            year2=req.year2,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Shapefile Download ─────────────────────────────────────────────────────────

@app.post("/change-shapefile", response_model=ShapefileResponse)
def change_shapefile(req: ChangeRequest):
    """
    Compute urban growth, extract polygons, and return a download URL for a Shapefile.
    This is extremely useful if the user wants to download the `.shp` locally.
    """
    bounds = req.bounds or ee_service.CITY_BOUNDS.get(req.city)
    if not bounds:
        raise HTTPException(status_code=400, detail=f"Unknown city: {req.city}")

    cache_key = f"{req.city}_shp_{req.year1}_{req.year2}"
    try:
        download_url = _cached_tile(
            cache_key,
            ee_service.compute_change_shapefile,
            req.year1, req.year2, bounds, req.city
        )
        return ShapefileResponse(
            download_url=download_url,
            city=req.city,
            year1=req.year1,
            year2=req.year2,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ── Area statistics ────────────────────────────────────────────────────────────

@app.post("/area-stats", response_model=AreaStatsResponse)
def area_stats(req: ChangeRequest):
    """Compute total new built-up area (sq km) between two years."""
    bounds = req.bounds or ee_service.CITY_BOUNDS.get(req.city)
    if not bounds:
        raise HTTPException(status_code=400, detail=f"Unknown city: {req.city}")

    try:
        result = ee_service.compute_area_stats(req.year1, req.year2, bounds)
        result["city"] = req.city
        return AreaStatsResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.delete("/cache")
def clear_cache():
    """Clear the tile-URL cache (admin use only)."""
    deleted = database.clear_cache()
    return {"deleted": deleted}
