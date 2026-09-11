"""
api/app.py — Minimal read-only API exposing published pipeline outputs.

This is intentionally small: it serves the already-published, versioned
outputs for each city (Match Register, Defaulters GeoJSON, Change-Detection
GeoJSON, Summary JSON) so the dashboard team can wire real data instead of
mocks, without standing up a database or job queue yet. It reads directly
from the `{city}_latest_manifest.json` pointer written by
`run_manager.publish_latest()` — it never triggers a pipeline run itself.

Run locally with:
    py -m pipeline_geoai.api
or:
    uvicorn pipeline_geoai.api.app:app --reload --port 8000

Configure the output directory to serve via the PIPELINE_OUTPUT_DIR
environment variable (defaults to "results", matching __main__.py's
--output default).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .. import manifest as manifest_mod

OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "results")

app = FastAPI(
    title="Property Tax Pipeline — Read-Only API",
    description=(
        "Serves the latest published monthly-refresh outputs per city. "
        "Read-only: does not trigger or modify pipeline runs."
    ),
    version="0.1.0",
)


def _latest_or_404(city: str) -> Dict[str, Any]:
    latest = manifest_mod.load_latest(OUTPUT_DIR, city)
    if not latest:
        raise HTTPException(
            status_code=404,
            detail=f"No published refresh found for city '{city}' in '{OUTPUT_DIR}'.",
        )
    return latest


def _read_json_output(city: str, filename_template: str) -> Any:
    _latest_or_404(city)  # ensures a run has actually been published
    path = os.path.join(OUTPUT_DIR, filename_template.format(city=city))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/cities/{city}/latest")
def latest_run(city: str) -> Dict[str, Any]:
    """Metadata about the most recently published refresh for a city --
    the pointer the dashboard should treat as the single source of truth."""
    return _latest_or_404(city)


@app.get("/cities/{city}/summary")
def summary(city: str) -> Dict[str, Any]:
    return _read_json_output(city, "{city}_summary.json")


@app.get("/cities/{city}/defaulters.geojson")
def defaulters_geojson(city: str) -> JSONResponse:
    data = _read_json_output(city, "{city}_Defaulters.geojson")
    return JSONResponse(content=data, media_type="application/geo+json")


@app.get("/cities/{city}/change-detection.geojson")
def change_detection_geojson(city: str) -> JSONResponse:
    data = _read_json_output(city, "{city}_Change_Detection.geojson")
    return JSONResponse(content=data, media_type="application/geo+json")


@app.get("/cities/{city}/match-register")
def match_register(city: str, limit: int = 500, offset: int = 0) -> Dict[str, Any]:
    """Paginated Match Register rows (CSV converted to JSON on the fly)."""
    import pandas as pd

    _latest_or_404(city)
    path = os.path.join(OUTPUT_DIR, f"{city}_Match_Register.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{path} not found.")
    df = pd.read_csv(path)
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": page.to_dict(orient="records"),
    }


@app.get("/cities/{city}/manifest-history")
def manifest_history(city: str) -> Dict[str, Any]:
    """Full provenance history for a city -- every run, success or failure."""
    return {"city": city, "runs": manifest_mod.load_manifest_history(OUTPUT_DIR, city)}
