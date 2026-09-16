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

import hmac
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .. import manifest as manifest_mod
from .. import run_registry
from ..contracts import SCHEMA_VERSION
from ..job_queue import JobQueue

OUTPUT_DIR = os.environ.get("PIPELINE_OUTPUT_DIR", "results")
job_queue = JobQueue(OUTPUT_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    job_queue.start()
    try:
        yield
    finally:
        job_queue.stop()

app = FastAPI(
    title="Property Tax Pipeline API",
    description=(
        "Submits durable monthly-refresh jobs and serves immutable outputs."
    ),
    version=SCHEMA_VERSION,
    lifespan=lifespan,
)


class RunSubmission(BaseModel):
    city: str
    state: str
    month: str
    mseva_path: str
    gis_path: str
    electricity_path: Optional[str] = None
    geoai_output_path: Optional[str] = None
    model_dir: Optional[str] = None
    force_retrain: bool = False
    change_detection_path: Optional[str] = None
    steps: Optional[List[str]] = None
    owner: str = "api"


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("PIPELINE_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PIPELINE_API_KEY is not configured",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _validate_submission(submission: RunSubmission) -> Dict[str, Any]:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", submission.month):
        raise HTTPException(status_code=422, detail="month must use YYYY-MM format")
    if not submission.city.strip() or not submission.state.strip():
        raise HTTPException(status_code=422, detail="city and state are required")
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9 _-]*", submission.city.strip()
    ):
        raise HTTPException(status_code=422, detail="city contains unsafe characters")

    required_paths = {
        "mseva_path": submission.mseva_path,
        "gis_path": submission.gis_path,
    }
    optional_paths = {
        "electricity_path": submission.electricity_path,
        "geoai_output_path": submission.geoai_output_path,
        "change_detection_path": submission.change_detection_path,
    }
    resolved: Dict[str, Optional[str]] = {}
    for key, value in {**required_paths, **optional_paths}.items():
        if not value:
            resolved[key] = None
            continue
        path = os.path.abspath(os.path.expanduser(value))
        if not os.path.isfile(path):
            raise HTTPException(status_code=422, detail=f"{key} not found: {path}")
        resolved[key] = path

    gis_path = resolved["gis_path"]
    if gis_path and gis_path.lower().endswith(".shp"):
        missing = [
            os.path.splitext(gis_path)[0] + ext
            for ext in (".shx", ".dbf")
            if not os.path.isfile(os.path.splitext(gis_path)[0] + ext)
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"GIS shapefile is incomplete; missing: {missing}",
            )

    allowed_steps = {
        "load", "train", "infer", "match", "defaulters", "report"
    }
    if submission.steps and not set(submission.steps).issubset(allowed_steps):
        raise HTTPException(status_code=422, detail="steps contains an unknown step")

    request = (
        submission.model_dump()
        if hasattr(submission, "model_dump")
        else submission.dict()
    )
    request.update(resolved)
    request["city"] = submission.city.strip()
    request["state"] = submission.state.strip()
    request["output_dir"] = os.path.abspath(OUTPUT_DIR)
    if submission.model_dir:
        request["model_dir"] = os.path.abspath(
            os.path.expanduser(submission.model_dir)
        )
    return request


def _latest_or_404(city: str) -> Dict[str, Any]:
    latest = manifest_mod.load_latest(OUTPUT_DIR, city)
    if not latest:
        raise HTTPException(
            status_code=404,
            detail=f"No published refresh found for city '{city}' in '{OUTPUT_DIR}'.",
        )
    return latest


def _output_path(city: str, filename: str) -> str:
    """Resolve a published output file's path through the
    {city}_latest_manifest.json pointer's stored `outputs[filename].path`
    (relative to OUTPUT_DIR) rather than assuming a flat layout — outputs
    now live under OUTPUT_DIR/{city}/{month}/{run_id}/."""
    latest = _latest_or_404(city)
    entry = latest.get("outputs", {}).get(filename)
    if not entry or "path" not in entry:
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' is not part of the latest published outputs for '{city}'.",
        )
    return _safe_output_path(entry["path"])


def _safe_output_path(relative_path: str) -> str:
    root = os.path.abspath(OUTPUT_DIR)
    path = os.path.abspath(os.path.join(root, relative_path))
    if os.path.commonpath([root, path]) != root:
        raise HTTPException(status_code=500, detail="Invalid output path in receipt")
    return path


def _read_json_output(city: str, filename_template: str) -> Any:
    path = _output_path(city, filename_template.format(city=city))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@app.post(
    "/runs",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
def submit_run(submission: RunSubmission) -> Dict[str, Any]:
    request = _validate_submission(submission)
    run_id = manifest_mod.new_run_id()
    record = run_registry.create_run(
        OUTPUT_DIR,
        run_id,
        request,
        owner=submission.owner.strip() or "api",
    )
    job_queue.enqueue(run_id)
    return {
        "run_id": run_id,
        "status": record["status"],
        "status_url": f"/runs/{run_id}",
    }


@app.get("/runs", dependencies=[Depends(require_api_key)])
def runs(
    city: Optional[str] = Query(default=None),
    run_status: Optional[str] = Query(default=None, alias="status"),
) -> Dict[str, Any]:
    return {
        "runs": run_registry.list_runs(
            OUTPUT_DIR, city=city, status=run_status
        )
    }


@app.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
def run_status(run_id: str) -> Dict[str, Any]:
    record = run_registry.load_run(OUTPUT_DIR, run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return record


@app.post(
    "/runs/{run_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
def retry_run(run_id: str) -> Dict[str, Any]:
    previous = run_registry.load_run(OUTPUT_DIR, run_id)
    if not previous:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if previous.get("status") != "FAILED":
        raise HTTPException(
            status_code=409,
            detail="Only a failed run can be retried",
        )
    request = dict(previous["request"])
    new_run_id = manifest_mod.new_run_id()
    request["resume_from_run_id"] = run_id
    record = run_registry.create_run(
        OUTPUT_DIR,
        new_run_id,
        request,
        owner=previous.get("owner", "api"),
    )
    job_queue.enqueue(new_run_id)
    return {
        "run_id": new_run_id,
        "resumed_from_run_id": run_id,
        "status": record["status"],
        "status_url": f"/runs/{new_run_id}",
    }


@app.get(
    "/runs/{run_id}/outputs/{filename}",
    dependencies=[Depends(require_api_key)],
)
def run_output(run_id: str, filename: str) -> FileResponse:
    record = run_registry.load_run(OUTPUT_DIR, run_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    entry = record.get("outputs", {}).get(filename)
    if not entry or "path" not in entry:
        raise HTTPException(
            status_code=404,
            detail=f"Output '{filename}' is not published for run '{run_id}'",
        )
    path = _safe_output_path(entry["path"])
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Published output is missing")
    return FileResponse(path, filename=filename)


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

    path = _output_path(city, f"{city}_Match_Register.csv")
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
