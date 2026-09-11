"""
run_manager.py — Orchestrates one full monthly-refresh execution with the
production-hardening behaviours required by the product doc:

  - Per-city lock so two refreshes for the same city can never run at once
    (ticket 4.4).
  - Isolated, atomic publish: outputs are written to a temporary run
    directory first and only copied into the real output directory if the
    entire run succeeds — a failed run never leaves half-finished data on
    the dashboard (tickets 1.14, 1.16).
  - A provenance manifest recording exact config/input hashes for every run,
    plus a "latest successful run" pointer per city (tickets 1.15, 1.21).
  - Field-list conformance validation against the Data Dictionary before a
    run is allowed to publish (ticket 1.18).
  - Alert hooks on failure and on anomalous swings vs. the last published
    run (tickets 4.5, 4.7).

This module does not change the internals of PropertyTaxPipeline /
change_detection / standardize — it wraps them. `__main__.py` calls
`execute_pipeline_run()` instead of driving those pieces directly.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from . import alerts, validate_output
from . import manifest as manifest_mod
from .config import CityConfig
from .lock import RefreshAlreadyRunningError, city_lock
from .orchestrator import PropertyTaxPipeline

logger = logging.getLogger(__name__)


class RefreshFailedError(RuntimeError):
    """Raised with a plain-language message when a run cannot be published."""


# Every file a refresh (pipeline + change detection + standardize) may
# produce. Listed once here so locking, publishing, and standardization all
# agree on the same set of output files.
_OUTPUT_FILENAME_TEMPLATES = [
    "{city}_Match_Register.csv",
    "{city}_Defaulters.csv",
    "{city}_Defaulters.geojson",
    "{city}_GeoAI_Geocoded.csv",
    "{city}_summary.json",
    "{city}_Change_Detection_Summary.csv",
    "{city}_Change_Detection.geojson",
]


def _output_filenames(city: str) -> List[str]:
    return [t.format(city=city) for t in _OUTPUT_FILENAME_TEMPLATES]


def execute_pipeline_run(
    config: CityConfig,
    steps: Optional[List[str]] = None,
    change_detection_path: Optional[str] = None,
    month: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one full monthly refresh with locking, isolation, provenance,
    validation, and alerting.

    Raises RefreshFailedError with a human-readable message on any failure;
    never leaves half-written output in `config.output_dir` — a failed run's
    partial files stay isolated in a `.runs/{city}_{run_id}/` subdirectory
    for debugging instead.

    Returns the manifest dict for the published run on success.
    """
    final_output_dir = config.output_dir
    city = config.name
    month = month or manifest_mod.default_month()
    run_id = manifest_mod.new_run_id()
    started_at = time.time()

    os.makedirs(final_output_dir, exist_ok=True)

    try:
        with city_lock(final_output_dir, city):
            return _run_locked(
                config=config, city=city, month=month, run_id=run_id,
                started_at=started_at, steps=steps,
                change_detection_path=change_detection_path,
                final_output_dir=final_output_dir,
            )
    except RefreshAlreadyRunningError as e:
        alerts.refresh_failed(city, str(e), context={"run_id": run_id})
        raise RefreshFailedError(str(e)) from e


def _run_locked(
    config: CityConfig,
    city: str,
    month: str,
    run_id: str,
    started_at: float,
    steps: Optional[List[str]],
    change_detection_path: Optional[str],
    final_output_dir: str,
) -> Dict[str, Any]:
    tmp_dir = os.path.join(final_output_dir, ".runs", f"{city}_{run_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    run_config = dataclasses.replace(config, output_dir=tmp_dir)

    try:
        pipeline = PropertyTaxPipeline(run_config)
        pipeline.run(steps=steps)

        if change_detection_path:
            _run_change_detection(config, tmp_dir, change_detection_path, city)

        from .standardize import standardize_csv

        for fname in _output_filenames(city):
            if fname.endswith(".csv"):
                fpath = os.path.join(tmp_dir, fname)
                if os.path.exists(fpath):
                    standardize_csv(fpath, city, config.state)

        results = validate_output.validate_city_outputs(tmp_dir, city)
        logger.info("\n%s", validate_output.format_report(results))
        if not validate_output.all_ok(results):
            raise RefreshFailedError(
                "Refresh produced outputs that don't match the signed-off Data "
                "Dictionary field list:\n" + validate_output.format_report(results)
            )

    except Exception as e:
        finished_at = time.time()
        error_message = f"Refresh for '{city}' failed: {e}"
        logger.error(error_message, exc_info=True)
        m = manifest_mod.build_manifest(
            config=config, run_id=run_id, month=month, steps=steps,
            started_at=started_at, finished_at=finished_at, status="failed",
            output_filenames=[], error=str(e),
        )
        manifest_mod.save_manifest(final_output_dir, city, m)
        alerts.refresh_failed(city, error_message, context={"run_id": run_id, "tmp_dir": tmp_dir})
        raise RefreshFailedError(
            f"{error_message}\nPartial outputs (if any) were left in {tmp_dir} for "
            f"debugging \u2014 nothing was published to {final_output_dir}."
        ) from e

    return _publish(config, city, month, run_id, steps, started_at, tmp_dir, final_output_dir)


def _run_change_detection(config: CityConfig, tmp_dir: str, change_detection_path: str, city: str) -> None:
    import geopandas as gpd
    import pandas as pd

    from .change_detection import process_change_detection

    gis_gdf = gpd.read_file(config.gis_path)
    register_path = os.path.join(tmp_dir, f"{city}_Match_Register.csv")
    if not os.path.exists(register_path):
        raise RefreshFailedError(
            f"Change detection was requested but no Match Register was produced at "
            f"{register_path} \u2014 run the 'match'/'report' steps first."
        )
    match_register = pd.read_csv(register_path)
    process_change_detection(
        change_shp_path=change_detection_path,
        gis_gdf=gis_gdf,
        match_register=match_register,
        gis_loc_col=(config.gis_columns or {}).get("locality", "Locality"),
        output_dir=tmp_dir,
        city_name=city,
    )


def _publish(
    config: CityConfig,
    city: str,
    month: str,
    run_id: str,
    steps: Optional[List[str]],
    started_at: float,
    tmp_dir: str,
    final_output_dir: str,
) -> Dict[str, Any]:
    """Copy a successful run's outputs into the real output directory
    atomically (per-file copy2, all-or-nothing at the manifest level — a
    failure here still leaves the tmp_dir intact for retry/debugging), then
    record provenance and check for anomalies vs. the last published run."""
    for fname in _output_filenames(city):
        src = os.path.join(tmp_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(final_output_dir, fname))

    finished_at = time.time()

    current_summary = None
    summary_path = os.path.join(final_output_dir, f"{city}_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                current_summary = json.load(f)
        except Exception:
            current_summary = None

    previous = manifest_mod.load_latest(final_output_dir, city)

    m = manifest_mod.build_manifest(
        config=config, run_id=run_id, month=month, steps=steps,
        started_at=started_at, finished_at=finished_at, status="success",
        output_filenames=_output_filenames(city),
        extra={"summary": current_summary} if current_summary else None,
    )
    manifest_path = manifest_mod.save_manifest(final_output_dir, city, m)
    manifest_mod.publish_latest(final_output_dir, city, m, manifest_path)

    if current_summary and previous:
        try:
            prev_manifest_path = os.path.join(final_output_dir, previous["manifest_path"])
            with open(prev_manifest_path, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
            prev_summary = prev_manifest.get("extra", {}).get("summary")
            if prev_summary:
                alerts.check_anomalies(city, current_summary, prev_summary)
        except Exception as e:
            logger.warning("Anomaly check skipped: %s", e)

    logger.info(
        "Refresh for '%s' (%s) published successfully. run_id=%s, elapsed=%.1fs",
        city, month, run_id, finished_at - started_at,
    )
    return m
