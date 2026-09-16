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
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import alerts, validate_output
from . import manifest as manifest_mod
from . import run_registry
from .contracts import SCHEMA_VERSION
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
    run_id: Optional[str] = None,
    owner: str = "cli",
    resume_from_run_id: Optional[str] = None,
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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", city):
        raise RefreshFailedError(
            "city may contain only letters, numbers, spaces, '-' and '_'"
        )
    month = month or manifest_mod.default_month()
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise RefreshFailedError("month must use YYYY-MM format")
    run_id = run_id or manifest_mod.new_run_id()
    started_at = time.time()

    os.makedirs(final_output_dir, exist_ok=True)
    request = {
        "city": config.name,
        "state": config.state,
        "mseva_path": config.mseva_path,
        "gis_path": config.gis_path,
        "electricity_path": config.electricity_path,
        "geoai_output_path": config.geoai_output_path,
        "model_dir": config.model_dir,
        "force_retrain": config.force_retrain,
        "output_dir": final_output_dir,
        "month": month,
        "steps": steps,
        "change_detection_path": change_detection_path,
        "resume_from_run_id": resume_from_run_id,
    }
    existing = run_registry.load_run(final_output_dir, run_id)
    if existing is None:
        run_registry.create_run(final_output_dir, run_id, request, owner=owner)

    terminal_manifest = manifest_mod.load_manifest(final_output_dir, city, run_id)
    if terminal_manifest and terminal_manifest.get("status") == "success":
        manifest_path = os.path.join(
            final_output_dir, ".manifests", city, f"{run_id}.json"
        )
        manifest_mod.publish_latest(
            final_output_dir, city, terminal_manifest, manifest_path
        )
        run_registry.update_run(
            final_output_dir,
            run_id,
            _allow_terminal_repair=True,
            status="SUCCEEDED",
            pid=None,
            outputs=terminal_manifest.get("outputs", {}),
            manifest_path=os.path.relpath(manifest_path, final_output_dir),
        )
        return terminal_manifest
    if terminal_manifest:
        run_registry.update_run(
            final_output_dir,
            run_id,
            _allow_terminal_repair=True,
            status="FAILED",
            pid=None,
            error=terminal_manifest.get("error"),
        )
        raise RefreshFailedError(f"Run '{run_id}' already has a failed receipt")
    if existing and existing.get("status") in run_registry.TERMINAL_STATUSES:
        raise RefreshFailedError(f"Run '{run_id}' is already {existing['status']}")

    config = dataclasses.replace(
        config,
        run_id=run_id,
        data_month=month,
        schema_version=SCHEMA_VERSION,
    )
    run_registry.update_run(
        final_output_dir,
        run_id,
        status="RUNNING",
        pid=os.getpid(),
        error=None,
    )

    try:
        with city_lock(final_output_dir, city, run_id=run_id):
            manifest = _run_locked(
                config=config, city=city, month=month, run_id=run_id,
                started_at=started_at, steps=steps,
                change_detection_path=change_detection_path,
                final_output_dir=final_output_dir,
                resume_from_run_id=resume_from_run_id,
            )
        run_registry.update_run(
            final_output_dir,
            run_id,
            status="SUCCEEDED",
            pid=None,
            outputs=manifest.get("outputs", {}),
            manifest_path=os.path.relpath(
                os.path.join(
                    final_output_dir, ".manifests", city, f"{run_id}.json"
                ),
                final_output_dir,
            ),
        )
        return manifest
    except Exception as e:
        finished_at = time.time()
        error_message = f"Refresh for '{city}' failed: {e}"
        logger.error(error_message, exc_info=True)
        if manifest_mod.load_manifest(final_output_dir, city, run_id) is None:
            failed_manifest = manifest_mod.build_manifest(
                config=config,
                run_id=run_id,
                month=month,
                steps=steps,
                started_at=started_at,
                finished_at=finished_at,
                status="failed",
                output_filenames=[],
                error=str(e),
            )
            manifest_mod.save_manifest(final_output_dir, city, failed_manifest)
        run_registry.update_run(
            final_output_dir,
            run_id,
            status="FAILED",
            pid=None,
            error=str(e),
        )
        alerts.refresh_failed(city, error_message, context={"run_id": run_id})
        if isinstance(e, RefreshAlreadyRunningError):
            raise RefreshFailedError(str(e)) from e
        if isinstance(e, RefreshFailedError):
            raise
        raise RefreshFailedError(
            f"{error_message}\nPartial outputs (if any) remain under "
            f"{os.path.join(final_output_dir, '.runs')}."
        ) from e


def _run_locked(
    config: CityConfig,
    city: str,
    month: str,
    run_id: str,
    started_at: float,
    steps: Optional[List[str]],
    change_detection_path: Optional[str],
    final_output_dir: str,
    resume_from_run_id: Optional[str],
) -> Dict[str, Any]:
    tmp_dir = os.path.join(final_output_dir, ".runs", f"{city}_{run_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    run_config = dataclasses.replace(config, output_dir=tmp_dir)
    checkpoint = _load_inference_checkpoint(tmp_dir, config)
    if checkpoint is None and resume_from_run_id:
        source_tmp_dir = os.path.join(
            final_output_dir, ".runs", f"{city}_{resume_from_run_id}"
        )
        source_checkpoint = _load_inference_checkpoint(source_tmp_dir, config)
        if source_checkpoint:
            source_df = pd.read_csv(
                source_checkpoint["artifact"], low_memory=False
            )
            source_df["City"] = config.name
            source_df["State"] = config.state
            source_df["month"] = config.data_month or ""
            source_df["run_id"] = config.run_id or ""
            source_df["schema_version"] = config.schema_version
            local_artifact = os.path.join(
                tmp_dir, f"{city}_GeoAI_Geocoded.csv"
            )
            source_df.to_csv(local_artifact, index=False)
            checkpoint = _save_inference_checkpoint(tmp_dir, config, city)
            run_registry.update_run(
                final_output_dir, run_id, checkpoint=checkpoint
            )
    run_steps = steps
    if checkpoint and (steps is None or "infer" in steps):
        run_steps = ["load", "match", "defaulters", "report"]
        logger.info("Resuming run %s from validated inference checkpoint", run_id)

    def progress(step: str, event: str) -> None:
        run_registry.update_run(
            final_output_dir,
            run_id,
            current_step=step if event == "started" else None,
            last_step=step if event == "completed" else None,
        )
        if step == "infer" and event == "completed":
            checkpoint_data = _save_inference_checkpoint(tmp_dir, config, city)
            run_registry.update_run(
                final_output_dir, run_id, checkpoint=checkpoint_data
            )

    pipeline = PropertyTaxPipeline(run_config, progress_callback=progress)
    if checkpoint:
        pipeline.load_geocoded_checkpoint(checkpoint["artifact"])
    pipeline.run(steps=run_steps)

    if change_detection_path:
        _run_change_detection(config, tmp_dir, change_detection_path, city)

    from .standardize import standardize_csv

    for fname in _output_filenames(city):
        if fname.endswith(".csv") and not fname.endswith("_GeoAI_Geocoded.csv"):
            fpath = os.path.join(tmp_dir, fname)
            if os.path.exists(fpath):
                standardize_csv(
                    fpath,
                    city,
                    config.state,
                    month,
                    run_id=run_id,
                    schema_version=SCHEMA_VERSION,
                )

    results = validate_output.validate_city_outputs(tmp_dir, city)
    logger.info("\n%s", validate_output.format_report(results))
    if not validate_output.all_ok(results):
        raise RefreshFailedError(
            "Refresh produced invalid or unreconciled outputs:\n"
            + validate_output.format_report(results)
        )

    return _publish(config, city, month, run_id, steps, started_at, tmp_dir, final_output_dir)


def _checkpoint_path(tmp_dir: str) -> Path:
    return Path(tmp_dir) / "checkpoint.json"


def _save_inference_checkpoint(
    tmp_dir: str,
    config: CityConfig,
    city: str,
) -> Dict[str, Any]:
    artifact = os.path.abspath(
        os.path.join(tmp_dir, f"{city}_GeoAI_Geocoded.csv")
    )
    if not os.path.exists(artifact):
        raise RefreshFailedError(
            "Inference completed without producing its checkpoint CSV"
        )
    checkpoint = {
        "completed_step": "infer",
        "artifact": artifact,
        "artifact_sha256": manifest_mod.compute_file_hash(artifact),
        "inputs": manifest_mod.build_input_manifest(config),
        "created_at": run_registry.utc_now(),
    }
    run_registry.atomic_write_json(_checkpoint_path(tmp_dir), checkpoint)
    return checkpoint


def _load_inference_checkpoint(
    tmp_dir: str,
    config: CityConfig,
) -> Optional[Dict[str, Any]]:
    path = _checkpoint_path(tmp_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        artifact = checkpoint["artifact"]
        if checkpoint.get("completed_step") != "infer":
            return None
        if not os.path.exists(artifact):
            return None
        if manifest_mod.compute_file_hash(artifact) != checkpoint.get(
            "artifact_sha256"
        ):
            return None
        if checkpoint.get("inputs") != manifest_mod.build_input_manifest(config):
            return None
        return checkpoint
    except (KeyError, OSError, json.JSONDecodeError):
        return None


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
    """Copy a successful run's outputs into a per-run directory nested under
    the real output directory (``{final_output_dir}/{city}/{month}/{run_id}/``)
    atomically (per-file copy2, all-or-nothing at the manifest level — a
    failure here still leaves the tmp_dir intact for retry/debugging), then
    record provenance and check for anomalies vs. the last published run.

    The provenance manifest history (``.manifests/{city}/``) and the
    ``{city}_latest_manifest.json`` pointer are deliberately kept at the flat
    `final_output_dir` root (per RUNBOOK.md) — only the data outputs
    themselves move into the nested, per-run path.
    """
    published_dir = os.path.join(final_output_dir, city, month, run_id)
    staging_dir = f"{published_dir}.publishing"
    os.makedirs(os.path.dirname(published_dir), exist_ok=True)
    if os.path.exists(published_dir):
        if manifest_mod.load_manifest(final_output_dir, city, run_id):
            raise RefreshFailedError(
                f"Refusing to overwrite existing run output: {published_dir}"
            )
        recovered = validate_output.validate_city_outputs(published_dir, city)
        if not validate_output.all_ok(recovered):
            raise RefreshFailedError(
                "An unreceipted published directory exists but is invalid: "
                f"{published_dir}"
            )
        logger.info("Recovering receipt for already-atomic run directory %s", published_dir)
    else:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=False)
        for fname in _output_filenames(city):
            src = os.path.join(tmp_dir, fname)
            if os.path.exists(src):
                destination = os.path.join(staging_dir, fname)
                shutil.copy2(src, destination)
                with open(destination, "r+b") as f:
                    os.fsync(f.fileno())
        os.rename(staging_dir, published_dir)

    finished_at = time.time()

    current_summary = None
    summary_path = os.path.join(published_dir, f"{city}_summary.json")
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
        published_dir=published_dir,
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
        "Refresh for '%s' (%s) published successfully to %s. run_id=%s, elapsed=%.1fs",
        city, month, published_dir, run_id, finished_at - started_at,
    )
    alerts.refresh_succeeded(
        city,
        f"Refresh {run_id} published successfully",
        context={
            "run_id": run_id,
            "month": month,
            "status": "SUCCEEDED",
            "manifest_path": os.path.relpath(manifest_path, final_output_dir),
            "outputs": m["outputs"],
        },
    )
    return m
