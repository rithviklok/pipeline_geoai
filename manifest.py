"""
manifest.py — Run provenance, hashing, and "publish immutable month" tracking.

Every pipeline run gets a manifest recording exactly what inputs and settings
produced its outputs, plus a content hash of each output file. Manifests are
kept indefinitely under {output_dir}/.manifests/{city}/ so a given month's
publish can always be traced back to the run (and inputs) that produced it.

The "latest" pointer file ({output_dir}/{city}_latest_manifest.json) is the
single source of truth a downstream API/dashboard should read to find the
current authoritative outputs for a city — it is only written after a run
finishes successfully end-to-end (see run_manager.py). This is the concrete
implementation of product-doc ticket 1.21 ("publish the month's list as a
fixed version") given there is no database yet.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

MANIFEST_DIR_NAME = ".manifests"


def compute_file_hash(path: str, algo: str = "sha256", chunk_size: int = 1 << 20) -> Optional[str]:
    """Stream-hash a file so we never load large CSV/GeoJSON outputs fully into memory."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    """Best-effort: record the exact code version that produced this run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def default_month(now: Optional[float] = None) -> str:
    """Return the current month as YYYY-MM (local time)."""
    return time.strftime("%Y-%m", time.localtime(now))


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def build_input_manifest(config) -> Dict[str, Any]:
    """Hash + stat every configured input file path so a run's exact inputs
    are always reconstructable later."""
    inputs: Dict[str, Any] = {}
    for key in ("mseva_path", "gis_path", "electricity_path", "geoai_output_path"):
        path = getattr(config, key, None)
        if not path:
            continue
        entry: Dict[str, Any] = {"path": path}
        if os.path.exists(path):
            entry["sha256"] = compute_file_hash(path)
            entry["size_bytes"] = os.path.getsize(path)
            entry["modified_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path))
            )
        else:
            entry["exists"] = False
        inputs[key] = entry
    return inputs


def build_output_manifest(
    output_dir: str,
    filenames: List[str],
    rel_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Hash + stat every output file found in `output_dir`.

    Each entry also records a `path` relative to `rel_base` (the flat,
    top-level output directory) so downstream consumers such as the API can
    resolve a published file's location even though it now lives under a
    nested {city}/{month}/{run_id}/ directory rather than directly in
    `rel_base`. When `rel_base` is omitted, `path` falls back to the bare
    filename (matching the historical flat layout).
    """
    outputs: Dict[str, Any] = {}
    for name in filenames:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            outputs[name] = {
                "sha256": compute_file_hash(path),
                "size_bytes": os.path.getsize(path),
                "path": os.path.relpath(path, rel_base) if rel_base else name,
            }
    return outputs


def build_manifest(
    config,
    run_id: str,
    month: str,
    steps: Optional[List[str]],
    started_at: float,
    finished_at: float,
    status: str,
    output_filenames: List[str],
    published_dir: Optional[str] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full provenance record for a single pipeline run.

    `status` is one of "success" | "failed". Output hashes are only computed
    for successful runs (a failed run's temp files are not the published
    truth and are left in place purely for debugging).

    `published_dir` is the actual directory the outputs were copied into
    (results/{city}/{month}/{run_id}/ — see run_manager._publish()). Output
    hashes are read from there, but each entry's `path` is recorded relative
    to `config.output_dir` (the flat, top-level directory), so the manifest
    stays a stable pointer regardless of the nested publish layout.
    """
    manifest = {
        "run_id": run_id,
        "city": config.name,
        "state": config.state,
        "month": month,
        "status": status,
        "steps_requested": steps,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at)),
        "elapsed_seconds": round(finished_at - started_at, 2),
        "host": platform.node(),
        "git_commit": _git_commit(),
        "config_snapshot": dataclasses.asdict(config),
        "inputs": build_input_manifest(config),
        "outputs": (
            build_output_manifest(published_dir, output_filenames, rel_base=config.output_dir)
            if status == "success" and published_dir else {}
        ),
        "error": error,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def _manifest_dir(output_dir: str, city: str) -> str:
    d = os.path.join(output_dir, MANIFEST_DIR_NAME, city)
    os.makedirs(d, exist_ok=True)
    return d


def save_manifest(output_dir: str, city: str, manifest: Dict[str, Any]) -> str:
    """Persist a manifest to the append-only manifest history. Never overwrites
    a prior run's record — every run (success or failure) gets its own file,
    satisfying "both runs are recorded separately" (ticket 1.14)."""
    d = _manifest_dir(output_dir, city)
    path = os.path.join(d, f"{manifest['run_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return path


def publish_latest(output_dir: str, city: str, manifest: Dict[str, Any], manifest_path: str) -> str:
    """Update the 'latest successful run' pointer for a city.

    This is the file a downstream API/dashboard should read to find the
    current authoritative published outputs. Only ever called after a run
    finishes with status == 'success'."""
    pointer = {
        "city": city,
        "run_id": manifest["run_id"],
        "month": manifest["month"],
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": os.path.relpath(manifest_path, output_dir),
        "outputs": manifest["outputs"],
    }
    pointer_path = os.path.join(output_dir, f"{city}_latest_manifest.json")
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2, ensure_ascii=False)
    return pointer_path


def load_latest(output_dir: str, city: str) -> Optional[Dict[str, Any]]:
    pointer_path = os.path.join(output_dir, f"{city}_latest_manifest.json")
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_history(output_dir: str, city: str) -> List[Dict[str, Any]]:
    """Return all historical manifests for a city, oldest first."""
    d = os.path.join(output_dir, MANIFEST_DIR_NAME, city)
    if not os.path.isdir(d):
        return []
    records = []
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".json"):
            with open(os.path.join(d, fname), "r", encoding="utf-8") as f:
                records.append(json.load(f))
    records.sort(key=lambda m: m.get("started_at", ""))
    return records
