"""
validate_output.py — Field-list conformance check against the signed-off
Data Dictionary (product-doc ticket 1.18: "Make the refresh output match the
agreed field list exactly. The shared checker passes on a full month for
both cities with nothing missing.").

Checks that a completed refresh's output files contain (at minimum) the
columns/keys documented in data_dictionary.pdf, §3, so the dashboard and any
other downstream consumer can rely on a stable contract. This checks for
*presence* of the required fields (extra columns are fine) rather than exact
equality, since the pipeline may add useful diagnostic columns over time.

If the Data Dictionary is amended, update the *_REQUIRED_* lists below to
match — they are the single source of truth this validator checks against.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# Data Dictionary §3.2 — Match Register CSV (columns beyond the pass-through
# original mSeva fields).
#
# `property_uid` and `month` are the cross-month join-key contract columns
# (property_uid added in orchestrator._step_report; month is stamped by
# standardize.standardize_csv during publish).
MATCH_REGISTER_REQUIRED_COLUMNS = [
    "propertyid", "matched_uid", "match_method",
    "gis_owner_name", "gis_mobile", "gis_locality",
    "property_uid", "month",
]

# Data Dictionary §3.3 — Defaulters CSV (now the unified GIS-parcel register:
# every loaded parcel, not just taxable-unmatched ones).
#
# `property_uid`/`tax_status`/`geo_status`/`ward_id` are added in
# orchestrator._step_defaulters; `month` is stamped by standardize_csv.
DEFAULTERS_CSV_REQUIRED_COLUMNS = [
    "gis_uid", "gis_owner_name", "gis_guardian_name", "gis_mobile", "gis_locality",
    "latitude", "longitude", "property_usage", "property_type", "status",
    "electricity_account_no", "electricity_holder_name",
    "property_uid", "tax_status", "geo_status", "ward_id", "month",
]

# Data Dictionary §3.4 — Defaulters GeoJSON feature properties (unified
# GIS-parcel register). No `month` here: GeoJSON files are not run through
# standardize_csv (CSV-only), so month is a CSV-only contract column.
DEFAULTERS_GEOJSON_REQUIRED_PROPERTIES = [
    "gis_uid", "gis_owner_name", "gis_guardian_name", "gis_mobile", "gis_locality",
    "property_usage", "property_type", "status",
    "electricity_account_no", "electricity_holder_name",
    "property_uid", "tax_status", "geo_status", "ward_id",
]

# Data Dictionary §3.5 — Summary JSON.
SUMMARY_JSON_REQUIRED_KEYS = [
    "city", "total_mseva", "total_gis", "matched_count", "unmatched_count",
    "match_rate", "layer_breakdown",
]


@dataclass
class ValidationResult:
    file: str
    ok: bool
    missing: List[str] = field(default_factory=list)
    note: str = ""


def _check_csv_columns(path: str, required: List[str]) -> ValidationResult:
    if not os.path.exists(path):
        return ValidationResult(file=path, ok=False, note="file not found")
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as e:
        return ValidationResult(file=path, ok=False, note=f"could not read: {e}")
    missing = [c for c in required if c not in columns]
    return ValidationResult(file=path, ok=not missing, missing=missing)


def _check_geojson_properties(path: str, required: List[str]) -> ValidationResult:
    if not os.path.exists(path):
        return ValidationResult(file=path, ok=False, note="file not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ValidationResult(file=path, ok=False, note=f"could not read: {e}")
    features = data.get("features", [])
    if not features:
        return ValidationResult(file=path, ok=True, note="no features to validate (empty output)")
    properties = set(features[0].get("properties", {}).keys())
    missing = [c for c in required if c not in properties]
    return ValidationResult(file=path, ok=not missing, missing=missing)


def _check_json_keys(path: str, required: List[str]) -> ValidationResult:
    if not os.path.exists(path):
        return ValidationResult(file=path, ok=False, note="file not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return ValidationResult(file=path, ok=False, note=f"could not read: {e}")
    missing = [c for c in required if c not in data]
    return ValidationResult(file=path, ok=not missing, missing=missing)


def validate_city_outputs(output_dir: str, city: str) -> Dict[str, ValidationResult]:
    """Validate every documented output file for one city's refresh.

    Returns a dict keyed by logical file name. Use `all_ok()` for a single
    pass/fail, or inspect `.missing`/`.note` per file for diagnostics.
    """
    return {
        "match_register": _check_csv_columns(
            os.path.join(output_dir, f"{city}_Match_Register.csv"),
            MATCH_REGISTER_REQUIRED_COLUMNS,
        ),
        "defaulters_csv": _check_csv_columns(
            os.path.join(output_dir, f"{city}_Defaulters.csv"),
            DEFAULTERS_CSV_REQUIRED_COLUMNS,
        ),
        "defaulters_geojson": _check_geojson_properties(
            os.path.join(output_dir, f"{city}_Defaulters.geojson"),
            DEFAULTERS_GEOJSON_REQUIRED_PROPERTIES,
        ),
        "summary_json": _check_json_keys(
            os.path.join(output_dir, f"{city}_summary.json"),
            SUMMARY_JSON_REQUIRED_KEYS,
        ),
    }


def all_ok(results: Dict[str, ValidationResult]) -> bool:
    return all(r.ok for r in results.values())


def format_report(results: Dict[str, ValidationResult]) -> str:
    lines = ["Field-list conformance check (vs. Data Dictionary):"]
    for name, r in results.items():
        if r.ok:
            lines.append(f"  [OK]   {name}" + (f" ({r.note})" if r.note else ""))
        else:
            reason = r.note or f"missing columns: {', '.join(r.missing)}"
            lines.append(f"  [FAIL] {name} \u2014 {reason}")
    return "\n".join(lines)
