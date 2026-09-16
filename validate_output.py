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
import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .contracts import GEO_STATUSES, SCHEMA_VERSION, TAX_STATUSES

# Data Dictionary §3.2 — Match Register CSV (columns beyond the pass-through
# original mSeva fields).
#
# `property_uid` and `month` are the cross-month join-key contract columns
# (property_uid added in orchestrator._step_report; month is stamped by
# standardize.standardize_csv during publish).
MATCH_REGISTER_REQUIRED_COLUMNS = [
    "propertyid", "matched_uid", "match_method",
    "gis_owner_name", "gis_mobile", "gis_locality",
    "property_uid", "month", "run_id", "schema_version",
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
    "run_id", "schema_version",
]

# Data Dictionary §3.4 — Defaulters GeoJSON feature properties (unified
# GIS-parcel register). No `month` here: GeoJSON files are not run through
# standardize_csv (CSV-only), so month is a CSV-only contract column.
DEFAULTERS_GEOJSON_REQUIRED_PROPERTIES = [
    "gis_uid", "gis_owner_name", "gis_guardian_name", "gis_mobile", "gis_locality",
    "property_usage", "property_type", "status",
    "electricity_account_no", "electricity_holder_name",
    "property_uid", "tax_status", "geo_status", "ward_id", "month",
    "run_id", "schema_version",
]

# Data Dictionary §3.5 — Summary JSON.
SUMMARY_JSON_REQUIRED_KEYS = [
    "city", "total_mseva", "total_gis", "matched_count", "unmatched_count",
    "match_rate", "layer_breakdown", "emitted_gis_rows", "potential_defaulters",
    "tax_status_counts", "month", "run_id", "schema_version",
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
    missing = sorted(
        {
            column
            for feature in features
            for column in required
            if column not in feature.get("properties", {})
        }
    )
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


def _check_reconciliation(output_dir: str, city: str) -> ValidationResult:
    """Verify counts, IDs and enums across all published contract files."""
    label = os.path.join(output_dir, f"{city} output bundle")
    try:
        parcels = pd.read_csv(
            os.path.join(output_dir, f"{city}_Defaulters.csv"),
            dtype=str,
            keep_default_na=False,
        )
        matches = pd.read_csv(
            os.path.join(output_dir, f"{city}_Match_Register.csv"),
            dtype=str,
            keep_default_na=False,
        )
        with open(
            os.path.join(output_dir, f"{city}_Defaulters.geojson"),
            "r",
            encoding="utf-8",
        ) as f:
            geojson = json.load(f)
        with open(
            os.path.join(output_dir, f"{city}_summary.json"),
            "r",
            encoding="utf-8",
        ) as f:
            summary = json.load(f)
    except Exception as e:
        return ValidationResult(file=label, ok=False, note=f"could not reconcile: {e}")

    problems = []
    features = geojson.get("features", [])
    expected_gis = int(summary.get("total_gis", -1))
    if len(parcels) != expected_gis:
        problems.append(f"parcel CSV rows {len(parcels)} != total_gis {expected_gis}")
    if len(features) != expected_gis:
        problems.append(f"GeoJSON features {len(features)} != total_gis {expected_gis}")
    if int(summary.get("emitted_gis_rows", -1)) != expected_gis:
        problems.append("summary emitted_gis_rows does not equal total_gis")
    if len(matches) != int(summary.get("total_mseva", -1)):
        problems.append("match-register rows do not equal total_mseva")

    gis_uids = parcels.get("gis_uid", pd.Series(dtype=str)).astype(str).str.strip()
    property_uids = parcels.get("property_uid", pd.Series(dtype=str)).astype(str).str.strip()
    if gis_uids.eq("").any():
        problems.append("parcel CSV contains blank gis_uid")
    if property_uids.eq("").any() or property_uids.duplicated().any():
        problems.append("parcel property_uid must be nonblank and unique")
    if len(parcels) and not (property_uids == "GIS:" + gis_uids).all():
        problems.append("parcel property_uid is not GIS:{gis_uid}")

    pt_ids = matches.get("propertyid", pd.Series(dtype=str)).astype(str).str.strip()
    pt_uids = matches.get("property_uid", pd.Series(dtype=str)).astype(str).str.strip()
    if pt_ids.eq("").any() or pt_uids.eq("").any():
        problems.append("match register contains blank property identity")
    if len(matches) and not (pt_uids == "PT:" + pt_ids).all():
        problems.append("match property_uid is not PT:{propertyid}")

    tax_status = parcels.get("tax_status", pd.Series(dtype=str)).astype(str)
    geo_status = parcels.get("geo_status", pd.Series(dtype=str)).astype(str)
    invalid_tax = sorted(set(tax_status) - TAX_STATUSES)
    invalid_geo = sorted(set(geo_status) - GEO_STATUSES)
    if invalid_tax:
        problems.append(f"invalid tax_status values: {invalid_tax}")
    if invalid_geo:
        problems.append(f"invalid geo_status values: {invalid_geo}")

    status_counts = {str(k): int(v) for k, v in tax_status.value_counts().items()}
    if sum(status_counts.values()) != expected_gis:
        problems.append("tax-status totals do not equal total_gis")
    if status_counts.get("SUSPECTED", 0) != int(
        summary.get("potential_defaulters", -1)
    ):
        problems.append("emitted SUSPECTED count differs from summary")
    if status_counts != summary.get("tax_status_counts", {}):
        problems.append("summary tax_status_counts differ from parcel CSV")

    months = set(parcels.get("month", pd.Series(dtype=str)).astype(str))
    run_ids = set(parcels.get("run_id", pd.Series(dtype=str)).astype(str))
    versions = set(
        parcels.get("schema_version", pd.Series(dtype=str)).astype(str)
    )
    if len(months) != 1 or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", next(iter(months), "")):
        problems.append("parcel month is not one canonical YYYY-MM value")
    if months != {str(summary.get("month", ""))}:
        problems.append("parcel and summary months differ")
    if run_ids != {str(summary.get("run_id", ""))} or "" in run_ids:
        problems.append("parcel and summary run_ids differ or are blank")
    if versions != {SCHEMA_VERSION} or summary.get("schema_version") != SCHEMA_VERSION:
        problems.append("unexpected schema_version")

    feature_props = [feature.get("properties", {}) for feature in features]
    if geojson.get("schema_version") != SCHEMA_VERSION:
        problems.append("GeoJSON has unexpected top-level schema_version")
    if geojson.get("run_id") != summary.get("run_id"):
        problems.append("GeoJSON and summary run_ids differ")
    if geojson.get("month") != summary.get("month"):
        problems.append("GeoJSON and summary months differ")
    if feature_props:
        geo_ids = [str(props.get("property_uid", "")) for props in feature_props]
        if geo_ids != property_uids.tolist():
            problems.append("GeoJSON and parcel CSV property order/IDs differ")
        geo_tax_counts = pd.Series(
            [props.get("tax_status", "") for props in feature_props]
        ).value_counts().to_dict()
        if geo_tax_counts != status_counts:
            problems.append("GeoJSON and parcel CSV tax-status counts differ")
        geo_geo_statuses = [
            str(props.get("geo_status", "")) for props in feature_props
        ]
        if geo_geo_statuses != geo_status.tolist():
            problems.append("GeoJSON and parcel CSV geo_status values differ")
        for feature, status_value in zip(features, geo_geo_statuses):
            geometry = feature.get("geometry")
            if status_value == "NONE" and geometry is not None:
                problems.append("geo_status NONE has a GeoJSON geometry")
                break
            if status_value in {"SHAPE", "DOT"} and geometry is None:
                problems.append(f"geo_status {status_value} has no GeoJSON geometry")
                break

    return ValidationResult(file=label, ok=not problems, note="; ".join(problems))


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
        "reconciliation": _check_reconciliation(output_dir, city),
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
