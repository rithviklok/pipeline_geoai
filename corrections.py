"""
corrections.py — Data contract for the field-check feedback loop
(product-doc tickets 3.13-3.15).

Design only, not fully wired: the phone-app/review backend that produces
approved field checks does not exist yet. This module defines the shape a
future refresh will consume, plus a stub `apply_corrections()`, so the
contract can be reviewed and agreed on paper before the rest of the loop
(review UI, approval workflow, storage) is built.

Hard rules this contract must uphold (per the product doc's 7 rules that
apply to every screen and every piece of data):
  - Corrections are ADDITIVE. The original mSeva/GIS record is never
    mutated or deleted — a correction is a new, separately-stored fact
    that a later step may choose to apply on top of the raw record.
  - Every correction is traceable to the specific field check that
    produced it, the officer, the approving supervisor, and the refresh
    run (see manifest.py's run_id) that first applied it.
  - A correction that was never approved (see `status`) must have zero
    effect on any refresh output (ticket 3.15).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FieldCheckCorrection:
    """One approved field-officer correction to a single property record.

    `gis_uid` or `mseva_property_id` (at least one) identifies the record
    this correction applies to. `field_name` is the canonical Data
    Dictionary key being corrected (e.g. "owner", "property_usage").
    """
    correction_id: str
    field_name: str
    corrected_value: Any
    status: str  # "APPROVED" | "REJECTED" | "PENDING" — only APPROVED is ever applied
    checked_by_officer: str
    checked_at: str  # ISO 8601
    source_check_id: str  # ID of the field-check submission this came from
    gis_uid: Optional[str] = None
    mseva_property_id: Optional[str] = None
    approved_by_supervisor: Optional[str] = None
    approved_at: Optional[str] = None
    applied_in_run_id: Optional[str] = None  # set once a refresh consumes it
    notes: str = ""


def load_corrections(path: Optional[str]) -> List[FieldCheckCorrection]:
    """Load approved corrections from a JSON file (a list of objects matching
    FieldCheckCorrection's fields). This is a placeholder for a future
    API/DB-backed source once the review backend exists — the on-disk JSON
    format is only meant for local testing of the contract.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [FieldCheckCorrection(**r) for r in raw]


def apply_corrections(
    gis_records: List[Dict[str, Any]],
    corrections: List[FieldCheckCorrection],
    uid_field: str,
    run_id: str,
) -> Dict[str, int]:
    """Preview what applying a set of corrections would do, without
    mutating `gis_records` in place.

    This is intentionally a stub: it validates and counts what *would* be
    applied so callers can log/report on the feedback loop before the full
    merge-and-persist behaviour (an additive corrections store, joined onto
    the raw GIS/mSeva records at query time) is built once the review
    backend exists. `run_id` is accepted now so the eventual implementation
    can stamp `applied_in_run_id` without changing this function's signature.

    Returns counts by outcome, suitable for the refresh's plain-language log.
    """
    counts = {"applied": 0, "skipped_not_approved": 0, "skipped_no_match": 0}
    uid_index = {r.get(uid_field): r for r in gis_records if r.get(uid_field)}

    for c in corrections:
        if c.status != "APPROVED":
            counts["skipped_not_approved"] += 1
            continue
        if c.gis_uid not in uid_index:
            counts["skipped_no_match"] += 1
            continue
        # TODO(full wiring): once the review backend exists, persist this as
        # an additive correction record keyed by (gis_uid, field_name) with
        # its own history, never overwriting uid_index[...] in place, and
        # mark c.applied_in_run_id = run_id on first application.
        counts["applied"] += 1

    return counts
