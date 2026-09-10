"""
Layer 2: Mobile number matching (global, text-based).

Reproduces the mobile-matching logic from Barnala full_match_pipeline.py
(lines 291-337).

Algorithm:
  1. Build a mobile → [polygon-index] inverted index from GIS
     (skipping already-matched UIDs).
  2. For each unmatched mSeva record, look up its mobile(s) in the index.
  3. If exactly one candidate polygon → TEXT_MOBILE.
  4. If multiple candidates → disambiguate by name scoring:
     - score ≥ 40 → TEXT_MOBILE+NAME
     - score < 40  → TEXT_MOBILE_WEAK
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Set

import pandas as pd

from ..config import CityConfig, MatchResult
from ..helpers import (
    best_name_score,
    normalize_mobile,
    normalize_name,
)

logger = logging.getLogger(__name__)


def match_mobile(
    mseva_df: pd.DataFrame,
    gis_records: list,
    config: CityConfig,
    existing_matches: dict,
) -> Dict[str, MatchResult]:
    """Layer 2: Mobile number matching.

    Parameters
    ----------
    mseva_df : pd.DataFrame
        Enriched mSeva records.
    gis_records : list[dict]
        GIS attribute dicts (one per polygon).
    config : CityConfig
        City-specific configuration (column names, etc.).
    existing_matches : dict
        Already-matched ``{propertyid: MatchResult}`` — skipped here.

    Returns
    -------
    dict[str, MatchResult]
        New matches found by this layer.
    """
    mcols = config.mseva_columns
    gcols = config.gis_columns
    pid_col = mcols.get("propertyid", "propertyid")
    uid_field = gcols.get("uid", "UID")

    # ── Junk mobile number filtering ──────────────────────────────────
    max_freq = getattr(config, 'mobile_max_frequency', 5)

    # First pass: count frequency of each normalised mobile across GIS
    mob_field1 = gcols.get("mobile", "Mobile_No")
    mob_field2 = gcols.get("mobile2", "Mobile_No_")
    freq_counter: Dict[str, int] = defaultdict(int)
    for r in gis_records:
        for m in (str(r.get(mob_field1, "")) + "|" + str(r.get(mob_field2, ""))).split("|"):
            mn = normalize_mobile(m)
            if mn:
                freq_counter[mn] += 1

    def _is_junk_mobile(number: str) -> bool:
        """Filter out junk mobile numbers."""
        if not number:
            return True
        # Pattern blocklist: all-same-digit (e.g. 9999999999)
        if len(set(number)) <= 2:
            return True
        # Common junk patterns
        if number in ('1234567890', '0123456789', '9876543210'):
            return True
        # Invalid Indian mobile format (must be 10 digits starting with 6-9)
        import re
        if not re.match(r'^[6-9]\d{9}$', number):
            return True
        # Frequency blocklist: appears too many times → almost certainly junk
        if freq_counter.get(number, 0) > max_freq:
            return True
        return False

    junk_blocked = 0

    # ── Build mobile → polygon-index map ────────────────────────────────
    mob_idx: Dict[str, list] = defaultdict(list)
    for i, r in enumerate(gis_records):
        for m in (str(r.get(mob_field1, "")) + "|" + str(r.get(mob_field2, ""))).split("|"):
            mn = normalize_mobile(m)
            if mn:
                if _is_junk_mobile(mn):
                    junk_blocked += 1
                    continue
                mob_idx[mn].append(i)

    logger.info("Mobile index: %d unique numbers (%d junk entries blocked, max_freq=%d)",
                len(mob_idx), junk_blocked, max_freq)

    # ── Scan unmatched mSeva records ────────────────────────────────────
    results: Dict[str, MatchResult] = {}
    mob_hits = 0

    mseva_mob_col = mcols.get("mobile", "mobileno")
    mseva_all_mob_col = mcols.get("all_mobiles", "all_mobilenos")
    mseva_owner_col = mcols.get("owner", "ownername")
    mseva_guardian_col = mcols.get("guardian", "guardianname")
    mseva_all_owners_col = mcols.get("all_owners", "all_ownernames")
    mseva_all_guardians_col = mcols.get("all_guardians", "all_guardiannames")

    gis_owner_field = gcols.get("owner", "Owner_Name")
    gis_guardian_field = gcols.get("guardian", "Father_Hus")
    gis_locality_field = gcols.get("locality", "Locality")

    for idx, row in mseva_df.iterrows():
        pid = str(row.get(pid_col, ""))
        if not pid or pid in existing_matches or pid in results:
            continue

        # Collect mSeva mobiles
        mobs: Set[str] = set()
        for field in [mseva_mob_col, mseva_all_mob_col]:
            raw = str(row.get(field, "")) if field in mseva_df.columns else ""
            for part in raw.split("|"):
                mn = normalize_mobile(part)
                if mn:
                    mobs.add(mn)
        if not mobs:
            continue

        # Look up candidates
        cands: Set[int] = set()
        for m in mobs:
            if m in mob_idx:
                cands.update(mob_idx[m])
        if not cands:
            continue

        if len(cands) == 1:
            # Unique mobile → direct match
            gi = list(cands)[0]
            results[pid] = MatchResult(
                matched_uid=gis_records[gi][uid_field],
                match_method="TEXT_MOBILE",
                gis_owner_name=gis_records[gi].get(gis_owner_field, ""),
                gis_mobile=gis_records[gi].get(gcols.get("mobile", "Mobile_No"), ""),
                gis_locality=gis_records[gi].get(gis_locality_field, ""),
            )
            mob_hits += 1
        else:
            # Multiple candidates → disambiguate by name
            mo = normalize_name(
                mseva_df.at[idx, mseva_owner_col]
                if mseva_owner_col in mseva_df.columns
                else ""
            )
            mg = normalize_name(
                mseva_df.at[idx, mseva_guardian_col]
                if mseva_guardian_col in mseva_df.columns
                else ""
            )
            mao = normalize_name(
                str(
                    mseva_df.at[idx, mseva_all_owners_col]
                    if mseva_all_owners_col in mseva_df.columns
                    else ""
                )
            )
            mag = normalize_name(
                str(
                    mseva_df.at[idx, mseva_all_guardians_col]
                    if mseva_all_guardians_col in mseva_df.columns
                    else ""
                )
            )

            bi = None
            bs = 0
            for gi in cands:
                s, _ = best_name_score(
                    mo,
                    mg,
                    mao,
                    mag,
                    normalize_name(gis_records[gi].get(gis_owner_field, "")),
                    normalize_name(gis_records[gi].get(gis_guardian_field, "")),
                )
                if s > bs:
                    bs, bi = s, gi

            if bi is not None:
                method = "TEXT_MOBILE+NAME" if bs >= 40 else "TEXT_MOBILE_WEAK"
                results[pid] = MatchResult(
                    matched_uid=gis_records[bi][uid_field],
                    match_method=method,
                    gis_owner_name=gis_records[bi].get(gis_owner_field, ""),
                    gis_mobile=gis_records[bi].get(gcols.get("mobile", "Mobile_No"), ""),
                    gis_locality=gis_records[bi].get(gis_locality_field, ""),
                )
                mob_hits += 1

    logger.info("Mobile matches: %d", mob_hits)
    return results
