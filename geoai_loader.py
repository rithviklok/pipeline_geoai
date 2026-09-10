"""
pipeline/geoai_loader.py
════════════════════════════════════════════════════════════════
Load GeoAI geocoded CSV output and convert matched records
into the pipeline's standard ``Dict[str, MatchResult]`` format.

This module is the bridge between GeoAI's inference output
and the existing matcher infrastructure.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from .config import MatchResult

logger = logging.getLogger(__name__)


def has_mobile_coverage(
    mseva_df: pd.DataFrame,
    mseva_columns: dict,
    min_coverage: float = 0.3,
) -> tuple:
    """Check whether enough mSeva records have a mobile number.

    Returns
    -------
    (bool, float)
        Whether coverage exceeds *min_coverage*, and the actual fill rate.
    """
    mob_col = mseva_columns.get("mobileno", "mobileno")
    if mob_col not in mseva_df.columns:
        return False, 0.0
    filled = (
        mseva_df[mob_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    # Filter out common non-mobile values
    filled = filled[~filled.isin(["0", "nan", "None", "null"])]
    rate = len(filled) / len(mseva_df) if len(mseva_df) > 0 else 0.0
    return rate >= min_coverage, rate


def load_geoai_matches(
    geoai_csv_path: str,
    gis_records: list,
    gis_columns: dict,
) -> Dict[str, MatchResult]:
    """Read GeoAI's geocoded CSV and extract match results.

    Only records with a non-empty ``Matched_UID`` are treated as real
    matches.  Records where ``Match_Type == 'LOCALITY_CENTROID'`` are
    excluded because they were not matched to a specific GIS polygon.

    Each match is converted into a :class:`MatchResult` keyed by the
    mSeva ``propertyid``.  The ``match_method`` is prefixed with
    ``GEOAI_`` to distinguish from our pipeline's own matcher labels.

    Parameters
    ----------
    geoai_csv_path : str
        Path to the GeoAI geocoded CSV (output of ``Inference.py``).
    gis_records : list[dict]
        Loaded GIS records (from ``data_loader.load_gis``).
    gis_columns : dict
        Auto-detected GIS column mappings.

    Returns
    -------
    dict[str, MatchResult]
        Matches keyed by mSeva ``propertyid``.
    """
    logger.info("Loading GeoAI geocoded output: %s", geoai_csv_path)
    df = pd.read_csv(geoai_csv_path, low_memory=False)
    logger.info("GeoAI records loaded: %d", len(df))

    # ── Clean Matched_UID ────────────────────────────────────────────
    df["_uid_clean"] = (
        df["Matched_UID"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    )

    # Filter to real matches (non-empty UID, not locality centroid)
    mask = df["_uid_clean"].notna()
    if "Match_Type" in df.columns:
        mask = mask & (df["Match_Type"] != "LOCALITY_CENTROID")

    matched_df = df[mask].copy()
    logger.info(
        "GeoAI matched records: %d / %d (%.1f%%)",
        len(matched_df),
        len(df),
        100.0 * len(matched_df) / len(df) if len(df) else 0,
    )

    # ── Build GIS UID → record lookup for enriching match results ────
    uid_field = gis_columns.get("uid", "UID")
    gis_owner_field = gis_columns.get("owner", "Owner_Name")
    gis_mob_field = gis_columns.get("mobile", "Mobile_No")
    gis_locality_field = gis_columns.get("locality", "Locality")

    gis_by_uid: Dict[str, dict] = {}
    for r in gis_records:
        uid = r.get(uid_field, "")
        if uid:
            gis_by_uid[uid] = r

    # ── Convert each row to a MatchResult ────────────────────────────
    results: Dict[str, MatchResult] = {}
    pid_col = "propertyid"  # GeoAI output always uses this column name

    for _, row in matched_df.iterrows():
        pid = str(row.get(pid_col, "")).strip()
        if not pid or pid == "nan":
            continue

        matched_uid = str(row["_uid_clean"])
        match_type = str(row.get("Match_Type", "UNKNOWN"))
        score = float(row.get("GeoAI_Match_Score", 0))

        # Look up GIS record for this UID to populate fields
        gis_rec = gis_by_uid.get(matched_uid, {})

        results[pid] = MatchResult(
            matched_uid=matched_uid,
            match_method=f"GEOAI_{match_type}",
            gis_owner_name=gis_rec.get(gis_owner_field, ""),
            gis_mobile=gis_rec.get(gis_mob_field, ""),
            gis_locality=gis_rec.get(gis_locality_field, ""),
            confidence=score / 100.0 if score else 0.0,
        )

    logger.info("GeoAI MatchResults created: %d", len(results))
    return results
