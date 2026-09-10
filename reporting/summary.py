"""
summary.py — Result formatting and stats computation.
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from ..config import CityConfig, MatchResult

logger = logging.getLogger(__name__)


def build_match_register(
    mseva_df: pd.DataFrame,
    match_results: Dict[str, MatchResult],
    config: CityConfig,
    mseva_columns: dict = None,
) -> pd.DataFrame:
    """Build the final match register DataFrame.

    Combines the original mSeva fields with the matched GIS information.
    For unmatched records, fills in appropriate status reason tags.
    """
    if mseva_columns is None:
        mseva_columns = config.mseva_columns or {}

    pid_col = mseva_columns.get("propertyid", "propertyid")
    old_pid_col = mseva_columns.get("old_propertyid", "oldpropertyid")
    owner_col = mseva_columns.get("owner", "ownername")
    guard_col = mseva_columns.get("guardian", "guardianname")
    mob_col = mseva_columns.get("mobile", "mobileno")
    loc_col = mseva_columns.get("locality", "localityname")
    addr_col = mseva_columns.get("address", "address")
    lat_col = mseva_columns.get("latitude", "latitude")
    lon_col = mseva_columns.get("longitude", "longitude")
    acc_col = mseva_columns.get("geocode_accuracy", "geocode_accuracy")

    has_coords = lat_col in mseva_df.columns and lon_col in mseva_df.columns

    # Detect city center coords to tag correctly
    cc_mask = pd.Series(False, index=mseva_df.index)
    missing_mask = pd.Series(False, index=mseva_df.index)
    if has_coords:
        if acc_col in mseva_df.columns:
            approx = mseva_df[mseva_df[acc_col] == "APPROXIMATE"]
            cc_coords = approx.groupby([lat_col, lon_col]).size().reset_index(name="c")
            cc_coords = cc_coords[cc_coords["c"] >= 100][[lat_col, lon_col]].values.tolist()
        else:
            cc_coords = []

        for clat, clon in cc_coords:
            cc_mask |= (
                (mseva_df[lat_col].round(4) == round(clat, 4))
                & (mseva_df[lon_col].round(4) == round(clon, 4))
            )
        missing_mask = mseva_df[lat_col].isna() | mseva_df[lon_col].isna()

    rows = []
    for idx in mseva_df.index:
        row = mseva_df.loc[idx]
        pid = str(row[pid_col])
        base = {
            "propertyid": pid,
            "oldpropertyid": row.get(old_pid_col, ""),
            "ownername": row.get(owner_col, ""),
            "guardianname": row.get(guard_col, ""),
            "mobileno": row.get(mob_col, ""),
            "localityname": row.get(loc_col, ""),
            "address": row.get(addr_col, ""),
            "latitude": row.get(lat_col, ""),
            "longitude": row.get(lon_col, ""),
            "geocode_accuracy": row.get(acc_col, ""),
            "google_formatted_address": row.get("google_formatted_address", ""),
        }
        
        # Include parsed fields if they were generated
        for f in ['parsed_house_no', 'parsed_building', 'parsed_street', 'parsed_locality', 'parsed_ward', 'completeness_score', 'confidence_tier']:
            if f in row.index:
                base[f] = row[f]

        if pid in match_results:
            mr = match_results[pid]
            base.update({
                "matched_uid": mr.matched_uid,
                "match_method": mr.match_method,
                "gis_owner_name": mr.gis_owner_name,
                "gis_mobile": mr.gis_mobile,
                "gis_locality": mr.gis_locality,
            })
        else:
            if cc_mask.loc[idx]:
                reason = "UNMATCHED_CITY_CENTER"
            elif missing_mask.loc[idx]:
                reason = "UNMATCHED_NO_COORDS"
            else:
                reason = "UNMATCHED"
            base.update({
                "matched_uid": "",
                "match_method": reason,
                "gis_owner_name": "",
                "gis_mobile": "",
                "gis_locality": "",
            })
        rows.append(base)

    return pd.DataFrame(rows)


def compute_summary(
    register: pd.DataFrame,
    gis_records: list,
    config: CityConfig,
    gis_columns: dict = None,
) -> dict:
    """Compute matches, unmatched, layers breakdown and defaulters statistics."""
    if gis_columns is None:
        gis_columns = config.gis_columns or {}

    uid_field = gis_columns.get("uid", "UID")

    has_uid = register["matched_uid"].astype(str).str.strip() != ""
    mc = register["match_method"].value_counts()

    matched_uids = set(register.loc[has_uid, "matched_uid"].unique())
    all_gis_uids = {r[uid_field] for r in gis_records if r.get(uid_field)}

    layer_breakdown = {method: int(count) for method, count in mc.items() if not method.startswith("UNMATCHED")}

    # Aggregate parser quality (confidence_tier)
    parser_quality = {}
    if "confidence_tier" in register.columns:
        parser_quality = {k: int(v) for k, v in register["confidence_tier"].value_counts().items()}

    # Aggregate geocoding quality (geocode_accuracy)
    geocode_quality = {}
    if "geocode_accuracy" in register.columns:
        # Filter out empty strings or nan
        valid_geocode = register[register["geocode_accuracy"].astype(str).str.strip() != ""]
        geocode_quality = {k: int(v) for k, v in valid_geocode["geocode_accuracy"].value_counts().items()}

    # Separate taxable vs exempted GIS polygons
    exempted_col = gis_columns.get("exempted")
    taxable_uids = set()
    exempted_uids = set()
    for r in gis_records:
        uid = r.get(uid_field)
        if not uid:
            continue
        exempt_val = str(r.get(exempted_col, "")).strip().lower() if exempted_col else ""
        if exempt_val == "exempted":
            exempted_uids.add(uid)
        else:
            taxable_uids.add(uid)

    taxable_defaulters = taxable_uids - matched_uids

    summary = {
        "city": config.name,
        "total_mseva": len(register),
        "total_gis": len(gis_records),
        "total_gis_taxable": len(taxable_uids),
        "total_gis_exempted": len(exempted_uids),
        "matched_count": int(has_uid.sum()),
        "unmatched_count": int((~has_uid).sum()),
        "match_rate": float(has_uid.sum() / len(register) * 100),
        "unique_gis_matched": len(matched_uids),
        "potential_defaulters": len(taxable_defaulters),
        "layer_breakdown": layer_breakdown,
        "parser_quality": parser_quality,
        "geocode_quality": geocode_quality,
    }

    # Add raw counts for unmatched categories
    for unmatched_cat in ["UNMATCHED", "UNMATCHED_CITY_CENTER", "UNMATCHED_NO_COORDS"]:
        summary[unmatched_cat.lower()] = int(mc.get(unmatched_cat, 0))

    return summary


def print_summary(summary: dict):
    """Print a formatted CLI dashboard representation of the results."""
    print("\n" + "=" * 70)
    print(f"MATCHING SUMMARY — {summary['city']}")
    print("=" * 70)
    print(f"Total mSeva records:   {summary['total_mseva']:,}")
    print(f"Total GIS polygons:    {summary['total_gis']:,}")
    if 'total_gis_taxable' in summary:
        print(f"  ├─ Taxable:          {summary['total_gis_taxable']:,}")
        print(f"  └─ Exempted:         {summary['total_gis_exempted']:,}")
    print(f"Matched records:       {summary['matched_count']:,} ({summary['match_rate']:.1f}%)")
    print(f"Unmatched records:     {summary['unmatched_count']:,} ({100 - summary['match_rate']:.1f}%)")
    print(f"Unique GIS matched:    {summary['unique_gis_matched']:,}")
    print(f"Potential defaulters:  {summary['potential_defaulters']:,} (taxable only)")
    print("\nLayer Breakdown:")
    for method, count in sorted(summary["layer_breakdown"].items()):
        pct = count / summary["total_mseva"] * 100
        print(f"  {method:30s}: {count:>6,} ({pct:>5.1f}%)")
        
    print("=" * 70)
