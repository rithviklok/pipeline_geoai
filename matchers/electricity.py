"""
Electricity matching layer.

Checks if an active electricity connection (meter GPS point) falls inside
an unmatched GIS polygon.  If so, confirms that building as a tax defaulter
(occupied but unpaid).
"""

from __future__ import annotations

import logging
from typing import Dict, Set

import pandas as pd
from shapely.geometry import Point

from ..config import CityConfig
from ..helpers import wgs84_to_utm

logger = logging.getLogger(__name__)


def confirm_defaulters(
    electricity_df: pd.DataFrame,
    gis_records: list,
    polygons: list,
    tree,
    config: CityConfig,
    matched_uids: Set[str],
    elec_columns: dict = None,
    gis_columns: dict = None,
) -> Dict[str, dict]:
    """Identify occupied properties (active electricity meters) that have no mSeva tax record.

    Parameters
    ----------
    electricity_df : pd.DataFrame
        Active electricity meter records with coordinates.
    gis_records : list[dict]
        GIS attribute dicts (one per polygon).
    polygons : list[shapely.geometry.BaseGeometry]
        GIS polygon geometries.
    tree : shapely.strtree.STRtree
        GIS spatial index.
    config : CityConfig
        City-specific configuration.
    matched_uids : Set[str]
        Set of GIS UIDs that already have matching mSeva records.
    elec_columns : dict, optional
        Column mappings for electricity DataFrame.
    gis_columns : dict, optional
        Column mappings for GIS records.

    Returns
    -------
    dict[str, dict]
        Mapping of ``{gis_uid: electricity_record_details}`` representing
        confirmed defaulters.
    """
    if elec_columns is None:
        elec_columns = config.elec_columns or {}
    if gis_columns is None:
        gis_columns = config.gis_columns or {}

    lat_col = elec_columns.get("latitude", "latitude")
    lon_col = elec_columns.get("longitude", "longitude")
    # account_no is often not auto-detected; fall back to scanning DataFrame columns
    acc_col = elec_columns.get("account_no")
    if not acc_col:
        acc_col = next(
            (c for c in electricity_df.columns if "account" in c.lower()), None
        )
    # Name may be mapped as 'owner' or 'name' depending on the detector
    name_col = elec_columns.get("owner") or elec_columns.get("name", "name")

    uid_field = gis_columns.get("uid", "UID")

    confirmed_defaulters: Dict[str, dict] = {}

    logger.info("Scanning %d electricity records for defaulters...", len(electricity_df))

    for idx in electricity_df.index:
        row = electricity_df.loc[idx]
        lat = row.get(lat_col)
        lon = row.get(lon_col)

        if pd.isna(lat) or pd.isna(lon):
            continue

        try:
            e, n = wgs84_to_utm(float(lat), float(lon), config.utm_zone)
        except Exception:
            continue

        pt = Point(e, n)
        cands = tree.query(pt)
        containing = [
            ci for ci in cands
            if polygons[ci].contains(pt) or polygons[ci].touches(pt)
        ]

        if not containing:
            continue

        # Find smallest containing polygon
        best_ci = min(containing, key=lambda ci: polygons[ci].area)
        ga = gis_records[best_ci]
        uid = ga.get(uid_field)

        if not uid or uid in matched_uids:
            continue

        # Confirmed defaulter!
        confirmed_defaulters[uid] = {
            "account_no": row.get(acc_col, ""),
            "holder_name": row.get(name_col, ""),
            "latitude": lat,
            "longitude": lon,
            "gis_uid": uid,
            "gis_owner": ga.get(gis_columns.get("owner", "Owner_Name"), ""),
        }

    logger.info(
        "Electricity confirmation - found %d confirmed occupied defaulters",
        len(confirmed_defaulters),
    )

    return confirmed_defaulters


def match_electricity_taxpayers(
    mseva_df: pd.DataFrame,
    gis_records: list,
    polygons: list,
    tree,
    electricity_df: pd.DataFrame,
    config: CityConfig,
    existing_matches: dict,
) -> Dict[str, MatchResult]:
    """Layer 5: Electricity-based taxpayer matching.

    Matches unmatched mSeva records against electricity account holders,
    then performs spatial checks of electricity meter GPS on GIS polygons.
    """
    from collections import defaultdict
    from ..config import MatchResult
    from ..helpers import normalize_name, token_set_ratio, wgs84_to_utm

    mcols = config.mseva_columns
    gcols = config.gis_columns
    elec_cols = config.elec_columns or {}

    pid_col = mcols.get("propertyid", "propertyid")
    uid_field = gcols.get("uid", "UID")

    elec_lat_col = elec_cols.get("latitude", "latitude")
    elec_lon_col = elec_cols.get("longitude", "longitude")
    elec_name_col = elec_cols.get("owner") or elec_cols.get("name") or "name"
    elec_acc_col = elec_cols.get("account_no") or next(
        (c for c in electricity_df.columns if "account" in c.lower()), "account_no"
    )

    # Filter electricity records to the bounding box of mSeva geocoded points
    # (only if lat/lon columns exist — they won't in GeoAI mode where mSeva is raw)
    lat_col = mcols.get("latitude", "latitude")
    lon_col = mcols.get("longitude", "longitude")
    if lat_col in mseva_df.columns and lon_col in mseva_df.columns:
        valid_coords = mseva_df[[lat_col, lon_col]].dropna()
        if not valid_coords.empty:
            lat_min = valid_coords.iloc[:, 0].min() - 0.05
            lat_max = valid_coords.iloc[:, 0].max() + 0.05
            lon_min = valid_coords.iloc[:, 1].min() - 0.05
            lon_max = valid_coords.iloc[:, 1].max() + 0.05

            elec_filtered = electricity_df[
                (electricity_df[elec_lat_col] >= lat_min) & (electricity_df[elec_lat_col] <= lat_max) &
                (electricity_df[elec_lon_col] >= lon_min) & (electricity_df[elec_lon_col] <= lon_max)
            ].copy()
            logger.info(
                "Filtered electricity records from %d to %d using mSeva coordinates bounding box",
                len(electricity_df), len(elec_filtered)
            )
        else:
            elec_filtered = electricity_df.copy()
    else:
        # No geocoded coordinates in mSeva — use all electricity records
        elec_filtered = electricity_df.copy()
        logger.info("No lat/lon in mSeva — using all %d electricity records (no bounding box filter)", len(elec_filtered))

    # Parse electricity names
    def parse_elec_name(name):
        if not name or pd.isna(name):
            return "", ""
        name_str = str(name)
        for sep in [" S/O ", " s/o ", " W/O ", " w/o ", " D/O ", " d/o ", " C/O ", " c/o "]:
            if sep in name_str:
                parts = name_str.split(sep, 1)
                return normalize_name(parts[0]), normalize_name(parts[1])
        return normalize_name(name_str), ""

    elec_name_idx = defaultdict(list)
    parsed_elec = []  # list of (idx, owner_norm, father_norm)
    for idx in elec_filtered.index:
        row = elec_filtered.loc[idx]
        o, f = parse_elec_name(row.get(elec_name_col, ""))
        parsed_elec.append((idx, o, f))
        if o:
            words = o.split()
            first_word = words[0] if words else ""
            if first_word and len(first_word) >= 3:
                elec_name_idx[first_word].append(len(parsed_elec) - 1)

    results: Dict[str, MatchResult] = {}
    mseva_owner_col = mcols.get("owner", "ownername")
    mseva_guardian_col = mcols.get("guardian", "guardianname")

    for idx in mseva_df.index:
        row = mseva_df.loc[idx]
        pid = str(row[pid_col])
        if pid in existing_matches or pid in results:
            continue

        mo = normalize_name(row.get(mseva_owner_col, ""))
        mg = normalize_name(row.get(mseva_guardian_col, ""))
        if not mo and not mg:
            continue

        # Get candidate electricity records by first-word blocking
        cands = set()
        for name in [mo, mg]:
            if name:
                words = name.split()
                first = words[0] if words else ""
                if first and len(first) >= 3 and first in elec_name_idx:
                    cands.update(elec_name_idx[first])

        if not cands:
            continue

        best_pe_idx = None
        best_score = 0
        best_method = ""

        for pe_idx in cands:
            _, eo, ef = parsed_elec[pe_idx]

            o_score = token_set_ratio(mo, eo)
            g_score = token_set_ratio(mg, ef) if ef else 0

            if o_score >= 60 and g_score >= 60:
                combined = o_score + g_score
                if combined > best_score:
                    best_score = combined
                    best_pe_idx = pe_idx
                    best_method = "ELEC_BOTH"
            elif o_score >= 80 and g_score >= 30:
                combined = o_score + g_score
                if combined > best_score:
                    best_score = combined
                    best_pe_idx = pe_idx
                    best_method = "ELEC_OWNER+GUARD"
            elif o_score >= 90:
                if o_score > best_score:
                    best_score = o_score
                    best_pe_idx = pe_idx
                    best_method = "ELEC_OWNER_STRONG"

        if best_pe_idx is not None:
            orig_idx, eo, ef = parsed_elec[best_pe_idx]
            elec_row = elec_filtered.loc[orig_idx]
            lat = elec_row.get(elec_lat_col)
            lon = elec_row.get(elec_lon_col)
            if pd.isna(lat) or pd.isna(lon):
                continue

            try:
                e, n = wgs84_to_utm(float(lat), float(lon), config.utm_zone)
            except Exception:
                continue

            pt = Point(e, n)
            cands_pip = tree.query(pt)
            containing = [
                ci for ci in cands_pip
                if polygons[ci].contains(pt) or polygons[ci].touches(pt)
            ]

            if containing:
                best_ci = min(containing, key=lambda ci: polygons[ci].area)
                ga = gis_records[best_ci]
                results[pid] = MatchResult(
                    matched_uid=ga[uid_field],
                    match_method=f"ELEC_PIP_{best_method}",
                    gis_owner_name=ga.get(gcols.get("owner", "Owner_Name"), ""),
                    gis_mobile=ga.get(gcols.get("mobile", "Mobile_No"), ""),
                    gis_locality=ga.get(gcols.get("locality", "Locality"), ""),
                )
            else:
                buf = pt.buffer(30)
                cands_buf = [ci for ci in tree.query(buf) if polygons[ci].intersects(buf)]
                if cands_buf:
                    best_ci = max(cands_buf, key=lambda ci: polygons[ci].intersection(buf).area)
                    ga = gis_records[best_ci]

                    raw_name = str(elec_row.get(elec_name_col, ""))
                    eo_clean = normalize_name(raw_name.split(' S/O ')[0] if ' S/O ' in raw_name else raw_name)
                    go = normalize_name(ga.get(gcols.get("owner", "Owner_Name"), ""))
                    name_score = token_set_ratio(eo_clean, go)

                    if name_score >= 40:
                        results[pid] = MatchResult(
                            matched_uid=ga[uid_field],
                            match_method=f"ELEC_BUFFER_{best_method}",
                            gis_owner_name=ga.get(gcols.get("owner", "Owner_Name"), ""),
                            gis_mobile=ga.get(gcols.get("mobile", "Mobile_No"), ""),
                            gis_locality=ga.get(gcols.get("locality", "Locality"), ""),
                        )

    logger.info("Electricity matching complete: found %d matches", len(results))
    return results
