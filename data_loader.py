"""
data_loader — Load mSeva CSV, GIS shapefiles, and electricity CSV.

Auto-detects column names from each source via configurable pattern
dictionaries, so the same pipeline works across Batala, Barnala, and
future cities without hard-coded column names.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import shapefile  # pyshp
from shapely.geometry import shape
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# COLUMN NAME AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════

#: Common column-name patterns across mSeva dumps, GIS shapefiles, and
#: electricity CSVs.  For each logical field the list is tried left-to-
#: right; the first match (case-insensitive) wins.
COLUMN_PATTERNS: dict[str, list[str]] = {
    "owner": [
        "ownername", "owner_name", "owner_nam", "Owner_Name",
        "name", "owner",
    ],
    "guardian": [
        "guardianname", "guardian_name", "Father_Hus", "father_hus",
        "father_name", "guardian", "fathername",
    ],
    "mobile": [
        "mobileno", "mobile_no", "Mobile_No", "mobile",
        "phone", "contact",
    ],
    "address": [
        "Proper_Address", "proper_address",
        "formatted_address", "full_address",
        "address", "Address", "PROPERTY ADDRESS",
        "propertyaddress",
    ],
    "locality": [
        "localityname", "locality_name", "Locality", "locality",
        "mohalla", "colony",
    ],
    "ward": [
        "ward", "Ward", "wardname", "ward_no",
        "blockname", "zonename", "ward_number",
    ],
    "property_id": [
        "propertyid", "property_id", "Property_ID", "prop_id",
    ],
    "old_property_id": [
        "oldpropertyid", "old_property_id", "Old_Uid",
        "old_uid", "OLD UID",
    ],
    "uid": ["UID", "uid", "Uid", "UNIQUE_ID"],
    "uid_old": ["uid_old", "old_uid", "Old_Uid", "OLD UID"],
    "latitude": ["latitude", "lat", "LAT", "Latitude", "y"],
    "longitude": ["longitude", "lon", "lng", "LONG", "Longitude", "x"],
    "all_owners": ["all_ownernames", "all_owners"],
    "all_guardians": ["all_guardiannames", "all_guardians"],
    "all_mobiles": ["all_mobilenos", "all_mobiles"],
    "geocode_accuracy": [
        "geocode_accuracy", "accuracy", "location_type",
    ],
    "electricity": [
        "Electricity", "electricity", "elec_conn",
        "electric_m", "meter",
    ],
    "occupancy": [
        "Occupancy", "occupancy", "occup_stat", "status",
    ],
    "exempted": [
        "Exempted", "exempted", "exempt", "tax_status",
        "taxable", "Taxable",
    ],
    "property_usage": [
        "property_u", "property_usage", "Property_Usage",
        "land_use", "usage_type", "property_use",
    ],
    "property_type": [
        "property_t", "property_type", "Property_Type",
        "type", "Type",
    ],
}


def detect_columns(
    df: pd.DataFrame,
    patterns: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """Auto-detect logical column names from a DataFrame.

    For each logical field in *patterns* the available DataFrame columns
    are scanned (case-insensitive) and the first hit is recorded.

    Parameters
    ----------
    df:
        DataFrame whose ``.columns`` will be inspected.
    patterns:
        Mapping of ``logical_name -> [candidate_column_name, ...]``.
        Falls back to :data:`COLUMN_PATTERNS` when *None*.

    Returns
    -------
    dict[str, str | None]
        ``{logical_name: actual_column_name}`` — the value is *None*
        when no candidate matched.
    """
    if patterns is None:
        patterns = COLUMN_PATTERNS

    # Build a lookup: lowercase column name -> actual column name
    col_lower: dict[str, str] = {c.lower().strip(): c for c in df.columns}

    detected: dict[str, str | None] = {}
    for logical, candidates in patterns.items():
        match: str | None = None
        for cand in candidates:
            actual = col_lower.get(cand.lower().strip())
            if actual is not None:
                match = actual
                break
        if match is not None:
            detected[logical] = match
    return detected


def _detect_columns_for_fields(
    field_names: list[str],
    patterns: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """Auto-detect logical column names from a *flat list* of field names.

    Useful for shapefile field headers, where there is no DataFrame.

    Parameters
    ----------
    field_names:
        Raw field names (e.g. from ``pyshp``).
    patterns:
        Same as :func:`detect_columns`.

    Returns
    -------
    dict[str, str | None]
        ``{logical_name: actual_field_name}`` — *None* when unmatched.
    """
    if patterns is None:
        patterns = COLUMN_PATTERNS

    name_lower: dict[str, str] = {n.lower().strip(): n for n in field_names}

    detected: dict[str, str | None] = {}
    for logical, candidates in patterns.items():
        match: str | None = None
        for cand in candidates:
            actual = name_lower.get(cand.lower().strip())
            if actual is not None:
                match = actual
                break
        if match is not None:
            detected[logical] = match
    return detected


# ══════════════════════════════════════════════════════════════════════
# MOBILE NUMBER NORMALISATION
# ══════════════════════════════════════════════════════════════════════

_NON_DIGIT_RE = re.compile(r"[^\d]")


def normalize_mobile(mob: Any) -> str:
    """Return the last 10 digits of a phone number, or ``""``."""
    if mob is None or (isinstance(mob, float) and pd.isna(mob)):
        return ""
    mob = str(mob).strip()
    if mob.endswith(".0"):
        mob = mob[:-2]
    mob = _NON_DIGIT_RE.sub("", mob)
    return mob[-10:] if len(mob) >= 10 else mob


def _normalize_mobile_column(series: pd.Series) -> pd.Series:
    """Vectorised mobile normalisation for a whole column."""
    return series.apply(normalize_mobile)


# ══════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════

# -- Sentinel to distinguish "no value passed" from an explicit None --
_UNSET = object()

# Common CSV encodings to try, in order.
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


def load_mseva(
    path: str | Path,
    column_overrides: dict[str, str] | None = None,
    *,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load an mSeva property-tax CSV dump.

    Parameters
    ----------
    path:
        File-system path to the CSV.
    column_overrides:
        Optional ``{logical_name: actual_column}`` map that takes
        precedence over auto-detection.
    encoding:
        Explicit encoding.  When *None* a small set of common encodings
        is tried automatically.

    Returns
    -------
    (df, columns)
        *df* — the loaded DataFrame with a ``mobileno_norm`` column
        added for the primary mobile field.
        *columns* — the detected (or overridden) column map.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file cannot be parsed with any attempted encoding.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mSeva CSV not found: {path}")

    # --- Read CSV with encoding fallback ---
    df: pd.DataFrame | None = None
    encodings_to_try = [encoding] if encoding else list(_ENCODINGS)

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            logger.info("Loaded %s with encoding=%s (%d rows)", path.name, enc, len(df))
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug("Encoding %s failed for %s, trying next…", enc, path.name)

    if df is None:
        raise ValueError(
            f"Could not decode {path.name} with any of {encodings_to_try}"
        )

    # --- Detect columns ---
    columns = detect_columns(df)
    if column_overrides:
        columns.update(column_overrides)

    # --- Normalise primary mobile column ---
    mob_col = columns.get("mobile")
    if mob_col and mob_col in df.columns:
        df["mobileno_norm"] = _normalize_mobile_column(df[mob_col])
        n_with = (df["mobileno_norm"] != "").sum()
        logger.info("Mobile normalised (%s): %d / %d non-empty", mob_col, n_with, len(df))
    else:
        df["mobileno_norm"] = ""
        logger.warning("No mobile column detected — mobileno_norm is empty")

    # --- Log detection summary ---
    logger.info(
        "Detected mSeva columns: %s",
        {k: v for k, v in columns.items() if v is not None},
    )

    return df, columns


def load_gis(
    path: str | Path,
    column_overrides: dict[str, str] | None = None,
) -> tuple[list[Any], list[dict[str, str]], STRtree, dict[str, str | None]]:
    """Load a GIS shapefile of property polygons.

    Parameters
    ----------
    path:
        Path to the ``.shp`` file (companion ``.dbf`` / ``.shx`` must
        sit alongside it).
    column_overrides:
        Optional ``{logical_name: actual_column}`` overrides.

    Returns
    -------
    (polygons, records, spatial_index, columns)
        *polygons* — list of :class:`shapely.geometry.BaseGeometry`
        objects.
        *records* — parallel list of dicts with original shapefile field keys.
        *spatial_index* — :class:`shapely.strtree.STRtree` built from
        *polygons*.
        *columns* — detected (or overridden) column mapping.

    Raises
    ------
    FileNotFoundError
        If the shapefile cannot be opened.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")

    sf_reader = shapefile.Reader(str(path))
    field_names: list[str] = [f[0] for f in sf_reader.fields[1:]]
    field_index: dict[str, int] = {f: i for i, f in enumerate(field_names)}

    # Detect logical → actual field mapping
    columns = _detect_columns_for_fields(field_names)
    if column_overrides:
        columns.update(column_overrides)

    logger.info(
        "GIS fields (%d): %s", len(field_names), field_names,
    )
    logger.info(
        "Detected GIS columns: %s",
        {k: v for k, v in columns.items() if v is not None},
    )

    # Sentinel values that should be treated as empty.
    _EMPTY_SENTINELS = {"NA", "None", "0", "nan", "N/A", ""}

    # Dedup rule (technical-problems review, P6): real GIS shapefiles have
    # been observed with duplicate UID values (e.g. 11 duplicate UIDs / 20
    # extra rows in Barnala's gissurvey.shp). Without a stated rule, a
    # duplicated UID silently produces multiple GIS records/polygons for
    # what should be one property, inflating downstream counts. Rule:
    # first occurrence wins; every later duplicate is dropped and counted.
    uid_field = columns.get("uid")
    seen_uids: set[str] = set()
    duplicate_uids = 0

    polygons: list[Any] = []
    records: list[dict[str, str]] = []
    skipped = 0

    for sr in sf_reader.iterShapeRecords():
        # --- Geometry ---
        try:
            geom = shape(sr.shape.__geo_interface__)
        except Exception:
            skipped += 1
            continue

        if geom.is_empty:
            skipped += 1
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                skipped += 1
                continue

        # --- Attributes (keep original shapefile field keys) ---
        rec: dict[str, str] = {}
        for fn in field_names:
            raw = sr.record[field_index[fn]]
            val = str(raw).strip() if raw is not None else ""
            rec[fn] = "" if val in _EMPTY_SENTINELS else val

        # --- Dedup: first occurrence of a UID wins ---
        if uid_field:
            uid_val = rec.get(uid_field, "")
            if uid_val:
                if uid_val in seen_uids:
                    duplicate_uids += 1
                    continue
                seen_uids.add(uid_val)

        polygons.append(geom)
        records.append(rec)

    logger.info(
        "GIS loaded: %d polygons, %d skipped", len(polygons), skipped,
    )
    if duplicate_uids:
        logger.warning(
            "GIS dedup: dropped %d duplicate-UID record(s) (first occurrence kept for each)",
            duplicate_uids,
        )

    # --- Spatial index ---
    spatial_index = STRtree(polygons)

    return polygons, records, spatial_index, columns


def load_electricity(
    path: str | Path,
    column_overrides: dict[str, str] | None = None,
    *,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load an electricity-connection CSV.

    Parameters
    ----------
    path:
        File-system path to the CSV.
    column_overrides:
        Optional ``{logical_name: actual_column}`` overrides.
    encoding:
        Explicit encoding (falls back to the same auto-detection as
        :func:`load_mseva`).

    Returns
    -------
    (df, columns)
        *df* — loaded DataFrame; a ``mobileno_norm`` column is added
        if a mobile field is detected.
        *columns* — the detected (or overridden) column map.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file cannot be parsed with any attempted encoding.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Electricity CSV not found: {path}")

    df: pd.DataFrame | None = None
    encodings_to_try = [encoding] if encoding else list(_ENCODINGS)

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            logger.info(
                "Loaded electricity CSV %s with encoding=%s (%d rows)",
                path.name, enc, len(df),
            )
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug("Encoding %s failed for %s", enc, path.name)

    if df is None:
        raise ValueError(
            f"Could not decode {path.name} with any of {encodings_to_try}"
        )

    # --- Detect columns ---
    columns = detect_columns(df)
    if column_overrides:
        columns.update(column_overrides)

    # --- Cast coordinates to float ---
    for col_key in ["latitude", "longitude"]:
        col_name = columns.get(col_key)
        if col_name and col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # --- Normalise mobile if present ---
    mob_col = columns.get("mobile")
    if mob_col and mob_col in df.columns:
        df["mobileno_norm"] = _normalize_mobile_column(df[mob_col])
    else:
        df["mobileno_norm"] = ""

    logger.info(
        "Detected electricity columns: %s",
        {k: v for k, v in columns.items() if v is not None},
    )

    return df, columns
