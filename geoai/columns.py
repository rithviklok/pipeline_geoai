"""
GIS Column Auto-Mapping & Standardisation.

Extracted from train_geoai_geocoder.py Part 1 (L242-L450).
Maps raw GIS column names → standardised names used throughout the pipeline.
"""

import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical mapping: raw GIS header → standardised name ────────────
# Covers multiple naming conventions across Punjab ULBs (Barnala, Amritsar, etc.)
GIS_COLUMN_MAPPING = {
    # Identity
    "UID": "uid",
    "UID OLD": "uid_old",
    "OLD UID": "uid_old",
    "Old_Uid": "uid_old",
    "PROPERTY ID": "property_id",
    "SURVEY ID": "survey_id",

    # Owner Information
    "OWNER NAME": "owner_name",
    "Property_O": "owner_name",
    "Owner_Name": "owner_name",
    "FATHER HUSBAND NAME": "father_husband_name",
    "Father_Nam": "father_husband_name",
    "Father_Hus": "father_husband_name",
    "Husband_Na": "father_husband_name",
    "MOBILE NO": "mobile_no",
    "Mobile_No": "mobile_no",
    "EMAIL ID": "email_id",

    # Address Information
    "ADDRESS": "address",
    "Property_A": "address",
    "LOCALITY": "locality",
    "Locality": "locality",
    "Locality_N": "locality",
    "ROAD NAME": "road_name",
    "Road_Name": "road_name",
    "WARD NO": "ward_no",
    "Ward_No": "ward_no",
    "Ward_no": "ward_no",
    "HOUSE FLAT NO": "house_flat",
    "HOUSE NO FLAT NO": "house_flat",
    "House_No": "house_flat",
    "House_No_F": "house_flat",
    "BUILDING NAME": "building_name",
    "APPARTMENT BUILDING NAME": "apartment_building_name",
    "Apartment_": "apartment_building_name",
    "SECTOR NO OR BLOCK NO": "sector_no",
    "Block_no_S": "sector_no",
    "PINCODE": "pincode",
    "Pin_Code": "pincode",
    "PIN CODE": "pincode",

    # Property Information
    "PROPERTY TYPE": "property_type",
    "Property_T": "property_type",
    "PROPERTY USE": "property_use",
    "Property_U": "property_use",
    "TYPE OF CONSTRUCTION": "type_of_construction",
    "Type_of_Co": "type_of_construction",
    "OCCUPANCY": "occupancy",
    "Occupancy_": "occupancy",
    "FLOORS": "floors",
    "No_of_Floo": "floors",
    "PLOT AREA": "plot_area",
    "Total_Plot": "plot_area",
    "BUILTUP AREA": "builtup_area",
    "Total_Buil": "builtup_area",

    # Utility Information
    "ELECTRIC METER NO": "electric_meter",
    "ELECTRIC METER": "electric_meter",
    "Electricit": "electric_meter",
    "WATER SUPPLY CONNECTION NO": "water_connection",
    "WATER SUPPLY CONNECTION NUMBER": "water_connection",
    "Water_conn": "water_connection",
    "SEWAGE CONNECTION NO": "sewage_connection",
    "SEWAGE CONNECTION NUMBER": "sewage_connection",
    "Sewarage_c": "sewage_connection",

    # Spatial
    "LAT": "Latitude",
    "LONG": "Longitude",
    "Latitude": "Latitude",
    "Longitude": "Longitude",

    # Exemption
    "Exempted": "exempted",
    "EXEMPTED": "exempted",
    "Exemption_": "exempted",

    # GIS Geometry
    "SHAPE AREA": "shape_area",
    "SHAPE_Area": "shape_area",
    "SHAPE LENGTH": "shape_length",
    "SHAPE_Leng": "shape_length",
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace & control chars from column names."""
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("\n", " ")
        .str.replace("  ", " ")
    )
    return df


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge columns that share the same name after mapping.

    When multiple raw headers map to the same standardised name (e.g.
    ``Father_Nam`` and ``Husband_Na`` both → ``father_husband_name``),
    pandas keeps both columns. This function coalesces them by taking
    the first non-null value.
    """
    seen = {}
    to_drop = []
    for i, col in enumerate(df.columns):
        if col in seen:
            # Coalesce: fill the first occurrence with values from this duplicate
            first_idx = seen[col]
            first_col = df.iloc[:, first_idx]
            dup_col = df.iloc[:, i]
            df.iloc[:, first_idx] = first_col.fillna(dup_col)
            to_drop.append(i)
        else:
            seen[col] = i

    if to_drop:
        df = df.drop(df.columns[to_drop], axis=1)
        logger.info("Coalesced %d duplicate columns: %s",
                     len(to_drop), [df.columns[i] for i in to_drop] if to_drop else [])
    return df


def standardise_gis_columns(
    df: pd.DataFrame,
    custom_mapping: dict = None,
) -> pd.DataFrame:
    """Apply column mapping and coalesce duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw GIS dataframe (from CSV, shapefile, etc.)
    custom_mapping : dict, optional
        Additional or override column mappings {raw_name: std_name}

    Returns
    -------
    pd.DataFrame with standardised column names
    """
    df = clean_column_names(df)

    mapping = dict(GIS_COLUMN_MAPPING)
    if custom_mapping:
        mapping.update(custom_mapping)

    # Only rename columns that actually exist
    applicable = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=applicable)

    logger.info("Mapped %d columns: %s", len(applicable), list(applicable.keys()))

    df = coalesce_duplicate_columns(df)
    return df
