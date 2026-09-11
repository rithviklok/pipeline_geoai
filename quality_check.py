"""
quality_check.py — Data quality assessment module.

Assesses data quality of mSeva, GIS, and optional electricity datasets
before running the matching pipeline. Calculates a feasibility score
and provides a GO/CAUTION/NO-GO verdict.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import shapefile
from shapely.geometry import shape

from .config import CityConfig
from .data_loader import load_electricity, load_gis, load_mseva
from .helpers import normalize_mobile, normalize_locality

logger = logging.getLogger(__name__)

# Indian address component detection regex
ADDRESS_PATTERNS = {
    "house_no": re.compile(
        r"(?:h\.?\s*no\.?|house\s*no\.?|plot\s*no\.?|flat\s*no\.?|door\s*no\.?|khasra\s*no\.?)"
        r"\s*[-:.]?\s*(\S+)", re.IGNORECASE
    ),
    "house_no_plain": re.compile(r"^(\d+[/-]?\d*[a-zA-Z]?)\s*[,\s]", re.IGNORECASE),
    "building": re.compile(
        r"(?:apartment|society|complex|tower|bhawan|mansion|plaza|arcade|court|residency|enclave|villa)",
        re.IGNORECASE
    ),
    "street": re.compile(
        r"(?:road|rd|street|st|gali|marg|lane|path|cross|main|chowk|bazaar|bazar|market|mandi)",
        re.IGNORECASE
    ),
    "locality": re.compile(
        r"(?:nagar|colony|mohalla|pura|basti|vihar|kunj|enclave|sector|phase|block|ward|wala|park|garden)",
        re.IGNORECASE
    ),
    "ward_ref": re.compile(r"(?:ward|zone|circle|block)\s*[-:.]?\s*(\d+)", re.IGNORECASE),
}

JUNK_VALUES = {
    "na", "n/a", "nil", "none", "-", "--", "...", ".", "same", "same as above",
    "test", "abc", "xyz", "dummy", "temp", "null", "unknown", "not available",
    "address", "no address",
}


def clean_address(addr) -> str:
    """Basic clean of address text."""
    if not addr or pd.isna(addr):
        return ""
    # Normalize unicode
    addr = unicodedata.normalize("NFC", str(addr))
    # Replace separators with spaces, collapse whitespace
    addr = re.sub(r"[\s,;:\-\./]+", " ", addr).strip()
    return addr


def is_junk_address(addr: str) -> bool:
    """Return True if address is placeholder or garbage."""
    if not addr:
        return True
    val = addr.lower().strip()
    if val in JUNK_VALUES:
        return True
    if len(val) < 3:
        return True
    if val.isdigit() and len(val) < 6:  # ignore short digit strings
        return True
    return False


def detect_duplicated_address(addr: str) -> bool:
    """Detect repeated parts of address (data-entry error)."""
    if not addr:
        return False
    parts = re.split(r"\s{2,}", addr)
    if len(parts) >= 2:
        unique = set(p.strip().lower() for p in parts if p.strip())
        if len(unique) == 1:
            return True
    return False


def extract_address_components(addr: str) -> Dict[str, bool]:
    """Identify if address components are present via regex."""
    a = addr.lower()
    return {
        "house_no": bool(ADDRESS_PATTERNS["house_no"].search(a) or ADDRESS_PATTERNS["house_no_plain"].search(a)),
        "building": bool(ADDRESS_PATTERNS["building"].search(a)),
        "street": bool(ADDRESS_PATTERNS["street"].search(a)),
        "locality": bool(ADDRESS_PATTERNS["locality"].search(a)),
        "ward": bool(ADDRESS_PATTERNS["ward_ref"].search(a)),
    }


def compute_address_completeness(components: Dict[str, bool]) -> float:
    """Calculate completeness score between 0.0 and 1.0."""
    weights = {"ward": 0.05, "locality": 0.45, "street": 0.25, "house_no": 0.25}
    return round(sum(weights[k] for k, v in components.items() if v and k in weights), 2)


def is_valid_mobile(mob: str) -> bool:
    """Verify if mobile number is valid Indian 10-digit format."""
    return bool(re.match(r"^[6789]\d{9}$", mob))


class QualityAssessor:
    """Performs data quality checks across input datasets."""

    def __init__(self, config: CityConfig):
        self.config = config

    def run(self) -> dict:
        """Run the full quality check suite."""
        logger.info("Running data quality check for %s...", self.config.name)

        report = {
            "city": self.config.name,
            "verdict": "NO-GO",
            "score": 0,
            "issues": [],
            "warnings": [],
            "mseva": {},
            "gis": {},
            "cross_dataset": {},
        }

        # 1. Assess mSeva
        try:
            mseva_df, m_cols = load_mseva(self.config.mseva_path, self.config.mseva_columns)
            self._assess_mseva(mseva_df, m_cols, report)
        except Exception as e:
            report["issues"].append(f"Failed to load/assess mSeva file: {e}")
            return report

        # 2. Assess GIS
        try:
            # We open with shapefile directly for low-level metadata
            sf_r = shapefile.Reader(self.config.gis_path)
            self._assess_gis(sf_r, report)
        except Exception as e:
            report["issues"].append(f"Failed to load/assess GIS shapefile: {e}")
            return report

        # 3. Assess Electricity (optional)
        if self.config.electricity_path and os.path.exists(self.config.electricity_path):
            try:
                elec_df, e_cols = load_electricity(self.config.electricity_path, self.config.elec_columns)
                self._assess_electricity(elec_df, e_cols, report)
            except Exception as e:
                report["warnings"].append(f"Failed to load/assess electricity file: {e}")

        # 4. Cross-dataset overlap
        if report["mseva"] and report["gis"]:
            self._assess_cross_dataset(mseva_df, m_cols, sf_r, report)

        # 5. Compute verdict
        self._compute_verdict(report)

        return report

    def _assess_mseva(self, df: pd.DataFrame, cols: dict, report: dict):
        n = len(df)
        report["mseva"] = {
            "total_records": n,
            "fields": {},
            "scores": {},
        }

        # Fill rates
        for key, col in cols.items():
            if col:
                filled = df[col].notna().sum()
                uniq = df[col].nunique()
                pct = filled / n * 100
                report["mseva"]["fields"][key] = {
                    "column": col,
                    "filled": int(filled),
                    "pct": round(pct, 1),
                    "unique": int(uniq),
                }
                if pct < 50 and key in ("owner", "mobile"):
                    report["issues"].append(f"mSeva critical field '{key}' has low fill rate ({pct:.1f}%)")
            else:
                report["mseva"]["fields"][key] = None
                if key in ("owner", "mobile", "address"):
                    report["issues"].append(f"mSeva critical field '{key}' not found/mapped")

        # Mobile validation
        mob_col = cols.get("mobile")
        if mob_col:
            mobs = df[mob_col].dropna()
            norm = mobs.apply(normalize_mobile)
            valid = norm.apply(is_valid_mobile)
            valid_pct = valid.sum() / n * 100
            report["mseva"]["scores"]["mobile_valid_pct"] = round(valid_pct, 1)
            if valid_pct < 50:
                report["issues"].append(f"mSeva valid mobile numbers fraction is very low ({valid_pct:.1f}%)")

        # Address completeness via regex (fast approximation)
        addr_col = cols.get("address")
        if addr_col:
            addrs = df[addr_col].fillna("")
            clean = addrs.apply(clean_address)
            junk = clean.apply(is_junk_address)
            dup = clean.apply(detect_duplicated_address)

            report["mseva"]["scores"]["address_junk_pct"] = round(junk.sum() / n * 100, 1)
            report["mseva"]["scores"]["address_dup_pct"] = round(dup.sum() / n * 100, 1)

            # Sample component assessment
            good = clean[~junk & (clean != "")]
            sample = good.sample(min(len(good), 2000), random_state=42) if len(good) > 0 else []
            scores = []
            for addr in sample:
                comps = extract_address_components(addr)
                scores.append(compute_address_completeness(comps))

            avg_score = sum(scores) / len(scores) if scores else 0.0
            report["mseva"]["scores"]["address_avg_completeness"] = round(avg_score, 2)

    def _assess_gis(self, sf_r, report: dict):
        n = len(sf_r)
        report["gis"] = {
            "total_polygons": n,
            "fields": {},
            "scores": {},
        }

        fields = [f[0] for f in sf_r.fields[1:]]
        # Auto detect columns on shapefile fields
        from .data_loader import detect_columns, COLUMN_PATTERNS
        # Create a dummy DataFrame with shapefile fields as columns to reuse detect_columns
        dummy_df = pd.DataFrame(columns=fields)
        cols = detect_columns(dummy_df, COLUMN_PATTERNS)

        for key, col in cols.items():
            if col:
                idx = fields.index(col)
                # Count filled
                filled = sum(
                    1 for rec in sf_r.iterRecords()
                    if str(rec[idx]).strip() not in ("", "NA", "None", "0", "nan", "N/A")
                )
                pct = filled / n * 100
                report["gis"]["fields"][key] = {
                    "column": col,
                    "filled": int(filled),
                    "pct": round(pct, 1),
                }
                if pct < 50 and key in ("owner", "mobile", "uid"):
                    report["issues"].append(f"GIS critical field '{key}' has low fill rate ({pct:.1f}%)")
            else:
                report["gis"]["fields"][key] = None
                if key in ("uid", "owner"):
                    report["issues"].append(f"GIS critical field '{key}' not found/mapped")

        # Geometry checks (sample of 2000)
        invalid = 0
        empty = 0
        checked = 0
        for sr in sf_r.iterShapeRecords():
            try:
                geom = shape(sr.shape.__geo_interface__)
                if geom.is_empty:
                    empty += 1
                elif not geom.is_valid:
                    invalid += 1
            except Exception:
                empty += 1
            checked += 1
            if checked >= 2000:
                break

        report["gis"]["scores"]["invalid_geom_pct"] = round(invalid / checked * 100, 1) if checked else 0
        report["gis"]["scores"]["empty_geom_pct"] = round(empty / checked * 100, 1) if checked else 0

        # Technical-problems review (P4): property_usage/property_type feed
        # the dashboard's per-category breakdown. They don't block matching,
        # so they're not in the critical-field list above, but a field that
        # is missing or near-empty in the SOURCE data (as observed for real
        # Barnala GIS data: Property_U was ~100% "NA"/blank) silently ships
        # a dashboard with an unusable category breakdown unless flagged
        # here first.
        for key in ("property_usage", "property_type"):
            field_info = report["gis"]["fields"].get(key)
            if field_info is None:
                report["warnings"].append(
                    f"GIS field '{key}' not found/mapped in the source shapefile — "
                    f"the dashboard's '{key}' category breakdown will be unavailable"
                )
            elif field_info["pct"] < 20:
                report["warnings"].append(
                    f"GIS field '{key}' is only {field_info['pct']:.1f}% filled in the source data — "
                    f"the dashboard's '{key}' category breakdown will be mostly blank"
                )

    def _assess_electricity(self, df: pd.DataFrame, cols: dict, report: dict):
        n = len(df)
        lat_col = cols.get("latitude")
        lon_col = cols.get("longitude")

        has_gps = 0
        if lat_col and lon_col:
            has_gps = (df[lat_col].notna() & df[lon_col].notna()).sum()

        report["electricity"] = {
            "total_records": n,
            "gps_coverage_pct": round(has_gps / n * 100, 1),
        }

    def _assess_cross_dataset(self, mseva_df: pd.DataFrame, mcols: dict, sf_r, report: dict):
        mob_col = mcols.get("mobile")
        # GIS mobile col
        fields = [f[0] for f in sf_r.fields[1:]]
        from .data_loader import detect_columns, COLUMN_PATTERNS
        dummy_df = pd.DataFrame(columns=fields)
        gcols = detect_columns(dummy_df, COLUMN_PATTERNS)
        gis_mob_field = gcols.get("mobile")

        overlap_pct = 0.0
        if mob_col and gis_mob_field:
            mseva_mobs = set(mseva_df[mob_col].dropna().apply(normalize_mobile))
            mseva_mobs.discard("")

            gis_mob_idx = fields.index(gis_mob_field)
            gis_mobs = set()
            for rec in sf_r.iterRecords():
                m = normalize_mobile(rec[gis_mob_idx])
                if m:
                    gis_mobs.add(m)

            overlap = mseva_mobs & gis_mobs
            if mseva_mobs:
                overlap_pct = len(overlap) / len(mseva_mobs) * 100

        report["cross_dataset"] = {
            "mobile_overlap_pct": round(overlap_pct, 1),
        }

    def _compute_verdict(self, report: dict):
        score = 100

        # Calculate score based on issues
        # Critical field missing in mseva
        mfields = report["mseva"].get("fields", {})
        gfields = report["gis"].get("fields", {})

        for f in ("owner", "mobile", "address"):
            if not mfields.get(f):
                score -= 30

        for f in ("uid", "owner"):
            if not gfields.get(f):
                score -= 30

        # Mobile validation
        mv = report["mseva"].get("scores", {}).get("mobile_valid_pct", 0)
        if mv < 50:
            score -= 15

        # Mobile overlap
        overlap = report["cross_dataset"].get("mobile_overlap_pct", 0)
        if overlap < 10:
            score -= 30
        elif overlap < 30:
            score -= 15

        # Address quality
        addr_comp = report["mseva"].get("scores", {}).get("address_avg_completeness", 0)
        if addr_comp < 0.30:
            score -= 20

        # Geometries
        inv = report["gis"].get("scores", {}).get("invalid_geom_pct", 0)
        if inv > 10:
            score -= 10

        # Near-empty downstream-important fields (P4): doesn't block
        # matching, but produces an unusable dashboard category breakdown.
        # Modest deduction (same scale as the address-completeness penalty)
        # so it can nudge GO -> CAUTION without alone forcing a NO-GO.
        for key in ("property_usage", "property_type"):
            field_info = gfields.get(key)
            pct = field_info["pct"] if field_info else 0
            if pct < 20:
                score -= 10

        report["score"] = max(0, score)

        if report["score"] >= 70:
            report["verdict"] = "GO"
        elif report["score"] >= 40:
            report["verdict"] = "CAUTION"
        else:
            report["verdict"] = "NO-GO"
