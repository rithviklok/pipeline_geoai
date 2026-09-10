"""
config.py — Configuration dataclasses for the Property Tax Pipeline.

Contains:
    CityConfig    – All parameters needed to run a city pipeline.
    MatchResult   – Result of matching a single mSeva record to GIS.
    PipelineResult – Aggregated output of a full pipeline run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# CityConfig — master configuration for one city run
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CityConfig:
    """Complete configuration for a single city's matching pipeline.

    Parameters
    ----------
    name : str
        City name (e.g. ``"Barnala"``, ``"Batala"``).  Used in logging,
        output filenames, and locality normalisation.
    state : str
        State name (e.g. ``"Punjab"``).
    mseva_path : str
        Path to the mSeva property-tax CSV (combined dump).
    gis_path : str
        Path to the GIS survey shapefile (``tax_area.shp``).
    electricity_path : str | None
        Optional path to the electricity-billing CSV.
    geocoded_path : str | None
        Optional path to a pre-geocoded mSeva CSV with lat/lon columns.
    output_dir : str
        Directory where pipeline outputs are written.  Created if absent.
    mseva_columns : dict | None
        Explicit column-name mapping for the mSeva CSV.
        Keys: ``propertyid``, ``ownername``, ``guardianname``, ``mobileno``,
        ``latitude``, ``longitude``, ``oldpropertyid``, ``address``,
        ``locality``.
        ``None`` ⇒ auto-detect at load time.
    gis_columns : dict | None
        Column-name mapping for GIS shapefile fields.
        Keys: ``uid``, ``uid_old``, ``owner_name``, ``father_hus``,
        ``mobile_no``, ``locality``, ``ward_no``, ``electric_m``.
        ``None`` ⇒ auto-detect at load time.
    elec_columns : dict | None
        Column-name mapping for the electricity CSV.
        ``None`` ⇒ auto-detect at load time.
    buffer_distance_m : float
        Spatial buffer radius in metres for proximity matching (Step 1b).
    name_threshold : int
        Minimum fuzzy-name score (0-100) to accept a text match.
    strong_name_threshold : int
        Score above which a name match is considered *strong* (no further
        verification needed).
    utm_zone : int | None
        UTM zone number used for the WGS-84 → UTM projection.
        ``None`` ⇒ auto-detect from GIS shapefile coordinates.
    gemini_api_key : str | None
        Google Gemini API key for LLM-assisted address parsing.
    geocoding_api_key : str | None
        Google Maps Geocoding API key.
    """

    # ── Required paths ───────────────────────────────────────────────────
    name: str
    state: str
    mseva_path: str

    # ── Optional paths ───────────────────────────────────────────────────
    gis_path: Optional[str] = None
    electricity_path: Optional[str] = None
    geoai_output_path: Optional[str] = None
    output_dir: str = "results"

    # ── Column mappings (None = auto-detect) ─────────────────────────────
    mseva_columns: Optional[Dict[str, str]] = None
    gis_columns: Optional[Dict[str, str]] = None
    elec_columns: Optional[Dict[str, str]] = None

    # ── Matching thresholds ──────────────────────────────────────────────
    buffer_distance_m: float = 50.0
    name_threshold: int = 60
    strong_name_threshold: int = 90
    utm_zone: Optional[int] = None

    # ── GeoAI training/inference ─────────────────────────────────────────
    model_dir: Optional[str] = None          # KB cache directory (default: models/{city}/)
    force_retrain: bool = False              # Force re-training even if KB is cached

    # ── Mobile number filtering ──────────────────────────────────────────
    mobile_max_frequency: int = 5            # Block numbers appearing > N times

    # ── Derived helpers ──────────────────────────────────────────────────

    @staticmethod
    def utm_zone_from_longitude(lon: float) -> int:
        """Compute the UTM zone number for a given longitude."""
        return int((lon + 180) / 6) + 1

    @property
    def utm_central_meridian(self) -> float:
        """Central meridian (degrees) for the configured UTM zone."""
        if self.utm_zone is None:
            raise ValueError(
                "utm_zone has not been set. Load GIS data first to auto-detect."
            )
        return (self.utm_zone - 1) * 6 - 180 + 3

    def ensure_output_dir(self) -> str:
        """Create the output directory if it doesn't exist and return its path."""
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def output_path(self, filename: str) -> str:
        """Return full path for a file inside the output directory."""
        self.ensure_output_dir()
        return os.path.join(self.output_dir, filename)


# ═══════════════════════════════════════════════════════════════════════════
# MatchResult — one mSeva-to-GIS match record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    """Result of matching a single mSeva property to a GIS polygon.

    Attributes
    ----------
    matched_uid : str
        GIS ``uid`` that the mSeva property was matched to.
    match_method : str
        Human-readable tag describing how the match was made
        (e.g. ``"PIP"``, ``"BUFFER_MOBILE"``, ``"NAME_LOCALITY"``).
    gis_owner_name : str
        Owner name from the GIS record.
    gis_mobile : str
        Mobile number from the GIS record.
    gis_locality : str
        Locality from the GIS record.
    confidence : float
        Match confidence in ``[0.0, 1.0]``.
    """

    matched_uid: str
    match_method: str
    gis_owner_name: str = ""
    gis_mobile: str = ""
    gis_locality: str = ""
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PipelineResult — aggregated output of a full pipeline run
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Container for the full output of a pipeline run.

    Attributes
    ----------
    city : str
        City name that was processed.
    total_mseva : int
        Total mSeva properties loaded.
    total_gis : int
        Total GIS polygons loaded.
    matched : int
        Number of mSeva properties successfully matched.
    unmatched : int
        Number of mSeva properties that remain unmatched.
    defaulters : int
        Number of GIS polygons identified as probable defaulters
        (present in GIS, absent from mSeva).
    match_register : Any
        The full match register as a ``pandas.DataFrame`` (or ``None``).
    defaulter_list : Any
        DataFrame of identified defaulters (or ``None``).
    step_stats : Dict[str, int]
        Per-step match counts, e.g.
        ``{"PIP": 1200, "BUFFER": 340, "MOBILE": 80, ...}``.
    output_files : List[str]
        Paths to all files written during the run.
    elapsed_seconds : float
        Wall-clock time for the entire pipeline run.
    """

    city: str = ""
    total_mseva: int = 0
    total_gis: int = 0
    matched: int = 0
    unmatched: int = 0
    defaulters: int = 0
    match_register: Any = None
    defaulter_list: Any = None
    step_stats: Dict[str, int] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def match_rate(self) -> float:
        """Fraction of mSeva properties that were matched."""
        return self.matched / self.total_mseva if self.total_mseva else 0.0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Pipeline Results — {self.city}",
            "=" * 50,
            f"  mSeva properties : {self.total_mseva:,}",
            f"  GIS polygons     : {self.total_gis:,}",
            f"  Matched          : {self.matched:,}  ({self.match_rate:.1%})",
            f"  Unmatched        : {self.unmatched:,}",
            f"  Defaulters       : {self.defaulters:,}",
            f"  Elapsed          : {self.elapsed_seconds:.1f}s",
        ]
        if self.step_stats:
            lines.append("  Step breakdown:")
            for step, count in self.step_stats.items():
                lines.append(f"    {step:30s} : {count:,}")
        return "\n".join(lines)
