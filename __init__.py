"""
Property Tax Defaulter Identification Pipeline
================================================
Modular, city-agnostic pipeline for matching mSeva property-tax records
against GIS survey data to identify defaulters.

Usage:
    from pipeline import CityConfig, PropertyTaxPipeline

    cfg = CityConfig(name="Barnala", state="Punjab",
                     mseva_path="path/to/mseva.csv",
                     gis_path="path/to/tax_area.shp")
    result = PropertyTaxPipeline(cfg).run()
"""

from .config import CityConfig, MatchResult, PipelineResult
from .orchestrator import PropertyTaxPipeline
from .helpers import (
    normalize_mobile,
    normalize_name,
    normalize_locality,
    token_set_ratio,
    best_name_score,
    wgs84_to_utm,
)

__all__ = [
    "CityConfig",
    "MatchResult",
    "PipelineResult",
    "PropertyTaxPipeline",
    "normalize_mobile",
    "normalize_name",
    "normalize_locality",
    "token_set_ratio",
    "best_name_score",
    "wgs84_to_utm",
]
