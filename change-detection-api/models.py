"""
models.py  —  Pydantic request / response models for the Change Detection API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class ChangeRequest(BaseModel):
    city:   str = Field(..., description="City name, e.g. Mohali, Barnala, Amritsar")
    year1:  int = Field(..., ge=2013, le=2030, description="Baseline year (GIS survey year)")
    year2:  int = Field(..., ge=2013, le=2030, description="Current / comparison year")
    bounds: Optional[list[float]] = Field(
        None,
        description="Optional [W, S, E, N] bounding box override. "
                    "If omitted, uses the default bounds for the city.",
    )


class TileResponse(BaseModel):
    tile_url: str
    city:     str
    year1:    int
    year2:    int


class AreaStatsResponse(BaseModel):
    city:              str
    year1:             int
    year2:             int
    new_builtup_sqkm:  float


class ShapefileResponse(BaseModel):
    download_url: str
    city:         str
    year1:        int
    year2:        int
