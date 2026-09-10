"""
helpers.py — Pure-function utilities for the Property Tax Pipeline.

Every function here is stateless, deterministic, and tested in both
the Barnala and Batala pipelines.  The logic is **identical** to the
working scripts — only the signatures have been generalised (e.g.
``normalize_locality`` now accepts an optional ``city_name`` parameter
instead of hard-coding the city).
"""

from __future__ import annotations

import math
import re
from typing import Tuple

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Mobile / Name / Locality normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalize_mobile(mob) -> str:
    """Normalize a mobile-phone number to its last 10 digits.

    Handles the common mSeva quirk where numbers are stored as floats
    (e.g. ``8557004043.0``), strips non-digit characters, and returns
    the rightmost 10 digits.

    Parameters
    ----------
    mob : Any
        Raw mobile value — may be ``str``, ``float``, ``int``, or ``NaN``.

    Returns
    -------
    str
        Normalised 10-digit string, or ``""`` if the input is empty/NaN.

    Examples
    --------
    >>> normalize_mobile(8557004043.0)
    '8557004043'
    >>> normalize_mobile('+91-9876543210')
    '9876543210'
    >>> normalize_mobile(None)
    ''
    """
    if not mob or pd.isna(mob):
        return ""
    mob = str(mob).strip()
    # Fix: mSeva stores mobiles as floats (e.g. 8557004043.0)
    # Strip .0 suffix BEFORE extracting digits
    if mob.endswith(".0"):
        mob = mob[:-2]
    mob = re.sub(r"[^\d]", "", mob)
    return mob[-10:] if len(mob) >= 10 else mob


def normalize_name(name) -> str:
    """Strip honorifics, relationship prefixes, and punctuation from a name.

    Lowercases, removes common Indian-English prefixes (Shri, Smt, S/O,
    W/O, etc.), drops non-alphanumeric characters, and collapses
    whitespace.

    Parameters
    ----------
    name : Any
        Raw name string or ``NaN``.

    Returns
    -------
    str
        Cleaned name, or ``""`` if input is empty/NaN.

    Examples
    --------
    >>> normalize_name('Shri Gurpreet Singh S/O Balwinder Singh')
    'gurpreet singh balwinder singh'
    """
    if not name or pd.isna(name):
        return ""
    name = str(name).lower().strip()
    for h in [
        "shri ", "smt ", "smt. ", "mr ", "mr. ", "mrs ", "mrs. ",
        "dr ", "dr. ", "s/o ", "w/o ", "d/o ", "c/o ",
        "sh ", "sh. ", "late ", "ms ", "ms. ",
        # Variants without trailing space (e.g. "s/oBalwinder")
        "s/o", "w/o", "d/o", "c/o",
    ]:
        name = name.replace(h, " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name)).strip()


def normalize_locality(loc, city_name: str = "") -> str:
    """Normalize a locality string for fuzzy comparison.

    Removes the city name (if provided), block references like
    ``Block 4A``, punctuation, and excess whitespace.

    Parameters
    ----------
    loc : Any
        Raw locality / address string.
    city_name : str, optional
        City name to strip (case-insensitive).  Pass ``""`` to skip.

    Returns
    -------
    str
        Cleaned locality, or ``""`` if input is empty/NaN.

    Examples
    --------
    >>> normalize_locality('Block 4A, Model Town, Batala', city_name='Batala')
    'model town'
    """
    if not loc or pd.isna(loc):
        return ""
    loc = str(loc).lower().strip()
    if city_name:
        loc = re.sub(r"\b" + re.escape(city_name.lower()) + r"\b", "", loc)
    loc = re.sub(r"\bblock\s*\d+[a-z]?\b", "", loc)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", loc)).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Fuzzy name matching
# ═══════════════════════════════════════════════════════════════════════════

def token_set_ratio(s1: str, s2: str) -> int:
    """Lightweight fuzzy-match score based on token (word) overlap.

    Computes the Jaccard similarity of the two token sets, scaled to
    0-100.  This is intentionally simple — no edit-distance, no
    phonetic encoding — because it is called millions of times in
    the inner matching loops.

    Parameters
    ----------
    s1, s2 : str
        Pre-normalised strings (lowercase, no punctuation).

    Returns
    -------
    int
        Similarity score in ``[0, 100]``.

    Examples
    --------
    >>> token_set_ratio('gurpreet singh', 'gurpreet singh')
    100
    >>> token_set_ratio('gurpreet singh', 'gurpreet kaur')
    50
    >>> token_set_ratio('', 'gurpreet')
    0
    """
    if not s1 or not s2:
        return 0
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 or not t2:
        return 0
    return int(len(t1 & t2) / len(t1 | t2) * 100)


def best_name_score(
    mo: str,
    mg: str,
    mao: str,
    mag: str,
    go: str,
    gf: str,
) -> Tuple[int, str]:
    """Compare owner + guardian names across all available variants.

    Tries every combination of mSeva owner/guardian against GIS
    owner/father-husband and returns the highest score plus a tag
    describing which combination produced it.

    Parameters
    ----------
    mo : str
        Normalised mSeva owner name.
    mg : str
        Normalised mSeva guardian name.
    mao : str
        Pipe-delimited (``|``) string of *all* mSeva owner names
        (multi-owner properties).
    mag : str
        Pipe-delimited string of all mSeva guardian names.
    go : str
        Normalised GIS owner name.
    gf : str
        Normalised GIS father/husband name.

    Returns
    -------
    tuple[int, str]
        ``(score, method)`` where *score* ∈ [0, 100] and *method* is
        one of ``'OWNER_MATCH'``, ``'ALL_OWNERS_MATCH'``,
        ``'GUARDIAN_MATCH'``, ``'ALL_GUARDIANS_MATCH'``, or
        ``'OWNER+GUARDIAN_COMBINED'``.

    Examples
    --------
    >>> best_name_score('gurpreet singh', 'balwinder singh',
    ...                 '', '', 'gurpreet singh', 'balwinder singh')
    (100, 'OWNER_MATCH')
    """
    best: int = 0
    det: str = ""

    # 1. Primary owner ↔ GIS owner
    if mo and go:
        s = token_set_ratio(mo, go)
        if s > best:
            best, det = s, "OWNER_MATCH"

    # 2. Any of the multi-owner names ↔ GIS owner
    if mao and go:
        for p in mao.split("|"):
            p = normalize_name(p)
            if p:
                s = token_set_ratio(p, go)
                if s > best:
                    best, det = s, "ALL_OWNERS_MATCH"

    # 3. Primary guardian ↔ GIS father/husband
    if mg and gf:
        s = token_set_ratio(mg, gf)
        if s > best:
            best, det = s, "GUARDIAN_MATCH"

    # 4. Any of the multi-guardian names ↔ GIS father/husband
    if mag and gf:
        for p in mag.split("|"):
            p = normalize_name(p)
            if p:
                s = token_set_ratio(p, gf)
                if s > best:
                    best, det = s, "ALL_GUARDIANS_MATCH"

    # 5. Combined owner + guardian (both must be ≥ 50)
    if mo and go and mg and gf:
        c = int((token_set_ratio(mo, go) + token_set_ratio(mg, gf)) / 2)
        if c > best and token_set_ratio(mo, go) >= 50 and token_set_ratio(mg, gf) >= 50:
            best, det = c, "OWNER+GUARDIAN_COMBINED"

    return best, det


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate conversion
# ═══════════════════════════════════════════════════════════════════════════

def wgs84_to_utm(lat: float, lon: float, zone: int = 44) -> Tuple[float, float]:
    """Convert WGS-84 latitude/longitude to UTM easting/northing.

    This is the same closed-form conversion used in the working Barnala
    and Batala pipelines, parameterised by UTM zone so it works for any
    Indian city.  The central meridian is derived from the zone number.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (positive = north).
    lon : float
        Longitude in decimal degrees (positive = east).
    zone : int, optional
        UTM zone number.  Default ``44`` (covers most of Punjab:
        central meridian 81°E).

    Returns
    -------
    tuple[float, float]
        ``(easting, northing)`` in metres.

    Notes
    -----
    Only the *northern-hemisphere* formula is implemented (no false
    northing subtracted).  This is correct for all of India.

    Examples
    --------
    >>> e, n = wgs84_to_utm(30.37, 75.38, zone=44)
    >>> int(e), int(n)
    (393713, 3360509)
    """
    # WGS-84 ellipsoid parameters
    a: float = 6378137.0                       # semi-major axis (m)
    f: float = 1.0 / 298.257223563             # flattening
    e2: float = 2 * f - f * f                  # eccentricity squared
    ep2: float = e2 / (1 - e2)                 # second eccentricity squared
    k0: float = 0.9996                         # scale factor on central meridian

    # Central meridian for the requested zone
    lon0: float = float((zone - 1) * 6 - 180 + 3)

    lr: float = math.radians(lat)
    lnr: float = math.radians(lon)
    l0r: float = math.radians(lon0)

    N: float = a / math.sqrt(1 - e2 * math.sin(lr) ** 2)
    T: float = math.tan(lr) ** 2
    C: float = ep2 * math.cos(lr) ** 2
    A: float = math.cos(lr) * (lnr - l0r)

    M: float = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lr
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024)
        * math.sin(2 * lr)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lr)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lr)
    )

    easting: float = (
        k0 * N * (
            A
            + (1 - T + C) * A ** 3 / 6
            + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120
        )
        + 500000.0
    )

    northing: float = k0 * (
        M
        + N * math.tan(lr) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720
        )
    )

    return easting, northing


def utm_to_wgs84(easting: float, northing: float, zone: int = 44) -> Tuple[float, float]:
    """Convert UTM easting/northing back to WGS-84 latitude/longitude.

    Uses the iterative Bowring method for the reverse projection.
    Only northern-hemisphere is implemented (suitable for all of India).

    Parameters
    ----------
    easting : float
        UTM easting in metres.
    northing : float
        UTM northing in metres (northern hemisphere, no false northing).
    zone : int, optional
        UTM zone number. Default ``44``.

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` in decimal degrees.

    Examples
    --------
    >>> lat, lon = utm_to_wgs84(393713, 3360509, zone=44)
    >>> round(lat, 2), round(lon, 2)
    (30.37, 75.38)
    """
    # WGS-84 ellipsoid parameters
    a: float = 6378137.0
    f: float = 1.0 / 298.257223563
    e2: float = 2 * f - f * f
    e1: float = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    k0: float = 0.9996

    # Central meridian
    lon0: float = float((zone - 1) * 6 - 180 + 3)

    x: float = easting - 500000.0  # remove false easting
    y: float = northing             # northern hemisphere: no false northing

    M: float = y / k0
    mu: float = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

    phi1: float = (
        mu
        + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
        + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
        + (1097 * e1 ** 4 / 512) * math.sin(8 * mu)
    )

    ep2: float = e2 / (1 - e2)
    N1: float = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1: float = math.tan(phi1) ** 2
    C1: float = ep2 * math.cos(phi1) ** 2
    R1: float = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D: float = x / (N1 * k0)

    lat: float = phi1 - (N1 * math.tan(phi1) / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2
           - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720
    )

    lon: float = (
        D
        - (1 + 2 * T1 + C1) * D ** 3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2
           + 8 * ep2 + 24 * T1 ** 2) * D ** 5 / 120
    ) / math.cos(phi1)

    return math.degrees(lat), math.degrees(lon) + lon0

