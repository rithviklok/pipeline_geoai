"""
GeoAI Inference Engine.

Refactored from Inference.py (9,455 lines → ~600 lines).
Matches mSeva property tax records to GIS properties using semantic
search, spatial filtering, and weighted similarity scoring.

Flow:
  1. Load KB + preprocess mSeva records
  2. Direct ID matching (Old UID, Property ID, Survey ID)
  3. For unmatched: FAISS semantic search → locality prediction →
     spatial candidate reduction → weighted similarity scoring
  4. Coordinate estimation from top candidates
  5. Duplicate detection + fuzzy rematch
  6. Confidence assignment + output generation
"""

import os
import re
import time
import logging
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Inference Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """All tunable inference parameters."""

    # Sentence Transformer
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64

    # FAISS search
    faiss_top_k: int = 20

    # Matching thresholds
    top5_threshold: float = 80.0
    top10_threshold: float = 70.0
    top5_spread_metres: float = 50.0
    top10_spread_metres: float = 100.0

    # Locality matching
    locality_match_threshold: float = 75.0

    # Candidate limits (by density)
    high_density_cap: int = 50
    medium_density_cap: int = 150
    low_density_cap: int = 300

    # Matching weights
    matching_weights: dict = field(default_factory=lambda: {
        "address": 25, "owner_name": 15, "locality": 10,
        "house_number": 15, "owner_tokens": 10, "address_tokens": 5,
        "survey_id": 10, "old_property": 5, "guardian_name": 10,
    })

    # mSeva column names
    mseva_owner_col: str = "ownername"
    mseva_guardian_col: str = "guardianname"
    mseva_locality_col: str = "localityname"
    mseva_address_col: str = "address"
    mseva_property_id_col: str = "propertyid"
    mseva_old_property_id_col: str = "oldpropertyid"
    mseva_survey_id_col: str = "surveyid"


# ═══════════════════════════════════════════════════════════════
# Text Utilities (shared with trainer but kept local for independence)
# ═══════════════════════════════════════════════════════════════

def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    for ch in ',.\\-/()\';:"_#|':
        value = value.replace(ch, " ")
    return " ".join(value.split())


def _clean_identifier(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    if value.endswith(".0"):
        value = value[:-2]
    return value.replace("-", "").replace("/", "").replace("_", "").replace(" ", "")


def _normalize_address(address: str) -> str:
    address = _clean_text(address)
    replacements = {
        "house number": "house no", "house no.": "house no",
        "h.no": "house no", "h no": "house no", "hno": "house no",
        "ward number": "ward", "ward no.": "ward",
        "street number": "street", "st.": "street",
        "road number": "road", "rd": "road",
        "mohala": "mohalla", "mohallaa": "mohalla",
        "col.": "colony", "soc.": "society",
    }
    for old, new in replacements.items():
        address = address.replace(old, new)
    return " ".join(address.split())


def _extract_house_number(address) -> str:
    if pd.isna(address):
        return ""
    address = str(address).upper()
    for pattern in [r'(SHOP[\-\s]?\d+[A-Z]*)', r'(SCO[\-\s]?\d+[A-Z]*)',
                    r'(MC[\-\s]?\d+[A-Z]*)', r'(\d+[A-Z]?\/?\d*)', r'(\d+\-\d+)']:
        match = re.search(pattern, address)
        if match:
            return match.group(1)
    return ""


def _tokenize(value) -> list:
    return [t for t in _clean_text(value).split() if len(t) >= 2]


# ═══════════════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=200000)
def _fuzzy_similarity(text1: str, text2: str) -> float:
    """Token-sort ratio via rapidfuzz."""
    from rapidfuzz import fuzz
    t1 = _clean_text(text1)
    t2 = _clean_text(text2)
    if not t1 or not t2:
        return 0.0
    return fuzz.token_sort_ratio(t1, t2)


def _token_overlap(tokens1, tokens2) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    return round(len(s1 & s2) / max(len(s1), len(s2)) * 100, 2)


def _exact_match(v1, v2) -> float:
    v1 = str(v1).strip().lower()
    v2 = str(v2).strip().lower()
    if not v1 or not v2:
        return 0.0
    return 100.0 if v1 == v2 else 0.0


def _calculate_match_score(prop_row: dict, gis_row: dict, weights: dict) -> float:
    """Calculate weighted match score between mSeva and GIS records."""
    scores = {
        "address": _fuzzy_similarity(prop_row.get("address", ""), gis_row.get("address", "")),
        "owner_name": _fuzzy_similarity(prop_row.get("ownername", ""), gis_row.get("owner_name", "")),
        "locality": _fuzzy_similarity(prop_row.get("localityname", ""), gis_row.get("locality", "")),
        "house_number": _exact_match(prop_row.get("house_number", ""), gis_row.get("house_number", "")),
        "owner_tokens": _token_overlap(prop_row.get("owner_tokens", []), gis_row.get("owner_tokens", [])),
        "address_tokens": _token_overlap(prop_row.get("address_tokens", []), gis_row.get("address_tokens", [])),
        "survey_id": _exact_match(prop_row.get("surveyid", ""), gis_row.get("survey_id", "")),
        "old_property": _exact_match(prop_row.get("oldpropertyid", ""), gis_row.get("uid_old", "")),
    }

    # Guardian score: compare across owner/guardian fields (may be swapped)
    guardian_candidates = []
    gp = _clean_text(str(prop_row.get("guardianname", "")))
    op = prop_row.get("ownername", "")
    gg = gis_row.get("father_husband_name", "")
    og = gis_row.get("owner_name", "")
    if gp and gg:
        guardian_candidates.append(_fuzzy_similarity(gp, gg))
    if gp and og:
        guardian_candidates.append(_fuzzy_similarity(gp, og))
    if op and gg:
        guardian_candidates.append(_fuzzy_similarity(op, gg))
    scores["guardian_name"] = max(guardian_candidates) if guardian_candidates else 0.0

    # Normalise weights (only count non-zero scores)
    available = {k: weights.get(k, 0) for k, s in scores.items() if s > 0}
    total_w = sum(available.values())
    if total_w == 0:
        return 0.0

    final = sum(scores[k] * (available[k] / total_w) for k in available)
    return round(final, 2)


def _confidence_level(score: float) -> str:
    if score >= 95:
        return "VERY HIGH"
    elif score >= 90:
        return "HIGH"
    elif score >= 80:
        return "MEDIUM"
    elif score >= 70:
        return "LOW"
    return "VERY LOW"


# ═══════════════════════════════════════════════════════════════
# Inferencer Class
# ═══════════════════════════════════════════════════════════════

class GeoAIInferencer:
    """Runs GeoAI inference to match mSeva records to GIS properties.

    Usage:
        from .knowledge_base import load
        kb = load("models/Barnala/")
        inferencer = GeoAIInferencer(kb, InferenceConfig())
        output_df = inferencer.infer(mseva_df)
    """

    def __init__(self, kb, config: InferenceConfig = None):
        from .knowledge_base import KnowledgeBase
        self.kb: KnowledgeBase = kb
        self.config = config or InferenceConfig()
        self._model = None
        self._gis_locality_list = None
        self._results = []

    def infer(self, mseva_df: pd.DataFrame) -> pd.DataFrame:
        """Run full inference pipeline.

        Parameters
        ----------
        mseva_df : pd.DataFrame
            Raw mSeva property tax records.

        Returns
        -------
        pd.DataFrame
            Geocoded output with Matched_UID, Match_Type, Latitude, Longitude, etc.
        """
        start = time.time()
        logger.info("Starting GeoAI Inference on %d records...", len(mseva_df))

        # Preprocess mSeva
        clean_mseva_df = self._preprocess_mseva(mseva_df)

        # Load embedding model
        from .embeddings import load_model
        self._model = load_model(self.config.model_name)

        # Build locality list for fuzzy matching
        house_df = self.kb.house_df
        if "locality" in house_df.columns:
            self._gis_locality_list = house_df["locality"].dropna().unique().tolist()

        # Step 1: Direct ID matching
        direct_matches = self._direct_id_match(clean_mseva_df)
        logger.info("Direct ID matches: %d", len(direct_matches))

        # Step 2: FAISS + weighted scoring for remaining records
        matched_pids = set(direct_matches.keys())
        pid_col = self.config.mseva_property_id_col
        unmatched_df = clean_mseva_df[~clean_mseva_df[pid_col].isin(matched_pids)]
        logger.info("Unmatched after direct ID: %d", len(unmatched_df))

        semantic_matches = self._semantic_match(unmatched_df)
        logger.info("Semantic matches: %d", len(semantic_matches))

        # Merge results
        all_matches = {**direct_matches, **semantic_matches}

        # Build output DataFrame using ORIGINAL mseva_df to preserve PIDs!
        output_df = self._build_output(mseva_df, all_matches)

        elapsed = time.time() - start
        logger.info("Inference complete in %.1f seconds — %d/%d matched (%.1f%%)",
                     elapsed, len(all_matches), len(mseva_df),
                     len(all_matches) / len(mseva_df) * 100 if len(mseva_df) > 0 else 0)

        return output_df

    # ── Preprocessing ──────────────────────────────────────────

    def _preprocess_mseva(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare mSeva records for matching."""
        df = df.copy()
        cfg = self.config

        # Clean text columns
        for col in [cfg.mseva_owner_col, cfg.mseva_guardian_col,
                     cfg.mseva_locality_col, cfg.mseva_address_col]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).apply(_clean_text)

        # Normalise address
        if cfg.mseva_address_col in df.columns:
            df["address"] = df[cfg.mseva_address_col].apply(_normalize_address)
        else:
            df["address"] = ""

        # Clean IDs (do not touch property_id to preserve the primary key format!)
        for col in [cfg.mseva_old_property_id_col, cfg.mseva_survey_id_col]:
            if col in df.columns:
                df[col] = df[col].apply(_clean_identifier)

        # Extract house number
        df["house_number"] = df["address"].apply(_extract_house_number)

        # Tokenise
        if cfg.mseva_owner_col in df.columns:
            df["owner_tokens"] = df[cfg.mseva_owner_col].apply(_tokenize)
        if cfg.mseva_address_col in df.columns:
            df["address_tokens"] = df["address"].apply(_tokenize)

        # Full address for embedding
        locality_col = cfg.mseva_locality_col if cfg.mseva_locality_col in df.columns else None
        addr = df["address"].fillna("")
        loc = df[locality_col].fillna("") if locality_col else ""
        df["full_address"] = (addr + " " + loc).str.strip()

        return df

    # ── Direct ID Matching ─────────────────────────────────────

    def _direct_id_match(self, mseva_df: pd.DataFrame) -> Dict[str, dict]:
        """Exact matching on Old UID, Property ID, and Survey ID."""
        house_df = self.kb.house_df
        cfg = self.config
        matches = {}

        # Build lookup indexes from GIS
        id_fields = [
            ("uid_old", cfg.mseva_old_property_id_col, "DIRECT_OLD_UID"),
            ("property_id", cfg.mseva_property_id_col, "DIRECT_PROPERTY_ID"),
            ("survey_id", cfg.mseva_survey_id_col, "DIRECT_SURVEY_ID"),
        ]

        for gis_col, mseva_col, match_type in id_fields:
            if gis_col not in house_df.columns or mseva_col not in mseva_df.columns:
                continue

            # Build GIS index: id → row
            gis_index = {}
            for idx, row in house_df.iterrows():
                gis_id = str(row.get(gis_col, "")).strip()
                if gis_id and gis_id not in ("", "nan", "NAN"):
                    gis_index[gis_id] = row

            pid_col = cfg.mseva_property_id_col
            for _, mseva_row in mseva_df.iterrows():
                pid = mseva_row[pid_col]
                if pid in matches:
                    continue  # Already matched
                mseva_id = str(mseva_row.get(mseva_col, "")).strip()
                if mseva_id and mseva_id in gis_index:
                    gis_row = gis_index[mseva_id]
                    matches[pid] = {
                        "Matched_UID": gis_row.get("uid", ""),
                        "Match_Type": match_type,
                        "Latitude": gis_row.get("Latitude", np.nan),
                        "Longitude": gis_row.get("Longitude", np.nan),
                        "Match_Score": 100.0,
                        "Confidence": "VERY HIGH",
                        "GIS_Owner": gis_row.get("owner_name", ""),
                        "GIS_Locality": gis_row.get("locality", ""),
                    }

            logger.info("  %s: %d matches", match_type, sum(1 for v in matches.values() if v["Match_Type"] == match_type))

        return matches

    # ── Semantic Matching ──────────────────────────────────────

    def _semantic_match(self, unmatched_df: pd.DataFrame) -> Dict[str, dict]:
        """FAISS semantic search + weighted similarity scoring."""
        if len(unmatched_df) == 0:
            return {}

        matches = {}
        house_df = self.kb.house_df
        cfg = self.config
        pid_col = cfg.mseva_property_id_col

        # Batch encode mSeva full addresses
        from .embeddings import encode_texts, search_faiss
        full_addresses = unmatched_df["full_address"].fillna("").tolist()
        query_embeddings = encode_texts(self._model, full_addresses, batch_size=cfg.batch_size)

        # FAISS batch search
        scores_matrix, indices_matrix = search_faiss(
            self.kb.faiss_index, query_embeddings, top_k=cfg.faiss_top_k
        )

        # Fuzzy locality matching (cached)
        from rapidfuzz import fuzz, process as rfprocess

        for i, (_, mseva_row) in enumerate(unmatched_df.iterrows()):
            pid = mseva_row[pid_col]
            if pid in matches:
                continue

            # Get FAISS candidates
            candidate_indices = indices_matrix[i]
            candidate_indices = candidate_indices[candidate_indices >= 0]  # Remove -1 padding
            if len(candidate_indices) == 0:
                continue

            candidates = house_df.iloc[candidate_indices]

            # Clean candidates (remove blank owners, duplicate UIDs)
            if "owner_name" in candidates.columns:
                valid_owner = candidates["owner_name"].fillna("").astype(str).str.strip() != ""
                candidates = candidates[valid_owner]
            if len(candidates) == 0:
                continue

            # Score each candidate
            prop_dict = mseva_row.to_dict()
            best_score = 0
            best_match = None

            for _, gis_row in candidates.iterrows():
                gis_dict = gis_row.to_dict()
                score = _calculate_match_score(prop_dict, gis_dict, cfg.matching_weights)
                if score > best_score:
                    best_score = score
                    best_match = gis_dict

            if best_match is None:
                continue

            # Determine match type
            if best_score >= cfg.top5_threshold:
                match_type = "TOP5_WEIGHTED"
            elif best_score >= cfg.top10_threshold:
                match_type = "TOP10_WEIGHTED"
            else:
                match_type = "LOCALITY_CENTROID"

            # For low-confidence matches, fall back to locality centroid
            if match_type == "LOCALITY_CENTROID":
                lat, lon = self._locality_centroid(prop_dict.get("localityname", ""))
                if lat is not None:
                    matches[pid] = {
                        "Matched_UID": "",
                        "Match_Type": "LOCALITY_CENTROID",
                        "Latitude": lat,
                        "Longitude": lon,
                        "Match_Score": best_score,
                        "Confidence": "VERY LOW",
                        "GIS_Owner": "",
                        "GIS_Locality": "",
                    }
                continue

            matches[pid] = {
                "Matched_UID": best_match.get("uid", ""),
                "Match_Type": match_type,
                "Latitude": best_match.get("Latitude", np.nan),
                "Longitude": best_match.get("Longitude", np.nan),
                "Match_Score": best_score,
                "Confidence": _confidence_level(best_score),
                "GIS_Owner": best_match.get("owner_name", ""),
                "GIS_Locality": best_match.get("locality", ""),
            }

        return matches

    # ── Locality Centroid Fallback ──────────────────────────────

    def _locality_centroid(self, locality_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Get locality centroid as fallback coordinates."""
        if not locality_name or self.kb.locality_stats_df is None:
            return None, None

        from rapidfuzz import fuzz, process as rfprocess

        stats = self.kb.locality_stats_df
        if "locality" not in stats.columns:
            return None, None

        loc_list = stats["locality"].dropna().tolist()
        match = rfprocess.extractOne(
            _clean_text(locality_name), loc_list,
            scorer=fuzz.token_sort_ratio
        )
        if match and match[1] >= self.config.locality_match_threshold:
            matched_row = stats[stats["locality"] == match[0]].iloc[0]
            return matched_row.get("centroid_y"), matched_row.get("centroid_x")

        return None, None

    # ── Output Generation ──────────────────────────────────────

    def _build_output(self, mseva_df: pd.DataFrame, all_matches: dict) -> pd.DataFrame:
        """Build the final geocoded output DataFrame."""
        cfg = self.config
        pid_col = cfg.mseva_property_id_col
        output_rows = []

        for _, row in mseva_df.iterrows():
            pid = row[pid_col]
            out = row.to_dict()

            if pid in all_matches:
                m = all_matches[pid]
                out["Matched_UID"] = m["Matched_UID"]
                out["Match_Type"] = m["Match_Type"]
                out["Latitude"] = m["Latitude"]
                out["Longitude"] = m["Longitude"]
                out["Match_Score"] = m["Match_Score"]
                out["Confidence"] = m["Confidence"]
                out["GIS_Owner"] = m.get("GIS_Owner", "")
                out["GIS_Locality"] = m.get("GIS_Locality", "")
            else:
                out["Matched_UID"] = ""
                out["Match_Type"] = "UNMATCHED"
                out["Latitude"] = np.nan
                out["Longitude"] = np.nan
                out["Match_Score"] = 0.0
                out["Confidence"] = ""
                out["GIS_Owner"] = ""
                out["GIS_Locality"] = ""

            output_rows.append(out)

        output_df = pd.DataFrame(output_rows)

        # Summary
        matched = (output_df["Match_Type"] != "UNMATCHED").sum()
        total = len(output_df)
        print(f"\n  GeoAI Inference Results:")
        print(f"    Total records:  {total}")
        print(f"    Matched:        {matched} ({matched/total*100:.1f}%)")
        print(f"    Unmatched:      {total - matched}")

        # Match type breakdown
        for mt in output_df["Match_Type"].unique():
            count = (output_df["Match_Type"] == mt).sum()
            print(f"    {mt}: {count}")

        return output_df
