"""
Layer 4: Name + Locality matching.

Matches remaining unmatched mSeva records against GIS polygons in the same
locality using fuzzy name comparison and optional locality crosswalk maps.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Set

import pandas as pd

from ..config import CityConfig, MatchResult
from ..helpers import normalize_locality, normalize_name, token_set_ratio

logger = logging.getLogger(__name__)


def match_name_locality(
    mseva_df: pd.DataFrame,
    gis_records: list,
    config: CityConfig,
    existing_matches: dict,
    crosswalk_path: str = None,
) -> Dict[str, MatchResult]:
    """Layer 4: Name + Locality matching.

    Parameters
    ----------
    mseva_df : pd.DataFrame
        Enriched mSeva records.
    gis_records : list[dict]
        GIS attribute dicts (one per polygon).
    config : CityConfig
        City-specific configuration.
    existing_matches : dict
        Already-matched ``{propertyid: MatchResult}`` — skipped here.
    crosswalk_path : str, optional
        Path to locality crosswalk CSV.

    Returns
    -------
    dict[str, MatchResult]
        New matches found by this layer.
    """
    mcols = config.mseva_columns
    gcols = config.gis_columns

    pid_col = mcols.get("property_id", "propertyid")
    uid_field = gcols.get("uid", "UID")
    gis_owner_field = gcols.get("owner", "Owner_Name")
    gis_guardian_field = gcols.get("guardian", "Father_Hus")
    gis_locality_field = gcols.get("locality", "Locality")

    # ── Build locality index ─────────────────────────────────────────────
    loc_idx: Dict[str, list] = defaultdict(list)
    for i, r in enumerate(gis_records):
        loc = normalize_locality(r.get(gis_locality_field, ""), config.name)
        if loc:
            loc_idx[loc].append(i)

    # ── Load locality crosswalk ──────────────────────────────────────────
    loc_crosswalk: Dict[str, Set[str]] = defaultdict(set)
    if crosswalk_path and os.path.exists(crosswalk_path):
        try:
            cw = pd.read_csv(crosswalk_path)
            # Find columns dynamically (expect mseva_normalized/gis_match/score or similar)
            m_col = next((c for c in cw.columns if 'mseva' in c.lower()), cw.columns[0])
            g_col = next((c for c in cw.columns if 'gis' in c.lower()), cw.columns[1])
            s_col = next((c for c in cw.columns if 'score' in c.lower()), cw.columns[2])

            for _, row in cw.iterrows():
                ml = str(row[m_col]).strip()
                gl = str(row[g_col]).strip()
                score = int(row[s_col])
                if score >= 60:
                    loc_crosswalk[ml].add(gl)
            logger.info("Locality crosswalk loaded from file: %d mappings", len(loc_crosswalk))
        except Exception as e:
            logger.error("Error loading locality crosswalk from file: %s", e)

    if not loc_crosswalk:
        logger.info("Building locality crosswalk dynamically...")
        mseva_locality_col = mcols.get("locality", "localityname")
        loc_col_to_use = 'parsed_locality' if 'parsed_locality' in mseva_df.columns else mseva_locality_col
        if loc_col_to_use in mseva_df.columns:
            mseva_locs = set(mseva_df[loc_col_to_use].dropna().unique())
            gis_locs = set(r.get(gis_locality_field, "") for r in gis_records if r.get(gis_locality_field))

            for ml in mseva_locs:
                ml_norm = normalize_locality(ml, config.name)
                if not ml_norm:
                    continue
                ml_tokens = set(ml_norm.split())
                if not ml_tokens:
                    continue

                for gl in gis_locs:
                    gl_norm = normalize_locality(gl, config.name)
                    if not gl_norm:
                        continue
                    gl_tokens = set(gl_norm.split())
                    if not gl_tokens:
                        continue

                    # Exact subset match or 70%+ token overlap
                    if ml_tokens.issubset(gl_tokens):
                        loc_crosswalk[ml_norm].add(gl_norm)
                    elif len(ml_tokens & gl_tokens) / len(ml_tokens) >= 0.7:
                        loc_crosswalk[ml_norm].add(gl_norm)
            logger.info("Dynamic locality crosswalk built: %d entries", len(loc_crosswalk))

    # ── Scan unmatched mSeva records ────────────────────────────────────
    results: Dict[str, MatchResult] = {}
    name_confirmed = 0
    name_strong_single = 0
    name_crosswalk = 0
    name_rejected = 0
    # Ticket 1.25: an mSeva locality with NO matching GIS locality (exact or
    # crosswalk) must never be silently dropped -- tracked here and reported
    # loudly below instead of disappearing into the generic rejected count.
    unknown_locality_counts: Counter = Counter()

    mseva_owner_col = mcols.get("owner", "ownername")
    mseva_guardian_col = mcols.get("guardian", "guardianname")
    mseva_all_owners_col = mcols.get("all_owners", "all_ownernames")
    mseva_all_guardians_col = mcols.get("all_guardians", "all_guardiannames")
    mseva_locality_col = mcols.get("locality", "localityname")

    remaining_pids = [
        str(mseva_df.at[idx, pid_col])
        for idx in mseva_df.index
        if str(mseva_df.at[idx, pid_col]) not in existing_matches
    ]
    total_r = len(remaining_pids)
    logger.info("Name+Locality matching for %d unmatched records...", total_r)

    # (Replaces a full-DataFrame scan that previously ran once per record.)
    mseva_indexed = mseva_df.copy()
    mseva_indexed[pid_col] = mseva_indexed[pid_col].astype(str)
    mseva_indexed = mseva_indexed.set_index(pid_col, drop=False)
    mseva_indexed = mseva_indexed[~mseva_indexed.index.duplicated(keep="first")]

    t_start = time.time()
    for prog, pid in enumerate(remaining_pids):
        if prog % 1000 == 0 and prog > 0:
            el = time.time() - t_start
            eta = (total_r - prog) / (prog / el)
            logger.info(
                "  Progress: %d/%d (%.1f%%) - matches: %d - ETA %.0fs",
                prog, total_r, prog/total_r*100,
                name_confirmed + name_strong_single + name_crosswalk, eta
            )

        row = mseva_indexed.loc[pid]
        mo = normalize_name(row.get(mseva_owner_col, ""))
        mg = normalize_name(row.get(mseva_guardian_col, ""))
        mao = normalize_name(str(row.get(mseva_all_owners_col, "")))
        mag = normalize_name(str(row.get(mseva_all_guardians_col, "")))
        # Prefer parsed_locality over raw localityname if available
        raw_loc = row.get('parsed_locality') if 'parsed_locality' in row.index else row.get(mseva_locality_col, "")
        if pd.isna(raw_loc):
            raw_loc = ""
        ml = normalize_locality(str(raw_loc), config.name)

        if not mo and not mg:
            continue

        # Get candidate GIS localities (exact + crosswalk)
        target_locs = {ml}
        if ml in loc_crosswalk:
            target_locs.update(loc_crosswalk[ml])

        exact_cands = set(loc_idx.get(ml, []))
        
        cands = []
        for tl in target_locs:
            cands.extend(loc_idx.get(tl, []))

        if not cands:
            if ml:
                unknown_locality_counts[ml] += 1
            continue

        # Cap candidates to avoid infinite loops on huge localities
        if len(cands) > 500:
            cands = cands[:500]

        best_ci = None
        best_method = None
        best_combined = 0

        for gi in cands:
            ga = gis_records[gi]
            pass

            go = normalize_name(ga.get(gis_owner_field, ""))
            gf = normalize_name(ga.get(gis_guardian_field, ""))

            # Owner scores
            o_score = token_set_ratio(mo, go)
            ao_score = 0
            if mao:
                for part in mao.split("|"):
                    pn = normalize_name(part)
                    if pn:
                        s = token_set_ratio(pn, go)
                        if s > ao_score:
                            ao_score = s
            owner_best = max(o_score, ao_score)

            # Guardian scores
            g_score = token_set_ratio(mg, gf)
            ag_score = 0
            if mag:
                for part in mag.split("|"):
                    pn = normalize_name(part)
                    if pn:
                        s = token_set_ratio(pn, gf)
                        if s > ag_score:
                            ag_score = s
            guard_best = max(g_score, ag_score)

            # TIER 1: Both owner AND guardian match (strongest)
            if owner_best >= 60 and guard_best >= 60:
                combined = owner_best + guard_best
                if combined > best_combined:
                    best_combined = combined
                    best_ci = gi
                    best_method = "NAME_VERIFIED_BOTH"

            # TIER 2: Owner matches strongly AND guardian has some similarity
            elif owner_best >= 75 and guard_best >= 30:
                combined = owner_best + guard_best
                if combined > best_combined:
                    best_combined = combined
                    best_ci = gi
                    best_method = "NAME_VERIFIED_OWNER+GUARD"

            # TIER 3: Guardian matches strongly AND owner has some similarity
            elif guard_best >= 75 and owner_best >= 30:
                combined = owner_best + guard_best
                if combined > best_combined:
                    best_combined = combined
                    best_ci = gi
                    best_method = "NAME_VERIFIED_GUARD+OWNER"

            # TIER 4: Very strong single match (90+)
            elif owner_best >= 90 or guard_best >= 90:
                combined = max(owner_best, guard_best)
                if combined > best_combined:
                    best_combined = combined
                    best_ci = gi
                    best_method = "NAME_STRONG_SINGLE"

        if best_ci is not None:
            used_crosswalk = best_ci not in exact_cands
            method_label = f"TEXT_{best_method}"
            if used_crosswalk:
                method_label += "_CROSSWALK"

            ga = gis_records[best_ci]
            results[pid] = MatchResult(
                matched_uid=ga[uid_field],
                match_method=method_label,
                gis_owner_name=ga.get(gis_owner_field, ""),
                gis_mobile=ga.get(gcols.get("mobile", "Mobile_No"), ""),
                gis_locality=ga.get(gis_locality_field, ""),
            )

            if used_crosswalk:
                name_crosswalk += 1
            elif "BOTH" in best_method or "OWNER+GUARD" in best_method or "GUARD+OWNER" in best_method:
                name_confirmed += 1
            else:
                name_strong_single += 1
        else:
            name_rejected += 1

    logger.info(
        "Name+Locality matches found: confirmed=%d, single=%d, crosswalk=%d, rejected=%d",
        name_confirmed, name_strong_single, name_crosswalk, name_rejected
    )

    if unknown_locality_counts:
        total_affected = sum(unknown_locality_counts.values())
        top = ", ".join(f"{loc!r} ({n})" for loc, n in unknown_locality_counts.most_common(10))
        logger.warning(
            "Name+Locality: %d unmatched mSeva record(s) reference %d locality name(s) with "
            "NO matching GIS locality (exact match or crosswalk). These are flagged here "
            "rather than silently dropped -- top unmapped localities: %s",
            total_affected, len(unknown_locality_counts), top,
        )

    return results
