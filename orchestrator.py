"""
pipeline/orchestrator.py
════════════════════════════════════════════════════════════════
Main pipeline orchestrator. Runs all steps in sequence or
individual steps on demand.
"""
import os
import time
import json
import logging
import pandas as pd
from typing import Optional, List
from dataclasses import asdict

from .config import CityConfig, PipelineResult
from .data_loader import load_mseva, load_gis, load_electricity, detect_columns
from .helpers import normalize_mobile
from .matchers.mobile import match_mobile
from .matchers.name_locality import match_name_locality
from .matchers.electricity import confirm_defaulters, match_electricity_taxpayers
from .reporting.summary import build_match_register, compute_summary, print_summary

logger = logging.getLogger(__name__)


class PropertyTaxPipeline:
    """
    Orchestrates the property tax defaulter identification pipeline.
    
    Usage:
        config = CityConfig(name="Barnala", state="Punjab",
                            mseva_path="data.csv", gis_path="taxable.shp")
        pipeline = PropertyTaxPipeline(config)
        result = pipeline.run()
        print(result.match_rate)
    """

    VALID_STEPS = [
        'load', 'quality_check', 'train', 'infer', 'match', 'defaulters', 'report'
    ]

    def __init__(self, config: CityConfig):
        self.config = config
        self._mseva_df = None
        self._mseva_columns = None
        self._gis_records = None
        self._polygons = None
        self._tree = None
        self._tree_idx_map = None
        self._gis_columns = None
        self._electricity_df = None
        self._elec_columns = None
        self._match_results = {}
        self._defaulter_list_df = None
        self._defaulters_path = None
        self._start_time = None
        # GeoAI training/inference state
        self._kb = None              # Trained Knowledge Base
        self._geocoded_df = None     # Geocoded output from inference

    def run(self, steps: Optional[List[str]] = None) -> PipelineResult:
        """
        Run the full pipeline or specific steps.
        
        Args:
            steps: List of step names to run. If None, runs all applicable steps.
                   Valid steps: load, quality_check, train, infer, match, defaulters, report
        
        Returns:
            PipelineResult with match statistics and file paths.
        """
        self._start_time = time.time()
        
        if steps is None:
            if self.config.geoai_output_path:
                # Pre-built mode: skip train/infer
                steps = ['load', 'match', 'defaulters', 'report']
            else:
                # Full mode: train KB + run inference
                steps = ['load', 'train', 'infer', 'match', 'defaulters', 'report']
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        self._log_header()

        if 'load' in steps:
            self._step_load()

        if 'train' in steps:
            self._step_train()

        if 'infer' in steps:
            self._step_infer()

        if 'match' in steps:
            self._step_match()

        if 'defaulters' in steps:
            self._step_defaulters()

        result = PipelineResult(
            city=self.config.name,
            total_mseva=len(self._mseva_df) if self._mseva_df is not None else 0,
            total_gis=len(self._gis_records) if self._gis_records is not None else 0,
            matched=len(self._match_results),
            unmatched=len(self._mseva_df) - len(self._match_results) if self._mseva_df is not None else 0,
            elapsed_seconds=time.time() - self._start_time,
        )
        result.defaulter_list = self._defaulter_list_df

        if 'report' in steps and self._mseva_df is not None:
            result = self._step_report(result)

        self._log_footer(result)
        return result

    # ─── Individual Steps ─────────────────────────────────────────────

    def _step_load(self):
        """Step 1: Load all data sources."""
        print("\n" + "=" * 70)
        print("STEP 1: LOADING DATA")
        print("=" * 70)

        # Load mSeva
        print(f"\n  Loading mSeva: {self.config.mseva_path}")
        self._mseva_df, self._mseva_columns = load_mseva(
            self.config.mseva_path, 
            self.config.mseva_columns
        )
        self.config.mseva_columns = self._mseva_columns
        print(f"  → {len(self._mseva_df):,} records loaded")
        print(f"  → Columns detected: {self._mseva_columns}")

        # Load GIS (optional — skipped for parse/geocode-only runs)
        if self.config.gis_path and os.path.exists(self.config.gis_path):
            print(f"\n  Loading GIS: {self.config.gis_path}")
            self._polygons, self._gis_records, self._tree, self._tree_idx_map, self._gis_columns = load_gis(
                self.config.gis_path,
                self.config.gis_columns
            )
            self.config.gis_columns = self._gis_columns
            geo_count = sum(1 for g in self._polygons if g is not None)
            print(f"  → {len(self._gis_records):,} records loaded ({geo_count:,} with geometry)")
            print(f"  → Columns detected: {self._gis_columns}")

            # Auto-detect UTM zone from GIS shapefile projection if not set
            if self.config.utm_zone is None and self._polygons:
                detected_zone = None

                # Strategy 1: Read the .prj file for the actual CRS definition
                import re
                prj_path = os.path.splitext(self.config.gis_path)[0] + ".prj"
                if os.path.exists(prj_path):
                    try:
                        with open(prj_path, "r") as f:
                            prj_text = f.read()
                        # Look for "UTM_Zone_XXN" or "UTM_Zone_XXS" pattern
                        m = re.search(r'UTM_Zone_(\d+)[NS]', prj_text, re.IGNORECASE)
                        if m:
                            detected_zone = int(m.group(1))
                        else:
                            # Try to extract Central_Meridian and compute zone
                            m = re.search(r'Central_Meridian["\s,]+(-?[\d.]+)', prj_text)
                            if m:
                                cm = float(m.group(1))
                                detected_zone = int((cm + 183) / 6)
                    except Exception as e:
                        print(f"  ⚠ Could not read .prj file: {e}")

                # Strategy 2: Fall back to coordinate analysis
                if detected_zone is None:
                    sample_polys = [p for p in self._polygons[:min(200, len(self._polygons))] if p is not None][:100]
                    sample_bounds = [p.bounds for p in sample_polys]
                    avg_x = sum(b[0] + b[2] for b in sample_bounds) / (2 * len(sample_bounds))
                    if avg_x > 100000:
                        # Projected CRS — assume zone 43 for Punjab (most common)
                        detected_zone = 43
                        print(f"  ⚠ No .prj file found. Assuming UTM zone 43 for projected data.")
                    else:
                        # WGS-84 geographic coordinates
                        detected_zone = CityConfig.utm_zone_from_longitude(avg_x)

                self.config.utm_zone = detected_zone
                print(f"  → UTM zone: {detected_zone} (auto-detected from GIS shapefile)")

            # Load electricity (optional)
            if self.config.electricity_path and os.path.exists(self.config.electricity_path):
                print(f"\n  Loading Electricity: {self.config.electricity_path}")
                self._electricity_df, self._elec_columns = load_electricity(
                    self.config.electricity_path,
                    self.config.elec_columns
                )
                self.config.elec_columns = self._elec_columns
                print(f"  → {len(self._electricity_df):,} records loaded")
            else:
                print("\n  No electricity data provided.")
        else:
            if self.config.gis_path:
                print(f"\n  ⚠ GIS path not found: {self.config.gis_path}")
            else:
                print("\n  ℹ No GIS shapefile provided — running in parse/geocode-only mode.")


    def _step_train(self):
        """Step T: Train GeoAI Knowledge Base (skipped if cached)."""
        print("\n" + "=" * 70)
        print("STEP T: GEOAI KNOWLEDGE BASE TRAINING")
        print("=" * 70)

        from .geoai.trainer import GeoAITrainer, TrainingConfig
        from .geoai import knowledge_base as kb_mod

        # Default to pipeline_geoai/models/CityName so it stays neatly inside the working directory
        kb_path = self.config.model_dir or os.path.join("pipeline_geoai", "models", self.config.name)

        # Check cache
        gis_path = self.config.gis_path
        if not self.config.force_retrain and kb_mod.is_valid(kb_path, gis_path):
            print(f"  → KB cached at {kb_path} — skipping training")
            self._kb = kb_mod.load(kb_path)
            return

        if not gis_path or not os.path.exists(gis_path):
            raise RuntimeError(
                f"GIS path required for training but not found: {gis_path}\n"
                "Provide --gis or use --geoai-output for pre-built mode."
            )

        trainer = GeoAITrainer(TrainingConfig())
        self._kb = trainer.train(gis_path)
        kb_mod.save(self._kb, kb_path)
        print(f"  → KB saved to {kb_path}")

    def _step_infer(self):
        """Step I: Run GeoAI inference to geocode mSeva records."""
        print("\n" + "=" * 70)
        print("STEP I: GEOAI INFERENCE")
        print("=" * 70)

        if self._kb is None:
            raise RuntimeError("Knowledge Base not loaded. Run 'train' step first.")
        if self._mseva_df is None:
            raise RuntimeError("mSeva data not loaded. Run 'load' step first.")

        from .geoai.inferencer import GeoAIInferencer, InferenceConfig

        inferencer = GeoAIInferencer(self._kb, InferenceConfig())
        self._geocoded_df = inferencer.infer(self._mseva_df)

        # Save intermediate output for debugging
        out_path = os.path.join(
            self.config.output_dir, f"{self.config.name}_GeoAI_Geocoded.csv"
        )
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._geocoded_df.to_csv(out_path, index=False)
        print(f"  → Geocoded CSV saved: {out_path}")

    def _step_match(self):
        """Step 2: GeoAI matching with funneling to downstream layers."""
        if self._mseva_df is None:
            raise RuntimeError("Data not loaded. Call run() with 'load' step first.")

        print("\n" + "=" * 70)
        print("STEP 2: MATCHING (GeoAI + funneling)")
        print("=" * 70)

        # Layer 1: GeoAI (primary — exact ID + semantic + spatial)
        print("\n  ── Layer 1: GeoAI Matching ──")

        if self._geocoded_df is not None:
            # In-memory mode (from _step_infer)
            print("  Using in-memory geocoded output from inference step")
            geoai_matches = self._load_geoai_from_dataframe(self._geocoded_df)
        elif self.config.geoai_output_path:
            # Pre-built CSV mode (backward compatible)
            print(f"  Loading: {self.config.geoai_output_path}")
            from .geoai_loader import load_geoai_matches, has_mobile_coverage as _has_mobile
            geoai_matches = load_geoai_matches(
                self.config.geoai_output_path,
                self._gis_records, self._gis_columns
            )
        else:
            print("  ⚠ No GeoAI output available — skipping Layer 1")
            geoai_matches = {}

        self._match_results.update(geoai_matches)
        print(f"  → {len(geoai_matches):,} GeoAI matches")
        print(f"  → Running total: {len(self._match_results):,}")

        # Layer 2: Mobile Number (only if coverage is sufficient)
        mobile_rate = self._check_mobile_coverage()
        if mobile_rate >= 0.30:
            print(f"\n  ── Layer 2: Mobile Number (coverage: {mobile_rate:.0%}) ──")
            mobile_matches = match_mobile(
                self._mseva_df, self._gis_records,
                self.config, self._match_results
            )
            self._match_results.update(mobile_matches)
            print(f"  → {len(mobile_matches):,} mobile matches")
            print(f"  → Running total: {len(self._match_results):,}")
        else:
            print(f"\n  ── Layer 2: Mobile Number — SKIPPED (coverage: {mobile_rate:.0%} < 30%) ──")

        # Layer 3: Electricity (only if data is available)
        if self._electricity_df is not None:
            print("\n  ── Layer 3: Electricity Coordinates ──")
            elec_matches = match_electricity_taxpayers(
                self._mseva_df, self._gis_records, self._polygons,
                self._tree, self._electricity_df, self.config, self._match_results,
                tree_idx_map=self._tree_idx_map,
            )
            self._match_results.update(elec_matches)
            print(f"  → {len(elec_matches):,} electricity matches")
            print(f"  → Running total: {len(self._match_results):,}")
        else:
            print("\n  ── Layer 3: Electricity — SKIPPED (no data provided) ──")

        # Layer 4: Name + Locality (fuzzy fallback)
        print("\n  ── Layer 4: Name + Locality ──")
        crosswalk_path = self._find_crosswalk()
        name_matches = match_name_locality(
            self._mseva_df, self._gis_records,
            self.config, self._match_results,
            crosswalk_path=crosswalk_path
        )
        self._match_results.update(name_matches)
        print(f"  → {len(name_matches):,} name+locality matches")
        print(f"  → Running total: {len(self._match_results):,}")

        total_matched = len(self._match_results)
        total_mseva = len(self._mseva_df)
        print(f"\n  ────────────────────────────────────────────────────")
        print(f"  Total matched: {total_matched:,} / {total_mseva:,} ({100.0 * total_matched / total_mseva:.1f}%)")

    def _load_geoai_from_dataframe(self, geocoded_df: pd.DataFrame) -> dict:
        """Build match results from in-memory geocoded DataFrame."""
        from .config import MatchResult
        matches = {}

        for _, row in geocoded_df.iterrows():
            match_type = str(row.get("Match_Type", ""))
            if match_type == "UNMATCHED" or not match_type:
                continue

            pid = str(row.get("propertyid", row.get("property_id", "")))
            if not pid or pid in matches:
                continue

            matched_uid = str(row.get("Matched_UID", "")).strip()
            if matched_uid.lower() in ("nan", "none") or not matched_uid:
                continue

            lat = row.get("Latitude", None)
            lon = row.get("Longitude", None)

            if pd.isna(lat) or pd.isna(lon):
                continue

            matches[pid] = MatchResult(
                matched_uid=matched_uid,
                match_method=f"GEOAI_{match_type.upper()}",
                gis_owner_name=str(row.get("GIS_Owner", "")),
                gis_mobile="", # Mobile is not populated by GeoAIInferencer
                gis_locality=str(row.get("GIS_Locality", "")),
                confidence=float(row.get("Match_Score", 0)),
            )

        return matches

    def _check_mobile_coverage(self) -> float:
        """Check mobile number coverage in mSeva data."""
        if self._mseva_df is None or self._mseva_columns is None:
            return 0.0
        mobile_col = self._mseva_columns.get("mobile", "mobileno")
        if mobile_col not in self._mseva_df.columns:
            return 0.0
        valid = self._mseva_df[mobile_col].dropna().astype(str).str.strip()
        valid = valid[valid.str.len() >= 10]
        return len(valid) / len(self._mseva_df) if len(self._mseva_df) > 0 else 0.0

    # ── Helpers ──────────────────────────────────────────────────────

    def _find_crosswalk(self):
        """Locate the locality crosswalk CSV near the mSeva file."""
        crosswalk_path = os.path.join(os.path.dirname(self.config.mseva_path), 'locality_crosswalk.csv')
        if not os.path.exists(crosswalk_path):
            crosswalk_path = os.path.join(os.path.dirname(self.config.mseva_path), 'Enriched Data', 'locality_crosswalk.csv')
        if not os.path.exists(crosswalk_path):
            crosswalk_path = os.path.join(os.path.dirname(os.path.dirname(self.config.mseva_path)), 'Enriched Data', 'locality_crosswalk.csv')
        if not os.path.exists(crosswalk_path):
            crosswalk_path = None
        return crosswalk_path

    def _step_defaulters(self):
        """Step 3: Identify and confirm defaulters."""
        print("\n" + "=" * 70)
        print("STEP 3: DEFAULTER IDENTIFICATION")
        print("=" * 70)

        matched_uids = set(mr.matched_uid for mr in self._match_results.values())
        all_uids = set()
        uid_col = self._gis_columns.get('uid', 'UID')
        for r in self._gis_records:
            uid = r.get(uid_col, r.get('UID', ''))
            if uid:
                all_uids.add(uid)
        
        unmatched_uids = all_uids - matched_uids

        # --- Detect the Exempted field and separate taxable vs exempted ---
        exempted_col = self._gis_columns.get("exempted")
        prop_usage_col = self._gis_columns.get("property_usage")
        prop_type_col = self._gis_columns.get("property_type")

        # Build a quick UID → record lookup for exemption status
        uid_to_record: dict = {}
        for r in self._gis_records:
            uid = r.get(uid_col, r.get('UID', ''))
            if uid:
                uid_to_record[uid] = r

        # Separate unmatched into taxable vs exempted
        taxable_unmatched = set()
        exempted_unmatched = set()
        for uid in unmatched_uids:
            rec = uid_to_record.get(uid, {})
            exempt_val = rec.get(exempted_col, "").strip().lower() if exempted_col else ""
            if exempt_val == "exempted":
                exempted_unmatched.add(uid)
            else:
                taxable_unmatched.add(uid)

        print(f"  Total GIS polygons:   {len(all_uids):,}")
        print(f"  Matched polygons:     {len(matched_uids):,}")
        print(f"  Unmatched polygons:   {len(unmatched_uids):,}")
        if exempted_col:
            print(f"    ├─ Taxable (defaulters):  {len(taxable_unmatched):,}")
            print(f"    └─ Exempted (excluded):   {len(exempted_unmatched):,}")
        else:
            print(f"  ⚠ No 'Exempted' field found — all unmatched treated as potential defaulters")

        # Electricity confirmation (optional — only when electricity data is present)
        confirmed: dict = {}
        if self._electricity_df is not None:
            print("\n  Running electricity confirmation...")
            confirmed = confirm_defaulters(
                self._electricity_df, self._gis_records,
                self._polygons, self._tree, self.config,
                matched_uids, self._elec_columns, self._gis_columns,
                tree_idx_map=self._tree_idx_map,
            )
            # Only count confirmations that are in the taxable set
            if exempted_col:
                confirmed = {uid: v for uid, v in confirmed.items() if uid in taxable_unmatched}
            print(f"  → {len(confirmed):,} confirmed defaulters (occupied, no tax record)")
        else:
            print("  No electricity data — all defaulters remain 'potential'.")

        # Persist the unified GIS-parcel register to disk — EVERY parcel
        # loaded (matched, suspected/potential, confirmed-occupied, and
        # exempt), per the Data Dictionary's tax_status/geo_status contract.
        gcols = self._gis_columns or {}
        g_owner = gcols.get("owner", "Owner_Name")
        g_guardian = gcols.get("guardian", "Father_Hus")
        g_mobile = gcols.get("mobile", "Mobile_No")
        g_locality = gcols.get("locality", "Locality")

        ward_col = gcols.get("ward")
        if not ward_col:
            logger.warning(
                "No ward column detected for GIS data (city=%s) — 'ward_id' "
                "will be left empty for every row in this run.",
                self.config.name,
            )

        from .helpers import utm_to_wgs84

        # Fix 2 contract columns: relabel the existing matched / taxable-
        # unmatched (POTENTIAL/CONFIRMED_OCCUPIED) / exempted-unmatched
        # outcome into the 3-way tax_status enum — the matching itself is
        # not re-derived, only relabeled per-parcel.
        parcel_meta = []
        for r in self._gis_records:
            uid = r.get(uid_col, r.get('UID', ''))
            conf = confirmed.get(uid, {}) if uid else {}
            if uid and uid in matched_uids:
                tax_status, status = "IN_TAX_NET", "MATCHED"
            elif uid in exempted_unmatched:
                tax_status, status = "EXEMPT", "EXEMPT"
            else:
                tax_status = "SUSPECTED"
                status = "CONFIRMED_OCCUPIED" if conf else "POTENTIAL"
            parcel_meta.append({
                "tax_status": tax_status,
                "status": status,
                "property_uid": f"GIS:{uid}",
                "ward_id": r.get(ward_col, "") if ward_col else "",
            })

        defaulter_rows = []
        for i, r in enumerate(self._gis_records):
            uid = r.get(uid_col, r.get('UID', ''))
            meta = parcel_meta[i]
            conf = confirmed.get(uid, {}) if uid else {}

            # Compute polygon centroid → WGS84 lat/lon. `i` indexes directly
            # into self._polygons since both lists are built in parallel by
            # data_loader.load_gis (one entry per successfully-loaded parcel).
            centroid_lat, centroid_lon = "", ""
            has_polygon = False
            if i < len(self._polygons):
                try:
                    c = self._polygons[i].centroid
                    clat, clon = utm_to_wgs84(c.x, c.y, self.config.utm_zone)
                    centroid_lat = round(clat, 6)
                    centroid_lon = round(clon, 6)
                    has_polygon = True
                except Exception:
                    pass
            geo_status = "SHAPE" if has_polygon else ("DOT" if centroid_lat != "" else "NONE")

            row_data = {
                "gis_uid": uid,
                "property_uid": meta["property_uid"],
                "gis_owner_name": r.get(g_owner, ""),
                "gis_guardian_name": r.get(g_guardian, ""),
                "gis_mobile": r.get(g_mobile, ""),
                "gis_locality": r.get(g_locality, ""),
                "ward_id": meta["ward_id"],
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "property_usage": r.get(prop_usage_col, "") if prop_usage_col else "",
                "property_type": r.get(prop_type_col, "") if prop_type_col else "",
                "status": meta["status"],
                "tax_status": meta["tax_status"],
                "geo_status": geo_status,
                "electricity_account_no": conf.get("account_no", ""),
                "electricity_holder_name": conf.get("holder_name", ""),
            }
            defaulter_rows.append(row_data)

        defaulter_df = pd.DataFrame(defaulter_rows)
        defaulters_path = os.path.join(
            self.config.output_dir, f"{self.config.name}_Defaulters.csv"
        )
        defaulter_df.to_csv(defaulters_path, index=False)
        status_counts = defaulter_df["tax_status"].value_counts().to_dict() if len(defaulter_df) else {}
        print(
            f"  GIS parcel register saved: {defaulters_path} "
            f"({len(defaulter_df):,} total parcels — {status_counts})"
        )

        # ── GeoJSON output with full polygon geometry (every parcel) ──
        geojson_features = []
        for i, r in enumerate(self._gis_records):
            uid = r.get(uid_col, r.get('UID', ''))
            meta = parcel_meta[i]
            conf = confirmed.get(uid, {}) if uid else {}

            # Convert polygon coordinates from UTM → WGS84
            geometry = None
            has_polygon = False
            if i < len(self._polygons):
                try:
                    poly = self._polygons[i]
                    # Convert exterior ring
                    exterior_coords = []
                    for x, y in poly.exterior.coords:
                        lat, lon = utm_to_wgs84(x, y, self.config.utm_zone)
                        exterior_coords.append([round(lon, 7), round(lat, 7)])  # GeoJSON is [lon, lat]
                    
                    # Convert interior rings (holes), if any
                    interior_rings = []
                    for interior in poly.interiors:
                        ring_coords = []
                        for x, y in interior.coords:
                            lat, lon = utm_to_wgs84(x, y, self.config.utm_zone)
                            ring_coords.append([round(lon, 7), round(lat, 7)])
                        interior_rings.append(ring_coords)
                    
                    coordinates = [exterior_coords] + interior_rings
                    geometry = {"type": "Polygon", "coordinates": coordinates}
                    has_polygon = True
                except Exception:
                    pass  # geometry stays None — valid GeoJSON

            geo_status = "SHAPE" if has_polygon else "NONE"

            properties = {
                "gis_uid": str(uid),
                "property_uid": meta["property_uid"],
                "gis_owner_name": str(r.get(g_owner, "")),
                "gis_guardian_name": str(r.get(g_guardian, "")),
                "gis_mobile": str(r.get(g_mobile, "")),
                "gis_locality": str(r.get(g_locality, "")),
                "ward_id": str(meta["ward_id"]),
                "property_usage": str(r.get(prop_usage_col, "")) if prop_usage_col else "",
                "property_type": str(r.get(prop_type_col, "")) if prop_type_col else "",
                "status": meta["status"],
                "tax_status": meta["tax_status"],
                "geo_status": geo_status,
                "electricity_account_no": str(conf.get("account_no", "")),
                "electricity_holder_name": str(conf.get("holder_name", "")),
            }

            geojson_features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            })

        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features,
        }
        geojson_path = os.path.join(
            self.config.output_dir, f"{self.config.name}_Defaulters.geojson"
        )
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)
        print(f"  GeoJSON saved: {geojson_path} ({len(geojson_features):,} features, every GIS parcel)")

        self._defaulter_list_df = defaulter_df
        self._defaulters_path = defaulters_path

    def _step_report(self, result: PipelineResult) -> PipelineResult:
        """Step 4: Build match register and generate reports."""
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING REPORTS")
        print("=" * 70)

        # Build match register
        register = build_match_register(
            self._mseva_df, self._match_results,
            self.config, self._mseva_columns
        )

        # Fix 2 contract column: stable join key so this file can be joined
        # against other months' outputs.
        register["property_uid"] = register["propertyid"].astype(str).map(lambda pid: f"PT:{pid}")

        register_path = os.path.join(
            self.config.output_dir, 
            f"{self.config.name}_Match_Register.csv"
        )
        register.to_csv(register_path, index=False)
        print(f"  Match register saved: {register_path}")

        # Compute summary
        summary = compute_summary(register, self._gis_records, self.config, self._gis_columns)
        print_summary(summary)

        # Save summary JSON
        summary_path = os.path.join(
            self.config.output_dir, 
            f"{self.config.name}_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")

        # Update result
        result.matched = summary['matched_count']
        result.unmatched = summary['unmatched_count']
        result.defaulters = summary.get('potential_defaulters', 0)
        result.step_stats = summary['layer_breakdown']
        result.output_files = [register_path, summary_path]
        if self._defaulters_path:
            result.output_files.append(self._defaulters_path)

        return result

    # ─── Logging ──────────────────────────────────────────────────────

    def _log_header(self):
        """Print pipeline header."""
        print("\n" + "═" * 70)
        print(f"  PROPERTY TAX DEFAULTER IDENTIFICATION PIPELINE")
        print(f"  City: {self.config.name}, {self.config.state}")
        print("═" * 70)

    def _log_footer(self, result: PipelineResult):
        """Print pipeline footer."""
        elapsed = time.time() - self._start_time
        print("\n" + "═" * 70)
        print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
        if result.match_rate > 0:
            print(f"  Match rate: {result.match_rate:.1%}")
            print(f"  Matched: {result.matched:,} / {result.total_mseva:,}")
        print("═" * 70)
