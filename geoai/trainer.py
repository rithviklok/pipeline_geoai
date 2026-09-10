"""
GeoAI Training Engine.

Refactored from train_geoai_geocoder.py (5,187 lines → ~400 lines).
Converts GIS data into a searchable Knowledge Base with embeddings and
spatial indexes.

Bug fixes applied:
  - DBSCAN eps=50 now operates on UTM-projected metres, not WGS84 degrees
  - FAISS index uses IndexFlatIP (cosine) instead of IndexFlatL2
"""

import os
import re
import json
import time
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree, BallTree

from .columns import standardise_gis_columns, clean_column_names
from .embeddings import load_model, encode_texts, build_faiss_index
from .knowledge_base import KnowledgeBase, compute_gis_hash, save

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """All tunable training parameters (previously hardcoded constants)."""

    # Sentence Transformer
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64

    # DBSCAN for locality polygon generation
    dbscan_eps_metres: float = 50.0       # Bug fix: now in metres, not degrees
    dbscan_min_samples: int = 2
    min_properties_for_polygon: int = 3   # Skip localities with < 3 points

    # Coordinate validation
    min_latitude: float = -90.0
    max_latitude: float = 90.0
    min_longitude: float = -180.0
    max_longitude: float = 180.0

    # Coordinate outlier detection (std deviations)
    outlier_std_multiplier: float = 3.0

    # Locality matching
    locality_match_threshold: int = 75

    # Spatial tree
    leaf_size: int = 40

    # Column mapping overrides (optional)
    custom_column_mapping: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Text Cleaning Utilities
# ═══════════════════════════════════════════════════════════════

def clean_text(value) -> str:
    """Clean and normalise a text field."""
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    for ch in ',.\\-/()\';:"_#|':
        value = value.replace(ch, " ")
    return " ".join(value.split())


def normalize_address(address: str) -> str:
    """Municipal address normalisation (Punjab conventions)."""
    address = clean_text(address)
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


def extract_house_number(address) -> str:
    """Extract house/shop number from address string."""
    if pd.isna(address):
        return ""
    address = str(address).upper()
    patterns = [
        r'(SHOP[\-\s]?\d+[A-Z]*)',
        r'(SCO[\-\s]?\d+[A-Z]*)',
        r'(MC[\-\s]?\d+[A-Z]*)',
        r'(\d+[A-Z]?\/?\d*)',
        r'(\d+\-\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, address)
        if match:
            return match.group(1)
    return ""


def clean_identifier(value) -> str:
    """Normalise a property/survey ID for exact matching."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    if value.endswith(".0"):
        value = value[:-2]
    value = value.replace("-", "").replace("/", "").replace("_", "").replace(" ", "")
    return value


def tokenize_text(value) -> list:
    """Tokenise cleaned text into words ≥ 2 characters."""
    tokens = clean_text(value).split()
    return [t for t in tokens if len(t) >= 2]


# ═══════════════════════════════════════════════════════════════
# Trainer Class
# ═══════════════════════════════════════════════════════════════

class GeoAITrainer:
    """Trains a GeoAI Knowledge Base from GIS data.

    Usage:
        trainer = GeoAITrainer(TrainingConfig())
        kb = trainer.train("path/to/gis.shp")
        save(kb, "models/Barnala/")
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self._house_df: pd.DataFrame = None
        self._gdf: gpd.GeoDataFrame = None
        self._kb = KnowledgeBase()
        self._stats = {}

    def train(
        self,
        gis_path: str,
        locality_shp_path: str = None,
    ) -> KnowledgeBase:
        """Run the full training pipeline.

        Parameters
        ----------
        gis_path : str
            Path to GIS data (shapefile, geodatabase, geopackage, or CSV with lat/lon).
        locality_shp_path : str, optional
            Path to official locality polygon shapefile. If None, locality
            polygons are auto-generated from DBSCAN clustering.

        Returns
        -------
        KnowledgeBase
        """
        start = time.time()
        logger.info("Starting GeoAI Training...")

        self._load_gis(gis_path)
        self._clean_and_normalize()
        self._create_geodataframe()
        self._learn_locality_polygons(locality_shp_path)
        self._generate_embeddings()
        self._build_spatial_indexes()
        self._compute_statistics(gis_path)

        elapsed = time.time() - start
        logger.info("Training complete in %.1f seconds", elapsed)
        self._kb.training_config["training_time_seconds"] = round(elapsed, 1)

        return self._kb

    # ── Step 1: Load GIS ──────────────────────────────────────

    def _load_gis(self, gis_path: str):
        """Load GIS data from shapefile/geodatabase/geopackage/CSV."""
        ext = os.path.splitext(gis_path)[1].lower()
        print(f"\n  Loading GIS: {gis_path}")

        if ext == ".csv":
            df = pd.read_csv(gis_path)
        elif ext in (".shp", ".gdb", ".gpkg"):
            # Read spatial file, extract centroids as lat/lon
            gdf = gpd.read_file(gis_path)
            # Ensure WGS84
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            # Compute centroids
            centroids = gdf.geometry.centroid
            gdf["Latitude"] = centroids.y
            gdf["Longitude"] = centroids.x
            # Convert to plain DataFrame (drop geometry for now, re-create later)
            df = pd.DataFrame(gdf.drop(columns=["geometry"]))
        else:
            raise ValueError(f"Unsupported GIS format: {ext}. Use .csv, .shp, .gdb, or .gpkg")

        # Standardise column names
        df = standardise_gis_columns(df, self.config.custom_column_mapping)
        logger.info("GIS loaded: %d records, columns: %s", len(df), list(df.columns[:10]))

        self._house_df = df

    # ── Step 2: Clean & Normalise ─────────────────────────────

    def _clean_and_normalize(self):
        """Clean text fields, normalise addresses, extract house numbers."""
        df = self._house_df
        print("  Cleaning and normalising GIS attributes...")

        # Remove invalid coordinates
        cfg = self.config
        valid = df.apply(
            lambda r: (
                not pd.isna(r.get("Latitude")) and not pd.isna(r.get("Longitude"))
                and cfg.min_latitude <= r["Latitude"] <= cfg.max_latitude
                and cfg.min_longitude <= r["Longitude"] <= cfg.max_longitude
            ), axis=1
        )
        invalid_count = (~valid).sum()
        df = df[valid].copy()
        logger.info("Removed %d records with invalid coordinates", invalid_count)

        # Remove duplicate UIDs
        if "uid" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["uid"])
            logger.info("Removed %d duplicate UIDs", before - len(df))

        # Clean text columns
        text_cols = ["owner_name", "locality", "road_name", "house_flat",
                     "address", "father_husband_name"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).apply(clean_text)

        # Clean identifier columns
        for col in ["property_id", "survey_id", "uid_old"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_identifier)

        # Normalise addresses
        if "address" in df.columns:
            df["address"] = df["address"].apply(normalize_address)

        # Extract house numbers
        if "address" in df.columns:
            df["house_number"] = df["address"].apply(extract_house_number)

        # Tokenise
        if "owner_name" in df.columns:
            df["owner_tokens"] = df["owner_name"].apply(tokenize_text)
        if "address" in df.columns:
            df["address_tokens"] = df["address"].apply(tokenize_text)
        if "locality" in df.columns:
            df["locality_tokens"] = df["locality"].apply(tokenize_text)

        # Create full_address
        addr = df.get("address", pd.Series("", index=df.index)).fillna("")
        loc = df.get("locality", pd.Series("", index=df.index)).fillna("")
        road = df.get("road_name", pd.Series("", index=df.index)).fillna("")
        df["full_address"] = (addr + " " + loc + " " + road).str.strip()

        # Coordinate outlier detection
        lat_mean, lat_std = df["Latitude"].mean(), df["Latitude"].std()
        lon_mean, lon_std = df["Longitude"].mean(), df["Longitude"].std()
        m = cfg.outlier_std_multiplier
        outlier_mask = (
            (df["Latitude"] < lat_mean - m * lat_std) |
            (df["Latitude"] > lat_mean + m * lat_std) |
            (df["Longitude"] < lon_mean - m * lon_std) |
            (df["Longitude"] > lon_mean + m * lon_std)
        )
        self._stats["outlier_count"] = int(outlier_mask.sum())
        self._stats["invalid_coords"] = int(invalid_count)

        self._house_df = df
        logger.info("After cleaning: %d records", len(df))

    # ── Step 3: Create GeoDataFrame ────────────────────────────

    def _create_geodataframe(self):
        """Convert cleaned DataFrame to GeoDataFrame with point geometry."""
        df = self._house_df
        df["geometry"] = [
            Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])
        ]
        self._gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        logger.info("Created GeoDataFrame: %d features", len(self._gdf))

    # ── Step 4: Learn Locality Polygons ────────────────────────

    def _learn_locality_polygons(self, locality_shp_path: str = None):
        """Build locality polygon boundaries.

        Uses official locality shapefile if provided, otherwise auto-generates
        from DBSCAN clustering of property coordinates.

        Bug fix: DBSCAN eps=50 is now applied in UTM metres, not WGS84 degrees.
        """
        gdf = self._gdf
        locality_polygons = []
        official_localities = []

        # Load official shapefile if available
        if locality_shp_path and os.path.exists(locality_shp_path):
            print(f"  Loading official locality shapefile: {locality_shp_path}")
            locality_shp = gpd.read_file(locality_shp_path)
            locality_shp = locality_shp[locality_shp.geometry.notna()]
            locality_shp["geometry"] = locality_shp.geometry.buffer(0)

            # Find locality name column
            loc_col = None
            for col in locality_shp.columns:
                if col.lower() in ("locality", "locality_name", "colony", "name"):
                    loc_col = col
                    break

            if loc_col:
                for _, row in locality_shp.iterrows():
                    name = clean_text(row[loc_col])
                    if row.geometry is not None:
                        locality_polygons.append({
                            "locality": name,
                            "geometry": row.geometry,
                            "source": "OFFICIAL_SHP",
                        })
                        official_localities.append(name)
                logger.info("Loaded %d official locality polygons", len(official_localities))

        # Auto-generate polygons for localities not in official shapefile
        gis_localities = gdf["locality"].dropna().unique()
        missing = [loc for loc in gis_localities if loc not in official_localities]
        logger.info("Auto-generating polygons for %d missing localities", len(missing))

        # Detect UTM zone for DBSCAN in metres
        utm_epsg = self._detect_utm_epsg()
        gdf_utm = gdf.to_crs(epsg=utm_epsg) if utm_epsg else gdf

        for locality_name in missing:
            subset = gdf_utm[gdf_utm["locality"] == locality_name].copy()
            if len(subset) < self.config.min_properties_for_polygon:
                continue

            # Extract coordinates (in UTM metres if available)
            coords = np.array([[pt.x, pt.y] for pt in subset.geometry])

            # DBSCAN clustering — eps in metres (bug fix)
            clustering = DBSCAN(
                eps=self.config.dbscan_eps_metres,
                min_samples=self.config.dbscan_min_samples,
            ).fit(coords)

            subset = subset.copy()
            subset["cluster"] = clustering.labels_
            subset = subset[subset["cluster"] != -1]

            if len(subset) < self.config.min_properties_for_polygon:
                continue

            # Convert back to WGS84 for the convex hull
            if utm_epsg:
                subset = subset.to_crs(epsg=4326)

            polygon = subset.unary_union.convex_hull
            locality_polygons.append({
                "locality": locality_name,
                "geometry": polygon,
                "source": "AUTO_GENERATED",
            })

        logger.info("Total locality polygons: %d (official: %d, auto: %d)",
                     len(locality_polygons), len(official_localities),
                     len(locality_polygons) - len(official_localities))

        # Build statistics
        locality_stats = []
        bbox_stats = []
        for pd_item in locality_polygons:
            name = pd_item["locality"]
            poly = pd_item["geometry"]
            centroid = poly.centroid
            minx, miny, maxx, maxy = poly.bounds
            prop_count = len(gdf[gdf["locality"] == name])
            locality_stats.append({
                "locality": name, "property_count": prop_count,
                "area": poly.area, "perimeter": poly.length,
                "centroid_x": centroid.x, "centroid_y": centroid.y,
                "polygon_source": pd_item["source"],
            })
            bbox_stats.append({
                "locality": name, "minx": minx, "miny": miny,
                "maxx": maxx, "maxy": maxy,
            })

        # Store in KB
        if locality_polygons:
            self._kb.locality_polygons_gdf = gpd.GeoDataFrame(
                locality_polygons, geometry="geometry", crs="EPSG:4326"
            )
        self._kb.locality_stats_df = pd.DataFrame(locality_stats)
        self._kb.locality_bbox_df = pd.DataFrame(bbox_stats)

    def _detect_utm_epsg(self) -> Optional[int]:
        """Auto-detect UTM zone from median longitude."""
        try:
            lon = self._gdf["Longitude"].median()
            zone = int((lon + 180) / 6) + 1
            lat = self._gdf["Latitude"].median()
            epsg = 32600 + zone if lat >= 0 else 32700 + zone
            logger.info("UTM zone detected: %d (EPSG:%d)", zone, epsg)
            return epsg
        except Exception:
            logger.warning("Could not detect UTM zone, DBSCAN will use WGS84")
            return None

    # ── Step 5: Generate Embeddings ────────────────────────────

    def _generate_embeddings(self):
        """Generate SentenceTransformer embeddings for all text fields."""
        print("  Generating embeddings...")
        model = load_model(self.config.model_name)
        gdf = self._gdf
        bs = self.config.batch_size

        fields = {
            "address": "address",
            "owner": "owner_name",
            "locality": "locality",
            "road": "road_name",
            "ward": "ward_no",
            "full_address": "full_address",
        }

        for emb_name, col_name in fields.items():
            if col_name in gdf.columns:
                texts = gdf[col_name].fillna("").astype(str).tolist()
                embeddings = encode_texts(model, texts, batch_size=bs)
                setattr(self._kb, f"{emb_name}_embeddings", embeddings)
                print(f"    {emb_name}: {embeddings.shape}")
            else:
                logger.warning("Column '%s' not found — skipping %s embeddings", col_name, emb_name)

    # ── Step 6: Build Spatial Indexes ──────────────────────────

    def _build_spatial_indexes(self):
        """Build KDTree, BallTree, FAISS, and locality RTree."""
        print("  Building spatial indexes...")
        gdf = self._gdf

        # Coordinate matrix
        coords = np.array(list(zip(gdf["Latitude"], gdf["Longitude"]))).astype(np.float32)
        self._kb.coordinates = coords

        # KDTree
        self._kb.kd_tree = KDTree(coords, leaf_size=self.config.leaf_size)
        print("    KDTree: done")

        # BallTree
        self._kb.ball_tree = BallTree(coords, leaf_size=self.config.leaf_size)
        print("    BallTree: done")

        # FAISS (cosine similarity via IndexFlatIP — bug fix)
        if self._kb.full_address_embeddings is not None:
            self._kb.faiss_index = build_faiss_index(self._kb.full_address_embeddings)
            print(f"    FAISS: {self._kb.faiss_index.ntotal} vectors")

        # Locality RTree
        if self._kb.locality_polygons_gdf is not None and len(self._kb.locality_polygons_gdf) > 0:
            try:
                from rtree import index as rtree_index
                idx = rtree_index.Index()
                for i, row in self._kb.locality_polygons_gdf.iterrows():
                    if row.geometry is not None:
                        idx.insert(i, row.geometry.bounds)
                self._kb.locality_rtree = idx
                print(f"    RTree: {len(self._kb.locality_polygons_gdf)} locality polygons indexed")
            except ImportError:
                logger.warning("rtree not installed — skipping locality RTree")

        # House database
        self._kb.house_df = self._gdf.drop(columns=["geometry"], errors="ignore").copy()

    # ── Step 7: Compute Statistics ──────────────────────────────

    def _compute_statistics(self, gis_path: str):
        """Compute and store municipality statistics + training config."""
        gdf = self._gdf
        stats = {
            "total_properties": len(gdf),
            "unique_localities": int(gdf["locality"].nunique()) if "locality" in gdf.columns else 0,
            "unique_roads": int(gdf["road_name"].nunique()) if "road_name" in gdf.columns else 0,
            "unique_wards": int(gdf["ward_no"].nunique()) if "ward_no" in gdf.columns else 0,
            "coordinate_outliers": self._stats.get("outlier_count", 0),
            "invalid_coordinates": self._stats.get("invalid_coords", 0),
        }
        self._kb.training_config = {
            "model_name": self.config.model_name,
            "dbscan_eps_metres": self.config.dbscan_eps_metres,
            "dbscan_min_samples": self.config.dbscan_min_samples,
            "faiss_index_type": "IndexFlatIP",
            "gis_hash": compute_gis_hash(gis_path),
            "gis_path": os.path.basename(gis_path),
            "statistics": stats,
        }

        print(f"\n  Training Summary:")
        print(f"    Properties:  {stats['total_properties']}")
        print(f"    Localities:  {stats['unique_localities']}")
        print(f"    Roads:       {stats['unique_roads']}")
        print(f"    Wards:       {stats['unique_wards']}")
