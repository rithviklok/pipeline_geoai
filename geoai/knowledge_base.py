"""
Knowledge Base persistence, caching, and validation.

The KB is a directory containing 18 artifacts produced by training:
  - house_database.csv, locality_statistics.csv, locality_bbox.csv
  - locality_polygons.geojson
  - 6 embedding pickles
  - kd_tree.pkl, ball_tree.pkl, locality_rtree.pkl
  - faiss_index.bin
  - municipality_coordinates.npy
  - municipality_statistics.json, municipality_summary.csv
  - training_config.json
"""

import os
import json
import pickle
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Files that MUST exist for a valid KB
REQUIRED_FILES = [
    "house_database.csv",
    "locality_statistics.csv",
    "locality_bbox.csv",
    "locality_polygons.geojson",
    "address_embeddings.pkl",
    "owner_embeddings.pkl",
    "locality_embeddings.pkl",
    "road_embeddings.pkl",
    "ward_embeddings.pkl",
    "full_address_embeddings.pkl",
    "kd_tree.pkl",
    "ball_tree.pkl",
    "faiss_index.bin",
    "municipality_coordinates.npy",
    "training_config.json",
]

# Optional files (nice-to-have but not critical)
OPTIONAL_FILES = [
    "locality_rtree.pkl",
    "municipality_statistics.json",
    "municipality_summary.csv",
    "gdf_checkpoint.pkl",
]


@dataclass
class KnowledgeBase:
    """In-memory representation of a trained GeoAI Knowledge Base."""

    # Core data
    house_df: pd.DataFrame = None               # house_database.csv
    locality_stats_df: pd.DataFrame = None       # locality_statistics.csv
    locality_bbox_df: pd.DataFrame = None        # locality_bbox.csv

    # Embeddings (np.ndarray, shape=(N, dim))
    address_embeddings: np.ndarray = None
    owner_embeddings: np.ndarray = None
    locality_embeddings: np.ndarray = None
    road_embeddings: np.ndarray = None
    ward_embeddings: np.ndarray = None
    full_address_embeddings: np.ndarray = None

    # Spatial indexes
    kd_tree: object = None                       # sklearn KDTree
    ball_tree: object = None                     # sklearn BallTree
    faiss_index: object = None                   # faiss.IndexFlatIP
    locality_rtree: object = None                # rtree.Index (optional)

    # Coordinate matrix
    coordinates: np.ndarray = None               # municipality_coordinates.npy

    # Locality polygons (GeoDataFrame)
    locality_polygons_gdf: object = None         # locality_polygons.geojson

    # Config used for training
    training_config: dict = field(default_factory=dict)

    # Where this KB is stored on disk
    path: Optional[str] = None


def compute_gis_hash(gis_path: str) -> str:
    """Compute a fast hash of the GIS file to detect data changes."""
    h = hashlib.md5()
    with open(gis_path, "rb") as f:
        # Read first 1MB + file size for fast fingerprinting
        h.update(f.read(1024 * 1024))
        f.seek(0, 2)
        h.update(str(f.tell()).encode())
    return h.hexdigest()


def is_valid(kb_path: str, gis_path: str = None) -> bool:
    """Check if a cached Knowledge Base is valid and up-to-date.

    Parameters
    ----------
    kb_path : str
        Directory where KB artifacts are stored.
    gis_path : str, optional
        Path to GIS source file. If provided, checks that the KB was trained
        on the same data (via hash comparison).

    Returns
    -------
    bool
    """
    if not os.path.isdir(kb_path):
        return False

    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(kb_path, fname)):
            logger.info("KB missing file: %s", fname)
            return False

    # Check GIS hash if source file provided
    if gis_path:
        config_path = os.path.join(kb_path, "training_config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
            stored_hash = config.get("gis_hash", "")
            current_hash = compute_gis_hash(gis_path)
            if stored_hash != current_hash:
                logger.info("KB stale: GIS hash mismatch (stored=%s, current=%s)",
                           stored_hash[:8], current_hash[:8])
                return False
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    logger.info("KB valid at: %s", kb_path)
    return True


def save(kb: KnowledgeBase, kb_path: str):
    """Save a KnowledgeBase to disk."""
    os.makedirs(kb_path, exist_ok=True)

    # CSVs
    kb.house_df.to_csv(os.path.join(kb_path, "house_database.csv"), index=False)
    if kb.locality_stats_df is not None:
        kb.locality_stats_df.to_csv(os.path.join(kb_path, "locality_statistics.csv"), index=False)
    if kb.locality_bbox_df is not None:
        kb.locality_bbox_df.to_csv(os.path.join(kb_path, "locality_bbox.csv"), index=False)

    # GeoJSON
    if kb.locality_polygons_gdf is not None:
        kb.locality_polygons_gdf.to_file(
            os.path.join(kb_path, "locality_polygons.geojson"), driver="GeoJSON"
        )

    # Embeddings
    for name in ["address", "owner", "locality", "road", "ward", "full_address"]:
        emb = getattr(kb, f"{name}_embeddings")
        if emb is not None:
            with open(os.path.join(kb_path, f"{name}_embeddings.pkl"), "wb") as f:
                pickle.dump(emb, f)

    # Spatial indexes
    if kb.kd_tree is not None:
        with open(os.path.join(kb_path, "kd_tree.pkl"), "wb") as f:
            pickle.dump(kb.kd_tree, f)
    if kb.ball_tree is not None:
        with open(os.path.join(kb_path, "ball_tree.pkl"), "wb") as f:
            pickle.dump(kb.ball_tree, f)
    if kb.faiss_index is not None:
        from .embeddings import save_faiss_index
        save_faiss_index(kb.faiss_index, os.path.join(kb_path, "faiss_index.bin"))
    if kb.locality_rtree is not None:
        with open(os.path.join(kb_path, "locality_rtree.pkl"), "wb") as f:
            pickle.dump(kb.locality_rtree, f)

    # Coordinates
    if kb.coordinates is not None:
        np.save(os.path.join(kb_path, "municipality_coordinates.npy"), kb.coordinates)

    # Config
    with open(os.path.join(kb_path, "training_config.json"), "w") as f:
        json.dump(kb.training_config, f, indent=2)

    kb.path = kb_path
    logger.info("KB saved to: %s (%d files)", kb_path, len(os.listdir(kb_path)))


def load(kb_path: str) -> KnowledgeBase:
    """Load a KnowledgeBase from disk."""
    import geopandas as gpd
    from .embeddings import load_faiss_index

    kb = KnowledgeBase(path=kb_path)

    # CSVs
    kb.house_df = pd.read_csv(os.path.join(kb_path, "house_database.csv"))
    stats_path = os.path.join(kb_path, "locality_statistics.csv")
    if os.path.exists(stats_path):
        kb.locality_stats_df = pd.read_csv(stats_path)
    bbox_path = os.path.join(kb_path, "locality_bbox.csv")
    if os.path.exists(bbox_path):
        kb.locality_bbox_df = pd.read_csv(bbox_path)

    # GeoJSON
    geojson_path = os.path.join(kb_path, "locality_polygons.geojson")
    if os.path.exists(geojson_path):
        kb.locality_polygons_gdf = gpd.read_file(geojson_path)

    # Embeddings
    for name in ["address", "owner", "locality", "road", "ward", "full_address"]:
        pkl_path = os.path.join(kb_path, f"{name}_embeddings.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                setattr(kb, f"{name}_embeddings", pickle.load(f))

    # Spatial indexes
    for name in ["kd_tree", "ball_tree", "locality_rtree"]:
        pkl_path = os.path.join(kb_path, f"{name}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                setattr(kb, name, pickle.load(f))

    # FAISS
    faiss_path = os.path.join(kb_path, "faiss_index.bin")
    if os.path.exists(faiss_path):
        kb.faiss_index = load_faiss_index(faiss_path)

    # Coordinates
    coords_path = os.path.join(kb_path, "municipality_coordinates.npy")
    if os.path.exists(coords_path):
        kb.coordinates = np.load(coords_path)

    # Config
    config_path = os.path.join(kb_path, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            kb.training_config = json.load(f)

    logger.info("KB loaded from: %s (house_df=%d rows, faiss=%s vectors)",
                kb_path, len(kb.house_df),
                kb.faiss_index.ntotal if kb.faiss_index else "N/A")
    return kb
