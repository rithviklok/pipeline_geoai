"""
database.py  —  SQLite tile-URL cache.

GEE map tile URLs are valid for ~24 hours; caching them avoids
re-running expensive Earth Engine computations on every request.
"""

from __future__ import annotations

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "cd_cache.db")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS tile_cache (
            cache_key  TEXT PRIMARY KEY,
            tile_url   TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    return con


def get_cached(cache_key: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT tile_url FROM tile_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    return row[0] if row else None


def save_cached(cache_key: str, tile_url: str) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO tile_cache (cache_key, tile_url) VALUES (?, ?)",
            (cache_key, tile_url),
        )
        con.commit()


def clear_cache() -> int:
    """Delete all cached entries; returns count of deleted rows."""
    with _conn() as con:
        cur = con.execute("DELETE FROM tile_cache")
        con.commit()
    return cur.rowcount
