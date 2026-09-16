"""
lock.py — Per-city lock preventing two refreshes for the same city from
running concurrently (product-doc ticket 4.4: "Starting a second one while
the first is running is refused with a clear reason. Never run twice, never
silently queued.").
"""
from __future__ import annotations

import contextlib
import json
import os
import time

from .run_registry import process_is_alive


class RefreshAlreadyRunningError(RuntimeError):
    """Raised when a lock for the requested city is already held by another run."""


def _lock_path(output_dir: str, city: str) -> str:
    return os.path.join(output_dir, f".{city}.refresh.lock")


@contextlib.contextmanager
def city_lock(
    output_dir: str,
    city: str,
    run_id: str | None = None,
    stale_after_seconds: int = 6 * 3600,
):
    """Acquire an exclusive lock for `city` for the duration of the `with` block.

    Raises RefreshAlreadyRunningError if another (non-stale) run holds the
    lock. A lock older than `stale_after_seconds` is treated as abandoned
    (e.g. left behind by a crashed process) and is reclaimed automatically
    rather than requiring manual cleanup.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = _lock_path(output_dir, city)

    while True:
        info = {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "started_at_epoch": time.time(),
        }
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
                f.flush()
                os.fsync(f.fileno())
            break
        except FileExistsError:
            existing = {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

            age = time.time() - existing.get("started_at_epoch", 0)
            owner_alive = process_is_alive(existing.get("pid"))
            owner_pid = existing.get("pid")
            if owner_alive or (not owner_pid and existing and age < stale_after_seconds):
                raise RefreshAlreadyRunningError(
                    f"Refresh for '{city}' is already running "
                    f"(run_id {existing.get('run_id') or 'unknown'}, "
                    f"pid {existing.get('pid')}, started {existing.get('started_at')}). "
                    "Not starting a second one."
                )
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    try:
        yield info
    finally:
        try:
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f)
            if current.get("pid") == info["pid"] and current.get("run_id") == run_id:
                os.remove(path)
        except (OSError, json.JSONDecodeError):
            pass
