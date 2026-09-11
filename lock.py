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


class RefreshAlreadyRunningError(RuntimeError):
    """Raised when a lock for the requested city is already held by another run."""


def _lock_path(output_dir: str, city: str) -> str:
    return os.path.join(output_dir, f".{city}.refresh.lock")


@contextlib.contextmanager
def city_lock(output_dir: str, city: str, stale_after_seconds: int = 6 * 3600):
    """Acquire an exclusive lock for `city` for the duration of the `with` block.

    Raises RefreshAlreadyRunningError if another (non-stale) run holds the
    lock. A lock older than `stale_after_seconds` is treated as abandoned
    (e.g. left behind by a crashed process) and is reclaimed automatically
    rather than requiring manual cleanup.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = _lock_path(output_dir, city)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            age = time.time() - info.get("started_at_epoch", 0)
            if age < stale_after_seconds:
                raise RefreshAlreadyRunningError(
                    f"Refresh for '{city}' is already running (pid {info.get('pid')}, "
                    f"started {info.get('started_at')}). Not starting a second one. "
                    f"If this is wrong (the other process crashed), remove {path} "
                    f"or wait {stale_after_seconds // 3600}h for it to be reclaimed automatically."
                )
        except (json.JSONDecodeError, OSError):
            pass  # corrupt lock file — treat as stale and reclaim it below

    info = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at_epoch": time.time(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f)

    try:
        yield
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
