"""Small, durable filesystem registry for submitted pipeline runs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_DIR_NAME = ".run_registry"
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED"})


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def registry_dir(output_dir: str) -> Path:
    path = Path(output_dir) / REGISTRY_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_path(output_dir: str, run_id: str) -> Path:
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError("run_id may contain only letters, numbers, '-' and '_'")
    return registry_dir(output_dir) / f"{run_id}.json"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def _record_lock(path: Path):
    """Serialize read-modify-write updates across API and worker processes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    token = f"{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
    deadline = time.time() + 10
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"token": token, "pid": os.getpid(), "time": time.time()}, f)
            break
        except FileExistsError:
            stale = False
            try:
                with lock_path.open("r", encoding="utf-8") as f:
                    owner = json.load(f)
                stale = (
                    time.time() - float(owner.get("time", 0)) > 30
                    or not process_is_alive(owner.get("pid"))
                )
            except (OSError, ValueError, json.JSONDecodeError):
                stale = True
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out updating run record: {path.name}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            with lock_path.open("r", encoding="utf-8") as f:
                owner = json.load(f)
            if owner.get("token") == token:
                lock_path.unlink()
        except (OSError, json.JSONDecodeError):
            pass


def create_run(
    output_dir: str,
    run_id: str,
    request: Dict[str, Any],
    owner: str = "cli",
) -> Dict[str, Any]:
    path = run_path(output_dir, run_id)
    with _record_lock(path):
        if path.exists():
            raise FileExistsError(f"Run '{run_id}' already exists")
        now = utc_now()
        record = {
            "run_id": run_id,
            "owner": owner,
            "city": request.get("city", ""),
            "month": request.get("month"),
            "status": "QUEUED",
            "current_step": None,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "finished_at": None,
            "pid": None,
            "error": None,
            "checkpoint": None,
            "outputs": {},
            "request": request,
        }
        atomic_write_json(path, record)
    return record


def load_run(output_dir: str, run_id: str) -> Optional[Dict[str, Any]]:
    path = run_path(output_dir, run_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_run(
    output_dir: str,
    run_id: str,
    *,
    _allow_terminal_repair: bool = False,
    **changes: Any,
) -> Dict[str, Any]:
    path = run_path(output_dir, run_id)
    with _record_lock(path):
        record = load_run(output_dir, run_id)
        if record is None:
            raise FileNotFoundError(f"Unknown run_id: {run_id}")
        if (
            not _allow_terminal_repair
            and record.get("status") in TERMINAL_STATUSES
            and changes.get("status") not in (None, record["status"])
        ):
            raise ValueError(f"Run '{run_id}' is already terminal")
        record.update(changes)
        record["updated_at"] = utc_now()
        if changes.get("status") == "RUNNING" and not record.get("started_at"):
            record["started_at"] = record["updated_at"]
        if changes.get("status") in TERMINAL_STATUSES:
            record["finished_at"] = record["updated_at"]
            record["current_step"] = None
        atomic_write_json(path, record)
    return record


def list_runs(
    output_dir: str,
    *,
    city: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in registry_dir(output_dir).glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if city and record.get("city", "").casefold() != city.casefold():
            continue
        if status and record.get("status") != status:
            continue
        records.append(record)
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records


def process_is_alive(pid: Any) -> bool:
    """Return True if a process with this PID currently exists.

    Uses ``os.kill(pid, 0)``, which does not send a signal. Python 3.8+
    implements that probe on Windows as well as POSIX, so there is no
    platform-specific OpenProcess / ctypes path here.
    """
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but is owned by another user (POSIX).
        return True
    except (TypeError, ValueError, OSError):
        return False

