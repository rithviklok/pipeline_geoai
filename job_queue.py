"""Persistent single-worker queue used by the FastAPI job interface."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from . import run_registry
from .lock import RefreshAlreadyRunningError, city_lock

logger = logging.getLogger(__name__)


class JobQueue:
    """Run submitted jobs one at a time and recover them after API restart."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._queued: set[str] = set()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._leader_context = None
        self._is_leader = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._leader_context = city_lock(
            self.output_dir,
            "_job_queue",
            run_id=f"api-{os.getpid()}",
        )
        try:
            self._leader_context.__enter__()
            self._is_leader = True
        except RefreshAlreadyRunningError:
            self._leader_context = None
            self._is_leader = False
            logger.info(
                "Another API process owns the persistent job worker; "
                "this process will serve requests only."
            )
            return
        self._recover()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="property-tax-job-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)
        if self._leader_context:
            self._leader_context.__exit__(None, None, None)
            self._leader_context = None
        self._is_leader = False

    def enqueue(self, run_id: str) -> None:
        if not self._is_leader:
            # A different API process owns the worker. The durable QUEUED
            # record will be discovered by that leader's periodic scan.
            return
        with self._lock:
            if run_id in self._queued:
                return
            self._queued.add(run_id)
            self._queue.put(run_id)

    def _recover(self) -> None:
        for record in reversed(run_registry.list_runs(self.output_dir)):
            status = record.get("status")
            if status == "QUEUED":
                self.enqueue(record["run_id"])
            elif status == "RUNNING" and not run_registry.process_is_alive(
                record.get("pid")
            ):
                run_registry.update_run(
                    self.output_dir,
                    record["run_id"],
                    status="QUEUED",
                    pid=None,
                    current_step=None,
                    error="Worker stopped; queued for checkpoint recovery",
                )
                self.enqueue(record["run_id"])

    def _run_loop(self) -> None:
        while True:
            try:
                run_id = self._queue.get(timeout=2)
            except queue.Empty:
                self._recover()
                continue
            if run_id is None:
                self._queue.task_done()
                return
            retry = False
            try:
                retry = self._run_job(run_id)
            except Exception:
                logger.exception("Job worker could not launch run %s", run_id)
                record = run_registry.load_run(self.output_dir, run_id)
                if record and record.get("status") not in run_registry.TERMINAL_STATUSES:
                    run_registry.update_run(
                        self.output_dir,
                        run_id,
                        status="FAILED",
                        pid=None,
                        error="Job worker could not launch the pipeline process",
                    )
            finally:
                with self._lock:
                    self._queued.discard(run_id)
                self._queue.task_done()
                if retry:
                    self.enqueue(run_id)

    def _run_job(self, run_id: str) -> bool:
        record = run_registry.load_run(self.output_dir, run_id)
        if not record or record.get("status") in run_registry.TERMINAL_STATUSES:
            return False

        logs_dir = Path(self.output_dir) / ".run_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{run_id}.log"
        package_parent = Path(__file__).resolve().parent.parent
        command = [
            sys.executable,
            "-m",
            "pipeline_geoai.worker",
            "--output-dir",
            self.output_dir,
            "--run-id",
            run_id,
        ]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(package_parent),
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            run_registry.update_run(
                self.output_dir,
                run_id,
                status="RUNNING",
                pid=process.pid,
                log_path=os.path.relpath(log_path, self.output_dir),
            )
            return_code = process.wait()

        final_record = run_registry.load_run(self.output_dir, run_id)
        if final_record and final_record.get("status") not in run_registry.TERMINAL_STATUSES:
            restart_count = int(final_record.get("restart_count", 0)) + 1
            if restart_count <= 1:
                run_registry.update_run(
                    self.output_dir,
                    run_id,
                    status="QUEUED",
                    pid=None,
                    current_step=None,
                    restart_count=restart_count,
                    error=(
                        f"Worker process exited unexpectedly with code {return_code}; "
                        "queued once for checkpoint recovery"
                    ),
                )
                return True
            run_registry.update_run(
                self.output_dir,
                run_id,
                status="FAILED",
                pid=None,
                restart_count=restart_count,
                error=f"Pipeline worker repeatedly exited with code {return_code}",
            )
        return False

