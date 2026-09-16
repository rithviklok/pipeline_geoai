"""Focused regression tests for the run-management shell."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient

from pipeline_geoai import (
    lock,
    manifest,
    run_manager,
    run_registry,
    standardize,
    validate_output,
)
from pipeline_geoai.api import app as api_module
from pipeline_geoai.config import CityConfig


class RunRegistryTests(unittest.TestCase):
    def test_process_is_alive_uses_portable_probe(self):
        self.assertTrue(run_registry.process_is_alive(os.getpid()))
        self.assertFalse(run_registry.process_is_alive(0))
        self.assertFalse(run_registry.process_is_alive(-1))
        self.assertFalse(run_registry.process_is_alive("not-a-pid"))
        with mock.patch("os.kill", side_effect=ProcessLookupError):
            self.assertFalse(run_registry.process_is_alive(12345))
        with mock.patch("os.kill", side_effect=PermissionError):
            self.assertTrue(run_registry.process_is_alive(12345))

    def test_registry_lifecycle_is_addressable(self):
        with tempfile.TemporaryDirectory() as output:
            created = run_registry.create_run(
                output,
                "run_1",
                {"city": "Mohali", "month": "2026-09"},
                owner="tester",
            )
            self.assertEqual(created["status"], "QUEUED")
            running = run_registry.update_run(
                output, "run_1", status="RUNNING", current_step="infer"
            )
            self.assertEqual(running["current_step"], "infer")
            complete = run_registry.update_run(
                output, "run_1", status="SUCCEEDED", outputs={"x": {}}
            )
            self.assertEqual(complete["status"], "SUCCEEDED")
            self.assertIsNotNone(complete["finished_at"])

    def test_city_lock_is_exclusive_and_records_run_id(self):
        with tempfile.TemporaryDirectory() as output:
            with lock.city_lock(output, "Mohali", run_id="first"):
                lock_path = Path(output) / ".Mohali.refresh.lock"
                info = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(info["run_id"], "first")
                with self.assertRaises(lock.RefreshAlreadyRunningError):
                    with lock.city_lock(output, "Mohali", run_id="second"):
                        pass
            self.assertFalse(lock_path.exists())

    def test_concurrent_registry_updates_do_not_clobber_fields(self):
        with tempfile.TemporaryDirectory() as output:
            run_registry.create_run(output, "run_2", {"city": "Mohali"})
            barrier = threading.Barrier(3)

            def update(field):
                barrier.wait()
                run_registry.update_run(output, "run_2", **{field: True})

            threads = [
                threading.Thread(target=update, args=("worker_field",)),
                threading.Thread(target=update, args=("api_field",)),
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()
            record = run_registry.load_run(output, "run_2")
            self.assertTrue(record["worker_field"])
            self.assertTrue(record["api_field"])


class ManifestTests(unittest.TestCase):
    def test_receipt_is_immutable_and_latest_pointer_is_replaceable(self):
        with tempfile.TemporaryDirectory() as output:
            receipt = {
                "run_id": "abc",
                "city": "Mohali",
                "month": "2026-09",
                "schema_version": "1.0",
                "outputs": {},
            }
            path = manifest.save_manifest(output, "Mohali", receipt)
            with self.assertRaises(FileExistsError):
                manifest.save_manifest(output, "Mohali", receipt)
            manifest.publish_latest(output, "Mohali", receipt, path)
            latest = manifest.load_latest(output, "Mohali")
            self.assertEqual(latest["run_id"], "abc")

    def test_inference_checkpoint_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            mseva = base / "mseva.csv"
            gis = base / "gis.shp"
            mseva.write_text("propertyid\n1\n", encoding="utf-8")
            gis.write_bytes(b"gis")
            config = CityConfig(
                name="Mohali",
                state="Punjab",
                mseva_path=str(mseva),
                gis_path=str(gis),
                output_dir=str(base / "results"),
            )
            tmp_dir = base / "work"
            tmp_dir.mkdir()
            artifact = tmp_dir / "Mohali_GeoAI_Geocoded.csv"
            artifact.write_text("propertyid,Matched_UID\n1,A\n", encoding="utf-8")

            run_manager._save_inference_checkpoint(
                str(tmp_dir), config, "Mohali"
            )
            self.assertIsNotNone(
                run_manager._load_inference_checkpoint(str(tmp_dir), config)
            )
            artifact.write_text("changed", encoding="utf-8")
            self.assertIsNone(
                run_manager._load_inference_checkpoint(str(tmp_dir), config)
            )

    def test_success_receipt_repairs_stale_failed_registry(self):
        with tempfile.TemporaryDirectory() as output:
            request = {"city": "Mohali", "month": "2026-09"}
            run_registry.create_run(output, "repair1", request)
            run_registry.update_run(output, "repair1", status="FAILED")
            receipt = {
                "schema_version": "1.0",
                "run_id": "repair1",
                "city": "Mohali",
                "state": "Punjab",
                "month": "2026-09",
                "status": "success",
                "outputs": {},
            }
            manifest.save_manifest(output, "Mohali", receipt)
            config = CityConfig(
                name="Mohali",
                state="Punjab",
                mseva_path="unused.csv",
                output_dir=output,
            )
            result = run_manager.execute_pipeline_run(
                config, month="2026-09", run_id="repair1"
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(
                run_registry.load_run(output, "repair1")["status"],
                "SUCCEEDED",
            )


class OutputContractTests(unittest.TestCase):
    def test_standardize_preserves_leading_zero_source_ids(self):
        with tempfile.TemporaryDirectory() as output:
            path = Path(output) / "ids.csv"
            path.write_text(
                "propertyid,property_uid\n00123,PT:00123\n",
                encoding="utf-8",
            )
            standardize.standardize_csv(
                path, "Mohali", "Punjab", "2026-09", run_id="run1"
            )
            result = pd.read_csv(path, dtype=str, keep_default_na=False)
            self.assertEqual(result.loc[0, "propertyid"], "00123")
            self.assertEqual(result.loc[0, "property_uid"], "PT:00123")

    def _write_valid_bundle(self, output: str):
        city = "Mohali"
        matches = pd.DataFrame(
            [
                {
                    "propertyid": "PT-1",
                    "matched_uid": "GIS-1",
                    "match_method": "TEST",
                    "gis_owner_name": "",
                    "gis_mobile": "",
                    "gis_locality": "",
                    "property_uid": "PT:PT-1",
                    "month": "2026-09",
                    "run_id": "run1",
                    "schema_version": "1.0",
                }
            ]
        )
        parcels = pd.DataFrame(
            [
                {
                    "gis_uid": "GIS-1",
                    "gis_owner_name": "",
                    "gis_guardian_name": "",
                    "gis_mobile": "",
                    "gis_locality": "",
                    "latitude": "",
                    "longitude": "",
                    "property_usage": "",
                    "property_type": "",
                    "status": "MATCHED",
                    "electricity_account_no": "",
                    "electricity_holder_name": "",
                    "property_uid": "GIS:GIS-1",
                    "tax_status": "IN_TAX_NET",
                    "geo_status": "SHAPE",
                    "ward_id": "1",
                    "month": "2026-09",
                    "run_id": "run1",
                    "schema_version": "1.0",
                },
                {
                    "gis_uid": "GIS-2",
                    "gis_owner_name": "",
                    "gis_guardian_name": "",
                    "gis_mobile": "",
                    "gis_locality": "",
                    "latitude": "",
                    "longitude": "",
                    "property_usage": "",
                    "property_type": "",
                    "status": "EXEMPT",
                    "electricity_account_no": "",
                    "electricity_holder_name": "",
                    "property_uid": "GIS:GIS-2",
                    "tax_status": "EXEMPT",
                    "geo_status": "NONE",
                    "ward_id": "2",
                    "month": "2026-09",
                    "run_id": "run1",
                    "schema_version": "1.0",
                },
            ]
        )
        matches.to_csv(Path(output) / f"{city}_Match_Register.csv", index=False)
        parcels.to_csv(Path(output) / f"{city}_Defaulters.csv", index=False)
        features = [
            {
                "type": "Feature",
                "geometry": {} if row["geo_status"] == "SHAPE" else None,
                "properties": row,
            }
            for row in parcels.to_dict(orient="records")
        ]
        (Path(output) / f"{city}_Defaulters.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "schema_version": "1.0",
                    "run_id": "run1",
                    "month": "2026-09",
                    "features": features,
                }
            ),
            encoding="utf-8",
        )
        (Path(output) / f"{city}_summary.json").write_text(
            json.dumps(
                {
                    "city": city,
                    "total_mseva": 1,
                    "total_gis": 2,
                    "emitted_gis_rows": 2,
                    "matched_count": 1,
                    "unmatched_count": 0,
                    "match_rate": 100.0,
                    "unique_gis_matched": 1,
                    "potential_defaulters": 0,
                    "tax_status_counts": {"IN_TAX_NET": 1, "EXEMPT": 1},
                    "layer_breakdown": {"TEST": 1},
                    "month": "2026-09",
                    "run_id": "run1",
                    "schema_version": "1.0",
                }
            ),
            encoding="utf-8",
        )

    def test_complete_bundle_passes_and_dropped_feature_fails(self):
        with tempfile.TemporaryDirectory() as output:
            self._write_valid_bundle(output)
            results = validate_output.validate_city_outputs(output, "Mohali")
            self.assertTrue(validate_output.all_ok(results), validate_output.format_report(results))

            path = Path(output) / "Mohali_Defaulters.geojson"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["features"].pop()
            path.write_text(json.dumps(data), encoding="utf-8")
            results = validate_output.validate_city_outputs(output, "Mohali")
            self.assertFalse(results["reconciliation"].ok)


class ApiTests(unittest.TestCase):
    def test_submission_requires_key_and_returns_run_id(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            for name in ("mseva.csv", "gis.shp", "gis.shx", "gis.dbf"):
                (base / name).write_bytes(b"x")
            old_output = api_module.OUTPUT_DIR
            api_module.OUTPUT_DIR = str(base / "results")
            try:
                with mock.patch.dict(os.environ, {"PIPELINE_API_KEY": "secret"}):
                    with mock.patch.object(api_module.job_queue, "enqueue") as enqueue:
                        client = TestClient(api_module.app)
                        payload = {
                            "city": "Mohali",
                            "state": "Punjab",
                            "month": "2026-09",
                            "mseva_path": str(base / "mseva.csv"),
                            "gis_path": str(base / "gis.shp"),
                        }
                        self.assertEqual(client.post("/runs", json=payload).status_code, 401)
                        response = client.post(
                            "/runs",
                            json=payload,
                            headers={"X-API-Key": "secret"},
                        )
                        self.assertEqual(response.status_code, 202, response.text)
                        run_id = response.json()["run_id"]
                        enqueue.assert_called_once_with(run_id)
                        status_response = client.get(
                            f"/runs/{run_id}",
                            headers={"X-API-Key": "secret"},
                        )
                        self.assertEqual(status_response.json()["status"], "QUEUED")
                        run_registry.update_run(
                            api_module.OUTPUT_DIR,
                            run_id,
                            status="FAILED",
                            error="test failure",
                        )
                        retry = client.post(
                            f"/runs/{run_id}/retry",
                            headers={"X-API-Key": "secret"},
                        )
                        self.assertEqual(retry.status_code, 202, retry.text)
                        self.assertEqual(
                            retry.json()["resumed_from_run_id"], run_id
                        )
            finally:
                api_module.OUTPUT_DIR = old_output


if __name__ == "__main__":
    unittest.main()

