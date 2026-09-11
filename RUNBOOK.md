# Operating Runbook — GeoAI Property Tax Pipeline

Written so a second engineer who has never touched this pipeline can run,
check, and recover a monthly refresh unaided (product-doc ticket 1.17).

## 1. Prerequisites

- Python >= 3.10, dependencies installed per `README.md` (`pip install -r requirements.txt`).
- Run all commands from the **parent** directory of `pipeline_geoai/`, so it's importable as a package.
- Input files for the city: mSeva CSV, GIS shapefile (`.shp`+`.shx`+`.dbf`+`.prj`), optional electricity CSV, optional Earth Engine change-detection shapefile.

## 2. Running a refresh

```powershell
py -m pipeline_geoai --city "Barnala" --state "Punjab" `
  --mseva "Barnala\mseva_enriched.csv" `
  --gis "Barnala\gissurvey.shp" `
  --electricity "Barnala\electricity.csv" `
  --output "pipeline_geoai\results" `
  --change-detection "Barnala\Barnala_CD\Barnala_New_Constructions.shp"
```

Add `--month 2026-09` if you need to record a specific month explicitly (default: current month). Add `--force-retrain` only if you need to rebuild the GeoAI Knowledge Base from scratch.

**What happens under the hood** (see `run_manager.py`):
1. A per-city lock is acquired at `{output}/.{city}.refresh.lock`. If a refresh for that city is already running, the command fails immediately with a clear message naming the other run's PID and start time — it never queues silently or runs twice at once.
2. All pipeline steps, change detection, and CSV standardization write into an isolated temp directory: `{output}/.runs/{City}_{run_id}/`.
3. Outputs are checked against the signed-off Data Dictionary field list (`validate_output.py`).
4. **Only if every step and the validation succeed**, the temp directory's files are copied into `{output}/` (overwriting the previous month's files there), and a provenance manifest is written and published.
5. If anything fails, nothing in `{output}/` changes — the partial run stays isolated in `.runs/{City}_{run_id}/` for debugging, and the command exits non-zero with a plain-language error.

**Running the same city+month twice is safe**: each attempt gets its own `run_id`, its own manifest, and its own isolated temp directory. A successful second run simply republishes over the first.

## 3. Checking a refresh's status and provenance

- `{output}/{City}_latest_manifest.json` — the current published run for a city: run ID, month, publish time, and a hash of every output file. This is the file any downstream consumer (including the API) should treat as the source of truth.
- `{output}/.manifests/{City}/{run_id}.json` — the full record for *every* run, success or failure: exact input file hashes, the complete `CityConfig` used, git commit, host, start/finish time, and (for successes) a snapshot of that run's summary stats. Nothing in this folder is ever overwritten.
- To see every run for a city in order, read all files under `{output}/.manifests/{City}/` sorted by `started_at`, or hit the API's `/cities/{city}/manifest-history` endpoint (see §5).

## 4. Recovering from a failed run

1. Read the error printed to the console (also logged in `{output}/.manifests/{City}/{run_id}.json` under `"error"`) — it names the step and the underlying exception in plain language.
2. Inspect the partial outputs, if any, in `{output}/.runs/{City}_{run_id}/` for debugging. This directory is never deleted automatically — clean it up manually once you're done with it.
3. Fix the underlying issue (bad input file, missing column, unreachable API, etc.) and re-run the exact same command. A fresh `run_id` will be used automatically.
4. If the command reports "Refresh for '{city}' is already running" but you know the previous process crashed, either delete `{output}/.{city}.refresh.lock` manually, or simply wait — the lock is treated as stale and reclaimed automatically after 6 hours.

## 5. Serving outputs to the dashboard (minimal API)

```powershell
$env:PIPELINE_OUTPUT_DIR = "pipeline_geoai\results"
py -m pipeline_geoai.api
```

This starts a small read-only FastAPI service (default `http://127.0.0.1:8000`) that serves whatever the `{city}_latest_manifest.json` pointer currently points to — it never triggers a refresh itself. Key endpoints:

- `GET /cities/{city}/latest` — the published-run pointer (run ID, month, output hashes).
- `GET /cities/{city}/summary` — `{city}_summary.json`.
- `GET /cities/{city}/defaulters.geojson` — defaulter polygons for the map.
- `GET /cities/{city}/change-detection.geojson` — new-construction polygons for the map (feeds the "B3 Mixed Use/Ongoing" category on the dashboard).
- `GET /cities/{city}/match-register?limit=&offset=` — paginated match register rows.
- `GET /cities/{city}/manifest-history` — every run for a city, success or failure.

## 6. Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `Refresh for '{city}' is already running (pid ...)` | Lock held by another (or crashed) process | Wait for it to finish, or delete `{output}/.{city}.refresh.lock` if you've confirmed the other process is dead |
| `Refresh produced outputs that don't match the signed-off Data Dictionary field list` | A code change or input-format change dropped a required column | Check the listed missing columns against `validate_output.py`'s `*_REQUIRED_*` lists and `data_dictionary.pdf` |
| `Change detection was requested but no Match Register was produced` | `--change-detection` was passed without the `match`/`report` steps running first | Run the full pipeline (default `--steps`), or include `match report` explicitly |
| `Name+Locality: N unmatched mSeva record(s) reference M locality name(s) with NO matching GIS locality` (WARNING, not a failure) | mSeva locality names that don't map to any GIS locality via the crosswalk | Extend `locality_crosswalk.csv` near the mSeva file with the missing mappings |
| `GIS path required for training but not found` | `--gis` path wrong or file missing | Confirm the shapefile bundle (`.shp`/`.shx`/`.dbf`/`.prj`) exists at the given path |

## 7. Alerting

Alerts (refresh failed, input missing/stale, anomalous jump vs. last month) are always logged at `ERROR` level. To also forward them to a webhook (Slack, an internal notifier, etc.), set:

```powershell
$env:ALERT_WEBHOOK_URL = "https://your-webhook-endpoint"
```

No code changes are needed once a transport is chosen — see `alerts.py`.
