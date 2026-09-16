# Property Tax Pipeline API

Use this API to submit a monthly refresh, poll it by `run_id`, and download the published files. Jobs keep running if the client disconnects.

The pipeline is OS-independent. Examples below are given for Linux/macOS first, with an equivalent Windows PowerShell block where the syntax differs.

Run all commands from the directory that **contains** `pipeline_geoai/`.

Interactive docs are available at `http://127.0.0.1:8000/docs` once the server is up.

## 1. Start the server

Linux / macOS:

```bash
export PIPELINE_OUTPUT_DIR="pipeline_geoai/results"
export PIPELINE_API_KEY="replace-with-a-secret"
export API_HOST="127.0.0.1"
export API_PORT="8000"
python -m pipeline_geoai.api
```

Windows PowerShell:

```powershell
$env:PIPELINE_OUTPUT_DIR = "pipeline_geoai/results"
$env:PIPELINE_API_KEY = "replace-with-a-secret"
$env:API_HOST = "127.0.0.1"
$env:API_PORT = "8000"
python -m pipeline_geoai.api
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PIPELINE_API_KEY` | yes, for job endpoints | none | Shared secret sent as `X-API-Key` |
| `PIPELINE_OUTPUT_DIR` | no | `results` | Where runs, logs, and published files live |
| `API_HOST` | no | `127.0.0.1` | Bind address |
| `API_PORT` | no | `8000` | Bind port |
| `ALERT_WEBHOOK_URL` | no | unset | Optional completion / failure webhook |

Check that it is up:

```bash
curl -s http://127.0.0.1:8000/health
```

Expected:

```json
{"status": "ok", "schema_version": "1.0"}
```

`GET /health` does not need an API key. Every `/runs` endpoint does.

## 2. Auth

Send the key on every job request:

```http
X-API-Key: replace-with-a-secret
```

| Status | Meaning |
|---|---|
| `401` | Missing or wrong key |
| `503` | Server started without `PIPELINE_API_KEY` |

The `/cities/{city}/...` read endpoints currently serve the latest published pointer without a key. Use `/runs/{run_id}/outputs/{filename}` when you need a specific run behind the key.

## 3. Submit a run

`POST /runs` → `202 Accepted`

The request is accepted immediately. The work happens in a background worker.

Linux / macOS:

```bash
curl -s -X POST http://127.0.0.1:8000/runs \
  -H "X-API-Key: $PIPELINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Mohali",
    "state": "Punjab",
    "month": "2026-09",
    "mseva_path": "/data/Mohali/MohaliDataset/mseva.csv",
    "gis_path": "/data/Mohali/MohaliDataset/mohaligis.shp",
    "electricity_path": "/data/Mohali/MohaliDataset/electricity.csv",
    "owner": "dashboard"
  }'
```

Windows PowerShell:

```powershell
$body = @{
  city = "Mohali"
  state = "Punjab"
  month = "2026-09"
  mseva_path = (Resolve-Path "Mohali/MohaliDataset/mseva.csv").Path
  gis_path = (Resolve-Path "Mohali/MohaliDataset/mohaligis.shp").Path
  electricity_path = (Resolve-Path "Mohali/MohaliDataset/electricity.csv").Path
  owner = "dashboard"
} | ConvertTo-Json

$run = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/runs" `
  -Headers @{"X-API-Key" = $env:PIPELINE_API_KEY} `
  -ContentType "application/json" `
  -Body $body
```

Response:

```json
{
  "run_id": "1c6ec26a877c",
  "status": "QUEUED",
  "status_url": "/runs/1c6ec26a877c"
}
```

Keep `run_id`. That is the handle for status, retry, and downloads.

### Request body

| Field | Required | Notes |
|---|---|---|
| `city` | yes | Letters, numbers, spaces, `-`, `_`. Used in filenames |
| `state` | yes | Free text, e.g. `Punjab` |
| `month` | yes | `YYYY-MM`, e.g. `2026-09` |
| `mseva_path` | yes | Path to the tax CSV, as seen by the server |
| `gis_path` | yes | Path to `.shp`. `.shx` and `.dbf` must sit next to it |
| `electricity_path` | no | Electricity CSV |
| `geoai_output_path` | no | Skip train/infer and use a prebuilt GeoAI CSV |
| `model_dir` | no | Knowledge-base cache directory |
| `force_retrain` | no | `true` rebuilds the knowledge base |
| `change_detection_path` | no | Change-detection shapefile |
| `steps` | no | Subset of `load`, `train`, `infer`, `match`, `defaulters`, `report` |
| `owner` | no | Who submitted the job. Default `api` |

Paths are resolved and checked on the machine running the API. A missing file returns `422` and no `run_id` is created.

The same city cannot run twice at once. The second job fails with a clear lock error; it never writes into the first run.

## 4. Poll status

`GET /runs/{run_id}`

```bash
curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
  http://127.0.0.1:8000/runs/1c6ec26a877c
```

Poll until the run reaches a terminal state:

```bash
while true; do
  status=$(curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
    http://127.0.0.1:8000/runs/1c6ec26a877c | python -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "$status"
  case "$status" in SUCCEEDED|FAILED) break;; esac
  sleep 20
done
```

Windows PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/runs/$($run.run_id)" `
  -Headers @{"X-API-Key" = $env:PIPELINE_API_KEY}
```

Statuses:

| Status | Meaning |
|---|---|
| `QUEUED` | Accepted, waiting for the worker |
| `RUNNING` | Pipeline process is active. `current_step` is the step now running |
| `SUCCEEDED` | Published. `outputs` lists downloadable files |
| `FAILED` | Nothing published to `latest`. Inspect `error` and the log |

Useful fields on the record:

- `current_step` / `last_step`
- `error`
- `log_path` (relative to `PIPELINE_OUTPUT_DIR`)
- `checkpoint` (set after inference finishes)
- `outputs` (filled only after success)

List runs:

```bash
curl -s -H "X-API-Key: $PIPELINE_API_KEY" http://127.0.0.1:8000/runs
curl -s -H "X-API-Key: $PIPELINE_API_KEY" "http://127.0.0.1:8000/runs?city=Mohali"
curl -s -H "X-API-Key: $PIPELINE_API_KEY" "http://127.0.0.1:8000/runs?status=FAILED"
```

## 5. Download outputs

Only after `SUCCEEDED`.

`GET /runs/{run_id}/outputs/{filename}`

Filenames use the city name:

| File | Contents |
|---|---|
| `{City}_Match_Register.csv` | Every mSeva row, with `property_uid` = `PT:{propertyid}` |
| `{City}_Defaulters.csv` | Every GIS parcel (taxpayers, suspected, exempt) |
| `{City}_Defaulters.geojson` | Same parcels, with geometry |
| `{City}_summary.json` | Counts used by the dashboard |
| `{City}_GeoAI_Geocoded.csv` | Inference checkpoint / debug |
| `{City}_Change_Detection_Summary.csv` | Only if change detection was requested |
| `{City}_Change_Detection.geojson` | Only if change detection was requested |

Linux / macOS:

```bash
RUN_ID=1c6ec26a877c
CITY=Mohali

for file in "${CITY}_summary.json" "${CITY}_Defaulters.csv" "${CITY}_Match_Register.csv"; do
  curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
    -o "$file" \
    "http://127.0.0.1:8000/runs/$RUN_ID/outputs/$file"
done
```

Windows PowerShell:

```powershell
$id = $run.run_id
$city = "Mohali"

Invoke-WebRequest `
  -Uri "http://127.0.0.1:8000/runs/$id/outputs/${city}_summary.json" `
  -Headers @{"X-API-Key" = $env:PIPELINE_API_KEY} `
  -OutFile "${city}_summary.json"
```

On disk the same files live at:

```text
{PIPELINE_OUTPUT_DIR}/{city}/{YYYY-MM}/{run_id}/
```

That directory is never overwritten. A later month or retry gets a new `run_id`.

## 6. Latest city data (dashboard)

These read the `{city}_latest_manifest.json` pointer. They always mean "newest successful publish", not a specific historical run.

| Method | Path | Auth | Returns |
|---|---|---|---|
| `GET` | `/cities/{city}/latest` | no | Pointer: `run_id`, `month`, output hashes |
| `GET` | `/cities/{city}/summary` | no | `{City}_summary.json` |
| `GET` | `/cities/{city}/defaulters.geojson` | no | Parcel GeoJSON |
| `GET` | `/cities/{city}/match-register?limit=500&offset=0` | no | Paginated match-register rows |
| `GET` | `/cities/{city}/change-detection.geojson` | no | Change-detection GeoJSON, if published |
| `GET` | `/cities/{city}/manifest-history` | no | Every terminal receipt for that city |

```bash
curl -s http://127.0.0.1:8000/cities/Mohali/latest
curl -s http://127.0.0.1:8000/cities/Mohali/summary
curl -s "http://127.0.0.1:8000/cities/Mohali/match-register?limit=50&offset=0"
```

Prefer `/runs/{run_id}/...` when you need last month's files after a newer run has become latest.

## 7. Retry a failed run

`POST /runs/{run_id}/retry` → `202`

Only `FAILED` runs can be retried (`409` otherwise). The retry gets a **new** `run_id`. If inference finished, that checkpoint is reused.

```bash
curl -s -X POST -H "X-API-Key: $PIPELINE_API_KEY" \
  http://127.0.0.1:8000/runs/1c6ec26a877c/retry
```

```json
{
  "run_id": "9b41e0c27a55",
  "resumed_from_run_id": "1c6ec26a877c",
  "status": "QUEUED",
  "status_url": "/runs/9b41e0c27a55"
}
```

The original failed receipt stays on disk. Do not edit it.

## 8. Output contract (what the dashboard can trust)

A successful GIS snapshot contains **every loaded parcel**, including exempt and no-geometry rows.

| Field | Meaning |
|---|---|
| `propertyid` / `gis_uid` / `matched_uid` | Source IDs, unchanged |
| `property_uid` | `PT:{propertyid}` on the match register; `GIS:{gis_uid}` on parcels |
| `tax_status` | `IN_TAX_NET`, `SUSPECTED`, or `EXEMPT` |
| `geo_status` | `SHAPE`, `DOT`, or `NONE` |
| `ward_id` | GIS ward column |
| `month` | `YYYY-MM` from the request |
| `run_id` | The job id |
| `schema_version` | Currently `1.0` |

Publish is refused unless:

- parcel CSV rows = GeoJSON features = `total_gis`
- `IN_TAX_NET + SUSPECTED + EXEMPT` = emitted GIS rows
- emitted `SUSPECTED` = `potential_defaulters`
- match-register rows = `total_mseva`

## 9. Typical integration loop

1. `POST /runs` with city, month, and input paths.
2. Poll `GET /runs/{run_id}` until `SUCCEEDED` or `FAILED`.
3. On success, either:
   - download `/runs/{run_id}/outputs/...` for that month, or
   - read `/cities/{city}/latest` and `/summary` for the live dashboard.
4. On failure, read `error`, then `POST /runs/{run_id}/retry` after fixing inputs.

See [RUNBOOK.md](RUNBOOK.md) for recovery, locks, and on-disk layout.
