# Property Tax Pipeline Runbook

## 1. Inputs

A full refresh needs:

- mSeva CSV
- GIS shapefile bundle (`.shp`, `.shx`, `.dbf`, preferably `.prj`)
- optional electricity CSV
- city, state, and data month in `YYYY-MM`

Run commands from the directory containing `pipeline_geoai/`.

The pipeline runs on Linux, macOS, and Windows. Examples use bash and `curl`;
a PowerShell equivalent is shown where the shell syntax differs.

## 2. Recommended operation: job API

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

The API uses one persistent FIFO worker. This keeps resource use predictable
and prevents two cities from competing for the model in one process.

Full request/response reference: [API.md](API.md).

Submit:

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
    "owner": "operator-name"
  }'
```

The response contains the permanent `run_id`.

## 3. Status and logs

```bash
curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
  http://127.0.0.1:8000/runs/<run_id>
```

Statuses are `QUEUED`, `RUNNING`, `SUCCEEDED`, or `FAILED`. The record also
contains the current step, timestamps, PID, error, checkpoint, log path, and
published output receipt.

On disk:

- `{output}/.run_registry/{run_id}.json` — live job state
- `{output}/.run_logs/{run_id}.log` — worker output
- `{output}/.runs/{city}_{run_id}/` — isolated work/checkpoint directory
- `{output}/.manifests/{city}/{run_id}.json` — immutable terminal receipt

## 4. Output contract

Every successful GIS parcel snapshot contains all loaded parcels, including
exempt parcels and records without drawable geometry.

Required identity/status fields include:

- source IDs: `propertyid`, `gis_uid`, and `matched_uid`
- `property_uid`: `PT:{propertyid}` or `GIS:{gis_uid}`
- `tax_status`: `IN_TAX_NET`, `SUSPECTED`, or `EXEMPT`
- `geo_status`: `SHAPE`, `DOT`, or `NONE`
- `ward_id`
- `month` in `YYYY-MM`
- `run_id`
- `schema_version`

Before publication, validation requires:

- parcel CSV rows = GeoJSON features = summary `total_gis`
- status totals = emitted GIS rows
- emitted `SUSPECTED` = summary `potential_defaulters`
- match-register rows = summary `total_mseva`
- source IDs, enums, month, run ID, and schema version are valid

A mismatch fails the run and does not update `latest`.

## 5. Immutable publication

Successful outputs are staged, flushed, and atomically renamed to:

```text
{output}/{city}/{YYYY-MM}/{run_id}/
```

That directory is never overwritten. A later run receives a new `run_id`.
Only after the directory and immutable manifest are complete is
`{city}_latest_manifest.json` atomically replaced.

Download an output from a specific run:

```bash
curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
  -o Mohali_summary.json \
  "http://127.0.0.1:8000/runs/<run_id>/outputs/Mohali_summary.json"
```

Existing latest-data endpoints remain available under `/cities/{city}/...`.

## 6. Recovery

- Client disconnect: no action; the worker continues.
- API restart: unfinished jobs whose worker PID is gone are requeued.
- Interruption after inference: the worker verifies the checkpoint hash and
  input hashes, then reruns matching/reporting without repeating inference.
- Failed run: inspect its registry error and `.run_logs/{run_id}.log`, correct
  the source/configuration, then call `POST /runs/{run_id}/retry`. The retry
  gets a new immutable run ID and reuses a valid inference checkpoint.
- Concurrent same-city request: the second run fails clearly; it never writes
  into the first run.

Do not edit a published run directory or terminal manifest.

## 7. Alerts

Set `ALERT_WEBHOOK_URL` to receive failure, anomaly, and successful-publication
events. Success is sent only after the immutable bundle and latest pointer are
complete.

## 8. Tests

Focused shell tests:

```bash
python -m unittest pipeline_geoai.test_pipeline_shell -v
```

For real Mohali acceptance, submit the job shown above and verify the
reconciliation fields in its summary and manifest.
