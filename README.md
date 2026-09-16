# Property Tax GeoAI Pipeline

The pipeline matches mSeva tax records to GIS parcels and publishes a complete,
versioned property snapshot. Matching logic remains in `matchers/`; the
run-management shell provides durable jobs, validation, and immutable outputs.

## Requirements

- Python 3.10+
- `pip install -r pipeline_geoai/requirements.txt`
- Run package commands from the directory containing `pipeline_geoai/`

The pipeline runs on Linux, macOS, and Windows. Examples show Linux/macOS
first, with a PowerShell equivalent where the shell syntax differs.

## Run from the CLI

Linux / macOS:

```bash
python -m pipeline_geoai \
  --city "Mohali" \
  --state "Punjab" \
  --month "2026-09" \
  --mseva "Mohali/MohaliDataset/mseva.csv" \
  --gis "Mohali/MohaliDataset/mohaligis.shp" \
  --electricity "Mohali/MohaliDataset/electricity.csv" \
  --output "pipeline_geoai/results" \
  --owner "operator-name"
```

Windows PowerShell:

```powershell
python -m pipeline_geoai `
  --city "Mohali" `
  --state "Punjab" `
  --month "2026-09" `
  --mseva "Mohali/MohaliDataset/mseva.csv" `
  --gis "Mohali/MohaliDataset/mohaligis.shp" `
  --electricity "Mohali/MohaliDataset/electricity.csv" `
  --output "pipeline_geoai/results" `
  --owner "operator-name"
```

The command prints its `run_id`. A successful run is published to:

```text
results/{city}/{YYYY-MM}/{run_id}/
```

Published directories and terminal manifests are immutable. The
`{city}_latest_manifest.json` file is an atomic pointer to the newest successful
run.

## Run as a job API

Linux / macOS:

```bash
export PIPELINE_OUTPUT_DIR="pipeline_geoai/results"
export PIPELINE_API_KEY="replace-with-a-secret"
python -m pipeline_geoai.api
```

Windows PowerShell:

```powershell
$env:PIPELINE_OUTPUT_DIR = "pipeline_geoai/results"
$env:PIPELINE_API_KEY = "replace-with-a-secret"
python -m pipeline_geoai.api
```

Submit a full job and poll it:

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

curl -s -H "X-API-Key: $PIPELINE_API_KEY" \
  http://127.0.0.1:8000/runs/<run_id>
```

The API returns `202` immediately. Jobs continue if the submitting client
disconnects. If the API process is restarted, unfinished jobs are requeued and
a valid inference checkpoint is reused.

Retry a failed run with `POST /runs/{run_id}/retry`. The retry receives a new
run ID and reuses the failed run's validated inference checkpoint when present;
the original failed receipt remains immutable.

See [API.md](API.md) for every endpoint, request body, and download URL.
See [RUNBOOK.md](RUNBOOK.md) for recovery, locks, and on-disk layout.
