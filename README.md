# Property Tax Defaulter Identification Pipeline

Modular, city-agnostic pipeline that matches mSeva property-tax records against
GIS survey polygons to identify likely tax **defaulters** (properties present in
the GIS survey but absent from the tax roll).

It runs a layered matcher — mobile → property-id → 
spatial (point-in-polygon + buffer) → optional electricity → name + locality →  — then writes a match
register, a summary, defaulter list in csv and GeoJSON format.

## Requirements

- **Python >= 3.10**
- Dependencies listed in `requirements.txt`: `pandas`, `pyshp`, `shapely` (>= 2.0), `openai`.

## Setup

All commands below assume you start in this `pipeline` directory.

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample dataset

```bash
python create_sample_data.py
```

This writes:

- `sample_data/mseva_sample.csv` — 20 synthetic property-tax records
- `sample_data/survey_sample.shp` (+ `.shx` / `.dbf`) — matching GIS survey polygons

## First run (using the sample data)

The package is invoked with `python -m pipeline`, which must be run from the
**parent** directory (the folder that *contains* this `pipeline` directory) so
that `pipeline` is importable as a package.

Windows (PowerShell):

> **Note:** On Windows, `python` may resolve to the Microsoft Store stub. Use the `py` launcher instead.

```powershell
cd ..
py -m pipeline --city Barnala --state Punjab `
  --mseva "pipeline/sample_data/mseva_sample.csv" `
  --gis "pipeline/sample_data/survey_sample.shp" `
  --output "pipeline/results"
```

macOS / Linux:

```bash
cd ..
python -m pipeline --city Barnala --state Punjab \
  --mseva "pipeline/sample_data/mseva_sample.csv" \
  --gis "pipeline/sample_data/survey_sample.shp" \
  --output "pipeline/results"
```

### Outputs

After the run, `pipeline/results/` contains:

- `Barnala_Match_Register.csv` — every mSeva record with its matched GIS UID (or an unmatched reason)
- `Barnala_summary.json` — match counts and per-layer breakdown
- `Barnala_Defaulters.csv` — GIS polygons with no matching tax record (potential defaulters)
- `Barnala_Defaulters.geojson` - GIS polygons to visualize on the map

## Pipeline steps

Steps run by default (when `--steps` is omitted): **`load` → `match` → `defaulters` → `report`**.

The default run expects the mSeva CSV to already contain `latitude` / `longitude`
columns (the sample data does). The optional `parse_addresses` and `geocode`
steps are **not** run by default and require API keys (see below). To run a
specific subset, pass e.g. `--steps load match report`. A standalone data-quality
check is available via `--steps quality_check`.

## API keys (optional)

API keys are only needed for the optional steps:

- `parse_addresses` (LLM address parsing) — talks to an OpenAI-compatible LLM gateway, configured via:
  - `LLM_API_KEY` — gateway API key (required)
  - `LLM_BASE_URL` — gateway base URL (default `http://35.154.241.163:8502`)
  - `LLM_PROVIDER` — provider name (default `airawat`)
  - `LLM_MODEL` — model name (default `qwen3-30b-a3b-instruct`)
- `geocode` — `GOOGLE_MAPS_API_KEY`

Copy `.env.example` to `.env`, fill in your values, and export them as
environment variables (the pipeline reads them from the environment). No keys
are required for the default sample run above (which omits those two steps).
