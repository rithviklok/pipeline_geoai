# Property Tax Defaulter Identification Pipeline

Modular, city-agnostic AI pipeline that matches mSeva property-tax records against GIS survey polygons to identify likely tax **defaulters** (properties present in the GIS survey but absent from the tax roll).

It runs a layered matcher:
1. **GeoAI Semantic Matching:** Vector-based embedding matching (FAISS + SentenceTransformers) against the GIS Knowledge Base.
2. **Mobile Number Matching:** Cross-referencing 10-digit mobile numbers.
3. **Electricity Matching:** Point-in-polygon matching of electricity billing coordinates against GIS building footprints.
4. **Name + Locality Matching:** Dynamic cross-referencing of names within localized regions.

It also integrates **Google Earth Engine Change Detection** to identify massive swaths of newly constructed un-taxed buildings (e.g. new colonies).

## Production Features

This pipeline is hardened for production monthly-refreshes:
- **Run Manager:** Enforces per-city locking to prevent concurrent runs from corrupting data.
- **Atomic Publish:** Outputs are built in an isolated .runs/ directory and only published to esults/ if the *entire* process succeeds.
- **Data Validation:** Enforces a strict, signed-off Data Dictionary schema before publishing.
- **Read-Only API:** A FastAPI backend server to serve the latest published GeoJSON and CSV outputs directly to the Vercel Dashboard.

For comprehensive operational instructions, troubleshooting, and API usage, see the **[Operating Runbook](RUNBOOK.md)**.

## Requirements

- **Python >= 3.10**
- Dependencies listed in equirements.txt: pandas, geopandas, scikit-learn, sentence-transformers, aiss-cpu, apidfuzz, tree, pyogrio, astapi, uvicorn, etc.

## Setup

### 1. Create and activate a virtual environment

Windows (PowerShell):
`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
`

macOS / Linux:
`ash
python3 -m venv .venv
source .venv/bin/activate
`

### 2. Install dependencies

`ash
pip install -r requirements.txt
`

## Running the Pipeline

The package is invoked with python -m pipeline_geoai, which must be run from the **parent** directory (the folder that *contains* this pipeline_geoai directory).

Windows (PowerShell):
`powershell
cd ..
py -m pipeline_geoai --city Barnala --state Punjab 
  --mseva "Barnala\mseva_enriched.csv" 
  --gis "Barnala\gissurvey.shp" 
  --electricity "Barnala\electricity.csv" 
  --output "pipeline_geoai\results" 
  --change-detection "Barnala\Barnala_CD\Barnala_New_Constructions.shp"
`

## Running the API Dashboard Backend

To serve the generated pipeline outputs to the frontend Vercel Dashboard:
`powershell
 = "pipeline_geoai\results"
py -m pipeline_geoai.api
`
This spins up a read-only FastAPI service (default http://127.0.0.1:8000) providing endpoints like /cities/{city}/defaulters.geojson and /cities/{city}/summary.

---
*Please refer to [RUNBOOK.md](RUNBOOK.md) for full documentation on alert hooks, failure recovery, and provenance manifests.*
