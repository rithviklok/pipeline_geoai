# Property Tax Defaulter Identification Pipeline

Modular, city-agnostic pipeline that matches mSeva property-tax records against
GIS survey polygons to identify likely tax **defaulters** (properties present in
the GIS survey but absent from the tax roll).

It runs a layered matcher:
1. **GeoAI Semantic Matching:** Vector-based embedding matching (FAISS + SentenceTransformers) against the GIS Knowledge Base.
2. **Mobile Number Matching:** Cross-referencing 10-digit mobile numbers.
3. **Electricity Matching:** Point-in-polygon matching of electricity billing coordinates against GIS building footprints.
4. **Name + Locality Matching:** Dynamic cross-referencing of names within localized regions.

It also integrates **Google Earth Engine Change Detection** to identify massive swaths of newly constructed un-taxed buildings (e.g. new colonies).

## Requirements

- **Python >= 3.10**
- Dependencies listed in equirements.txt: pandas, geopandas, scikit-learn, sentence-transformers, aiss-cpu, apidfuzz, tree, pyogrio, etc.

## Setup

All commands below assume you start in the pipeline_geoai directory.

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

The package is invoked with python -m pipeline_geoai, which must be run from the
**parent** directory (the folder that *contains* this pipeline_geoai directory) so
that pipeline_geoai is importable as a package.

Windows (PowerShell):

> **Note:** On Windows, python may resolve to the Microsoft Store stub. Use the py launcher instead.

`powershell
cd ..
py -m pipeline_geoai --city Barnala --state Punjab 
  --mseva "Barnala/mseva_enriched.csv" 
  --gis "Barnala/gissurvey.shp" 
  --electricity "Barnala/electricity.csv" 
  --output "pipeline_geoai/results" 
  --change-detection "Barnala/Barnala_CD/Barnala_New_Constructions.shp"
`

macOS / Linux:

`ash
cd ..
python -m pipeline_geoai --city Barnala --state Punjab \
  --mseva "Barnala/mseva_enriched.csv" \
  --gis "Barnala/gissurvey.shp" \
  --electricity "Barnala/electricity.csv" \
  --output "pipeline_geoai/results" \
  --change-detection "Barnala/Barnala_CD/Barnala_New_Constructions.shp"
`

### Force Retraining the GeoAI Knowledge Base
If you want to clear the AI cache and force it to re-embed all 50,000+ addresses, append --force-retrain.

### Outputs

After the run, the output directory contains:

- [City]_Match_Register.csv — every mSeva record with its matched GIS UID
- [City]_Defaulters.csv — GIS polygons with no matching tax record (potential defaulters)
- [City]_Defaulters.geojson - GIS polygons to visualize on the map
- [City]_Change_Detection_Summary.csv - The 'Locality Bucket' math subtracting matched taxpayers from new construction polygons.
- [City]_summary.json — match counts and per-layer breakdown

*(Note: CSV files will be automatically stamped with City, State, and Report_Month columns at the end of the run to facilitate bulk-ingestion into PostgreSQL databases).*
