# Punjab Change Detection API

On-the-fly urban growth detection for Punjab cities using Google Earth Engine.
Translates the AlphaEarth + NDBI/NDVI script into a REST API
that the Vercel dashboard calls directly.

## Architecture

```
Vercel Dashboard                This Service              Google Earth Engine
 (Himanshu)                    (Cloud Run)                  (Google Cloud)
     │                              │                             │
     │  POST /change-map            │                             │
     │  { city, year1, year2 }      │                             │
     │─────────────────────────────>│                             │
     │                              │  ee.Initialize() + compute │
     │                              │────────────────────────────>│
     │                              │         tile_url            │
     │                              │<────────────────────────────│
     │    { tile_url }              │                             │
     │<─────────────────────────────│                             │
     │                              │                             │
     │  L.tileLayer(tile_url)       │                             │
     │──────────────────────────────────────────────────────────>│
     │          map tiles rendered                                │
     │<──────────────────────────────────────────────────────────│
```

## Run locally

```powershell
cd change-detection-api
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs to see the interactive Swagger UI.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | No | Service health + supported cities |
| `GET` | `/cities` | No | List cities with bounds and GIS survey years |
| `POST` | `/change-map` | No | Compute change detection, return map tile URL |
| `POST` | `/area-stats` | No | Compute new built-up area in sq km |
| `DELETE` | `/cache` | No | Clear cached tile URLs |

### `POST /change-map`

**Request:**
```json
{
  "city": "Mohali",
  "year1": 2014,
  "year2": 2025
}
```

**Response:**
```json
{
  "tile_url": "https://earthengine.googleapis.com/v1/projects/.../tiles/{z}/{x}/{y}",
  "city": "Mohali",
  "year1": 2014,
  "year2": 2025
}
```

**Frontend usage (Leaflet):**
```javascript
const res = await fetch('https://<DEPLOYED_URL>/change-map', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ city: 'Mohali', year1: 2014, year2: 2025 })
});
const { tile_url } = await res.json();
L.tileLayer(tile_url).addTo(map);
```

## Deploy to Cloud Run

```bash
gcloud run deploy punjab-cd-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars GEE_PROJECT_ID=your-gee-project-id
```

The Cloud Run service account must have Earth Engine access.
No API key files are needed — Cloud Run injects credentials automatically.
