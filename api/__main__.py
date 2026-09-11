"""Run the read-only API with `py -m pipeline_geoai.api`.

Environment variables:
    PIPELINE_OUTPUT_DIR  Output directory to serve (default: "results")
    API_HOST              Bind host (default: "127.0.0.1")
    API_PORT              Bind port (default: 8000)
    API_RELOAD             Set to any non-empty value to enable autoreload
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "pipeline_geoai.api.app:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=bool(os.environ.get("API_RELOAD")),
    )
