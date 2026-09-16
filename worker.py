"""Execute one persisted API job in an isolated Python process."""

from __future__ import annotations

import argparse
import sys

from .config import CityConfig
from .run_manager import RefreshFailedError, execute_pipeline_run
from .run_registry import load_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    record = load_run(args.output_dir, args.run_id)
    if not record:
        print(f"Unknown run_id: {args.run_id}", file=sys.stderr)
        return 2
    request = record["request"]
    config = CityConfig(
        name=request["city"],
        state=request["state"],
        mseva_path=request["mseva_path"],
        gis_path=request.get("gis_path"),
        electricity_path=request.get("electricity_path"),
        geoai_output_path=request.get("geoai_output_path"),
        output_dir=args.output_dir,
        model_dir=request.get("model_dir"),
        force_retrain=bool(request.get("force_retrain", False)),
    )
    try:
        manifest = execute_pipeline_run(
            config,
            steps=request.get("steps"),
            change_detection_path=request.get("change_detection_path"),
            month=request.get("month"),
            run_id=args.run_id,
            owner=record.get("owner", "api"),
            resume_from_run_id=request.get("resume_from_run_id"),
        )
    except RefreshFailedError as e:
        print(e, file=sys.stderr)
        return 1

    print(
        f"Run {manifest['run_id']} succeeded; "
        f"published {len(manifest.get('outputs', {}))} outputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

