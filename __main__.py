"""
__main__.py — CLI entry point for the GeoAI Property Tax Pipeline.

Supports two modes:
  1. Full pipeline: train → infer → match → defaulters → report
  2. Pre-built mode: --geoai-output → match → defaulters → report (backward compatible)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import CityConfig
from .quality_check import QualityAssessor

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main():
    parser = argparse.ArgumentParser(
        description="GeoAI Property Tax Defaulter Identification Pipeline"
    )
    parser.add_argument("--city", required=True, help="City name (e.g. Barnala, Batala)")
    parser.add_argument("--state", required=True, help="State name (e.g. Punjab)")
    parser.add_argument("--mseva", required=True, help="Path to mSeva property tax CSV")
    parser.add_argument("--gis", required=True, help="Path to GIS shapefile (.shp/.gdb/.gpkg)")
    parser.add_argument("--geoai-output",
        help="Path to pre-built GeoAI geocoded CSV. If omitted, pipeline runs "
             "training + inference internally.")
    parser.add_argument("--electricity", help="Optional path to electricity CSV")
    parser.add_argument("--output", default="results", help="Output directory for results")

    # GeoAI training/inference options
    parser.add_argument("--model-dir",
        help="Directory for Knowledge Base cache (default: models/{city}/)")
    parser.add_argument("--force-retrain", action="store_true",
        help="Force re-training even if Knowledge Base is cached")
    parser.add_argument("--change-detection", type=str,
        help="Path to Earth Engine Change Detection Shapefile (.shp) for the New Constructions report.")

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["load", "quality_check", "train", "infer", "match", "defaulters", "report"],
        help="Optional specific pipeline steps to run (default: run all standard steps)",
    )
    parser.add_argument(
        "--month", type=str, default=None,
        help="Month this refresh represents, format YYYY-MM (default: current month). "
             "Recorded in the run's provenance manifest; does not filter input data.",
    )

    args = parser.parse_args()

    config = CityConfig(
        name=args.city,
        state=args.state,
        mseva_path=args.mseva,
        gis_path=args.gis,
        geoai_output_path=getattr(args, 'geoai_output', None),
        electricity_path=args.electricity,
        output_dir=args.output,
        model_dir=args.model_dir,
        force_retrain=args.force_retrain,
    )

    if args.steps and "quality_check" in args.steps:
        # Run quality check only
        assessor = QualityAssessor(config)
        report = assessor.run()
        print("\n" + "=" * 70)
        print(f"QUALITY CHECK REPORT - {config.name}")
        print("=" * 70)
        print(f"Verdict: {report['verdict']} (Score: {report['score']}/100)")
        print("\nIssues:")
        for issue in report["issues"]:
            print(f"  [ERROR] {issue}")
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  [WARNING] {warning}")

        # Save JSON output if requested
        output_path = config.output_path(f"{config.name.lower()}_quality_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {output_path}")
        return

    # Run the pipeline via run_manager: this adds a per-city lock, an
    # isolated/atomic publish (no half-finished data ever lands in
    # --output), a provenance manifest, field-list conformance validation,
    # and failure/anomaly alerting on top of the pipeline steps, change
    # detection, and CSV standardization that used to be driven inline here.
    from .run_manager import RefreshFailedError, execute_pipeline_run

    try:
        execute_pipeline_run(
            config=config,
            steps=args.steps,
            change_detection_path=args.change_detection,
            month=args.month,
        )
    except RefreshFailedError as e:
        print(f"\n[REFRESH FAILED] {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
