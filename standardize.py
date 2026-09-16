import pandas as pd

from .contracts import SCHEMA_VERSION


def standardize_csv(
    filepath,
    city,
    state,
    month,
    run_id="",
    schema_version=SCHEMA_VERSION,
):
    """Insert the standard City/State/month columns for downstream/database
    ingestion.

    `month` must be the run's YYYY-MM string (see manifest.default_month()
    or the --month CLI arg) — it is never derived from datetime.now() here,
    so re-running or re-publishing an old run's outputs never silently
    relabels them with today's date.
    """
    # Contract IDs must remain byte-for-byte strings (for example "00123").
    # Default pandas inference would coerce these to integers and break the
    # property_uid/source-ID reconciliation.
    df = pd.read_csv(
        filepath,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    values = {
        "City": city,
        "State": state,
        "month": month,
        "run_id": run_id,
        "schema_version": schema_version,
    }
    for column, value in reversed(values.items()):
        if column in df.columns:
            df.pop(column)
        df.insert(0, column, value)
    df.to_csv(filepath, index=False)
    print(f"Standardized {filepath}")

