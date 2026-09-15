import pandas as pd

def standardize_csv(filepath, city, state, month):
    """Insert the standard City/State/month columns for downstream/database
    ingestion.

    `month` must be the run's YYYY-MM string (see manifest.default_month()
    or the --month CLI arg) — it is never derived from datetime.now() here,
    so re-running or re-publishing an old run's outputs never silently
    relabels them with today's date.
    """
    try:
        df = pd.read_csv(filepath)
        df.insert(0, "City", city)
        df.insert(1, "State", state)
        df.insert(2, "month", month)
        df.to_csv(filepath, index=False)
        print(f"Standardized {filepath}")
    except Exception as e:
        print(f"Could not standardize {filepath}: {e}")

