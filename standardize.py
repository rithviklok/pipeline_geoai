import pandas as pd
from datetime import datetime

def standardize_csv(filepath, city, state):
    try:
        df = pd.read_csv(filepath)
        df.insert(0, "City", city)
        df.insert(1, "State", state)
        df.insert(2, "Report_Month", datetime.now().strftime("%B %Y"))
        df.to_csv(filepath, index=False)
        print(f"Standardized {filepath}")
    except Exception as e:
        print(f"Could not standardize {filepath}: {e}")

