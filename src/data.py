import pandas as pd

from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Put students.csv here: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def validate_data(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is not a CSV column. Available: {', '.join(df.columns)}. "
            "Update TARGET_COLUMN in src/config.py."
        )
    features = FEATURE_COLUMNS or [c for c in df.columns if c != TARGET_COLUMN]
    if not features:
        raise ValueError("No input columns found.")
    return features
