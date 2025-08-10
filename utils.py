import os
import pandas as pd
import numpy as np

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_first_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # Ignore case matching
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def to_datetime_safe(s, tz=None):
    # Just hand it over to pandas for full fault tolerance. tz did not enforce it to avoid confusion
    return pd.to_datetime(s, errors="coerce", utc=False)

def minutes(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 60.0

def service_day(dt: pd.Series) -> pd.Series:
    # Simplified service day: Directly take the date (If it crosses midnight, please provide the "operating day" time when entering the reference)
    return pd.to_datetime(dt).dt.date
