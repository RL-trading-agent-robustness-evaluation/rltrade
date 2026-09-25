"""Data adapter providing unified canonical market data adhering to Regulation V1 §2 & §4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.fred_cash import load_fred_rates, merge_cash_yields


CANONICAL_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dividend_amount",
    "split_factor",
    "DGS3MO",
    "calendar_days_to_next",
    "cash_return_to_next",
]


def load_raw_tiingo(csv_path: str | Path = "data/raw/spy_price_tiingo.csv") -> pd.DataFrame:
    """Loads and standardizes raw Tiingo CSV into intermediate schema."""
    path = Path(csv_path)
    if not path.exists():
        # Fallback to data/spy_price.csv
        path = Path("data/spy_price.csv")
    if not path.exists():
        raise FileNotFoundError(f"Tiingo raw file not found at {path}")

    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "divCash": "dividend_amount",
            "splitFactor": "split_factor",
        }
    )
    if "dividend_amount" not in df.columns:
        df["dividend_amount"] = 0.0
    if "split_factor" not in df.columns:
        df["split_factor"] = 1.0

    df["dividend_amount"] = df["dividend_amount"].fillna(0.0)
    df["split_factor"] = df["split_factor"].fillna(1.0)

    # Standardize date format to YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def validate_canonical_dataset(df: pd.DataFrame) -> Tuple[bool, list[str]]:
    """Runs Gate D1 assertions per Regulation V1 §15."""
    issues = []

    # 1. Monotonic increasing dates & unique
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        issues.append("Dates are not monotonically increasing")
    if df["date"].duplicated().any():
        issues.append(f"Found {df['date'].duplicated().sum()} duplicate dates")

    # 2. Positive prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            issues.append(f"Non-positive values found in {col}")
        if df[col].isna().any():
            issues.append(f"NaN values found in {col}")

    # 3. OHLC bounds
    bad_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
    bad_low = (df["low"] > df["open"]) | (df["low"] > df["close"])
    if bad_high.any():
        issues.append(f"High constraint violated on {int(bad_high.sum())} rows")
    if bad_low.any():
        issues.append(f"Low constraint violated on {int(bad_low.sum())} rows")

    # 4. Non-negative volume
    if (df["volume"] < 0).any():
        issues.append("Negative volume found")

    # 5. Cash returns finite
    if df["cash_return_to_next"].isna().any() or np.isinf(df["cash_return_to_next"]).any():
        issues.append("Invalid cash_return_to_next values")

    return len(issues) == 0, issues


def build_canonical_dataset(
    price_csv: str | Path = "data/raw/spy_price_tiingo.csv",
    fred_csv: str | Path = "data/raw/DGS3MO.csv",
    output_csv: str | Path = "data/interim/spy_canonical.csv",
) -> pd.DataFrame:
    """Builds, validates, and saves canonical market data table with cash yields."""
    raw_df = load_raw_tiingo(price_csv)
    fred_df = load_fred_rates(fred_csv)

    merged = merge_cash_yields(raw_df, fred_df, rate_col="DGS3MO")

    # Filter to canonical columns
    for col in CANONICAL_COLUMNS:
        if col not in merged.columns:
            raise KeyError(f"Missing canonical column: {col}")

    canonical_df = merged[CANONICAL_COLUMNS].copy()

    # Run Gate D1 validation
    is_valid, issues = validate_canonical_dataset(canonical_df)
    if not is_valid:
        raise ValueError(f"Gate D1 validation failed: {issues}")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_csv(out_path, index=False)
    print(f"Canonical dataset written to {out_path} ({len(canonical_df)} rows, {canonical_df['date'].iloc[0]} to {canonical_df['date'].iloc[-1]})")
    return canonical_df


if __name__ == "__main__":
    df = build_canonical_dataset()
