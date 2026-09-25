"""Data adapter providing unified canonical market data adhering to Regulation V1 §2, §4 & §15."""

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
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_raw_crsp(
    permno: int = 84398,
    crsp_csv: str | Path = "data/raw/crsp_dsf_spy_tlt_gld_2009_2025.csv",
    dsedist_csv: Optional[str | Path] = "data/raw/dsedist_spy_tlt_gld_2009_2025.csv",
) -> pd.DataFrame:
    """Loads and standardizes canonical CRSP 50-col daily master data for a given PERMNO."""
    path = Path(crsp_csv)
    if not path.exists():
        path = Path("data/crsp_dsf_spy_tlt_gld_2009_2025.csv")
    if not path.exists():
        raise FileNotFoundError(f"CRSP raw file not found at {path}")

    df = pd.read_csv(path)
    sub = df[df["permno"] == permno].copy()
    if sub.empty:
        raise ValueError(f"PERMNO {permno} not found in CRSP dataset")

    # Map CRSP 50-col master fields per Regulation V1 §2.3
    sub["date"] = pd.to_datetime(sub["dlycaldt"]).dt.strftime("%Y-%m-%d")
    sub["open"] = sub["dlyopen"].astype(float)
    sub["high"] = sub["dlyhigh"].astype(float)
    sub["low"] = sub["dlylow"].astype(float)
    sub["close"] = sub["dlyclose"].astype(float)
    sub["volume"] = sub["dlyvol"].fillna(0.0).astype(float)

    # Corporate actions: ordinary cash dividends
    # dlyorddivamt contains dividend amounts on ex-dates directly from CRSP master
    if "dlyorddivamt" in sub.columns:
        sub["dividend_amount"] = sub["dlyorddivamt"].fillna(0.0).astype(float)
    else:
        sub["dividend_amount"] = 0.0

    # Split factors: CRSP cumulative split factor
    sub["split_factor"] = 1.0

    sub = sub.sort_values("date").reset_index(drop=True)
    return sub[["date", "open", "high", "low", "close", "volume", "dividend_amount", "split_factor"]]


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
    source: str = "crsp",
    permno: int = 84398,  # SPY
    price_csv: Optional[str | Path] = None,
    fred_csv: str | Path = "data/raw/DGS3MO.csv",
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Builds, validates, and saves canonical market data table with cash yields."""
    if source == "crsp":
        if price_csv is None:
            price_csv = "data/raw/crsp_dsf_spy_tlt_gld_2009_2025.csv"
        raw_df = load_raw_crsp(permno=permno, crsp_csv=price_csv)
        if output_csv is None:
            name_map = {84398: "spy", 89468: "tlt", 90448: "gld"}
            asset_name = name_map.get(permno, f"permno_{permno}")
            output_csv = f"data/interim/{asset_name}_canonical_crsp.csv"
    else:
        if price_csv is None:
            price_csv = "data/raw/spy_price_tiingo.csv"
        raw_df = load_raw_tiingo(price_csv)
        if output_csv is None:
            output_csv = "data/interim/spy_canonical.csv"

    fred_df = load_fred_rates(fred_csv)
    merged = merge_cash_yields(raw_df, fred_df, rate_col="DGS3MO")

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
    print(f"Canonical {source.upper()} dataset written to {out_path} ({len(canonical_df)} rows, {canonical_df['date'].iloc[0]} to {canonical_df['date'].iloc[-1]})")
    return canonical_df


if __name__ == "__main__":
    # Build canonical CRSP dataset for SPY, TLT, and GLD
    for p, name in [(84398, "SPY"), (89468, "TLT"), (90448, "GLD")]:
        build_canonical_dataset(source="crsp", permno=p)
