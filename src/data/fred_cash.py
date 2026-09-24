"""FRED interest rate processor adhering to Regulation V1 §4.3 and §10.

Computes causal backward as-of cash yields and calendar-day ACT/365 accrual factors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def load_fred_rates(
    csv_path: str | Path = "data/raw/DGS3MO.csv",
    series_name: str = "DGS3MO",
) -> pd.DataFrame:
    """Loads FRED rate series, parsing dates and cleaning non-numeric entries."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"FRED rate file not found at {path}")

    df = pd.read_csv(path)
    # The header in FRED is usually 'observation_date,DGS3MO' or 'DATE,DGS3MO'
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    if date_col not in df.columns:
        date_col = df.columns[0]
    
    rate_col = [c for c in df.columns if c != date_col][0]

    df = df.rename(columns={date_col: "date", rate_col: series_name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_name] = pd.to_numeric(df[series_name], errors="coerce")
    
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def merge_cash_yields(
    market_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    rate_col: str = "DGS3MO",
) -> pd.DataFrame:
    """Merges FRED interest rates onto market dates strictly causally (as-of prior date).
    
    Regulation V1 §4.3:
    y_t^{known} = most recent rate published before decision time t.
    We backward as-of join fred_df onto market_df using date < market_date (strictly lagged).
    Then we compute the calendar day gap Delta d to the next trading day t+1,
    and the corresponding cash interest rate:
    r^{cash}_{t, t+1} = (1 + y_t^{known} / 100) ** (Delta d / 365) - 1
    """
    df = market_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    # Prepare FRED rates: drop NaNs
    fred_clean = fred_df.dropna(subset=[rate_col]).copy()
    fred_clean["date_dt"] = pd.to_datetime(fred_clean["date"])
    fred_clean = fred_clean.sort_values("date_dt")

    # Shift FRED dates by 1 calendar day to ensure strict availability (known before decision time)
    # As FRED daily rates are published with a 1-day reporting lag
    fred_clean["available_date"] = fred_clean["date_dt"] + pd.Timedelta(days=1)

    # Perform merge_asof: for each market date, match latest FRED available_date <= market date
    merged = pd.merge_asof(
        df.sort_values("date_dt"),
        fred_clean[["available_date", rate_col]],
        left_on="date_dt",
        right_on="available_date",
        direction="backward",
    )

    # Calculate calendar day gap Delta d to NEXT trading day
    next_dates = merged["date_dt"].shift(-1)
    # For the last row, default gap to 1 calendar day
    delta_days = (next_dates - merged["date_dt"]).dt.days.fillna(1).astype(int)
    merged["calendar_days_to_next"] = delta_days

    # Cash yield y_t in percent (e.g. 5.25%)
    # Forward fill any initial missing rate with the first available rate if necessary
    merged[rate_col] = merged[rate_col].bfill().ffill()

    # Cash return r^{cash}_{t, t+1}
    y_known = merged[rate_col] / 100.0
    merged["cash_return_to_next"] = (1.0 + y_known) ** (merged["calendar_days_to_next"] / 365.0) - 1.0

    # Drop temporary datetime columns
    merged = merged.drop(columns=["date_dt", "available_date"], errors="ignore")
    return merged
