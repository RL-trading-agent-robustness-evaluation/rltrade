from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]


def _resolve_csv_path(raw_csv_path: str | Path) -> Path:
    path = Path(raw_csv_path)
    if path.exists():
        return path

    candidate_roots = [
        Path("data"),
        Path("data/raw"),
        Path("."),
    ]
    for root in candidate_roots:
        candidate = root / path.name
        if candidate.exists():
            return candidate
        if "raw_" not in path.name and "price" in path.name:
            raw_candidate = root / f"raw_{path.name}"
            if raw_candidate.exists():
                return raw_candidate
    return path


def read_raw_csv(raw_csv_path: str | Path) -> pd.DataFrame:
    path = _resolve_csv_path(raw_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def sanity_bounds_check(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue

        if df[col].isnull().any():
            issues.append(f"Null values detected in {col}: {int(df[col].isnull().sum())}")

        if col in {"open", "high", "low", "close"}:
            bad = df[col].le(0)
            if bad.any():
                issues.append(f"Non-positive values in {col}: {int(bad.sum())}")
        elif col == "volume":
            bad = df[col].lt(0)
            if bad.any():
                issues.append(f"Negative volume values: {int(bad.sum())}")
            zero_volume = df[col].eq(0)
            if zero_volume.any():
                issues.append(f"Zero volume values: {int(zero_volume.sum())}")

    return issues


def ohlc_constraints_check(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []

    for idx, row in df.iterrows():
        high, low, open_, close = row["high"], row["low"], row["open"], row["close"]

        if pd.isna(high) or pd.isna(low) or pd.isna(open_) or pd.isna(close):
            continue

        checks = [
            (high >= open_, f"High < Open on {row['date']}"),
            (high >= close, f"High < Close on {row['date']}"),
            (low <= open_, f"Low > Open on {row['date']}"),
            (low <= close, f"Low > Close on {row['date']}"),
        ]
        for ok, message in checks:
            if not ok:
                issues.append(message)

    return issues[:20]


def random_sampling_check(
    df: pd.DataFrame,
    sample_size: int = 20,
    benchmark_df: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    verified_dates: List[str] = []
    n = min(sample_size, len(df))
    rng = random.Random(random_seed)
    sample_dates = sorted(rng.sample(df["date"].tolist(), n))

    for date_value in sample_dates:
        row = df.loc[df["date"] == date_value].iloc[0]
        verified_dates.append(date_value)

        if row["volume"] <= 0:
            issues.append(f"Sampled date has non-positive volume: {date_value}")

        for col in ["open", "high", "low", "close"]:
            if pd.isna(row[col]):
                issues.append(f"Sampled date has null {col}: {date_value}")

        if benchmark_df is not None and "date" in benchmark_df.columns:
            benchmark_row = benchmark_df.loc[benchmark_df["date"] == date_value]
            if benchmark_row.empty:
                issues.append(f"Benchmark missing for sampled date: {date_value}")
                continue
            benchmark = benchmark_row.iloc[0]
            for col in ["open", "high", "low", "close"]:
                if col in benchmark_df.columns and pd.notna(benchmark[col]):
                    if abs(float(row[col]) - float(benchmark[col])) > 1e-6:
                        issues.append(f"Benchmark mismatch on {date_value} for {col}")

    return issues, verified_dates


def corporate_event_check(
    df: pd.DataFrame,
    reference_dividend_dates: Optional[Sequence[str]] = None,
    reference_split_dates: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    observed_dividends = df.loc[df.get("divCash", 0) > 0, "date"].astype(str).tolist()
    observed_splits = df.loc[df.get("splitFactor", 1) != 1, "date"].astype(str).tolist()

    issues: List[str] = []

    if reference_dividend_dates is not None:
        expected = set(pd.Series(reference_dividend_dates).astype(str))
        missing = sorted(expected - set(observed_dividends))
        if missing:
            issues.append(f"Missing expected dividend ex-dates: {missing}")

    if reference_split_dates is not None:
        expected = set(pd.Series(reference_split_dates).astype(str))
        missing = sorted(expected - set(observed_splits))
        if missing:
            issues.append(f"Missing expected split dates: {missing}")

    return {
        "observed_dividend_dates": observed_dividends,
        "observed_split_dates": observed_splits,
        "issues": issues,
    }


def validate_raw_csv(
    raw_csv_path: str | Path,
    ticker: str = "SPY",
    sample_size: int = 20,
    benchmark_df: Optional[pd.DataFrame] = None,
    reference_dividend_dates: Optional[Sequence[str]] = None,
    reference_split_dates: Optional[Sequence[str]] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    df = read_raw_csv(raw_csv_path)
    issues: List[str] = []

    issues.extend(sanity_bounds_check(df))
    issues.extend(ohlc_constraints_check(df))

    sample_issues, verified_dates = random_sampling_check(
        df,
        sample_size=sample_size,
        benchmark_df=benchmark_df,
        random_seed=random_seed,
    )
    issues.extend(sample_issues)

    event_data = corporate_event_check(
        df,
        reference_dividend_dates=reference_dividend_dates,
        reference_split_dates=reference_split_dates,
    )
    issues.extend(event_data["issues"])

    result = {
        "ticker": ticker.upper(),
        "passed": len(issues) == 0,
        "rows": int(len(df)),
        "date_range": [df["date"].min(), df["date"].max()],
        "sanity_issues": issues,
        "crosscheck_dates_verified": verified_dates,
        "observed_dividend_dates": event_data["observed_dividend_dates"],
        "observed_split_dates": event_data["observed_split_dates"],
    }

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw market CSV files before manifest creation.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol, e.g. SPY")
    parser.add_argument("--csv", default="data/spy_price.csv", help="Path to raw CSV")
    parser.add_argument("--sample-size", type=int, default=20, help="Random sample size for validation")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = validate_raw_csv(args.csv, ticker=args.ticker, sample_size=args.sample_size)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
