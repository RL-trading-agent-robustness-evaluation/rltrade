from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

from manifest.validate import validate_raw_csv


def generate_manifest(ticker: str, raw_csv_path: str, manifest_out_path: str) -> Dict[str, Any]:
    ticker_upper = ticker.upper()
    csv_path = Path(raw_csv_path)
    if not csv_path.exists():
        data_candidates = [
            Path("data") / f"{ticker_upper.lower()}_price.csv",
            Path("data") / f"raw_{ticker_upper.lower()}.csv",
            Path("data") / f"raw_{ticker_upper}.csv",
            Path("data") / f"{ticker_upper}.csv",
        ]
        for candidate in data_candidates:
            if candidate.exists():
                csv_path = candidate
                break

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found for ticker {ticker_upper}: {raw_csv_path}")

    validation_result = validate_raw_csv(csv_path, ticker=ticker_upper, sample_size=20)
    if not validation_result["passed"]:
        raise ValueError(
            f"Validation failed for {ticker_upper}: {validation_result['sanity_issues'][:10]}"
        )

    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "ticker": ticker_upper,
        "source": "tiingo",
        "acquired_date": date.today().isoformat(),
        "sha256": sha256,
        "rows": validation_result["rows"],
        "date_range": validation_result["date_range"],
        "adjustment": "raw_unadjusted",
        "crosscheck_source": "CRSP/Manual",
        "crosscheck_dates_verified": validation_result["crosscheck_dates_verified"],
        "missing_data_decisions": "forward-fill NONE; drop non-trading days",
    }

    out_path = Path(manifest_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cryptographic metadata manifest for raw market data.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol, e.g. SPY")
    parser.add_argument("--csv", default="data/spy_price.csv", help="Path to raw CSV file")
    parser.add_argument("--out", default="manifests/data_manifest_SPY.json", help="Output manifest path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest = generate_manifest(args.ticker, args.csv, args.out)
    print(json.dumps(manifest, indent=2))
