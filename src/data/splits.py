"""Data splitting protocol adhering to Regulation V1 §3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2019-01-02"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-09-14"


def split_dataset(
    df: pd.DataFrame,
    cfg: SplitConfig = SplitConfig(),
) -> Dict[str, pd.DataFrame]:
    """Splits canonical DataFrame into Train, Validation, and Locked Test sets."""
    df_clean = df.copy()
    df_clean["date_dt"] = pd.to_datetime(df_clean["date"])

    train_mask = (df_clean["date_dt"] >= cfg.train_start) & (df_clean["date_dt"] <= cfg.train_end)
    val_mask = (df_clean["date_dt"] >= cfg.val_start) & (df_clean["date_dt"] <= cfg.val_end)
    test_mask = (df_clean["date_dt"] >= cfg.test_start) & (df_clean["date_dt"] <= cfg.test_end)

    train_df = df_clean[train_mask].drop(columns=["date_dt"]).reset_index(drop=True)
    val_df = df_clean[val_mask].drop(columns=["date_dt"]).reset_index(drop=True)
    test_df = df_clean[test_mask].drop(columns=["date_dt"]).reset_index(drop=True)

    print(f"Data Splits: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows")
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def save_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: str | Path = "data/processed",
) -> None:
    """Saves splits to data/processed directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Splits saved to {out_dir}")
