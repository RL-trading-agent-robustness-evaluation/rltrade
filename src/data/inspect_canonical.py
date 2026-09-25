import hashlib
import pandas as pd
from pathlib import Path

files = [
    "crsp_dsf_spy_tlt_gld_2009_2025.csv",
    "dsedist_spy_tlt_gld_2009_2025.csv",
    "DGS3MO.csv",
    "DTB3.csv"
]
data_dir = Path("data")

print("=== CANONICAL DATA AUDIT ===")
for f in files:
    p = data_dir / f
    if not p.exists():
        print(f"Missing: {f}")
        continue
    content = p.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    df = pd.read_csv(p)
    print(f"File: {f}")
    print(f"  Size: {len(content):,} bytes")
    print(f"  SHA-256: {sha}")
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns[:10])}")
    date_cols = [c for c in df.columns if "date" in c.lower() or "dt" in c.lower()]
    if date_cols:
        dcol = date_cols[0]
        print(f"  Date column ({dcol}): {df[dcol].min()} to {df[dcol].max()}")
    if "PERMNO" in df.columns:
        print(f"  PERMNOs: {df['PERMNO'].unique().tolist()}")
    print("-" * 50)
