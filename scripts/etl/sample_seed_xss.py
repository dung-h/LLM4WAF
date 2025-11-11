#!/usr/bin/env python3
"""
Sample N rows from data/raw/seed_xss.csv (deduplicated) and overwrite it.
Usage:
    python scripts/etl/sample_seed_xss.py 500
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "seed_xss.csv"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    if not SRC.exists():
        raise SystemExit(f"Missing seed file: {SRC}")
    df = pd.read_csv(SRC)
    if "payload" not in df.columns:
        if df.shape[1] == 1:
            df.columns = ["payload"]
        else:
            raise SystemExit("seed_xss.csv must have a 'payload' column or single column")
    df = df.dropna(subset=["payload"]).drop_duplicates()
    if len(df) <= n:
        print(f"Seed size {len(df)} <= {n}, keeping all.")
        df.to_csv(SRC, index=False)
        return
    df_sample = df.sample(n=n, random_state=42)
    df_sample.to_csv(SRC, index=False)
    print(f"Wrote {n} sampled seeds to {SRC}")


if __name__ == "__main__":
    main()

