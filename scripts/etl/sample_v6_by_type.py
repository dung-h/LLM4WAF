#!/usr/bin/env python3
"""
Create a small v6 training set by sampling N examples per attack_type (SQLi/XSS).

Inputs:
  - data/processed/red_train_v6_multi.jsonl (combined)

Outputs:
  - data/processed/red_train_v6_small.jsonl (N per type, shuffled)

Usage:
  python scripts/etl/sample_v6_by_type.py 100
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    # Usage: python sample_v6_by_type.py [N] [SRC]
    # Default N=100, SRC=data/processed/red_train_v6_multi.jsonl
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    src = Path(sys.argv[2]) if len(sys.argv) > 2 else (PROCESSED / "red_train_v6_multi.jsonl")
    data = read_jsonl(src)

    buckets = {"SQLi": [], "XSS": []}
    for r in data:
        t = str(r.get("attack_type", "")).upper()
        if t in ("SQLI", "XSS"):
            buckets["SQLi" if t == "SQLI" else "XSS"].append(r)

    rng = random.Random(42)
    small: List[Dict] = []
    for k in ["SQLi", "XSS"]:
        rows = buckets.get(k, [])
        rng.shuffle(rows)
        small.extend(rows[:n])

    rng.shuffle(small)
    out = PROCESSED / "red_train_v6_small.jsonl"
    write_jsonl(out, small)
    print(f"Wrote {len(small)} rows to {out} from {src}")


if __name__ == "__main__":
    main()
