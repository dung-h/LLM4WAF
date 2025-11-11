#!/usr/bin/env python3
"""
Combine existing SQLi dataset with enriched XSS dataset to create v6 multi-attack splits.

Inputs:
  - data/processed/red_train.jsonl
  - data/processed/red_val.jsonl
  - data/processed/red_test.jsonl
  - data/processed/xss_enriched_deepseek.jsonl

Outputs:
  - data/processed/red_train_v6_multi.jsonl
  - data/processed/red_val_v6_multi.jsonl
  - data/processed/red_test_v6_multi.jsonl

The script preserves original fields and adds `attack_type` when missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
import random

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tag_attack_type(rows: List[Dict], default_type: str) -> List[Dict]:
    out = []
    for r in rows:
        t = r.get("attack_type")
        if not t:
            r["attack_type"] = default_type
        out.append(r)
    return out


def main() -> None:
    # Load SQLi baseline splits
    train_sql = read_jsonl(PROCESSED / "red_train.jsonl")
    val_sql = read_jsonl(PROCESSED / "red_val.jsonl")
    test_sql = read_jsonl(PROCESSED / "red_test.jsonl")

    train_sql = tag_attack_type(train_sql, "SQLi")
    val_sql = tag_attack_type(val_sql, "SQLi")
    test_sql = tag_attack_type(test_sql, "SQLi")

    # Load XSS enriched set (single pool; we will split 80/10/10)
    xss_all = read_jsonl(PROCESSED / "xss_enriched_deepseek.jsonl")

    # Shuffle deterministically for reproducibility
    rng = random.Random(42)
    rng.shuffle(xss_all)

    n = len(xss_all)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    xss_train = xss_all[:n_train]
    xss_val = xss_all[n_train:n_train + n_val]
    xss_test = xss_all[n_train + n_val:]

    # Combine
    train = train_sql + xss_train
    val = val_sql + xss_val
    test = test_sql + xss_test

    # Write
    write_jsonl(PROCESSED / "red_train_v6_multi.jsonl", train)
    write_jsonl(PROCESSED / "red_val_v6_multi.jsonl", val)
    write_jsonl(PROCESSED / "red_test_v6_multi.jsonl", test)

    print("Done.")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")


if __name__ == "__main__":
    main()

