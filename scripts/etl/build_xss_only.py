#!/usr/bin/env python3
from __future__ import annotations

"""
Build XSS-only training set from enriched XSS records.

Inputs:
  - data/processed/xss_enriched_deepseek.jsonl

Outputs:
  - data/processed/red_train_xss_only.jsonl (all)
  - data/processed/red_train_xss_only_small.jsonl (if --limit provided)
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for small set")
    args = ap.parse_args()

    src = PROCESSED / "xss_enriched_deepseek.jsonl"
    if not src.exists():
        raise SystemExit("Missing data/processed/xss_enriched_deepseek.jsonl")
    rows = read_jsonl(src)

    # Ensure attack_type is XSS (enricher already sets it)
    for r in rows:
        r["attack_type"] = "XSS"

    out_all = PROCESSED / "red_train_xss_only.jsonl"
    write_jsonl(out_all, rows)
    print(f"Wrote all XSS records: {len(rows)} -> {out_all}")

    if args.limit and args.limit > 0:
        small = rows[: args.limit]
        out_small = PROCESSED / "red_train_xss_only_small.jsonl"
        write_jsonl(out_small, small)
        print(f"Wrote small XSS set: {len(small)} -> {out_small}")


if __name__ == "__main__":
    main()

