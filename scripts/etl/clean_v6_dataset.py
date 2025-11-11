#!/usr/bin/env python3
from __future__ import annotations

"""
Clean v6 multi-attack datasets by removing placeholder/noisy payloads and
enforcing minimal validity rules per attack type.

Inputs (expected):
  - data/processed/red_train_v6_multi.jsonl
  - data/processed/red_val_v6_multi.jsonl
  - data/processed/red_test_v6_multi.jsonl

Outputs:
  - data/processed/red_train_v6_multi_clean.jsonl
  - data/processed/red_val_v6_multi_clean.jsonl
  - data/processed/red_test_v6_multi_clean.jsonl
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


BAD_SUBSTRINGS = {
    "(*)",
    "#__TIME__",
    "#__FILE__",
    "#__LINE__",
    "::text||",
    "chr (*)",
}

# Simple regex for long hex blobs (likely garbage): 0x followed by >=16 hex chars
HEX_BLOB_RE = re.compile(r"0x[0-9a-fA-F]{16,}")


def looks_like_sqli(payload: str) -> bool:
    s = payload.lower()
    if any(b in s for b in BAD_SUBSTRINGS):
        return False
    if HEX_BLOB_RE.search(s):
        return False
    # Must contain at least one classic SQLi indicator
    good = [
        " or ",
        " and ",
        " union select",
        " order by ",
        "--",
        "@@version",
        " sleep(",
        " benchmark(",
        " extractvalue(",
        " updatexml(",
    ]
    return any(k in s for k in good)


def looks_like_xss(payload: str) -> bool:
    s = payload.lower()
    if any(b in s for b in BAD_SUBSTRINGS):
        return False
    if HEX_BLOB_RE.search(s):
        return False
    # keep vectors short and with typical markers
    if len(payload) > 180:
        return False
    xss_keys = ["<script", "onerror", "onload", "<img", "<svg", "javascript:", "<iframe", "<details", "<style", "onanimationstart"]
    return any(k in s for k in xss_keys)


def clean_file(src: Path, dst: Path) -> Dict[str, int]:
    kept: List[Dict] = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    if not src.exists():
        return stats
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                row = json.loads(line)
            except Exception:
                stats["dropped"] += 1
                continue
            p = (row.get("payload") or "").strip()
            if not p:
                stats["dropped"] += 1
                continue
            t = str(row.get("attack_type", "")).upper()
            ok = False
            if t == "SQLI":
                ok = looks_like_sqli(p)
            elif t == "XSS":
                ok = looks_like_xss(p)
            else:
                # If missing label, attempt generic accept
                ok = looks_like_sqli(p) or looks_like_xss(p)
            if not ok:
                stats["dropped"] += 1
                continue
            kept.append(row)
            stats["kept"] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    pairs = [
        (PROCESSED / "red_train_v6_multi.jsonl", PROCESSED / "red_train_v6_multi_clean.jsonl"),
        (PROCESSED / "red_val_v6_multi.jsonl", PROCESSED / "red_val_v6_multi_clean.jsonl"),
        (PROCESSED / "red_test_v6_multi.jsonl", PROCESSED / "red_test_v6_multi_clean.jsonl"),
    ]
    for src, dst in pairs:
        stats = clean_file(src, dst)
        print(f"Cleaned {src.name}: total={stats['total']} kept={stats['kept']} dropped={stats['dropped']} -> {dst.name}")


if __name__ == "__main__":
    main()

