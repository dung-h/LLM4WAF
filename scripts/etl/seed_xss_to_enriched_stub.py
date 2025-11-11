#!/usr/bin/env python3
"""
Fallback: convert data/raw/seed_xss.csv into xss_enriched_deepseek.jsonl
without calling external APIs. Useful when DEEPSEEK_API_KEY is not set.

Fields: instruction, context, constraints, payload, reasoning, attack_type
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    src = RAW / "seed_xss.csv"
    if not src.exists():
        raise SystemExit("Missing data/raw/seed_xss.csv")
    df = pd.read_csv(src)
    if "payload" not in df.columns:
        # Assume first column is payload
        df = df.rename(columns={df.columns[0]: "payload"})
    df = df.dropna(subset=["payload"]).drop_duplicates()

    out = PROCESSED / "xss_enriched_deepseek.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            payload = str(row["payload"]).strip()
            if not payload:
                continue
            rec = {
                "instruction": "Generate an XSS payload to test a DVWA reflected XSS endpoint.",
                "context": "Target: /vulnerabilities/xss_r/ parameter 'name'. Environment: ModSecurity CRS PL1.",
                "constraints": "Keep length <=120 chars; avoid quotes if possible; robust to basic filters; output only the payload.",
                "payload": payload,
                "reasoning": (str(row.get("description", "seed stub")).strip() or "seed stub"),
                "attack_type": "XSS",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote stub XSS records to {out}")


if __name__ == "__main__":
    main()

