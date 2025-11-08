"""
Audit parser (skeleton) for ModSecurity v3 audit logs in JSON mode.
Parses rule id, message, data, anomaly score, and joins with replay results by transaction id if available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def parse_audit_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            tx = obj.get("transaction", {})
            m = obj.get("messages", [])
            for msg in m:
                rows.append({
                    "transaction_id": tx.get("id"),
                    "uri": tx.get("requestUri"),
                    "rule_id": msg.get("details", {}).get("ruleId"),
                    "message": msg.get("message"),
                    "data": msg.get("details", {}).get("data"),
                    "anomaly": msg.get("details", {}).get("anomalyScore"),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python replay/audit_parser.py <modsec_audit.jsonl>")
        raise SystemExit(1)
    df = parse_audit_jsonl(Path(sys.argv[1]))
    print(df.head())

