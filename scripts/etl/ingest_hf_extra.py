from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def require_env_token(var: str = "HF_TOKEN") -> None:
    if not os.environ.get(var):
        raise SystemExit(f"Environment variable {var} not set. Aborting for safety.")


def try_get_columns(df: pd.DataFrame, names: List[str]) -> List[str]:
    cols = []
    for n in names:
        if n in df.columns:
            cols.append(n)
    return cols


def main() -> None:
    require_env_token()
    token = os.environ.get("HF_TOKEN")
    # Shengqin/Web-Attacks-Dataset may have different configurations; we try default
    ds = load_dataset("Shengqin/Web-Attacks-Dataset", split="train", use_auth_token=token)
    df = ds.to_pandas()

    # Heuristics: look for label columns and payload-like columns
    label_col = None
    for c in ["label", "Label", "attack_type", "Attack_type", "category", "Category"]:
        if c in df.columns:
            label_col = c
            break

    payload_cols = try_get_columns(df, ["payload", "Payload", "request", "Request", "value", "Value"]) or [c for c in df.columns if df[c].dtype == object]

    xss_payloads: List[str] = []
    sqli_payloads: List[str] = []

    for _, row in df.iterrows():
        label = str(row.get(label_col, "")) if label_col else ""
        cat = label.lower()
        for c in payload_cols:
            val = row.get(c)
            if not isinstance(val, str):
                continue
            v = val.strip()
            if not v:
                continue
            if "xss" in cat:
                xss_payloads.append(v)
            elif "sql" in cat:
                sqli_payloads.append(v)

    # Write to seed files (append or create)
    RAW.mkdir(parents=True, exist_ok=True)
    if xss_payloads:
        pd.DataFrame({"payload": xss_payloads}).drop_duplicates().to_csv(RAW / "seed_xss_hf.csv", index=False)
    if sqli_payloads:
        pd.DataFrame({"payload": sqli_payloads}).drop_duplicates().to_csv(RAW / "seed_sqli_hf.csv", index=False)
    print(f"Shengqin seeds: XSS={len(xss_payloads)} SQLi={len(sqli_payloads)}")


if __name__ == "__main__":
    main()

