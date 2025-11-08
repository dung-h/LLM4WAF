"""
Ingest core datasets for Red/Blue pipelines.

Priority sources:
- PurpleAILAB/reasoning-base-SQLi (HF)
- payloadbox/xss-payload-list and payloadbox/sql-injection-payload-list (local clone or manual copy under data/raw/seeds)

All functions avoid printing secrets. `HF_TOKEN` must be set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

from datasets import load_dataset
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def require_env_token(var: str = "HF_TOKEN") -> None:
    if not os.environ.get(var):
        raise SystemExit(f"Environment variable {var} not set. Aborting for safety.")


def load_purpleaillab_sqli() -> pd.DataFrame:
    """Load PurpleAILAB/reasoning-base-SQLi from HF (requires token)."""
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("PurpleAILAB/reasoning-base-SQLi", split="train", use_auth_token=token)
    return ds.to_pandas()


def load_payloadbox_local() -> Dict[str, List[str]]:
    """Load payload seeds from local files under data/raw/seeds if present."""
    seeds_dir = RAW / "seeds"
    res: Dict[str, List[str]] = {"xss": [], "sqli": []}
    files = {
        "xss": [
            seeds_dir / "xss-payload-list" / "payloads.txt",
        ],
        "sqli": [
            seeds_dir / "sql-injection-payload-list" / "payloads.txt",
        ],
    }
    for k, paths in files.items():
        for p in paths:
            if p.exists():
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    res[k].extend([line.strip() for line in f if line.strip()])
    return res


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    require_env_token()
    RAW.mkdir(parents=True, exist_ok=True)

    print("Loading PurpleAILAB/reasoning-base-SQLi…")
    sqli_df = load_purpleaillab_sqli()
    write_csv(sqli_df, RAW / "purpleaillab_reasoning_sqli.csv")

    print("Collecting local payload seeds (optional)…")
    seeds = load_payloadbox_local()
    if any(seeds.values()):
        pd.DataFrame({"payload": seeds["xss"]}).to_csv(RAW / "seed_xss.csv", index=False)
        pd.DataFrame({"payload": seeds["sqli"]}).to_csv(RAW / "seed_sqli.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()

