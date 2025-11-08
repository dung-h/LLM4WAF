from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "kaggle"


DATASETS = [
    "syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning",
    "atulanandjha/web-application-payloads",
]


def creds_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Check config file locations
    cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    cand = []
    if cfg_dir:
        cand.append(Path(cfg_dir) / "kaggle.json")
    cand.append(Path.home() / ".kaggle" / "kaggle.json")
    return any(p.exists() for p in cand)


def kaggle_download(dataset: str, dest: Path) -> None:
    import kaggle  # type: ignore
    dest.mkdir(parents=True, exist_ok=True)
    # Try CLI first, then python -m fallback
    rc = os.system(f"kaggle datasets download -d {dataset} -p {dest}")
    if rc != 0:
        os.system(f"python -m kaggle datasets download -d {dataset} -p {dest}")


def extract_all_zips(root: Path) -> None:
    for z in root.glob("*.zip"):
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(root / z.stem)


def collect_payloads_from_csv(root: Path) -> pd.DataFrame:
    payloads: List[str] = []
    for csv in root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        # Identify label column if present
        label_col = None
        for c in df.columns:
            cl = c.lower()
            if cl in ("label", "attack", "class", "target"):
                label_col = c
                break
        pos_idx = None
        if label_col is not None:
            vals = set(pd.to_numeric(df[label_col], errors='coerce').dropna().astype(int).unique().tolist())
            if vals:
                # Assume positive is non-zero
                pos_idx = df[pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int) > 0].index
        # Heuristic: look for likely text columns
        keys = ["payload", "xss", "attack", "request", "value", "query", "sentence", "text"]
        cand_cols = [c for c in df.columns if any(k in c.lower() for k in keys)]
        for c in cand_cols:
            series = df[c].dropna().astype(str)
            if pos_idx is not None:
                series = series.loc[pos_idx]
            for v in series.head(200000):  # cap large files
                v = v.strip()
                if v:
                    payloads.append(v)
    return pd.DataFrame({"payload": list(dict.fromkeys(payloads))})


def main() -> None:
    if not creds_available():
        raise SystemExit("No Kaggle credentials found (env or kaggle.json).")
    for ds in DATASETS:
        target = RAW / ds.split("/")[-1]
        kaggle_download(ds, target)
        extract_all_zips(target)
    df = collect_payloads_from_csv(RAW)
    if not df.empty:
        out = ROOT / "data" / "raw" / "seed_mixed_kaggle.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote {out} with {len(df)} rows")
    else:
        print("No payloads collected from Kaggle datasets.")


if __name__ == "__main__":
    main()
