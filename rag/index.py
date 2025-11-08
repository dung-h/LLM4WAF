from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from rag.retrievers import TFIDFRetriever, Doc


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
IDX = ROOT / "rag" / "indexes"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_payload_corpus() -> List[Doc]:
    docs: List[Doc] = []
    # Red datasets (SQLi reasoning base)
    for split in ["red_train.jsonl", "red_val.jsonl", "red_test.jsonl"]:
        p = PROC / split
        if not p.exists():
            continue
        for i, row in enumerate(load_jsonl(p)):
            payload = row.get("payload", "")
            if not payload:
                continue
            context = row.get("context", "")
            instruction = row.get("instruction", "")
            meta = {
                "source": split,
                "category": "SQLi",
                "context": context,
                "instruction": instruction,
                "payload": payload,
                "reason": row.get("reasoning", ""),
            }
            docs.append(Doc(id=f"{split}:{i}", text=str(payload), meta=meta))

    # Optional seeds: include all seed_*.csv under data/raw
    for seed_csv in (ROOT / "data" / "raw").glob("seed_*.csv"):
        try:
            df = pd.read_csv(seed_csv)
        except Exception:
            continue
        category = "Mixed"
        name = seed_csv.name.lower()
        if "xss" in name:
            category = "XSS"
        elif "sqli" in name or "sql" in name:
            category = "SQLi"
        for i, row in df.iterrows():
            payload = str(row.get("payload", ""))
            if not payload:
                continue
            docs.append(Doc(id=f"{seed_csv.name}:{i}", text=payload, meta={"source": seed_csv.name, "category": category, "payload": payload}))

    return docs


def build_payload_index(out_path: Path | None = None) -> Path:
    IDX.mkdir(parents=True, exist_ok=True)
    out = out_path or (IDX / "payloads_tfidf.joblib")
    docs = build_payload_corpus()
    retr = TFIDFRetriever(analyzer="char_wb", ngram_range=(3, 6)).build(docs)
    retr.save(str(out))
    return out


def build_crs_index(crs_parquet: Path, out_path: Path | None = None) -> Path:
    """Build a TF-IDF index for CRS rule messages/regex for Blue RAG."""
    IDX.mkdir(parents=True, exist_ok=True)
    out = out_path or (IDX / "crs_tfidf.joblib")
    df = pd.read_parquet(crs_parquet)
    docs: List[Doc] = []
    for i, row in df.iterrows():
        text = f"{row.get('msg','')} | {row.get('regex','')} | phase:{row.get('phase','')}"
        meta = {
            "id": row.get("id"),
            "phase": row.get("phase"),
            "tags": row.get("tags"),
            "t": row.get("transforms"),
        }
        docs.append(Doc(id=str(row.get("id", i)), text=text, meta=meta))
    retr = TFIDFRetriever(analyzer="word", ngram_range=(1, 2)).build(docs)
    retr.save(str(out))
    return out
