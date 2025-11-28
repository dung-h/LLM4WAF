from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import json
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
    # for split in ["red_train.jsonl", "red_val.jsonl", "red_test.jsonl"]:
    #     p = PROC / split
    #     if not p.exists():
    #         continue
    #     for i, row in enumerate(load_jsonl(p)):
    #         payload = row.get("payload", "")
    #         if not payload:
    #             continue
    #         context = row.get("context", "")
    #         instruction = row.get("instruction", "")
    #         meta = {
    #             "source": split,
    #             "category": "SQLi",
    #             "context": context,
    #             "instruction": instruction,
    #             "payload": payload,
    #             "reason": row.get("reasoning", ""),
    #         }
    #         docs.append(Doc(id=f"{split}:{i}", text=str(payload), meta=meta))

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

    # PayloadBox SQLi Payloads
    payloadbox_sqli_path = ROOT / "data" / "raw" / "payloadbox_sqli"
    for txt_file in payloadbox_sqli_path.glob("**/*.txt"):
        dbms_type = txt_file.parent.name if txt_file.parent.name not in ["detect", "exploit", "payloads-sql-blind"] else "Generic"
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                payload = line.strip()
                # Filter out short or generic payloads, and incomplete/confusing examples
                if not payload or payload.startswith("#") or len(payload) <= 10 or \
                   payload.lower() in ["sqlvuln", "sqlvuln;"] or "as injectx" in payload.lower():
                    continue
                meta = {
                    "source": f"payloadbox_sqli:{txt_file.relative_to(payloadbox_sqli_path)}",
                    "category": "SQLi",
                    "payload": payload,
                    "dbms": dbms_type
                }
                docs.append(Doc(id=f"payloadbox_sqli:{txt_file.name}:{i}", text=payload, meta=meta))

    # PayloadBox XSS Payloads
    payloadbox_xss_path = ROOT / "data" / "raw" / "payloadbox_xss"
    for txt_file in payloadbox_xss_path.glob("**/*.txt"):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                payload = line.strip()
                # Filter out short or generic payloads
                if not payload or payload.startswith("#") or len(payload) <= 10 or payload.lower() in ["xss", "xss;"]:
                    continue
                meta = {
                    "source": f"payloadbox_xss:{txt_file.relative_to(payloadbox_xss_path)}",
                    "category": "XSS",
                    "payload": payload,
                }
                docs.append(Doc(id=f"payloadbox_xss:{txt_file.name}:{i}", text=payload, meta=meta))

    # Rare SQLi corpus (enriched) if present
    rare_path = ROOT / "data" / "processed" / "rare_sqli_corpus.jsonl"
    if rare_path.exists():
        for i, row in enumerate(load_jsonl(rare_path)):
            payload = (row.get("payload") or "").strip()
            if not payload:
                continue
            meta = {
                "source": row.get("source", "rare_sqli"),
                "category": "SQLi",
                "payload": payload,
                "reason": row.get("reason", ""),
                "rarity_score": row.get("rarity_score", 0.0),
            }
            docs.append(Doc(id=f"rare_sqli:{i}", text=payload, meta=meta))

    return docs


def build_payload_index(out_path: Path | None = None) -> Path:
    IDX.mkdir(parents=True, exist_ok=True)
    out = out_path or (IDX / "payloads_tfidf.joblib")
    print("Building payload corpus...")
    docs = build_payload_corpus()
    print(f"Corpus contains {len(docs)} documents.")
    print("Building TF-IDF index...")
    retr = TFIDFRetriever(analyzer="char_wb", ngram_range=(3, 6)).build(docs)
    print(f"Saving index to {out}...")
    retr.save(str(out))
    print("Index saved successfully.")
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

if __name__ == "__main__":
    build_payload_index()
