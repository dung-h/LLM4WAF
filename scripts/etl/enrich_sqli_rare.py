"""
Collect rare/obfuscated SQLi payloads from multiple public sources, deduplicate, score heuristics,
and write to data/processed/rare_sqli_corpus.jsonl

Run:
  wsl -e bash -lc "source .venv/bin/activate; python scripts/etl/enrich_sqli_rare.py"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict

import httpx


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "rare_sqli_corpus.jsonl"


SOURCES: List[Dict[str, str]] = [
    # PayloadsAllTheThings
    {
        "name": "PayloadsAllTheThings_SQLi_Generic",
        "url": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/SQL%20Injection/Intruder/Generic.txt",
    },
    {
        "name": "PayloadsAllTheThings_SQLi_MySQL",
        "url": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/SQL%20Injection/Intruder/MySQL.txt",
    },
    # FuzzDB
    {
        "name": "FuzzDB_SQLi_Payloads",
        "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/MySQLDetect.txt",
    },
    {
        "name": "FuzzDB_SQLi_ErrorBased",
        "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/GenericBlind.txt",
    },
    # SecLists
    {
        "name": "SecLists_SQLi_Generic",
        "url": "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/SQLi/quick-SQLi.txt",
    },
]


def fetch_text(url: str) -> str:
    try:
        r = httpx.get(url, timeout=20.0)
        r.raise_for_status()
        return r.text
    except Exception:
        return ""


def is_interesting(payload: str) -> bool:
    s = payload.lower()
    # rudimentary filter to skip too-simple or too-short items
    if len(payload) < 8:
        return False
    simple = [" or 1=1", " and 1=1", " union select"]
    if any(x in s for x in simple):
        # keep some proportion only if obfuscated
        if any(t in s for t in ["/*", "*/", "%2f", "%2a", "/*!", "0x", "char("]):
            return True
        return False
    return True


def rarity_score(payload: str) -> float:
    s = payload.lower()
    score = 0.0
    if "/*!" in s or "/**/" in s or "%2f%2a" in s or "%2a%2f" in s:
        score += 0.4
    if "0x" in s or "char(" in s:
        score += 0.3
    if "concat" in s or "regexp" in s or "geometrycollection" in s or "linestring" in s:
        score += 0.2
    if "sleep(" in s or "benchmark(" in s:
        score += 0.2
    if len(payload) > 120:
        score -= 0.2
    return max(0.0, min(1.0, score))


def normalize(payload: str) -> str:
    return re.sub(r"\s+", " ", payload.strip())


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows: List[Dict] = []

    for src in SOURCES:
        text = fetch_text(src["url"]) or ""
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            # drop placeholders
            if line.lower() in {"sqlvuln", "sqlvuln;"}:
                continue
            p = normalize(line)
            if not is_interesting(p):
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "payload": p,
                "source": src["name"],
                "rarity_score": rarity_score(p),
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()

