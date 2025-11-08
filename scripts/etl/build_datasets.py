from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
import sys

import pandas as pd

# Import sibling normalize when run as a script
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
try:
    from normalize import canonicalize  # type: ignore
except Exception:
    # Fallback minimal canonicalizer
    import html as _html
    import re as _re
    import unicodedata as _unicodedata
    from urllib.parse import unquote_plus as _unquote_plus

    def canonicalize(s: str) -> str:  # type: ignore
        out = s
        for _ in range(2):
            try:
                out = _unquote_plus(out)
            except Exception:
                pass
            try:
                out = _html.unescape(out)
            except Exception:
                pass
        out = _unicodedata.normalize("NFKC", out)
        out = _re.sub(r"\s+", " ", out.strip()).lower()
        return out


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def dedup_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[key]).reset_index(drop=True)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def build_red() -> None:
    src = RAW / "purpleaillab_reasoning_sqli.csv"
    if not src.exists():
        raise SystemExit("Missing purpleaillab_reasoning_sqli.csv — run ingest first.")

    df = pd.read_csv(src)
    # Expected columns (approx): instruction, reasoning, payload or similar naming
    # Normalize into our schema; fallbacks if names differ
    df = df.rename(columns={
        "prompt": "instruction",
        "input": "instruction",
        "user": "instruction",
        "response": "payload",
        "answer": "payload",
        "assistant": "payload",
        "explanation": "reasoning",
    })

    if "instruction" not in df.columns:
        df["instruction"] = "Generate SQLi payload for testing"
    if "payload" not in df.columns:
        df["payload"] = ""
    if "reasoning" not in df.columns:
        df["reasoning"] = ""

    df["context"] = df.get("context", pd.Series(["DB=MySQL; sink=query; CRS=PL2"] * len(df)))
    df["constraints"] = df.get("constraints", pd.Series(["lab-only; avoid destructive ops"] * len(df)))

    # Canonical key for dedup
    df["canon"] = [canonicalize(str(p)) for p in df["payload"].astype(str)]
    df = dedup_by_key(df, "canon")

    # Minimal split 80/10/10
    n = len(df)
    tr = int(n * 0.8)
    va = int(n * 0.1)
    red_train = df.iloc[:tr][["instruction", "context", "constraints", "payload", "reasoning"]].copy()
    red_val = df.iloc[tr:tr+va][["instruction", "context", "constraints", "payload", "reasoning"]].copy()
    red_test = df.iloc[tr+va:][["instruction", "context", "constraints", "payload", "reasoning"]].copy()

    write_jsonl(red_train, PROC / "red_train.jsonl")
    write_jsonl(red_val, PROC / "red_val.jsonl")
    write_jsonl(red_test, PROC / "red_test.jsonl")


def build_blue() -> None:
    # Placeholder: Blue set derived from replay results after baseline run
    # Keep fields as strings to match training format
    rule_str = (
        'SecRule ARGS:q "(?i:(sleep\\s*\\()|benchmark\\s*\\())" '
        '"phase:2,deny,id:950001,log,msg:\'Block time-based SQLi\','
        't:lowercase,ctl:ruleEngine=On"'
    )
    input_text = (
        "payload: ' OR SLEEP(5)-- -\n"
        "context: DB=MySQL; sink=query; method=GET; param=q\n"
        "waf_context: CRS=PL2\n"
        "missed_rules: []\n"
        "evidence: app responded 200 within ~5s delay"
    )
    output_text = (
        f"rule: {rule_str}\n"
        "normalizer: t:urlDecodeUni,t:lowercase\n"
        "phase: 2"
    )
    rows = [
        {
            "input": input_text,
            "output": output_text,
            "rationale": "Blocks time-based SQLi function names; low FP on benign query terms.",
        }
    ]

    df = pd.DataFrame(rows)
    write_jsonl(df, PROC / "blue_train.jsonl")
    write_jsonl(df.iloc[:0], PROC / "blue_val.jsonl")
    write_jsonl(df.iloc[:0], PROC / "blue_test.jsonl")


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    build_red()
    build_blue()
    print("Built red/blue jsonl datasets in data/processed/.")


if __name__ == "__main__":
    main()
