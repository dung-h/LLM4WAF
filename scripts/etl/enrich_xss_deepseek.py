#!/usr/bin/env python3
"""
Enrich XSS seed payloads using the DeepSeek Chat API.

Inputs:
  - data/raw/seed_xss.csv           (required; columns: payload[, type][, description])

Outputs:
  - data/processed/xss_enriched_deepseek.jsonl

Environment:
  - DEEPSEEK_API_KEY  (required)

Notes:
  - Keeps costs/time low by default (max_seeds=20, variants_per_seed=2)
  - Produces records with fields compatible with training formatter:
      instruction, context, constraints, payload, reasoning, attack_type
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
LOGS = ROOT / "logs"


SYSTEM_PROMPT = (
    "You are a security researcher generating ADVANCED XSS payloads designed to bypass "
    "ModSecurity + OWASP CRS (PL1). Prefer obfuscated techniques (entity/Unicode encoding, constructor/atob, "
    "string concatenation, bracket access, JS URL indirection, CSS/animation/on* events, DOM attributes, srcdoc, polyglot). "
    "Avoid trivial patterns like <script>alert(1)</script> or plain <img onerror=alert(1)>. Output JSON objects only."
)


def ensure_env() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY not set. Put it in your environment or .env."
        )
    return api_key


def load_seeds() -> pd.DataFrame:
    # Prefer curated seed file; fall back to HF/payloadbox seeds if available
    candidates = [RAW / "seed_xss.csv", RAW / "seed_xss_hf.csv"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize expected column name
            if "payload" not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: "payload"})
            df = df.dropna(subset=["payload"]).drop_duplicates()
            return df
    raise SystemExit("No XSS seed file found. Create data/raw/seed_xss.csv first.")


def build_user_prompt(example: str, n_variants: int) -> str:
    return (
        "Task: Generate XSS payload variants that bypass typical WAF filters (CRS PL1).\n"
        "Context: Target is DVWA 'Reflected XSS' at /vulnerabilities/xss_r/ with parameter 'name'.\n"
        "Hard constraints:\n"
        "- Length <= 120 chars; output ONLY the payload + a one-line reasoning (JSON object per line)\n"
        "- Prefer obfuscation: HTML entities, Unicode escapes, template literals/backticks, String.fromCharCode, atob, constructor, bracket access (top['al'+'ert']), CSS/animation/on* events, details/ontoggle, srcdoc/iframe, math/svg quirks\n"
        "- Avoid trivial payloads: direct <script>alert(1)</script>, plain <img onerror=alert(1)>\n"
        "- No markdown fences\n\n"
        f"Seed inspiration (do not copy literally): {example}\n\n"
        "Output: JSON lines, one object per variant:\n"
        "{\"payload\": <string>, \"reasoning\": <short string>}\n"
        f"Generate exactly {n_variants} variants."
    )


def deepseek_chat(api_key: str, user_prompt: str) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 400,
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def parse_variants(text: str) -> List[Dict[str, str]]:
    # Heuristic: try to parse as JSON lines or a JSON array; fallback to line parsing
    text = text.strip()
    variants: List[Dict[str, str]] = []
    # Try JSON array
    if text.startswith("["):
        try:
            arr = json.loads(text)
            for obj in arr:
                if isinstance(obj, dict) and "payload" in obj:
                    variants.append({
                        "payload": str(obj["payload"]).strip(),
                        "reasoning": str(obj.get("reasoning", "")).strip(),
                    })
            return variants
        except Exception:
            pass
    # Try JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "payload" in obj:
                variants.append({
                    "payload": str(obj["payload"]).strip(),
                    "reasoning": str(obj.get("reasoning", "")).strip(),
                })
        except Exception:
            # Fallback: treat whole line as payload
            if any(tag in line.lower() for tag in ("<script", "onerror", "svg", "img", "javascript:")):
                variants.append({"payload": line, "reasoning": "heuristic parsed"})
    return variants


ADV_KEYWORDS = [
    "fromcharcode", "atob", "constructor", "[\"al\"+\"ert\"]", "['al'+'ert']", "javascript:",
    "onanimationstart", "ontoggle", "srcdoc", "template", "backtick", "`", "&#x", "%3c", "xlink:href",
]

def looks_advanced(payload: str) -> bool:
    s = payload.lower()
    # Reject trivial patterns
    if "<script>alert(1)</script>" in s:
        return False
    if "<img" in s and "onerror=alert(1)" in s:
        return False
    # Prefer any advanced keyword/signal
    return any(k in s for k in ADV_KEYWORDS)


def main() -> None:
    api_key = ensure_env()
    seeds = load_seeds()

    max_seeds = int(os.environ.get("XSS_ENRICH_MAX_SEEDS", 20))
    variants_per_seed = int(os.environ.get("XSS_ENRICH_VARIANTS", 4))

    out_path = PROCESSED / "xss_enriched_deepseek.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, str]] = []

    use = seeds.head(max_seeds)
    # progress log
    LOGS.mkdir(parents=True, exist_ok=True)
    progress_path = LOGS / "enrich_xss_progress.log"
    try:
        progress_f = open(progress_path, "a", encoding="utf-8")
    except Exception:
        progress_f = None
    for i, row in use.iterrows():
        seed_payload = str(row["payload"]).strip()
        user_prompt = build_user_prompt(seed_payload, variants_per_seed)

        try:
            content = deepseek_chat(api_key, user_prompt)
        except Exception as e:
            print(f"[WARN] DeepSeek request failed at seed {i}: {e}")
            continue

        variants = parse_variants(content)
        if not variants:
            print(f"[WARN] No variants parsed for seed {i}")
            continue

        # Keep at most N
        variants = variants[:variants_per_seed]

        for v in variants:
            rec = {
                "instruction": (
                    "Generate an XSS payload to test a DVWA reflected XSS endpoint."
                ),
                "context": (
                    "Target: /vulnerabilities/xss_r/ parameter 'name'. Environment: ModSecurity CRS PL1."
                ),
                "constraints": (
                    "Keep length <=120 chars; avoid quotes if possible; robust to basic filters; output only the payload."
                ),
                "payload": v.get("payload", "").strip(),
                "reasoning": v.get("reasoning", "").strip(),
                "attack_type": "XSS",
            }
            # Execution-based selection: accept all non-empty payloads; filtering happens at replay.
            if rec["payload"]:
                records.append(rec)

        msg = f"[INFO] Seed {i}: generated {len(variants)} variants"
        print(msg, flush=True)
        if progress_f is not None:
            progress_f.write(msg + "\n")
            progress_f.flush()

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} XSS records to {out_path}")
    if progress_f is not None:
        progress_f.write(f"SAVED {len(records)} -> {out_path}\n")
        progress_f.flush(); progress_f.close()


if __name__ == "__main__":
    main()
