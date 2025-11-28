import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import requests


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"

# WARNING: Hardcoded keys at user request. Rotate/remove after use.
DEEPSEEK_KEYS: List[str] = [
    "sk-ffc6fa18776a475c8aba7d5457df2824",
    "sk-a6eae5b6fe93460380793444d1677478",
    "sk-92ba51e5f1cb4034b2cdcf0ec444f5dd",
    "sk-5c06b1843f2c40b9a7612f6f8cfd0afa",
    "sk-5624a8638a7442b4a98aa03382eb72ce",
    "sk-2098c65133c8421689da4bb227eb65e5",
    "sk-5d82535cd52f46948ee9d6b6363696ec",
    "sk-eb47b3807ffb45bf821de90c06e6d5e4",
    "sk-afd8f7921f60497fa28129fae291f3c7",
    "sk-afd7ed4ba2ce45639fd5ecf931f8a6b3",
]


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def make_sql_instr_prompt(rec: Dict[str, Any]) -> str:
    instruction = rec.get("instruction", "")
    context = rec.get("context", "")
    constraints = rec.get("constraints", "")
    payload = rec.get("payload", "")
    attack_subtype = rec.get("attack_subtype", "unknown")
    db_type = rec.get("db_type", "Unknown")
    waf_name = rec.get("waf_name", "Unknown")

    lines = [
        "You are helping to diversify a supervised finetuning dataset for SQL injection payload generation.",
        "",
        f"Database type: {db_type}",
        f"WAF name or hints: {waf_name}",
        f"Attack subtype: {attack_subtype}",
        f"Current instruction: {instruction}",
        f"Current context: {context}",
        f"Current constraints: {constraints}",
        f"Example payload (for your understanding, do NOT include it in the output): {payload}",
        "",
        "TASK:",
        "- Write a NEW instruction (1–2 sentences) that asks for a similar SQL injection payload,",
        "  consistent with the same attack subtype and environment, but phrased differently.",
        "- Write a NEW context string (1 short sentence) that describes the target/setting.",
        "- Write NEW constraints (1–2 short sentences) that are precise and helpful (e.g. about output format, length, avoiding explanations).",
        "- The new texts must not be trivial paraphrases; change wording and structure enough to look like different authors.",
        "",
        "Return JSON with a single object, with keys:",
        '  "instruction", "context", "constraints".',
        "Do not include the payload in the output.",
    ]
    return "\n".join(lines)


def make_xss_instr_prompt(rec: Dict[str, Any]) -> str:
    instruction = rec.get("instruction", "")
    context = rec.get("context", "")
    constraints = rec.get("constraints", "")
    payload = rec.get("payload", "")

    lines = [
        "You are helping to diversify a supervised finetuning dataset for XSS payload generation.",
        "",
        f"Current instruction: {instruction}",
        f"Current context: {context}",
        f"Current constraints: {constraints}",
        f"Example payload (for your understanding, do NOT include it in the output): {payload}",
        "",
        "TASK:",
        "- Write a NEW instruction (1–2 sentences) that asks for an XSS payload in a similar scenario (DVWA-style reflected XSS),",
        "  but with different wording and possibly slightly different focus (e.g. different filter assumptions).",
        "- Write a NEW context string (1 short sentence) that mentions the target URL/parameter and WAF level in your own words.",
        "- Write NEW constraints (1–2 short sentences) that emphasize output-only payload, maybe length limits, quote avoidance, etc.",
        "- The new texts must not be trivial paraphrases; change wording and structure enough to look like different authors.",
        "",
        "Return JSON with a single object, with keys:",
        '  "instruction", "context", "constraints".',
        "Do not include the payload in the output.",
    ]
    return "\n".join(lines)


def call_deepseek_json(prompt: str, api_key: str, max_retries: int = 3) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful cybersecurity assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=body, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek JSON call failed after {max_retries} attempts: {last_err}")


def worker(idx: int, rec: Dict[str, Any], api_key: str) -> Tuple[int, Dict[str, Any]]:
    attack_type = rec.get("attack_type", "SQLi")
    if attack_type == "XSS":
        prompt = make_xss_instr_prompt(rec)
    else:
        prompt = make_sql_instr_prompt(rec)

    data = call_deepseek_json(prompt, api_key)

    new_instr = str(data.get("instruction", rec.get("instruction", ""))).strip()
    new_ctx = str(data.get("context", rec.get("context", ""))).strip()
    new_constraints = str(data.get("constraints", rec.get("constraints", ""))).strip()

    out_rec = dict(rec)
    out_rec["instruction"] = new_instr or rec.get("instruction", "")
    out_rec["context"] = new_ctx or rec.get("context", "")
    out_rec["constraints"] = new_constraints or rec.get("constraints", "")
    out_rec["instr_source"] = "deepseek_instr_v1"
    out_rec["instr_parent_index"] = idx
    return idx, out_rec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diversify instruction/context/constraints for SQLi/XSS records using DeepSeek."
    )
    ap.add_argument(
        "--input",
        default="data/processed/v14_sft_data.jsonl",
        help="Path to base JSONL (can contain SQLi and/or XSS).",
    )
    ap.add_argument(
        "--output",
        default="data/processed/v14_sft_data_diversified_instr.jsonl",
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--max_records",
        type=int,
        default=200,
        help="Maximum number of records to diversify (random subset).",
    )
    ap.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum parallel DeepSeek calls.",
    )
    args = ap.parse_args()

    # Collect candidates directly from input (both SQLi and XSS)
    candidates: List[Dict[str, Any]] = []
    for rec in iter_jsonl(args.input):
        attack_type = rec.get("attack_type", "SQLi")
        if attack_type in ("SQLi", "XSS"):
            candidates.append(rec)

    if not candidates:
        raise SystemExit("No SQLi/XSS records selected for diversification.")

    random.shuffle(candidates)
    candidates = candidates[: args.max_records]

    keys = [k for k in DEEPSEEK_KEYS if k]
    if not keys:
        raise SystemExit("No DeepSeek keys configured in DEEPSEEK_KEYS.")
    max_workers = max(1, min(args.max_workers, len(keys)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Selected {len(candidates)} records for diversification from {args.input}")
    print(f"[INFO] Using {len(keys)} DeepSeek key(s), up to {max_workers} workers")

    out_f = open(args.output, "w", encoding="utf-8")
    try:
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, rec in enumerate(candidates):
                key = keys[idx % len(keys)]
                tasks.append(ex.submit(worker, idx, rec, key))

            done_count = 0

            for fut in as_completed(tasks):
                try:
                    idx, out_rec = fut.result()
                except Exception as e:
                    print(f"[WARN] diversify worker error: {e}")
                    continue
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                done_count += 1
                if done_count and done_count % 20 == 0:
                    print(f"[INFO] Diversified {done_count}/{len(candidates)} samples...")

    finally:
        out_f.close()

    print(f"[INFO] Diversification completed. Wrote diversified subset to {args.output}")


if __name__ == "__main__":
    main()
