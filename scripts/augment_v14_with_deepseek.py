import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import requests


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"

# WARNING: Keys hardcoded at user request. Rotate/remove after use.
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


def make_sql_prompt(rec: Dict[str, Any], variants: int) -> str:
    instruction = rec.get("instruction", "")
    context = rec.get("context", "")
    constraints = rec.get("constraints", "")
    payload = rec.get("payload", "")
    attack_subtype = rec.get("attack_subtype", "unknown")
    db_type = rec.get("db_type", "Unknown")
    waf_name = rec.get("waf_name", "Unknown")

    lines = [
        "You are an expert offensive security researcher.",
        "Given the original SQL injection payload and context, generate new, high-quality variants.",
        "",
        f"Database type: {db_type}",
        f"WAF name or hints: {waf_name}",
        f"Attack subtype: {attack_subtype}",
        f"Original instruction: {instruction}",
        f"Context: {context}",
        f"Constraints: {constraints}",
        f"Original payload: {payload}",
        "",
        f"Generate {variants} new SQL injection payloads that:",
        "- Keep the same overall goal and subtype (e.g., time-based, error-based, union-based, stacked, tautology).",
        "- Use different syntax, operators, comments, or obfuscation techniques.",
        "- Are realistic and syntactically valid for MySQL.",
        "- Do NOT include any explanation; only the payload strings themselves.",
        "",
        'Return JSON with a single key "payloads" whose value is a list of strings.',
    ]
    return "\n".join(lines)


def call_deepseek(prompt: str, api_key: str, model: str = "deepseek-coder", max_retries: int = 3) -> List[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
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
            payloads = parsed.get("payloads", [])
            if isinstance(payloads, list):
                return [str(p).strip() for p in payloads if str(p).strip()]
            # Fallback: if "payloads" missing, treat whole content as one payload
            return [content.strip()]
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DeepSeek API failed after {max_retries} attempts: {last_err}")


def worker_sql(idx: int, rec: Dict[str, Any], api_key: str, variants: int) -> Tuple[int, List[Dict[str, Any]]]:
    prompt = make_sql_prompt(rec, variants)
    new_payloads = call_deepseek(prompt, api_key)

    augmented: List[Dict[str, Any]] = []
    for p in new_payloads:
        aug = dict(rec)
        aug["payload"] = p
        # Mark source + parent index for traceability
        aug["source"] = "deepseek_sql_augment_v1"
        aug["parent_index"] = idx
        augmented.append(aug)
    return idx, augmented


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment v14 SFT data with DeepSeek-generated SQLi variants.")
    ap.add_argument(
        "--input",
        default="data/processed/v14_sft_data.jsonl",
        help="Path to base v14 SFT dataset.",
    )
    ap.add_argument(
        "--output",
        default="data/processed/v14_sft_data_augmented_deepseek.jsonl",
        help="Output JSONL for augmented samples.",
    )
    ap.add_argument(
        "--max_records",
        type=int,
        default=100,
        help="Maximum number of base records to augment.",
    )
    ap.add_argument(
        "--variants_per_sample",
        type=int,
        default=3,
        help="Number of new payload variants per base sample.",
    )
    ap.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum parallel DeepSeek calls.",
    )
    args = ap.parse_args()

    # Prepare base records (for now: only SQLi)
    base_records: List[Dict[str, Any]] = []
    for rec in iter_jsonl(args.input):
        if rec.get("attack_type", "SQLi") != "SQLi":
            continue
        base_records.append(rec)

    if not base_records:
        raise SystemExit("No SQLi records found in input dataset.")

    # Subsample
    random.shuffle(base_records)
    base_records = base_records[: args.max_records]

    # Keys & workers
    keys = [k for k in DEEPSEEK_KEYS if k]
    if not keys:
        raise SystemExit("No DeepSeek API keys configured in DEEPSEEK_KEYS.")
    max_workers = max(1, min(args.max_workers, len(keys)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Loaded {len(base_records)} base SQLi records from {args.input}")
    print(f"[INFO] Using {len(keys)} DeepSeek key(s), up to {max_workers} workers.")
    print(f"[INFO] Will generate {args.variants_per_sample} variants per record.")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(args.output, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(base_records):
            key = keys[idx % len(keys)]
            tasks.append(
                ex.submit(worker_sql, idx, rec, key, args.variants_per_sample)
            )

        total_aug = 0
        for fut in as_completed(tasks):
            try:
                idx, aug_list = fut.result()
            except Exception as e:
                print(f"[WARN] augment worker error: {e}")
                continue
            for aug in aug_list:
                out_f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                total_aug += 1
            if total_aug and total_aug % 20 == 0:
                print(f"[INFO] Generated {total_aug} augmented samples so far...")

    print(f"[INFO] Done. Total augmented samples written: {total_aug} -> {args.output}")


if __name__ == "__main__":
    main()

