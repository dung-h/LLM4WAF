import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import requests

# --- Configuration ---
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
# User-provided DeepSeek API Keys (Rotation pool)
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

# --- Helper Functions ---

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def call_deepseek(prompt: str, api_key: str, model: str = "deepseek-chat", max_retries: int = 3) -> Dict[str, Any]:
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
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }

    last_err: Exception | None = None
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
    print(f"[WARN] DeepSeek API failed: {last_err}")
    return {}

# --- Prompt Builders ---

def build_reasoning_prompt(payload: str, attack_type: str) -> str:
    """Prompt to generate reasoning, instruction, and context for an existing payload."""
    return f"""Bạn là một chuyên gia nghiên cứu bảo mật.
Với payload {attack_type} sau: `{payload}`

Hãy tạo một đối tượng JSON với các khóa sau:
- instruction: một mô tả rõ ràng yêu cầu model tạo payload này, bằng tiếng Việt tự nhiên (ví dụ: "Hãy tạo một payload SQLi để bypass bộ lọc khoảng trắng").
- context: mô tả ngắn gọn về mục tiêu (ví dụ: MySQL backend, WAF chặn khoảng trắng).
- constraints: các ràng buộc định dạng (ví dụ: không dùng khoảng trắng, dùng comment thay thế).
- reasoning: Phân tích chiến thuật: Giải thích chi tiết cơ chế hoạt động, kỹ thuật né tránh WAF, và cách payload này khai thác lỗ hổng.
- attack_subtype: loại tấn công chi tiết (ví dụ: time_based, error_based, reflected_xss, dom_xss).

Chỉ trả về JSON hợp lệ."""

def build_variant_prompt(payload: str, attack_type: str) -> str:
    """Prompt to generate advanced variants of a payload."""
    return f"""Bạn là chuyên gia Red Teaming.
Payload gốc ({attack_type}): `{payload}`

Hãy tạo ra một biến thể nâng cao (advanced variant) của payload này để bypass WAF mạnh hơn.
Trả về JSON với các khóa:
- payload: chuỗi payload mới.
- reasoning: giải thích tại sao biến thể này mạnh hơn bản gốc.
- technique: kỹ thuật obfuscation đã dùng (ví dụ: double encoding, sql comments, unicode)."""

# --- Main Worker ---

def worker(seed: Dict[str, Any], api_key: str, task: str) -> Dict[str, Any]:
    payload = seed.get("payload", "")
    attack_type = seed.get("attack_type", "SQLI")
    
    if not payload: return {}

    if task == "reasoning":
        prompt = build_reasoning_prompt(payload, attack_type)
    elif task == "variant":
        prompt = build_variant_prompt(payload, attack_type)
    else:
        return {}

    result = call_deepseek(prompt, api_key)
    
    # Merge result with original seed
    enriched_record = seed.copy()
    
    if task == "reasoning":
        enriched_record.update(result) # Add instruction, context, reasoning...
        enriched_record["source"] = "deepseek_reasoning_enriched"
        
        # Auto-format for Gemma training if fields exist
        if "instruction" in enriched_record:
            user_block = f"{enriched_record['instruction']}\nContext: {enriched_record.get('context','')}\nConstraints: {enriched_record.get('constraints','')}"
            enriched_record["prompt"] = f"<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n"
            enriched_record["chosen"] = f"{enriched_record['prompt']}{payload}<end_of_turn>\n"

    elif task == "variant":
        # For variants, we create a NEW record mostly
        if "payload" in result:
            enriched_record["original_payload"] = payload
            enriched_record["payload"] = result["payload"]
            enriched_record["variant_reasoning"] = result.get("reasoning", "")
            enriched_record["technique"] = result.get("technique", "")
            enriched_record["source"] = "deepseek_variant_generated"
            # Remove old result status as this is a new payload
            enriched_record.pop("result", None) 
            enriched_record.pop("waf_result", None)

    return enriched_record

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Universal tool to augment datasets using DeepSeek API.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument("--task", type=str, choices=["reasoning", "variant"], required=True, help="Task type: 'reasoning' (add context/instruction) or 'variant' (generate harder payloads).")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers (max 10).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found.")
        return

    seeds = list(iter_jsonl(args.input))
    print(f"[INFO] Loaded {len(seeds)} seeds from {args.input}")

    # Resume logic
    processed_payloads = set()
    if args.resume and args.output.exists():
        print(f"[INFO] Resuming... Scanning {args.output}")
        for rec in iter_jsonl(args.output):
            # We use 'payload' (for reasoning task) or 'original_payload' (for variant task) to track progress
            if args.task == "reasoning":
                if "payload" in rec: processed_payloads.add(rec["payload"])
            elif args.task == "variant":
                if "original_payload" in rec: processed_payloads.add(rec["original_payload"])
        print(f"[INFO] Found {len(processed_payloads)} already processed records.")
    
    # Filter seeds
    seeds_to_process = [s for s in seeds if s.get("payload") not in processed_payloads]
    print(f"[INFO] {len(seeds_to_process)} seeds remaining to process.")

    if not seeds_to_process:
        print("[INFO] Nothing to do.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume else "w"
    
    keys = [k for k in DEEPSEEK_KEYS if k]
    max_workers = min(args.max_workers, len(keys))

    print(f"[INFO] Starting processing with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex, args.output.open(open_mode, encoding="utf-8") as out_f:
        futures = []
        for i, seed in enumerate(seeds_to_process):
            key = keys[i % len(keys)]
            futures.append(ex.submit(worker, seed, key, args.task))
        
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 10 == 0:
                    print(f"[PROG] Processed {completed}/{len(seeds_to_process)}")

    print(f"[INFO] Completed. Output saved to {args.output}")

if __name__ == "__main__":
    main()
