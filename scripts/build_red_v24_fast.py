import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

def iter_jsonl(path: Path):
    if not path.is_file():
        print(f"[WARN] File not found: {path}")
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def build_prompt_and_chosen(instruction: str, context: str, constraints: str, payload: str) -> Tuple[str, str]:
    user_parts = []
    if instruction: user_parts.append(instruction)
    if context: user_parts.append(f"Context: {context}")
    if constraints: user_parts.append(f"Constraints: {constraints}")
    user_block = "\n\n".join(user_parts)
    prompt = f"<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n"
    chosen = f"{prompt}{payload}<end_of_turn>\n"
    return prompt, chosen

def main():
    parser = argparse.ArgumentParser(description="Fast build of RED v24 unified dataset.")
    args = parser.parse_args()
    
    v20_path = Path("data/processed/red_v20_unified_mysql_xss.jsonl")
    high_quality_passed_seeds_path = Path("data/processed/seeds_high_quality_passed.jsonl")
    enriched_passed_path = Path("data/processed/v24_passed_enriched_by_deepseek.jsonl")
    output_file = Path("data/processed/red_v24_unified_mysql_xss.jsonl")
    
    # Key: (attack_type, payload), Value: record dict
    unified_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    print(f"[INFO] Loading base dataset: {v20_path}")
    for rec in iter_jsonl(v20_path):
        atk = str(rec.get("attack_type", "SQLI")).upper()
        payload = str(rec.get("payload", "")).strip()
        if not payload: continue
        unified_records[(atk, payload)] = rec
    print(f"[INFO] Loaded {len(unified_records)} records from v20 as base.")

    print(f"[INFO] Merging high-quality seeds (pre-DeepSeek): {high_quality_passed_seeds_path}")
    count_hq_seeds = 0
    for rec in iter_jsonl(high_quality_passed_seeds_path):
        atk = str(rec.get("attack_type", "SQLI")).upper()
        payload = str(rec.get("payload", "")).strip()
        if not payload: continue
        key = (atk, payload)
        
        # Keep reflected if already there
        if key in unified_records and unified_records[key].get("result") == "reflected":
            continue
        
        # If new is passed, and existing is not passed or reflected, update
        if rec.get("result") == "passed" and key in unified_records and unified_records[key].get("result") not in ["passed", "reflected"]:
            rec["source"] = "v24_hq_seed_upgrade_base"
            unified_records[key] = rec
        elif key not in unified_records:
            rec["source"] = "v24_hq_seed_new_base"
            unified_records[key] = rec
        count_hq_seeds += 1
    print(f"[INFO] Processed {count_hq_seeds} high-quality seeds.")

    print(f"[INFO] Merging DeepSeek enriched passed payloads: {enriched_passed_path}")
    count_enriched = 0
    for rec in iter_jsonl(enriched_passed_path):
        atk = str(rec.get("attack_type", "SQLI")).upper()
        payload = str(rec.get("payload", "")).strip()
        if not payload: continue
        key = (atk, payload)
        
        # Always prioritize DeepSeek enriched info, unless existing is reflected
        if key in unified_records and unified_records[key].get("result") == "reflected":
            # If reflected exists, only update instruction/context/reasoning if DeepSeek is better
            # For simplicity, we'll just skip DeepSeek record if reflected is there for this payload
            # Or, we could merge fields carefully. For now, simple skip to preserve reflected.
            continue 
        
        unified_records[key] = rec # DeepSeek enriched records take precedence
        count_enriched += 1
    print(f"[INFO] Processed {count_enriched} DeepSeek enriched passed payloads.")

    # Finalize: Ensure prompt/chosen fields exist
    print("[INFO] Finalizing records...")
    final_list = []
    for rec in unified_records.values():
        if not rec.get("chosen"):
            instruction = rec.get("instruction", f"Generate a {rec.get('attack_type', 'SQLI')} payload.")
            context = rec.get("context", "")
            constraints = rec.get("constraints", "")
            payload = rec.get("payload", "").strip()
            prompt_text, chosen_text = build_prompt_and_chosen(instruction, context, constraints, payload)
            rec["prompt"] = prompt_text
            rec["chosen"] = chosen_text
        final_list.append(rec)

    print(f"[INFO] Writing {len(final_list)} records to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for rec in final_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print("[INFO] Build complete.")

if __name__ == "__main__":
    main()
