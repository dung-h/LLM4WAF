import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

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

def build_prompt_gemma(instruction: str, context: str, constraints: str, payload: str) -> Tuple[str, str]:
    user_parts = []
    if instruction: user_parts.append(instruction)
    if context: user_parts.append(f"Context: {context}")
    if constraints: user_parts.append(f"Constraints: {constraints}")
    user_block = "\n\n".join(user_parts)
    prompt = f"<start_of_turn>user\n{user_block}<end_of_turn>\n<start_of_turn>model\n"
    chosen = f"{prompt}{payload}<end_of_turn>\n"
    return prompt, chosen

def build_prompt_phi3(instruction: str, context: str, constraints: str, payload: str) -> Tuple[str, str]:
    user_parts = []
    if instruction: user_parts.append(instruction)
    if context: user_parts.append(f"Context: {context}")
    if constraints: user_parts.append(f"Constraints: {constraints}")
    user_block = "\n\n".join(user_parts)
    prompt = f"<|user|>\n{user_block}<|end|>\n<|assistant|>\n"
    chosen = f"{prompt}{payload}<|end|>\n"
    return prompt, chosen

def main():
    parser = argparse.ArgumentParser(description="Standard tool to build/merge unified datasets.")
    parser.add_argument("--base", type=Path, help="Base dataset (lowest priority).")
    parser.add_argument("--inputs", type=Path, nargs='+', help="List of input datasets to merge (later ones override earlier ones).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument("--format", choices=["gemma", "phi3", "none"], default="none", help="Auto-format prompt/chosen fields.")
    args = parser.parse_args()
    
    # Key: (attack_type, payload), Value: record dict
    unified_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    # 1. Load Base
    if args.base:
        print(f"[INFO] Loading base: {args.base}")
        for rec in iter_jsonl(args.base):
            atk = str(rec.get("attack_type", "SQLI")).upper()
            payload = str(rec.get("payload", "")).strip()
            if not payload: continue
            unified_records[(atk, payload)] = rec
        print(f"[INFO] Base loaded: {len(unified_records)} records.")

    # 2. Merge Inputs
    if args.inputs:
        for input_path in args.inputs:
            print(f"[INFO] Merging: {input_path}")
            count = 0
            for rec in iter_jsonl(input_path):
                atk = str(rec.get("attack_type", "SQLI")).upper()
                payload = str(rec.get("payload", "")).strip()
                if not payload: continue
                
                key = (atk, payload)
                
                # Conflict Resolution Logic
                if key in unified_records:
                    existing_rec = unified_records[key]
                    existing_result = existing_rec.get("result", existing_rec.get("waf_result", "unknown"))
                    
                    # Preserve 'reflected'
                    if existing_result == "reflected":
                        continue
                    
                    # Allow overwrite
                    unified_records[key] = rec
                else:
                    unified_records[key] = rec
                count += 1
            print(f"[INFO] Processed {count} records.")

    # 3. Finalize & Format
    print("[INFO] Finalizing...")
    final_list = []
    for rec in unified_records.values():
        if args.format != "none":
            instruction = rec.get("instruction", f"Generate a {rec.get('attack_type', 'SQLI')} payload.")
            context = rec.get("context", "")
            constraints = rec.get("constraints", "")
            payload = rec.get("payload", "").strip()
            
            if args.format == "gemma":
                p, c = build_prompt_gemma(instruction, context, constraints, payload)
            else:
                p, c = build_prompt_phi3(instruction, context, constraints, payload)
            
            rec["prompt"] = p
            rec["chosen"] = c
            
        final_list.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in final_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"[INFO] Done. Written {len(final_list)} records to {args.output}")

if __name__ == "__main__":
    main()
