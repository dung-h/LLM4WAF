import argparse
import json
from pathlib import Path

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

def build_phi3_prompt_and_chosen(instruction: str, context: str, constraints: str, payload: str):
    user_parts = []
    if instruction: user_parts.append(instruction)
    if context: user_parts.append(f"Context: {context}")
    if constraints: user_parts.append(f"Constraints: {constraints}")
    
    user_block = "\n\n".join(user_parts)

    # Phi-3 Instruct format
    # <|user|>
    # {prompt}<|end|>
    # <|assistant|>
    # {response}<|end|>
    
    prompt = f"<|user|>\n{user_block}<|end|>\n<|assistant|>\n"
    chosen = f"{prompt}{payload}<|end|>\n"
    return prompt, chosen

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to Phi-3 format.")
    parser.add_argument(
        "--input", 
        type=Path,
        default=Path("data/processed/red_v25_unified_mysql_xss.jsonl"),
        help="Input JSONL dataset (Gemma format or generic)."
    )
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path("data/processed/red_v25_phi3_format.jsonl"),
        help="Output JSONL dataset in Phi-3 format."
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: Input file not found: {args.input}")
        return

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with args.output.open("w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(args.input):
            # Extract core fields
            instruction = rec.get("instruction", "")
            context = rec.get("context", "")
            constraints = rec.get("constraints", "")
            payload = rec.get("payload", "")
            
            # If instruction is missing but prompt exists, try to reverse engineer or just use default
            if not instruction and not context:
                # Fallback: Generate a generic instruction based on attack type
                atk = rec.get("attack_type", "SQLI")
                instruction = f"Generate a {atk} payload."

            # Build new prompt/chosen
            prompt_text, chosen_text = build_phi3_prompt_and_chosen(instruction, context, constraints, payload)
            
            new_rec = dict(rec)
            new_rec["prompt"] = prompt_text
            new_rec["chosen"] = chosen_text
            
            out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} records to Phi-3 format at {args.output}")

if __name__ == "__main__":
    main()
