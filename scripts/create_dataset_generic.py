import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# --- Configuration & Constants ---

DEFAULT_INSTRUCTIONS = {
    "sqli": [
        "Generate a SQL injection payload.",
        "Craft a SQLi attack string.",
        "Provide a SQL injection vector."
    ],
    "xss": [
        "Generate a Cross-Site Scripting (XSS) payload.",
        "Craft an XSS attack string.",
        "Provide a payload for XSS testing."
    ],
    "default": [
        "Generate a security testing payload.",
        "Provide a payload for penetration testing."
    ]
}

FORMAT_TEMPLATES = {
    "gemma": {
        "prompt": "<start_of_turn>user\n{instruction}\n{context}\n{constraints}<end_of_turn>\n<start_of_turn>model\n",
        "chosen": "{payload}<end_of_turn>\n"
    },
    "phi3": {
        "prompt": "<|user|}\n{instruction}\n{context}\n{constraints}<|end|>\n<|assistant|}\n",
        "chosen": "{payload}<|end|>\n"
    },
    "simple": {
        "prompt": "Payload: ",
        "chosen": "{payload}"
    }
}

# --- Helper Functions ---

def categorize_payload(payload: str) -> str:
    """Simple heuristic to categorize payload type."""
    payload_lower = payload.lower()
    if any(k in payload_lower for k in ["union select", "sleep(", "benchmark(", "extractvalue", "updatexml"]):
        return "sqli"
    if any(k in payload_lower for k in ["<script", "onerror=", "onload=", "javascript:", "alert("]):
        return "xss"
    return "default"

def build_chat_prompt(template_name: str, instruction: str, context: str = "", constraints: str = "") -> str:
    """Builds the user prompt part based on the selected template."""
    template = FORMAT_TEMPLATES.get(template_name, FORMAT_TEMPLATES["gemma"])[ "prompt"]
    # Only add newlines if fields are not empty to keep it clean
    context_str = f"Context: {context}" if context else ""
    constraints_str = f"Constraints: {constraints}" if constraints else ""
    
    return template.format(
        instruction=instruction,
        context=context_str,
        constraints=constraints_str
    ).strip() + "\n" # Ensure trailing newline for chat formats

def build_full_entry(template_name: str, prompt: str, payload: str) -> Dict[str, str]:
    """Builds the full dataset entry with 'prompt' and 'chosen' fields."""
    template = FORMAT_TEMPLATES.get(template_name, FORMAT_TEMPLATES["gemma"])[ "chosen"]
    chosen_text = template.format(payload=payload)
    
    if template_name == "simple":
        return {"text": f"{prompt}{payload}"}
    
    return {
        "prompt": prompt,
        "chosen": f"{prompt}{chosen_text}", # Standard format often includes prompt in chosen for some trainers
        "completion": chosen_text # Some trainers prefer separated completion
    }

# --- Main Logic ---

def process_jsonl_mode(input_path: Path, mode: str) -> List[Dict[str, Any]]:
    """Reads records from a JSONL file."""
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def process_raw_mode(input_path: Path) -> List[Dict[str, Any]]:
    """Reads raw payloads from .txt files in a directory."""
    records = []
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob("*.txt"))
    
    for file in files:
        with file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                payload = line.strip()
                if payload and len(payload) > 3: # Basic filter
                    records.append({
                        "payload": payload,
                        "attack_type": categorize_payload(payload)
                    })
    return records

def main():
    parser = argparse.ArgumentParser(description="Generic script to create SFT datasets from various sources.")
    
    parser.add_argument("--input", type=Path, required=True, help="Input path (JSONL file or directory of TXT files).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path.")
    parser.add_argument("--mode", choices=["jsonl", "raw"], default="jsonl", help="Input mode: 'jsonl' for existing records, 'raw' for raw payload files.")
    parser.add_argument("--format", choices=["gemma", "phi3", "simple"], default="gemma", help="Output format template.")
    parser.add_argument("--balance", action="store_true", help="Whether to balance the dataset across categories.")
    parser.add_argument("--reasoning", action="store_true", help="If set, creates a reasoning dataset (requires 'reasoning' field in input).")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"[INFO] Loading data from {args.input} in {args.mode} mode...")
    if args.mode == "jsonl":
        data = process_jsonl_mode(args.input, args.mode)
    else:
        data = process_raw_mode(args.input)
    
    print(f"[INFO] Loaded {len(data)} raw records.")

    # 2. Process & Categorize
    categorized_data = {}
    for rec in data:
        cat = rec.get("attack_type", "default").lower()
        if cat not in categorized_data:
            categorized_data[cat] = []
        categorized_data[cat].append(rec)

    # 3. Balance (Optional)
    if args.balance:
        min_count = min(len(v) for v in categorized_data.values())
        print(f"[INFO] Balancing dataset to {min_count} records per category.")
        for cat in categorized_data:
            categorized_data[cat] = random.sample(categorized_data[cat], min_count)

    # 4. Format Output
    final_dataset = []
    
    for cat, records in categorized_data.items():
        for rec in records:
            payload = rec.get("payload", "")
            
            if args.reasoning:
                # Special handling for reasoning dataset
                reasoning = rec.get("reasoning", "")
                if not reasoning: continue
                
                # Logic similar to create_v14_reasoning_sft_dataset.py
                instruction = f"Explain the following {cat.upper()} payload."
                context = rec.get("context", "")
                prompt_text = build_chat_prompt(args.format, instruction, context, f"Payload: {payload}")
                entry = build_full_entry(args.format, prompt_text, reasoning)
                
            else:
                # Standard SFT dataset
                instruction = rec.get("instruction")
                if not instruction:
                    # Pick a random default instruction if none exists
                    instr_list = DEFAULT_INSTRUCTIONS.get(cat, DEFAULT_INSTRUCTIONS["default"])
                    instruction = random.choice(instr_list)
                
                context = rec.get("context", "")
                constraints = rec.get("constraints", "")
                
                prompt_text = build_chat_prompt(args.format, instruction, context, constraints)
                entry = build_full_entry(args.format, prompt_text, payload)
            
            final_dataset.append(entry)

    # 5. Shuffle and Save
    random.shuffle(final_dataset)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Successfully created dataset at {args.output} with {len(final_dataset)} records.")

if __name__ == "__main__":
    main()
