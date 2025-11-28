import json
import os
import re

# --- Configuration ---
FILTERED_DATA_PATH = "data/processed/red_train_v7_filtered.jsonl"
ENRICHED_DATA_V2_PATH = "data/processed/deepseek_generated_sqli_v2.jsonl"
FINAL_OUTPUT_PATH = "data/processed/red_train_v8_final.jsonl"

def extract_payload_from_text(text: str) -> str:
    """Extracts the payload from the 'text' field of a JSONL entry."""
    match = re.search(r"Payload:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def format_sft_entry(payload: str) -> str:
    """Formats a payload into the SFT JSONL structure, escaping braces."""
    safe_payload = payload.replace("{{", "{{").replace("}}", "}}")
    template = {
        "text": (
            "<|system|>\nYou are a helpful AI assistant.\n<|end|>\n"
            "<|user|>\nGenerate a single, effective MySQL SQL injection payload.\n"
            "Output only the payload itself, with no other text or explanation.\n<|end|>\n"
            "<|assistant|>\nPayload: {payload}"
        )
    }
    return json.dumps({"text": template["text"].format(payload=safe_payload)})

def main():
    print("--- Starting Final Dataset Consolidation ---")
    
    unique_payloads = set()
    
    # 1. Read filtered data
    print(f"[+] Reading filtered data from '{FILTERED_DATA_PATH}'...")
    try:
        with open(FILTERED_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    payload = extract_payload_from_text(data.get('text', ''))
                    if payload:
                        unique_payloads.add(payload)
                except json.JSONDecodeError:
                    continue
        print(f"  - Found {len(unique_payloads)} unique payloads so far.")
    except FileNotFoundError:
        print(f"  [!] Warning: Filtered data file not found at '{FILTERED_DATA_PATH}'. Skipping.")

    # 2. Read new enriched data (v2)
    initial_count = len(unique_payloads)
    print(f"\n[+] Reading new enriched data from '{ENRICHED_DATA_V2_PATH}'...")
    try:
        with open(ENRICHED_DATA_V2_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    payload = extract_payload_from_text(data.get('text', ''))
                    if payload:
                        unique_payloads.add(payload)
                except json.JSONDecodeError:
                    continue
        newly_added = len(unique_payloads) - initial_count
        print(f"  - Added {newly_added} new unique payloads.")
    except FileNotFoundError:
        print(f"  [!] Error: New enriched data file not found at '{ENRICHED_DATA_V2_PATH}'. Aborting.")
        return

    # 3. Write to final output file
    print(f"\n[+] Writing {len(unique_payloads)} total unique payloads to '{FINAL_OUTPUT_PATH}'...")
    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for payload in sorted(list(unique_payloads)): # Sort for consistent output
            f.write(format_sft_entry(payload) + "\n")
            
    print("\n--- Final Consolidation Complete ---")
    print(f"Payloads from filtered dataset: {initial_count}")
    print(f"Newly added unique payloads: {newly_added}")
    print(f"Total unique payloads in final dataset: {len(unique_payloads)}")
    
    # 4. Clean up intermediate files
    for f_path in [FILTERED_DATA_PATH, ENRICHED_DATA_V2_PATH, "scripts/etl/filter_low_quality.py", "scripts/etl/enrich_sqli_deepseek_v2.py"]:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"[+] Cleaned up intermediate file: '{f_path}'")

if __name__ == "__main__":
    main()
