import json
import os
import re

# --- Configuration ---
ORIGINAL_DATA_PATH = "processed/advanced_sqli_finetune_data.jsonl"
NEW_DATA_PATH = "data/processed/deepseek_generated_sqli.jsonl"
OUTPUT_PATH = "data/processed/red_train_v7_enriched.jsonl"

def extract_payload_from_text(text: str) -> str:
    """Extracts the payload from the 'text' field of a JSONL entry."""
    match = re.search(r"Payload:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def format_sft_entry(payload: str) -> str:
    """Formats a payload into the SFT JSONL structure, escaping braces."""
    # Escape braces in the payload itself to avoid format string errors
    safe_payload = payload.replace("{", "{{").replace("}", "}}")
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
    print("--- Starting Dataset Consolidation and Deduplication ---")
    
    unique_payloads = set()
    
    # 1. Read original data
    print(f"[+] Reading original data from '{ORIGINAL_DATA_PATH}'...")
    try:
        with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    full_text = data.get('text', '')
                    payload = extract_payload_from_text(full_text)
                    if payload:
                        unique_payloads.add(payload)
                except json.JSONDecodeError:
                    continue
        print(f"  - Found {len(unique_payloads)} unique payloads so far.")
    except FileNotFoundError:
        print(f"  [!] Warning: Original data file not found at '{ORIGINAL_DATA_PATH}'. Skipping.")

    # 2. Read new data
    initial_count = len(unique_payloads)
    print(f"\n[+] Reading new data from '{NEW_DATA_PATH}'...")
    try:
        with open(NEW_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    full_text = data.get('text', '')
                    payload = extract_payload_from_text(full_text)
                    if payload:
                        unique_payloads.add(payload)
                except json.JSONDecodeError:
                    continue
        newly_added = len(unique_payloads) - initial_count
        print(f"  - Added {newly_added} new unique payloads.")
    except FileNotFoundError:
        print(f"  [!] Error: New data file not found at '{NEW_DATA_PATH}'. Aborting.")
        return

    # 3. Write to output file
    print(f"\n[+] Writing {len(unique_payloads)} total unique payloads to '{OUTPUT_PATH}'...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for payload in sorted(list(unique_payloads)): # Sort for consistent output
            f.write(format_sft_entry(payload) + "\n")
            
    print("\n--- Consolidation Complete ---")
    print(f"Original unique payloads: {initial_count}")
    print(f"Newly added unique payloads: {newly_added}")
    print(f"Total unique payloads in final dataset: {len(unique_payloads)}")
    
    # 4. Clean up intermediate files
    if os.path.exists(NEW_DATA_PATH):
        os.remove(NEW_DATA_PATH)
        print(f"[+] Cleaned up intermediate file: '{NEW_DATA_PATH}'")
    if os.path.exists("scripts/etl/enrich_sqli_deepseek.py"):
        os.remove("scripts/etl/enrich_sqli_deepseek.py")
        print("[+] Cleaned up enrichment script.")


if __name__ == "__main__":
    main()
