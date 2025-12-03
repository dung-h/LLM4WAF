
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
import argparse

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def extract_patterns(input_file, output_file):
    print(f"Extracting patterns from {input_file}...")
    
    # Dictionary to store groups: keys = (attack_type, technique), value = list of payloads
    patterns = defaultdict(list)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    attack_type = sample.get("attack_type", "UNKNOWN")
                    technique = sample.get("technique", "UNKNOWN")
                    
                    # Extract payload
                    payload = ""
                    if "messages" in sample:
                        for m in sample["messages"]:
                            if m["role"] == "assistant":
                                payload = m["content"]
                                break
                    elif "payload" in sample: # Fallback
                        payload = sample["payload"]
                    
                    if payload:
                        patterns[(attack_type, technique)].append(payload)
                        
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return None

    print(f"Found {len(patterns)} unique patterns (attack_type + technique).")
    
    # Create documents
    documents = []
    for idx, ((attack_type, technique), payloads) in enumerate(patterns.items()):
        # Sanitize technique for ID
        sanitized_tech = "".join([c if c.isalnum() else "_" for c in technique])[:50]
        
        doc = {
            "doc_id": f"pattern_{attack_type}_{sanitized_tech}_{idx}",
            "kind": "payload_pattern",
            "attack_type": attack_type,
            "technique": technique,
            "source": ["red_phase1_enriched_v2"],
            "count": len(payloads),
            "short_desc": f"{attack_type} pattern using {technique}",
            "example_payloads": payloads[:10] # Take top 10 examples
        }
        documents.append(doc)
    
    # Save to output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
            
    print(f"Saved {len(documents)} pattern documents to {output_file}")
    return len(documents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/processed/red_phase1_enriched_v2.jsonl")
    parser.add_argument("--output_file", default="data/rag/internal_payload_patterns.jsonl")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_extract_internal_patterns.py --input_file {args.input_file}"
    
    try:
        count = extract_patterns(args.input_file, args.output_file)
        if count is not None:
            log_message(cmd_str, "OK", args.output_file)
        else:
            log_message(cmd_str, "FAIL", "Input file not found")
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))