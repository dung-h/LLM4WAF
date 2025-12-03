
import json
import os
import sys
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

def generate_rag_sft_dataset(input_eval_file, output_sft_file):
    print(f"Generating RAG-aware SFT dataset from {input_eval_file}...")
    
    sft_candidates = []
    
    records = []
    if os.path.exists(input_eval_file):
        with open(input_eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue

    for record in records:
        if record["mode"] == "rag_v2" and record["result"] == "passed": # Only take RAG-mode passed samples
            sft_candidate = {
                "attack_type": record["attack_type"],
                "waf_profile": record["waf_profile"],
                "rag_docs_used": record.get("rag_docs_used", []),
                # Assuming history_payloads is from the build_red_prompt_with_rag calls,
                # which in this mini eval is empty. For real SFT, this would be from actual history.
                "history_payloads": [], 
                "final_payload": record["payload"],
                "result": record["result"]
            }
            sft_candidates.append(sft_candidate)
    
    # Save SFT candidates
    os.makedirs(os.path.dirname(output_sft_file), exist_ok=True)
    with open(output_sft_file, 'w', encoding='utf-8') as f:
        for candidate in sft_candidates:
            f.write(json.dumps(candidate) + "\n")
            
    print(f"Generated {len(sft_candidates)} RAG-aware SFT candidates to {output_sft_file}")
    return len(sft_candidates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_eval_file", type=str, default="eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl")
    parser.add_argument("--output_sft_file", type=str, default="data/processed/red_phase2_rag_sft_candidates.jsonl")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_rag_generate_sft_dataset.py"
    try:
        count = generate_rag_sft_dataset(args.input_eval_file, args.output_sft_file)
        if count is not None:
            log_message(cmd_str, "OK", args.output_sft_file)
        else:
            log_message(cmd_str, "FAIL", "Error generating SFT dataset")
    except Exception as e:
        print(f"Error running SFT dataset generation: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))
