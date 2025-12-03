
import json
import os
import sys
import glob
from collections import defaultdict, Counter
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

def extract_strategies(eval_dir, output_cases_file, output_strategies_file):
    print(f"Scanning {eval_dir} for eval logs...")
    log_files = glob.glob(os.path.join(eval_dir, "red_phase*.jsonl"))
    
    # Structure: (attack_type, waf_profile) -> list of records
    clusters = defaultdict(list)
    
    total_records = 0
    for log_file in log_files:
        print(f"Processing {log_file}...")
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    attack_type = record.get("attack_type", "UNKNOWN")
                    waf_profile = record.get("waf_profile", "unknown")
                    
                    clusters[(attack_type, waf_profile)].append(record)
                    total_records += 1
                except:
                    continue

    print(f"Processed {total_records} records into {len(clusters)} clusters.")

    cases_docs = []
    strategies_docs = []

    for idx, ((attack_type, waf_profile), records) in enumerate(clusters.items()):
        # --- Create Attack Case Doc ---
        status_counts = Counter([r.get("status", "unknown") for r in records])
        payloads = [r.get("payload") or r.get("generated_payload") for r in records]
        payloads = [p for p in payloads if p][:10] # Top 10 examples
        
        case_doc = {
            "doc_id": f"case_{attack_type}_{waf_profile}_{idx}",
            "kind": "attack_case",
            "attack_type": attack_type,
            "waf_profile": waf_profile,
            "stats": dict(status_counts),
            "observed_behavior": {
                "notes": f"Aggregated results from {len(records)} attempts against {waf_profile}.",
                "top_statuses": [s[0] for s in status_counts.most_common(3)]
            },
            "example_payloads": payloads
        }
        cases_docs.append(case_doc)

        # --- Create Attack Strategy Doc (Heuristic) ---
        # Infer strategy: what worked?
        passed_records = [r for r in records if r.get("status") in ["passed", "sql_error_bypass", "reflected", "reflected_no_exec"]]
        
        if passed_records:
            # If we have successes, this cluster reveals a working strategy
            success_payloads = [r.get("payload") or r.get("generated_payload") for r in passed_records][:5]
            
            strategy_doc = {
                "doc_id": f"strategy_{attack_type}_{waf_profile}_{idx}",
                "kind": "attack_strategy",
                "attack_type": attack_type,
                "idea": f"Effective strategy against {waf_profile} for {attack_type}.",
                "when_to_use": [f"Targeting {waf_profile}", "Bypassing standard filters"],
                "worked_on_profiles": [waf_profile],
                "example_payloads": success_payloads
            }
            strategies_docs.append(strategy_doc)

    # Save
    os.makedirs(os.path.dirname(output_cases_file), exist_ok=True)
    
    with open(output_cases_file, 'w', encoding='utf-8') as f:
        for doc in cases_docs:
            f.write(json.dumps(doc) + "\n")
            
    with open(output_strategies_file, 'w', encoding='utf-8') as f:
        for doc in strategies_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Saved {len(cases_docs)} cases to {output_cases_file}")
    print(f"Saved {len(strategies_docs)} strategies to {output_strategies_file}")
    return len(cases_docs) + len(strategies_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="eval")
    parser.add_argument("--output_cases", default="data/rag/internal_attack_cases.jsonl")
    parser.add_argument("--output_strategies", default="data/rag/internal_attack_strategies.jsonl")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_extract_internal_strategies.py"
    
    try:
        count = extract_strategies(args.eval_dir, args.output_cases, args.output_strategies)
        log_message(cmd_str, "OK", f"{args.output_cases}, {args.output_strategies}")
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))
