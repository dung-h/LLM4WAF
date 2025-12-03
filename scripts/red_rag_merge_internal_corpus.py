
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

def merge_corpus(pattern_file, case_file, strategy_file, output_file):
    print("Merging internal corpus...")
    merged_docs = []
    
    # 1. Process Patterns
    if os.path.exists(pattern_file):
        with open(pattern_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                # Create embedding text
                text = f"Payload Pattern: {doc['short_desc']}. Attack Type: {doc['attack_type']}. Technique: {doc['technique']}. Count: {doc['count']}."
                doc['text_for_embedding'] = text
                merged_docs.append(doc)
    
    # 2. Process Cases
    if os.path.exists(case_file):
        with open(case_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                stats_str = ", ".join([f"{k}: {v}" for k, v in doc['stats'].items()])
                text = f"Attack Case: {doc['attack_type']} against {doc['waf_profile']}. Results: {stats_str}. Notes: {doc['observed_behavior']['notes']}"
                doc['text_for_embedding'] = text
                merged_docs.append(doc)

    # 3. Process Strategies
    if os.path.exists(strategy_file):
        with open(strategy_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                text = f"Attack Strategy: {doc['attack_type']}. Idea: {doc['idea']}. Best for: {', '.join(doc['worked_on_profiles'])}."
                doc['text_for_embedding'] = text
                merged_docs.append(doc)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in merged_docs:
            f.write(json.dumps(doc) + "\n")
            
    print(f"Merged {len(merged_docs)} documents into {output_file}")
    return len(merged_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_file", default="data/rag/internal_payload_patterns.jsonl")
    parser.add_argument("--case_file", default="data/rag/internal_attack_cases.jsonl")
    parser.add_argument("--strategy_file", default="data/rag/internal_attack_strategies.jsonl")
    parser.add_argument("--output_file", default="data/rag/red_corpus_internal.jsonl")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_merge_internal_corpus.py"
    
    try:
        count = merge_corpus(args.pattern_file, args.case_file, args.strategy_file, args.output_file)
        log_message(cmd_str, "OK", args.output_file)
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))
