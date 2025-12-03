
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

def merge_with_writeups(corpus_v1_file, strategies_file, writeups_file, strategies_from_writeups_file, output_file):
    print("Merging corpus with new strategies and writeups (if any).")
    merged_docs = []
    
    # Load Corpus V1
    if os.path.exists(corpus_v1_file):
        with open(corpus_v1_file, 'r', encoding='utf-8') as f:
            for line in f:
                merged_docs.append(json.loads(line.strip()))
    
    # Load Strategies from patterns (from TASK 2)
    if os.path.exists(strategies_file):
        with open(strategies_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                # Ensure text_for_embedding is consistent
                text = f"Attack Strategy: {doc['attack_type']}. Idea: {doc['idea']}. Notes: {doc['notes']}"
                doc['text_for_embedding'] = text
                merged_docs.append(doc)
    
    # Load Writeups (if any)
    if os.path.exists(writeups_file):
        with open(writeups_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                text = f"Attack Writeup: {doc['title']}. Summary: {doc['short_summary']}. WAF Behavior: {doc['waf_behavior_notes']}. Strategy Ideas: {', '.join(doc['strategy_ideas'])}"
                doc['text_for_embedding'] = text
                merged_docs.append(doc)

    # Load Strategies from Writeups (if any)
    if os.path.exists(strategies_from_writeups_file):
        with open(strategies_from_writeups_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                text = f"Attack Strategy (from writeup): {doc['attack_type']}. Idea: {doc['idea']}. Notes: {doc['notes']}"
                doc['text_for_embedding'] = text
                merged_docs.append(doc)

    # Ensure unique doc_id
    final_docs = {}
    for doc in merged_docs:
        final_docs[doc['doc_id']] = doc

    merged_count = len(final_docs)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, doc in final_docs.items():
            f.write(json.dumps(doc) + "\n")
            
    print(f"Merged {merged_count} documents into {output_file}")
    return merged_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_v1_file", default="data/rag/red_corpus_internal.jsonl")
    parser.add_argument("--strategies_file", default="data/rag/internal_attack_strategies.jsonl")
    parser.add_argument("--writeups_file", default="data/rag/internal_writeups.jsonl")
    parser.add_argument("--strategies_from_writeups_file", default="data/rag/internal_strategies_from_writeups.jsonl")
    parser.add_argument("--output_file", default="data/rag/red_corpus_internal_v2.jsonl")
    args = parser.parse_args()
    
    cmd_str = f"python scripts/red_rag_merge_with_writeups.py"
    
    try:
        count = merge_with_writeups(args.corpus_v1_file, args.strategies_file, args.writeups_file, args.strategies_from_writeups_file, args.output_file)
        if count is not None:
            log_message(cmd_str, "OK", args.output_file)
        else:
            log_message(cmd_str, "FAIL", "Error in merging corpus")
    except Exception as e:
        print(f"Error: {e}")
        log_message(cmd_str, "FAIL", str(e))
