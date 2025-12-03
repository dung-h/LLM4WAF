
import json
import os
import sys
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

def analyze_doc_impact(input_eval_file, output_impact_file, output_summary_file):
    print(f"Analyzing RAG doc impact from {input_eval_file}...")
    
    doc_stats = defaultdict(lambda: {"used_count": 0, "success_count": 0, "pass_rate": 0.0, "doc_info": {}})
    
    records = []
    if os.path.exists(input_eval_file):
        with open(input_eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    for record in records:
        if record["mode"] == "rag_v2" and record.get("rag_docs_used"):
            is_success = (record["result"] == "passed")
            for doc_meta in record["rag_docs_used"]:
                doc_id = doc_meta["doc_id"]
                doc_stats[doc_id]["used_count"] += 1
                if is_success:
                    doc_stats[doc_id]["success_count"] += 1
                
                # Store doc info once
                if not doc_stats[doc_id]["doc_info"]:
                    doc_stats[doc_id]["doc_info"] = doc_meta

    # Calculate pass rates
    impact_docs = []
    for doc_id, stats in doc_stats.items():
        if stats["used_count"] > 0:
            stats["pass_rate"] = stats["success_count"] / stats["used_count"]
        impact_docs.append({
            "doc_id": doc_id,
            "kind": stats["doc_info"].get("kind", "unknown"),
            "attack_type": stats["doc_info"].get("attack_type", "unknown"),
            "technique": stats["doc_info"].get("technique"),
            "waf_profile": stats["doc_info"].get("waf_profile"),
            "used_count": stats["used_count"],
            "success_count": stats["success_count"],
            "pass_rate": stats["pass_rate"],
            "notes": f"Impact analysis for {doc_id}" # Placeholder
        })
    
    # Save detailed doc impact
    os.makedirs(os.path.dirname(output_impact_file), exist_ok=True)
    with open(output_impact_file, 'w', encoding='utf-8') as f:
        for doc in impact_docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Detailed RAG doc impact saved to {output_impact_file}")

    # Generate summary
    summary_content = []
    summary_content.append("--- RAG Document Impact Summary ---")
    summary_content.append(f"Analyzed {len(records)} RAG-mode attacks.")
    summary_content.append(f"Identified impact for {len(impact_docs)} unique RAG documents.\n")

    # Filter out docs used less than 5 times for meaningful success rate
    meaningful_impact_docs = [doc for doc in impact_docs if doc["used_count"] >= 5]

    summary_content.append("Top 10 Documents by Usage Count:")
    for doc in sorted(impact_docs, key=lambda x: x["used_count"], reverse=True)[:10]:
        summary_content.append(f"- {doc['doc_id']} (Kind: {doc['kind']}, Type: {doc['attack_type']}): Used={doc['used_count']}, Passed={doc['success_count']}, Rate={doc['pass_rate']:.2%}")

    summary_content.append("\nTop 10 Documents by Success Rate (used >= 5 times):")
    if not meaningful_impact_docs:
        summary_content.append("- No documents used enough times for meaningful success rate.")
    else:
        for doc in sorted(meaningful_impact_docs, key=lambda x: x["pass_rate"], reverse=True)[:10]:
            summary_content.append(f"- {doc['doc_id']} (Kind: {doc['kind']}, Type: {doc['attack_type']}): Used={doc['used_count']}, Passed={doc['success_count']}, Rate={doc['pass_rate']:.2%}")
    
    # Save summary
    os.makedirs(os.path.dirname(output_summary_file), exist_ok=True)
    with open(output_summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_content))
    print(f"RAG doc impact summary saved to {output_summary_file}")
    
    return output_impact_file, output_summary_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_eval_file", type=str, default="eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl")
    parser.add_argument("--output_impact_file", type=str, default="data/rag/red_rag_doc_impact.jsonl")
    parser.add_argument("--output_summary_file", type=str, default="data/rag/red_rag_doc_impact_summary.txt")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_rag_analyze_doc_impact.py"
    try:
        impact_file, summary_file = analyze_doc_impact(args.input_eval_file, args.output_impact_file, args.output_summary_file)
        log_message(cmd_str, "OK", f"{impact_file}, {summary_file}")
    except Exception as e:
        print(f"Error running doc impact analysis: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))
