
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

def read_summary_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return f"File not found: {filepath}\n"

def count_jsonl_samples(filepath):
    count = 0
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvwa_pl1_extended_summary", type=str, default="eval/red_rag_eval_dvwa_pl1_extended_summary.txt")
    parser.add_argument("--multiwaf_extended_summary", type=str, default="eval/red_rag_eval_multiwaf_extended_summary.txt")
    parser.add_argument("--doc_impact_summary", type=str, default="data/rag/red_rag_doc_impact_summary.txt")
    parser.add_argument("--sft_candidates_file", type=str, default="data/processed/red_phase2_rag_sft_candidates.jsonl")
    parser.add_argument("--output_report_file", type=str, default="eval/red_rag_overall_report_v2.txt")
    args = parser.parse_args()

    cmd_str = f"python scripts/red_rag_report_v2.py"
    
    report_content = []
    report_content.append("--- RED RAG Overall Report V2 ---\n")

    # 1. DVWA + modsec_pl1 Extended Summary
    report_content.append("## 1. DVWA + ModSecurity PL1 Extended Evaluation Summary\n")
    report_content.append(read_summary_file(args.dvwa_pl1_extended_summary))
    report_content.append("\n")

    # 2. Multi-WAF Extended Summary
    report_content.append("## 2. Multi-WAF Extended Evaluation Summary\n")
    report_content.append(read_summary_file(args.multiwaf_extended_summary))
    report_content.append("\n")

    # 3. RAG Doc Impact Summary
    report_content.append("## 3. RAG Document Impact Analysis Summary\n")
    report_content.append(read_summary_file(args.doc_impact_summary))
    report_content.append("\n")

    # 4. SFT/RL Candidates Count
    report_content.append("## 4. SFT/RL Dataset Candidates\n")
    sft_count = count_jsonl_samples(args.sft_candidates_file)
    report_content.append(f"Number of RAG-aware SFT/RL candidates generated: {sft_count} samples in {args.sft_candidates_file}\n")
    report_content.append("\n")

    # Save overall report
    os.makedirs(os.path.dirname(args.output_report_file), exist_ok=True)
    with open(args.output_report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    print(f"Overall report saved to {args.output_report_file}")
    
    log_message(cmd_str, "OK", args.output_report_file)

if __name__ == "__main__":
    main()
