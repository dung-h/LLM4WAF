
import json
import os
import sys
import argparse
from collections import Counter
from datetime import datetime

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

def main():
    parser = argparse.ArgumentParser(description="Compare BLUE LLM suggestions from different backends.")
    parser.add_argument("--gemma_suggestions_file", type=str, default="data/blue/blue_phase3_suggestions.jsonl",
                        help="Path to Gemma2 suggestions JSONL file.")
    parser.add_argument("--phi3_suggestions_file", type=str, default="data/blue/blue_phase3_suggestions_phi3.jsonl",
                        help="Path to Phi-3 Mini suggestions JSONL file.")
    parser.add_argument("--output_file", type=str, default="data/blue/blue_phase3_compare_backends.txt",
                        help="Path to save the comparison report.")
    args = parser.parse_args()

    cmd_str = f"python blue/phase3_compare_backends.py --gemma_suggestions_file {args.gemma_suggestions_file} --phi3_suggestions_file {args.phi3_suggestions_file} --output_file {args.output_file}"

    report_content = []
    report_content.append("---")
    report_content.append("BLUE Phase 3 Backend Comparison")
    report_content.append("=====================================\n")

    def analyze_suggestions(filepath, backend_name):
        if not os.path.exists(filepath):
            report_content.append(f"Error: Suggestions file not found for {backend_name} at {filepath}.")
            return None, None, None

        suggestions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                suggestions.append(json.loads(line.strip()))
        
        total_episodes = len(suggestions)
        json_valid_count = 0
        rule_recommendation_count = 0
        vuln_effect_counts = Counter()

        for s in suggestions:
            blue_output = s["blue_recommendation"]
            
            # Check for successful JSON parsing (not a fallback error)
            if blue_output.get("recommended_actions") and not \
               any("LLM_ERROR" in action for action in blue_output["recommended_actions"]):
                json_valid_count += 1
                
                # Count rules if JSON was valid
                if blue_output.get("recommended_rules"):
                    rule_recommendation_count += len(blue_output["recommended_rules"])
                
                vuln_effect_counts[blue_output.get("vuln_effect", "UNKNOWN")] += 1
            
        return total_episodes, json_valid_count, rule_recommendation_count, vuln_effect_counts

    # Analyze Gemma2 results
    report_content.append(f"--- Gemma 2 Results (File: {args.gemma_suggestions_file}) ---")
    gemma_total, gemma_json_valid, gemma_rule_recs, gemma_vuln_effects = analyze_suggestions(args.gemma_suggestions_file, "Gemma2")
    if gemma_total is not None:
        report_content.append(f"  Total episodes processed: {gemma_total}")
        report_content.append(f"  JSON Valid Responses: {gemma_json_valid}/{gemma_total} ({gemma_json_valid/gemma_total:.2%})")
        report_content.append(f"  Total Recommended Rules (from valid JSONs): {gemma_rule_recs}")
        report_content.append(f"  Top Vulnerability Effects Identified: {gemma_vuln_effects.most_common(3) if gemma_vuln_effects else 'N/A'}")
        report_content.append("\n")

    # Analyze Phi-3 Mini results
    report_content.append(f"--- Phi-3 Mini Results (File: {args.phi3_suggestions_file}) ---")
    phi3_total, phi3_json_valid, phi3_rule_recs, phi3_vuln_effects = analyze_suggestions(args.phi3_suggestions_file, "Phi-3 Mini")
    if phi3_total is not None:
        report_content.append(f"  Total episodes processed: {phi3_total}")
        report_content.append(f"  JSON Valid Responses: {phi3_json_valid}/{phi3_total} ({phi3_json_valid/phi3_total:.2%})")
        report_content.append(f"  Total Recommended Rules (from valid JSONs): {phi3_rule_recs}")
        report_content.append(f"  Top Vulnerability Effects Identified: {phi3_vuln_effects.most_common(3) if phi3_vuln_effects else 'N/A'}")
        report_content.append("\n")

    # Overall Comparison
    report_content.append("---")
    report_content.append("Overall Comparison")
    if gemma_total is not None and phi3_total is not None:
        report_content.append(f"  Gemma2 JSON Validity: {gemma_json_valid/gemma_total:.2%} vs Phi-3 Mini: {phi3_json_valid/phi3_total:.2%}")
        report_content.append(f"  Gemma2 Recommended Rules: {gemma_rule_recs} vs Phi-3 Mini: {phi3_rule_recs}")
    else:
        report_content.append("  Comparison not possible due to missing data.")

    # Save report
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    print(f"\nComparison report saved to {args.output_file}")
    print("\n" + "\n".join(report_content))
    log_message(cmd_str, "OK", args.output_file)

if __name__ == "__main__":
    main()
