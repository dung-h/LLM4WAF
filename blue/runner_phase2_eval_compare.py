import json
import os
import sys
import argparse
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blue.rag_index import CRSKnowledgeBase
from blue.rag_retriever import retrieve_for_episode
from blue.prompts import BLUE_PROMPT_TEMPLATE
from blue.llm_client import call_blue_llm, BlueModelBackend, get_blue_backend # Import BlueModelBackend and get_blue_backend

# --- Configuration ---
GOLDEN_SET_FILE = "data/blue/blue_phase1_golden.jsonl"
KB_FILE = "data/blue/blue_phase1_crs_kb.jsonl"

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def format_episode_for_prompt(episode: Dict) -> Dict:
    """Extracts relevant fields from a blue_episode for the prompt."""
    return {
        "app_context": {
            "app_name": episode["app_context"]["app_name"],
            "endpoint": episode["app_context"]["endpoint"],
            "http_method": episode["app_context"]["http_method"],
            "injection_point": episode["app_context"]["injection_point"],
            "tech_stack": episode["app_context"]["tech_stack"]
        },
        "attack": {
            "attack_type": episode["attack"]["attack_type"],
            "technique": episode["attack"]["technique"],
            "payload": episode["attack"]["payload"]
        },
        "waf_observation": {
            "blocked": episode["waf_observation"]["blocked"],
            "http_status": episode["waf_observation"]["http_status"]
        },
        "app_observation": {
            "http_status": episode["app_observation"]["http_status"],
            "resp_body_snippet": episode["app_observation"]["resp_body_snippet"],
            "error_class": episode["app_observation"]["error_class"]
        },
        "blue_label": {
            "vuln_effect": episode["blue_label"]["vuln_effect"],
            "is_false_negative": episode["blue_label"]["is_false_negative"]
        }
    }

def format_kb_entry_for_prompt(kb_entry: Dict) -> Dict:
    """Extracts relevant fields from a KB entry for the prompt."""
    return {
        "rule_id": kb_entry.get("rule_id"),
        "test_description": kb_entry.get("test_description"),
        "attack_type": kb_entry.get("attack_type"),
        "variables": kb_entry.get("variables"),
        "operator": kb_entry.get("operator")
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLUE LLM performance on golden set for different backends.")
    parser.add_argument("--output_prefix", type=str, default="blue_phase2_eval",
                        help="Prefix for output files (e.g., 'blue_phase2_eval').")
    parser.add_argument("--backends", nargs='*', default=[backend.value for backend in BlueModelBackend],
                        help="List of backends to evaluate (e.g., 'gemma2 phi3_mini'). Evaluates all if not specified.")
    args = parser.parse_args()

    # 1. Load data
    golden_set = []
    with open(GOLDEN_SET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            golden_set.append(json.loads(line.strip()))
    print(f"Loaded {len(golden_set)} golden set episodes.")

    kb = CRSKnowledgeBase(kb_path=KB_FILE)
    print(f"Loaded {len(kb.entries)} KB entries.")
    
    # --- Loop through specified backends ---
    for backend_str in args.backends:
        try:
            current_backend = BlueModelBackend(backend_str)
        except ValueError:
            print(f"Skipping unknown backend: {backend_str}", file=sys.stderr)
            continue

        os.environ["BLUE_BACKEND"] = current_backend.value
        print(f"\n--- Evaluating with {current_backend.value.upper()} backend ---")
        cmd_str = f"python blue/runner_phase2_eval_compare.py --output_prefix {args.output_prefix} --backends {current_backend.value}"
        
        eval_results = []
        rule_hit_counts = Counter()
        total_golden_cases = 0
        rule_hit_summary = {"hit": 0, "total": 0}
        json_valid_count = 0
        json_error_count = 0

        # 2. Iterate through golden set
        for i, episode in enumerate(golden_set):
            total_golden_cases += 1
            
            # Retrieve KB snippets
            retrieved_kb_entries = retrieve_for_episode(episode, kb, top_k=2)
            
            # Build prompt
            episode_json_for_prompt = json.dumps(format_episode_for_prompt(episode), indent=2)
            kb_snippets_for_prompt = json.dumps([format_kb_entry_for_prompt(e) for e in retrieved_kb_entries], indent=2)
            
            full_prompt = BLUE_PROMPT_TEMPLATE.format(
                EPISODE_JSON=episode_json_for_prompt,
                KB_SNIPPETS=kb_snippets_for_prompt
            )
            
            # Call LLM
            llm_response = call_blue_llm(full_prompt)

            # Check JSON validity status
            if llm_response.get("recommended_actions") not in (["LLM_ERROR_OR_INVALID_JSON_GEMMA2"], ["LLM_ERROR_OR_INVALID_JSON_PHI3_MINI"], ["LLM_RESPONSE_VALIDATION_ERROR"]):
                json_valid_count += 1
            else:
                json_error_count += 1
            
            # Evaluation (against blue_label.expected_crs_rules)
            expected_rules = episode["blue_label"].get("expected_crs_rules", [])
            
            is_rule_hit = False
            if expected_rules:
                rule_hit_summary["total"] += 1
                for rec_rule in llm_response.get("recommended_rules", []):
                    if rec_rule.get("rule_id") in expected_rules:
                        is_rule_hit = True
                        break
            if is_rule_hit:
                rule_hit_summary["hit"] += 1

            eval_results.append({
                "original_episode_id": i,
                "prompt": full_prompt,
                "llm_response": llm_response,
                "expected_label": episode["blue_label"],
                "retrieved_kb": retrieved_kb_entries,
                "is_rule_hit": is_rule_hit,
                "blue_backend": current_backend.value # Add backend info to results
            })

        # 3. Save raw eval results
        raw_output_file = f"data/blue/{args.output_prefix}_{current_backend.value}_raw.jsonl"
        with open(raw_output_file, 'w', encoding='utf-8') as f_out:
            for res in eval_results:
                f_out.write(json.dumps(res) + "\n")
        print(f"\nRaw evaluation results saved to {raw_output_file}")

        # 4. Generate summary
        summary_output_file = f"data/blue/{args.output_prefix}_{current_backend.value}_summary.txt"
        summary_lines = []
        summary_lines.append(f"--- BLUE Phase 2 Evaluation Summary for {current_backend.value.upper()} ---")
        summary_lines.append(f"Total golden set cases: {total_golden_cases}")
        summary_lines.append(f"JSON Valid Responses: {json_valid_count}/{total_golden_cases} ({json_valid_count/total_golden_cases:.2%})")
        
        if rule_hit_summary["total"] > 0:
            hit_rate = (rule_hit_summary["hit"] / rule_hit_summary["total"]) * 100
            summary_lines.append(f"Rule Hit Rate (cases with expected rules): {rule_hit_summary['hit']}/{rule_hit_summary['total']} ({hit_rate:.2f}%)")
        else:
            summary_lines.append("No golden set cases with 'expected_crs_rules' for rule hit evaluation.")

        print("\n" + "\n".join(summary_lines))
        with open(summary_output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("\n".join(summary_lines) + "\n")
        print(f"Evaluation summary saved to {summary_output_file}")
        
        log_message(cmd_str, "OK", raw_output_file)

if __name__ == "__main__":
    main()