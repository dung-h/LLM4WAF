
import json
import os
import sys
from typing import List, Dict, Any
from collections import Counter

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from blue.rag_index import CRSKnowledgeBase
from blue.rag_retriever import retrieve_for_episode
from blue.prompts import BLUE_PROMPT_TEMPLATE
from blue.llm_client import call_blue_llm

# --- Configuration ---
GOLDEN_SET_FILE = "data/blue/blue_phase1_golden.jsonl"
KB_FILE = "data/blue/blue_phase1_crs_kb.jsonl"
EVAL_RAW_OUTPUT_FILE = "data/blue/blue_phase2_eval_raw.jsonl"
EVAL_SUMMARY_OUTPUT_FILE = "data/blue/blue_phase2_eval_summary.txt"

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
    # 1. Load data
    golden_set = []
    with open(GOLDEN_SET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            golden_set.append(json.loads(line.strip()))
    print(f"Loaded {len(golden_set)} golden set episodes.")

    kb = CRSKnowledgeBase(kb_path=KB_FILE)
    print(f"Loaded {len(kb.entries)} KB entries.")

    eval_results = []
    rule_hit_counts = Counter()
    total_golden_cases = 0
    rule_hit_summary = {"hit": 0, "total": 0}

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
        
        # Call stub LLM
        llm_response = call_blue_llm(full_prompt)
        
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
            "is_rule_hit": is_rule_hit
        })

    # 3. Save raw eval results
    with open(EVAL_RAW_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for res in eval_results:
            f_out.write(json.dumps(res) + "\n")
    print(f"\nRaw evaluation results saved to {EVAL_RAW_OUTPUT_FILE}")

    # 4. Generate summary
    summary = []
    summary.append(f"--- BLUE Phase 2 Evaluation Summary ---")
    summary.append(f"Total golden set cases: {total_golden_cases}")
    
    if rule_hit_summary["total"] > 0:
        hit_rate = (rule_hit_summary["hit"] / rule_hit_summary["total"]) * 100
        summary.append(f"Rule Hit Rate (cases with expected rules): {rule_hit_summary['hit']}/{rule_hit_summary['total']} ({hit_rate:.2f}%)")
    else:
        summary.append("No golden set cases with 'expected_crs_rules' for rule hit evaluation.")

    print("\n" + "\n".join(summary))
    with open(EVAL_SUMMARY_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(summary) + "\n")
    print(f"Evaluation summary saved to {EVAL_SUMMARY_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
