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
from blue.llm_client import call_blue_llm, BlueModelBackend # Import BlueModelBackend

# --- Configuration Defaults ---
EPISODES_FILE = "data/blue/blue_phase1_episodes.jsonl"
KB_FILE = "data/blue/blue_phase1_crs_kb.jsonl"
DEFAULT_SUGGESTIONS_OUTPUT_FILE = "data/blue/blue_phase3_suggestions.jsonl"
MAX_EPISODES_TO_PROCESS = 200 # Limit to avoid excessive processing
TOP_K_KB_SNIPPETS = 3 # Number of KB entries to retrieve

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def format_episode_for_prompt(episode: Dict) -> Dict:
    """Extracts relevant fields from a blue_episode for the prompt, avoiding sensitive/too verbose fields."""
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
        "operator": kb_entry.get("operator"),
        "example_payload": kb_entry.get("example_payload")
    }

def main():
    parser = argparse.ArgumentParser(description="Generate BLUE LLM suggestions for false negative episodes.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_SUGGESTIONS_OUTPUT_FILE,
                        help="Path to save the generated suggestions JSONL file.")
    parser.add_argument("--backend", type=str, choices=[b.value for b in BlueModelBackend], required=True,
                        help="LLM backend to use for generating suggestions (gemma2 or phi3_mini).")
    args = parser.parse_args()

    # Set the BLUE_BACKEND environment variable for llm_client.py
    os.environ["BLUE_BACKEND"] = args.backend
    current_backend = BlueModelBackend(args.backend)

    cmd_str = f"python blue/runner_phase3_suggest.py --backend {args.backend} --output_file {args.output_file}"

    try:
        # 1. Load data
        all_episodes = []
        with open(EPISODES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                all_episodes.append(json.loads(line.strip()))
        print(f"Loaded {len(all_episodes)} blue_episodes from {EPISODES_FILE}.")

        kb = CRSKnowledgeBase(kb_path=KB_FILE)
        print(f"Loaded {len(kb.entries)} KB entries from {KB_FILE}.")

        # 2. Select target episodes (is_false_negative == true)
        target_episodes = [ep for ep in all_episodes if ep["blue_label"]["is_false_negative"]]
        
        if len(target_episodes) > MAX_EPISODES_TO_PROCESS:
            target_episodes = target_episodes[:MAX_EPISODES_TO_PROCESS] # Limit for processing speed
        
        print(f"Selected {len(target_episodes)} false negative episodes for Phase 3 processing using {current_backend.value.upper()}.")

        # 3. Process episodes and get recommendations
        suggestions = []
        
        for i, episode in enumerate(target_episodes):
            print(f"Processing episode {i+1}/{len(target_episodes)} (vuln_effect: {episode['blue_label']['vuln_effect']})...")
            
            # Retrieve KB snippets
            retrieved_kb_entries = retrieve_for_episode(episode, kb, top_k=TOP_K_KB_SNIPPETS)
            
            # Build prompt
            episode_json_for_prompt = json.dumps(format_episode_for_prompt(episode), indent=2)
            kb_snippets_for_prompt = json.dumps([format_kb_entry_for_prompt(e) for e in retrieved_kb_entries], indent=2)
            
            full_prompt = BLUE_PROMPT_TEMPLATE.format(
                EPISODE_JSON=episode_json_for_prompt,
                KB_SNIPPETS=kb_snippets_for_prompt
            )
            
            # Call LLM
            llm_output = call_blue_llm(full_prompt)
            
            suggestions.append({
                "episode_id": f"ep_{i}", # Simple ID
                "original_episode_summary": { # Store a summary of the original episode
                    "attack_type": episode["attack"]["attack_type"],
                    "payload": episode["attack"]["payload"],
                    "waf_engine": episode["waf_env"]["engine"],
                    "app_name": episode["app_context"]["app_name"],
                    "vuln_effect": episode["blue_label"]["vuln_effect"]
                },
                "kb_hits": retrieved_kb_entries, # Store full KB hits for context
                "blue_recommendation": llm_output,
                "full_prompt_sent": full_prompt, # Optional: store full prompt for debugging
                "blue_backend": current_backend.value # Add backend info to results
            })
            print(f"  Generated recommendation for episode {i+1}. Vuln Effect: {llm_output['vuln_effect']}. Backend: {current_backend.value.upper()}")


        # 4. Save suggestions
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for s in suggestions:
                f_out.write(json.dumps(s) + "\n")
        print(f"\nSaved {len(suggestions)} suggestions to {args.output_file}")

        # 5. Report (brief)
        print("\n--- Phase 3 BLUE Suggestions Report ---")
        print(f"Total episodes processed: {len(suggestions)}")
        print(f"Backend used: {current_backend.value.upper()}")
        
        # Example to confirm structure
        if suggestions:
            print("\n--- Example Suggestion (first 1) ---")
            print(json.dumps(suggestions[0], indent=2))
        
        log_message(cmd_str, "OK", args.output_file)

    except Exception as e:
        print(f"Error running runner_phase3_suggest.py for backend {args.backend}: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()