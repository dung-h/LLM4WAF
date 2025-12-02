
import json
import os
from collections import Counter, defaultdict

SUGGESTIONS_FILE = "data/blue/blue_phase3_suggestions.jsonl"
REPORT_OUTPUT_FILE = "data/blue/blue_phase3_report.txt"

def main():
    if not os.path.exists(SUGGESTIONS_FILE):
        print(f"Error: Suggestions file not found at {SUGGESTIONS_FILE}. Please run runner_phase3_suggest.py first.")
        return

    suggestions = []
    with open(SUGGESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            suggestions.append(json.loads(line.strip()))
    
    total_analyzed_episodes = len(suggestions)
    
    # Statistics
    false_negative_count = 0
    vuln_effect_counts = Counter()
    recommended_rules_by_engine = defaultdict(Counter) # {engine: {rule_id: count}}
    top_actions = Counter() # Most frequent recommended actions

    for s in suggestions:
        blue_output = s["blue_recommendation"]
        
        if blue_output.get("is_false_negative"):
            false_negative_count += 1
        
        vuln_effect_counts[blue_output.get("vuln_effect", "UNKNOWN")] += 1
        
        for rule in blue_output.get("recommended_rules", []):
            engine = rule.get("engine", "UNKNOWN")
            rule_id = rule.get("rule_id", "UNKNOWN")
            recommended_rules_by_engine[engine][rule_id] += 1
            
        for action in blue_output.get("recommended_actions", []):
            top_actions[action] += 1

    # Generate Report
    report_content = []
    report_content.append("BLUE Phase 3 – WAF Tuning Summary")
    report_content.append("=================================")
    report_content.append(f"\nTotal analyzed episodes: {total_analyzed_episodes}")
    report_content.append(f"Total false_negative flagged: {false_negative_count}")
    
    report_content.append("\nVuln Effect Distribution:")
    for effect, count in vuln_effect_counts.most_common():
        report_content.append(f"- {effect}: {count} episodes")

    report_content.append("\nPer-engine recommended rules:")
    if not recommended_rules_by_engine:
        report_content.append("- No specific rules recommended by current LLM stub.")
    else:
        for engine, rules_counter in recommended_rules_by_engine.items():
            report_content.append(f"- {engine}:")
            for rule_id, count in rules_counter.most_common():
                report_content.append(f"    - {rule_id}: {count} episodes")
    
    report_content.append("\nTop Recommended Actions:")
    if not top_actions:
        report_content.append("- No specific actions recommended by current LLM stub.")
    else:
        for action, count in top_actions.most_common(5): # Top 5 actions
            report_content.append(f"- {action} ({count} episodes)")

    # Example Notes section
    report_content.append("\nNotes:")
    report_content.append("- The current LLM client is a stub, so recommendations are based on simple keyword matching.")
    report_content.append("- For accurate results, connect 'blue/llm_client.py' to a real LLM (e.g., Gemma 2B running locally).")
    
    # Save report
    with open(REPORT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))

    print(f"\nReport saved to {REPORT_OUTPUT_FILE}")
    print("\n" + "\n".join(report_content))


if __name__ == "__main__":
    main()
