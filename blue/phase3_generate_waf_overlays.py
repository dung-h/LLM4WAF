
import json
import os
from collections import defaultdict
import datetime

SUGGESTIONS_FILE = "data/blue/blue_phase3_suggestions.jsonl"
MODSECURITY_OVERLAY_FILE = "waf/blue_modsecurity_suggestions.conf"
CORAZA_OVERLAY_FILE = "waf/blue_coraza_suggestions.yaml" # Placeholder, actual format needs to be determined

def generate_modsecurity_rule(rule_id, reason, vuln_effect, severity="WARNING", phase=2, payload_example=""):
    """Generates a ModSecurity SecRule skeleton."""
    # Escape payload example for use in comments
    escaped_payload = payload_example.replace("\"", "\\\"")
    
    return f"""
# BLUE Phase 3 Suggestion: {vuln_effect}
# Reason: {reason}
# Original Rule ID: {rule_id}
# Example Payload: {escaped_payload[:100]}...

SecRule RESPONSE_BODY \"@rx {payload_example}\" \
  \"id:{rule_id}, \
   phase:{phase}, \
   block, \
   log, \
   msg:'BLUE Suggestion: {vuln_effect} detected. Rule {rule_id} triggered.', \
   severity:{severity}\" 
"""

def main():
    if not os.path.exists(SUGGESTIONS_FILE):
        print(f"Error: Suggestions file not found at {SUGGESTIONS_FILE}. Please run runner_phase3_suggest.py first.")
        return

    suggestions = []
    with open(SUGGESTIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            suggestions.append(json.loads(line.strip()))

    modsecurity_rules = defaultdict(list) # {rule_id: [suggestions]}
    coraza_rules = defaultdict(list)

    for s in suggestions:
        blue_output = s["blue_recommendation"]
        original_episode = s["original_episode_summary"]

        for rule in blue_output.get("recommended_rules", []):
            engine = rule.get("engine", "UNKNOWN").lower()
            rule_id = rule.get("rule_id", "UNKNOWN")
            reason = rule.get("reason", "")
            
            # Simple aggregation to avoid too many duplicate rules if stub LLM is repetitive
            # For a real LLM, each suggestion might be unique enough
            unique_key = (engine, rule_id)
            if unique_key not in [r["unique_key"] for r_list in modsecurity_rules.values() for r in r_list] and \
               unique_key not in [r["unique_key"] for r_list in coraza_rules.values() for r in r_list]: # Check across both
                
                rule_info = {
                    "rule_id": rule_id,
                    "reason": reason,
                    "vuln_effect": blue_output.get("vuln_effect", "UNKNOWN"),
                    "severity": blue_output.get("severity", "WARNING"),
                    "payload_example": original_episode.get("payload", ""),
                    "unique_key": unique_key # For de-duplication
                }
                
                if engine == "modsecurity":
                    modsecurity_rules[rule_id].append(rule_info)
                elif engine == "coraza":
                    coraza_rules[rule_id].append(rule_info)
                # Other WAFs if any


    # Generate ModSecurity Overlay
    with open(MODSECURITY_OVERLAY_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("# ModSecurity WAF Overlay Suggestions from BLUE Phase 3\n")
        f_out.write("# Generated: " + datetime.datetime.now().isoformat() + "\n\n")
        for rule_id, suggestions_list in modsecurity_rules.items():
            # Taking the first suggestion for rule generation (since stub LLM might repeat)
            s = suggestions_list[0] 
            f_out.write(generate_modsecurity_rule(
                s["rule_id"], s["reason"], s["vuln_effect"], s["severity"],
                payload_example=s["payload_example"] # Using payload as regex pattern example, not ideal but for skeleton
            ))
            f_out.write("\n")
    print(f"Generated ModSecurity overlay config to {MODSECURITY_OVERLAY_FILE}")

    # Generate Coraza Overlay (Placeholder for now)
    with open(CORAZA_OVERLAY_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("# Coraza WAF Overlay Suggestions from BLUE Phase 3\n")
        f_out.write("# Generated: " + datetime.datetime.now().isoformat() + "\n\n")
        if not coraza_rules:
            f_out.write("# No specific Coraza rules recommended by current LLM stub.\n")
        else:
            for rule_id, suggestions_list in coraza_rules.items():
                s = suggestions_list[0]
                f_out.write(f"# Coraza Rule Suggestion for {s['vuln_effect']} (ID: {s['rule_id']})\n")
                f_out.write(f"# Reason: {s['reason']}\n")
                f_out.write(f"# Payload Example: {s['payload_example']}\n")
                f_out.write(f"""

# Rule Example (placeholder, needs manual review):
# SecRule ARGS|REQUEST_BODY \"@contains {s['payload_example']}\" \"id:{s['rule_id']},phase:request,block,msg:'BLUE Suggestion: {s['vuln_effect']}',severity:{s['severity']}\" 
"""
)
    print(f"Generated Coraza overlay config (skeleton) to {CORAZA_OVERLAY_FILE}")


if __name__ == "__main__":
    main()
