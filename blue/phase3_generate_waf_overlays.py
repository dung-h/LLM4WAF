import json
import os
import sys
import argparse
from collections import defaultdict
import datetime

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Mes.log function ---
def log_message(cmd, status, output_file=""):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} CMD=\"{cmd}\" STATUS={status}"
    if output_file:
        log_entry += f" OUTPUT=\"{output_file}\""
    with open("mes.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def generate_modsecurity_rule(rule_id, reason, vuln_effect, severity="WARNING", phase=2, payload_example=""):
    """Generates a ModSecurity SecRule skeleton."""
    # Escape payload example for use in comments and regex if payload is used directly
    escaped_payload_for_comment = payload_example.replace("\"", "\\\"")
    # For actual regex, payload needs proper regex escaping and un-encoding,
    # but for this POC, we'll use it as-is in the regex for demonstration.
    # In real world, one would derive a generic regex from the payload type.
    
    # Try to derive phase from vuln_effect or attack_type
    if "xss" in vuln_effect.lower() or "reflected" in vuln_effect.lower():
        phase = 4 # ModSecurity RESPONSE_BODY phase
    elif "sqli" in vuln_effect.lower() or "injection" in vuln_effect.lower():
        phase = 2 # ModSecurity REQUEST_BODY/ARGS phase
    else:
        phase = 2 # Default to request phase

    # Generate a unique ID for the generated rule to avoid conflicts
    # Use a range outside of CRS (e.g., 800,000s or 990,000s)
    generated_rule_id = f"99{rule_id}" 

    return f"""
# BLUE Phase 3 Suggestion: {vuln_effect}
# Reason: {reason}
# Original Rule ID (from LLM): {rule_id}
# Example Payload: {escaped_payload_for_comment[:100]}...

SecRule REQUEST_FILENAME|ARGS|ARGS_NAMES|REQUEST_BODY "@rx {payload_example}" \
  "id:{generated_rule_id}, \
   phase:{phase}, \
   block, \
   capture, \
   log, \
   msg:'BLUE Suggestion: {vuln_effect} detected. Rule {generated_rule_id} triggered by payload: {escaped_payload_for_comment[:50]}', \
   severity:{severity}" 
"""

def main():
    parser = argparse.ArgumentParser(description="Generate WAF overlay configuration files from BLUE LLM suggestions.")
    parser.add_argument("--input_file", type=str, default="data/blue/blue_phase3_suggestions.jsonl",
                        help="Path to the suggestions JSONL file.")
    parser.add_argument("--modsecurity_output", type=str, default="waf/blue_modsecurity_suggestions.conf",
                        help="Path to save the generated ModSecurity overlay config.")
    parser.add_argument("--coraza_output", type=str, default="waf/blue_coraza_suggestions.yaml",
                        help="Path to save the generated Coraza overlay config.")
    args = parser.parse_args()

    cmd_str = f"python blue/phase3_generate_waf_overlays.py --input_file {args.input_file} --modsecurity_output {args.modsecurity_output} --coraza_output {args.coraza_output}"

    try:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Error: Suggestions file not found at {args.input_file}. Please run runner_phase3_suggest.py first.")

        suggestions = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                suggestions.append(json.loads(line.strip()))

        modsecurity_rules_to_generate = [] 
        coraza_rules_to_generate = []
        
        # Keep track of generated rules to avoid duplicates if LLM suggests same rule multiple times
        generated_rule_ids = set() 

        for s in suggestions:
            blue_output = s["blue_recommendation"]
            original_episode = s["original_episode_summary"]

            if "LLM_ERROR" in str(blue_output.get("recommended_actions", [])):
                continue # Skip suggestions that were due to LLM parsing errors

            for rule in blue_output.get("recommended_rules", []):
                engine = rule.get("engine", "UNKNOWN").lower()
                rule_id = rule.get("rule_id", "UNKNOWN")
                reason = rule.get("reason", "")
                vuln_effect = blue_output.get("vuln_effect", "UNKNOWN")
                payload_example = original_episode.get("payload", "") # Use original payload as example

                # Ensure rule_id is a string
                rule_id_str = str(rule_id)
                generated_id_for_this_rule = f"99{rule_id_str}" # Custom ID to avoid CRS conflicts

                if generated_id_for_this_rule in generated_rule_ids:
                    continue # Skip if this specific rule has already been generated

                generated_rule_ids.add(generated_id_for_this_rule)

                if engine == "modsecurity":
                    modsecurity_rules_to_generate.append(generate_modsecurity_rule(
                        rule_id_str, reason, vuln_effect,
                        payload_example=payload_example
                    ))
                elif engine == "coraza":
                    # For Coraza, we'll just capture the rule ID and payload example for now
                    # Real Coraza rule generation is more complex and depends on format
                    coraza_rules_to_generate.append({
                        "id": generated_id_for_this_rule,
                        "secrule": f"REQUEST_FILENAME|ARGS|ARGS_NAMES|REQUEST_BODY \"@rx {payload_example}\"",
                        "msg": f"BLUE Suggestion: {vuln_effect} detected. Rule {generated_id_for_this_rule} triggered.",
                        "severity": "WARNING",
                        "phase": 2 # Default phase
                    })


        # Generate ModSecurity Overlay
        with open(args.modsecurity_output, 'w', encoding='utf-8') as f_out:
            f_out.write("# ModSecurity WAF Overlay Suggestions from BLUE Phase 3\n")
            f_out.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
            if not modsecurity_rules_to_generate:
                f_out.write("# No specific ModSecurity rules recommended by the LLM (or all JSONs were invalid).\n")
            else:
                for rule_content in modsecurity_rules_to_generate:
                    f_out.write(rule_content)
                    f_out.write("\n")
        print(f"Generated ModSecurity overlay config to {args.modsecurity_output}")

        # Generate Coraza Overlay (YAML format)
        with open(args.coraza_output, 'w', encoding='utf-8') as f_out:
            f_out.write("# Coraza WAF Overlay Suggestions from BLUE Phase 3\n")
            f_out.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
            if not coraza_rules_to_generate:
                f_out.write("# No specific Coraza rules recommended by the LLM (or all JSONs were invalid).\n")
            else:
                f_out.write("rules:\n")
                for rule_data in coraza_rules_to_generate:
                    f_out.write(f"  - id: {rule_data['id']}\n")
                    f_out.write(f"    secrule: \"{rule_data['secrule']}\"\n")
                    f_out.write(f"    msg: \"{rule_data['msg']}\"\n")
                    f_out.write(f"    severity: \"{rule_data['severity']}\"\n")
                    f_out.write(f"    phase: {rule_data['phase']}\n")
                    f_out.write("\n")
        print(f"Generated Coraza overlay config (skeleton) to {args.coraza_output}")
        log_message(cmd_str, "OK", f"{args.modsecurity_output}, {args.coraza_output}")

    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))
    except Exception as e:
        print(f"Error running phase3_generate_waf_overlays.py: {e}", file=sys.stderr)
        log_message(cmd_str, "FAIL", str(e))

if __name__ == "__main__":
    main()