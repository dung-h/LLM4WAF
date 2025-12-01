import json
import random
import os
from collections import Counter

# --- Configuration ---
PASSED_SAMPLES_FILE = "data/processed/red_v40_passed_waf_only.jsonl"
ALL_SAMPLES_FILE = "data/processed/red_v40_balanced_final_v13.jsonl" # This file should contain 'result' field
CLASSIC_PAYLOADS_FILE = "data/resources/classic_payloads.json"
OUTPUT_FILE = "data/processed/red_v40_phase2_reasoning.jsonl"
TARGET_SAMPLES_COUNT = 5000

# WAF context
WAF_TYPE = "ModSecurity + OWASP CRS 3.3 (PL1)"
INJECTION_POINT = "query parameter"

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}. Returning empty list.")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line.strip()}")
                continue
    return data

def build_payload_history_list(payload_history):
    history_str = ""
    for i, item in enumerate(payload_history):
        history_str += f"{i+1}) {item['payload']}\n"
    return history_str.strip()

def generate_phase2_prompt(waf_type, attack_type, injection_point, payload_history_list, target_technique):
    attack_type_str = "SQL injection" if attack_type == "SQLI" else ("XSS" if attack_type == "XSS" else attack_type)
    
    return f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}.
- Attack type: {attack_type_str}.
- Injection point: {injection_point}.

Previously tried payloads against this target (all blocked by the WAF):
{payload_history_list}

Your task:
Generate a NEW {attack_type_str} payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {target_technique}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."""

def main():
    print(f"Loading passed samples from {PASSED_SAMPLES_FILE}...")
    passed_samples = load_jsonl(PASSED_SAMPLES_FILE)
    
    print(f"Loading all samples from {ALL_SAMPLES_FILE} to identify blocked payloads...")
    all_samples = load_jsonl(ALL_SAMPLES_FILE)

    if not passed_samples or not all_samples:
        print("Required input files are missing or empty. Aborting.")
        return

    # Filter for blocked samples
    blocked_samples = []
    for sample in all_samples:
        # Check for 'status' (from waf_test_results.jsonl) or 'result' (from original v13)
        if sample.get("status") == "blocked" or sample.get("result") == "blocked":
            blocked_samples.append(sample)
    
    print(f"Found {len(blocked_samples)} blocked samples.")

    # Group blocked samples by attack_type
    blocked_by_attack_type = {
        "SQLI": [],
        "XSS": [],
        "OS_INJECTION": [] # Ensure all relevant types are covered
    }
    for sample in blocked_samples:
        at = sample.get("attack_type")
        if at in blocked_by_attack_type:
            blocked_by_attack_type[at].append(sample)
    
    print(f"Blocked samples grouped by type: { {k: len(v) for k, v in blocked_by_attack_type.items()} }")

    # Load classic payloads if available
    classic_payloads = {}
    if os.path.exists(CLASSIC_PAYLOADS_FILE):
        print(f"Loading classic payloads from {CLASSIC_PAYLOADS_FILE}...")
        with open(CLASSIC_PAYLOADS_FILE, 'r', encoding='utf-8') as f:
            classic_payloads = json.load(f)
        print(f"Loaded classic payloads: { {k: len(v) for k, v in classic_payloads.items()} }")
    else:
        print(f"No classic payloads file found at {CLASSIC_PAYLOADS_FILE}. Will not use.")

    # Select passed samples to build Phase 2 dataset
    # Filter passed_samples to ensure they have an assistant payload or 'payload' field directly
    filtered_passed_samples = []
    for sample in passed_samples:
        payload_content = None
        if "messages" in sample and isinstance(sample["messages"], list):
            for msg in sample["messages"]:
                if msg.get("role") == "assistant":
                    payload_content = msg.get("content", "")
                    break
        elif "payload" in sample: # Assuming this is from red_v40_passed_waf_only.jsonl directly
             payload_content = sample["payload"]

        if payload_content: # Only add if a valid payload is found
            filtered_passed_samples.append(sample)
    
    if len(filtered_passed_samples) > TARGET_SAMPLES_COUNT:
        selected_passed_samples = random.sample(filtered_passed_samples, TARGET_SAMPLES_COUNT)
    else:
        selected_passed_samples = filtered_passed_samples
    
    print(f"Selected {len(selected_passed_samples)} passed samples for Phase 2 dataset generation.")

    # Build Phase 2 dataset
    phase2_dataset = []
    for i, p_sample in enumerate(selected_passed_samples):
        attack_type_p = p_sample.get("attack_type", "UNKNOWN")
        technique_p = p_sample.get("technique", "UNKNOWN_TECHNIQUE")
        payload_passed = None
        
        # Extract payload_passed robustly
        if "messages" in p_sample and isinstance(p_sample["messages"], list):
             for msg in p_sample["messages"]:
                if msg.get("role") == "assistant":
                    payload_passed = msg.get("content")
                    break
        elif "payload" in p_sample: # For samples from red_v40_passed_waf_only.jsonl
            payload_passed = p_sample["payload"]
        
        if not payload_passed:
            print(f"Warning: Passed payload not found for sample {i}. Skipping.")
            continue

        # Create payload_history
        current_payload_history = []
        possible_blocked = blocked_by_attack_type.get(attack_type_p, [])
        
        num_blocked_to_pick = random.randint(2, 9)
        
        # Prioritize picking from actual blocked samples
        picked_from_blocked = random.sample(possible_blocked, min(num_blocked_to_pick, len(possible_blocked)))
        for blocked_s in picked_from_blocked:
            blocked_payload_content = None
            if "messages" in blocked_s and isinstance(blocked_s["messages"], list):
                for msg in blocked_s["messages"]:
                    if msg.get("role") == "assistant":
                        blocked_payload_content = msg.get("content")
                        break
            elif "payload" in blocked_s: # For waf_test_results.jsonl or similar structure
                blocked_payload_content = blocked_s["payload"]
            
            if blocked_payload_content:
                current_payload_history.append({"payload": blocked_payload_content, "blocked": True})
        
        # Supplement with classic payloads if needed and available
        if len(current_payload_history) < num_blocked_to_pick and attack_type_p in classic_payloads:
            num_to_add = num_blocked_to_pick - len(current_payload_history)
            for classic_p in random.sample(classic_payloads[attack_type_p], min(num_to_add, len(classic_payloads[attack_type_p]))):
                current_payload_history.append({"payload": classic_p, "blocked": True})

        if not current_payload_history: # Fallback: Ensure history is never empty
            fallback_payload = "generic_blocked_payload_placeholder"
            if attack_type_p in classic_payloads and len(classic_payloads[attack_type_p]) > 0:
                fallback_payload = classic_payloads[attack_type_p][0]
            current_payload_history.append({"payload": fallback_payload, "blocked": True})


        # Build the actual Phase 2 JSON object
        phase2_entry = {
            "waf_type": WAF_TYPE.lower().replace(" ", "_").replace("+", ""),
            "attack_type": attack_type_p,
            "injection_point": INJECTION_POINT,
            "payload_history": current_payload_history,
            "target_technique": technique_p,
            "messages": [
                {
                    "role": "user",
                    "content": generate_phase2_prompt(
                        WAF_TYPE,
                        attack_type_p,
                        INJECTION_POINT,
                        build_payload_history_list(current_payload_history),
                        technique_p
                    )
                },
                {
                    "role": "assistant",
                    "content": payload_passed
                }
            ],
            "expected_response": {
                "payload": payload_passed
            }
        }
        phase2_dataset.append(phase2_entry)
        
        if len(phase2_dataset) >= TARGET_SAMPLES_COUNT:
            break

    # Save output
    print(f"\nSaving {len(phase2_dataset)} Phase 2 samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in phase2_dataset:
            f.write(json.dumps(entry) + '\n')
    print("Done.")

    # --- Report ---
    print("\n--- Phase 2 Dataset Report ---")
    print(f"Total samples generated: {len(phase2_dataset)}")
    
    attack_type_counts = Counter(s["attack_type"] for s in phase2_dataset)
    print("Samples by Attack Type:")
    for atype, count in attack_type_counts.most_common():
        print(f"- {atype}: {count}")

    if phase2_dataset:
        print("\n--- Example Phase 2 Sample (first 1) ---")
        print(json.dumps(phase2_dataset[0], indent=2))

if __name__ == "__main__":
    main()