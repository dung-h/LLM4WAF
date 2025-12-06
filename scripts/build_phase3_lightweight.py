import json
import random
import os
from tqdm import tqdm
from collections import defaultdict

# Config
INPUT_FILE = "data/processed/red_phase1_enriched_v2.jsonl"
OUTPUT_FILE = "data/processed/red_phase3_lightweight.jsonl"
NUM_SAMPLES = 5000 # Reduced sample size for quick testing/debugging
HELPFUL_RATIO = 0.6

def load_and_group_data(filepath):
    print(f"Loading data from {filepath}...")
    data_by_tech = defaultdict(list)
    all_data = []
    
    if not os.path.exists(filepath):
        print(f"Error: Input file {filepath} not found.")
        return [], {{}}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                technique = item.get('technique', 'Unknown')
                attack_type = item.get('attack_type', 'Unknown')
                
                payload = None
                if 'messages' in item and len(item['messages']) > 1:
                    for msg in item['messages']:
                        if msg['role'] == 'assistant':
                            payload = msg['content']
                            break
                
                if payload is None and 'payload' in item:
                    payload = item['payload']
                
                if payload is None:
                    continue

                entry = {
                    "payload": payload,
                    "technique": technique,
                    "attack_type": attack_type
                }
                
                data_by_tech[technique].append(entry)
                all_data.append(entry)
            except:
                continue
    return all_data, data_by_tech

def build_lightweight_prompt(target_entry, all_data, data_by_tech, is_helpful):
    # 1. Select Payloads
    blocked_list = []
    passed_list = []
    
    same_tech_candidates = data_by_tech.get(target_entry['technique'], [])
    
    # Helpful Scenario: Add a PASSED payload of same technique
    if is_helpful and len(same_tech_candidates) >= 2:
        # Pick 1 passed example (different from target)
        candidates = [p for p in same_tech_candidates if p['payload'] != target_entry['payload']]
        if candidates:
            passed_list.append(random.choice(candidates)['payload'])
            
    # Fill BLOCKED list with random distractions
    num_blocked = random.randint(2, 3)
    for _ in range(num_blocked):
        p = random.choice(all_data)
        # Simplified logic: just assume random ones are blocked for this simulation
        if p['payload'] not in passed_list and p['payload'] != target_entry['payload']:
             blocked_list.append(p['payload'])

    # 2. Construct Compact Prompt
    blocked_str = ", ".join([f"`{p}`" for p in blocked_list])
    passed_str = ", ".join([f"`{p}`" for p in passed_list]) if passed_list else "None"
    
    prompt = f"""Target: {target_entry['attack_type']} on ModSecurity PL1.
Technique: {target_entry['technique']}

[Observations]
- BLOCKED: {blocked_list}
- PASSED: {passed_list}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload."""
    
    return prompt

def main():
    all_data, data_by_tech = load_and_group_data(INPUT_FILE)
    output_data = []
    
    print(f"Generating {NUM_SAMPLES} lightweight samples...")
    for _ in tqdm(range(NUM_SAMPLES)):
        target_entry = random.choice(all_data)
        is_helpful = random.random() < HELPFUL_RATIO
        
        instruction = build_lightweight_prompt(target_entry, all_data, data_by_tech, is_helpful)
        
        sample = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": target_entry['payload']}
            ]
        }
        output_data.append(sample)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
