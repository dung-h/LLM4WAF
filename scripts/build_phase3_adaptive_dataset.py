import json
import random
import os
from tqdm import tqdm
from collections import defaultdict

# Config
INPUT_FILE = "data/processed/red_phase1_enriched_v2.jsonl"
OUTPUT_FILE = "data/processed/red_phase3_adaptive_sft.jsonl"
NUM_SAMPLES = 10000
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
                
                # Extract payload from various formats
                payload = None
                if 'messages' in item and len(item['messages']) > 1:
                    # Standard chat format: user, assistant
                    # Assuming assistant turn is the payload
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
            except Exception as e:
                continue
                
    print(f"Loaded {len(all_data)} samples across {len(data_by_tech)} techniques.")
    return all_data, data_by_tech

def generate_probe_history(target_entry, all_data, data_by_tech, is_helpful):
    history = []
    num_probes = random.randint(3, 5)
    
    same_tech_candidates = data_by_tech.get(target_entry['technique'], [])
    
    if is_helpful and len(same_tech_candidates) < 2: 
        is_helpful = False 
        
    if is_helpful:
        num_pass = random.randint(1, 2)
        passed_probes = random.sample(same_tech_candidates, min(num_pass, len(same_tech_candidates)))
        
        for p in passed_probes:
            if p['payload'] != target_entry['payload']:
                history.append({"payload": p['payload'], "technique": p['technique'], "result": "PASSED"})
    
    attempts = 0
    while len(history) < num_probes and attempts < 20:
        attempts += 1
        p = random.choice(all_data)
        
        if any(h['payload'] == p['payload'] for h in history):
            continue
            
        history.append({"payload": p['payload'], "technique": p['technique'], "result": "BLOCKED"})
            
    random.shuffle(history)
    return history

def build_prompt(target_entry, history):
    waf_type = "ModSecurity + OWASP CRS 3.3 (PL1)" 
    
    history_str = ""
    for i, h in enumerate(history):
        history_str += f"{i+1}. Payload: `{h['payload']}` (Technique: {h['technique']}) -> RESULT: {h['result']}\n"
    
    prompt = f"""You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {waf_type}.
- Attack type: {target_entry['attack_type']}.
- Injection point: query parameter.

[Probing History - Previous Attempts]
{history_str}

Your Task:
Analyze the Probing History above.
- If a specific technique or pattern PASSED, analyze why and generate a similar but new payload.
- If all attempts were BLOCKED, try a completely DIFFERENT technique from the ones listed.
- Generate a NEW payload targeting: {target_entry['technique']}.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations.
- Do NOT wrap in code fences."""
    return prompt

def main():
    all_data, data_by_tech = load_and_group_data(INPUT_FILE)
    
    if len(all_data) == 0:
        print("Error: No valid data found. Check input file.")
        return

    output_data = []
    
    print(f"Generating {NUM_SAMPLES} adaptive samples (Helpful: {HELPFUL_RATIO*100}%)...")
    for _ in tqdm(range(NUM_SAMPLES)):
        target_entry = random.choice(all_data)
        is_helpful = random.random() < HELPFUL_RATIO
        history = generate_probe_history(target_entry, all_data, data_by_tech, is_helpful)
        instruction = build_prompt(target_entry, history)
        
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
            
    print(f"Saved {len(output_data)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()