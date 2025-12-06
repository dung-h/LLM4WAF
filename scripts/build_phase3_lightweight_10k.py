#!/usr/bin/env python3
"""
Build Phase 3 Lightweight 10k Dataset
Reduced from 20k to 10k samples while maintaining technique diversity
"""
import json
import random
import os
from collections import defaultdict
from tqdm import tqdm

# Configuration
INPUT_FILE = "data/processed/red_phase1_enriched_v2.jsonl"
OUTPUT_FILE = "data/processed/red_phase3_lightweight_10k.jsonl"
NUM_SAMPLES = 10000
HELPFUL_RATIO = 0.6  # 60% with PASSED examples
ADD_SYSTEM_PROMPT = True

def load_and_group_data(filepath):
    print(f"Loading data from {filepath}...")
    data_by_tech = defaultdict(list)
    data_by_attack = defaultdict(list)
    all_data = []
    
    if not os.path.exists(filepath):
        print(f"Error: Input file {filepath} not found.")
        return [], {}, {}

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
                data_by_attack[attack_type].append(entry)
                all_data.append(entry)
            except:
                continue
    return all_data, data_by_tech, data_by_attack

def sanitize_payload(payload):
    """Remove problematic Unicode characters that cause encoding issues."""
    if not isinstance(payload, str):
        return str(payload)
    
    try:
        payload.encode('utf-8', errors='ignore').decode('utf-8')
        return payload
    except:
        return payload.encode('ascii', errors='ignore').decode('ascii')

def build_lightweight_prompt(target_entry, all_data, data_by_tech, data_by_attack, is_helpful, add_system=True):
    """Build a lightweight adaptive prompt with attack type consistency."""
    blocked_list = []
    passed_list = []
    
    same_tech_candidates = data_by_tech.get(target_entry['technique'], [])
    same_tech_candidates = [p for p in same_tech_candidates 
                           if p['attack_type'] == target_entry['attack_type']]
    
    if is_helpful and len(same_tech_candidates) >= 2:
        candidates = [p for p in same_tech_candidates if p['payload'] != target_entry['payload']]
        if candidates:
            num_passed = min(2, len(candidates))
            passed_samples = random.sample(candidates, num_passed)
            passed_list = [sanitize_payload(p['payload']) for p in passed_samples]
    
    num_blocked = random.randint(2, 4)
    same_attack_candidates = data_by_attack.get(target_entry['attack_type'], [])
    different_tech_candidates = [p for p in same_attack_candidates
                                  if p['technique'] != target_entry['technique'] 
                                  and p['payload'] not in passed_list 
                                  and p['payload'] != target_entry['payload']]
    
    if len(different_tech_candidates) >= num_blocked:
        blocked_samples = random.sample(different_tech_candidates, num_blocked)
        blocked_list = [sanitize_payload(p['payload']) for p in blocked_samples]
    else:
        fallback_candidates = [p for p in same_attack_candidates
                              if p['payload'] not in passed_list 
                              and p['payload'] != target_entry['payload']]
        if fallback_candidates:
            num_to_pick = min(num_blocked, len(fallback_candidates))
            blocked_list = [sanitize_payload(p['payload']) for p in random.sample(fallback_candidates, num_to_pick)]

    system_prefix = ""
    if add_system:
        system_prefix = "Generate WAF-evasion payloads.\n\n"
    
    passed_str = "[]" if not passed_list else json.dumps(passed_list, ensure_ascii=False)
    blocked_str = json.dumps(blocked_list, ensure_ascii=False)
    
    prompt = f"""{system_prefix}Target: {target_entry['attack_type']} on ModSecurity PL1.
Technique: {target_entry['technique']}

[Observations]
- BLOCKED: {blocked_str}
- PASSED: {passed_str}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload."""
    
    return prompt

def main():
    print("="*60)
    print("Building Phase 3 Lightweight 10k Dataset")
    print("="*60)
    
    all_data, data_by_tech, data_by_attack = load_and_group_data(INPUT_FILE)
    
    if not all_data:
        print("Error: No data loaded. Exiting.")
        return
    
    print(f"\nDataset Statistics:")
    print(f"  Total payloads: {len(all_data):,}")
    print(f"  Unique techniques: {len(data_by_tech):,}")
    print(f"  Attack types: {len(data_by_attack):,}")
    for atype, payloads in data_by_attack.items():
        print(f"    - {atype}: {len(payloads):,} payloads")
    print(f"  Target samples: {NUM_SAMPLES:,}")
    print(f"  PASSED ratio: {HELPFUL_RATIO*100:.0f}%")
    print(f"  Add system prompt: {ADD_SYSTEM_PROMPT}")
    
    output_data = []
    
    print(f"\nGenerating {NUM_SAMPLES:,} lightweight adaptive samples...")
    for _ in tqdm(range(NUM_SAMPLES)):
        target_entry = random.choice(all_data)
        is_helpful = random.random() < HELPFUL_RATIO
        
        instruction = build_lightweight_prompt(
            target_entry, 
            all_data, 
            data_by_tech,
            data_by_attack, 
            is_helpful,
            add_system=ADD_SYSTEM_PROMPT
        )
        
        target_payload = sanitize_payload(target_entry['payload'])
        
        sample = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": target_payload}
            ]
        }
        output_data.append(sample)
    
    # Save output
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Statistics
    helpful_count = sum(1 for item in output_data if '- PASSED: [' in item['messages'][0]['content'] and '- PASSED: []' not in item['messages'][0]['content'])
    
    print(f"\n{'='*60}")
    print(f"✅ DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total samples: {len(output_data):,}")
    print(f"Samples with PASSED: {helpful_count:,} ({helpful_count/len(output_data)*100:.1f}%)")
    print(f"Samples without PASSED: {len(output_data)-helpful_count:,}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
