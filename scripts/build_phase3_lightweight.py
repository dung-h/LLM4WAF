import json
import random
import os
from tqdm import tqdm
from collections import defaultdict

# Config
INPUT_FILE = "data/processed/red_phase1_enriched_v2.jsonl"
OUTPUT_FILE = "data/processed/red_phase3_lightweight.jsonl"
NUM_SAMPLES = 20000  # Enriched dataset for better coverage
HELPFUL_RATIO = 0.6  # 60% samples have PASSED examples
ADD_SYSTEM_PROMPT = True  # Add lightweight system prompt for better context

def load_and_group_data(filepath):
    print(f"Loading data from {filepath}...")
    data_by_tech = defaultdict(list)
    data_by_attack = defaultdict(list)  # Group by attack type
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
    
    # Replace common problematic characters
    try:
        # Try to encode/decode to catch issues
        payload.encode('utf-8', errors='ignore').decode('utf-8')
        return payload
    except:
        # If fails, use ASCII-safe version
        return payload.encode('ascii', errors='ignore').decode('ascii')

def build_lightweight_prompt(target_entry, all_data, data_by_tech, data_by_attack, is_helpful, add_system=True):
    """
    Build a lightweight adaptive prompt with optional system context.
    CRITICAL: Ensures BLOCKED and PASSED payloads are from SAME attack type as target.
    
    Args:
        target_entry: Target payload entry to generate prompt for
        all_data: All available payloads
        data_by_tech: Payloads grouped by technique
        data_by_attack: Payloads grouped by attack type
        is_helpful: Whether to include PASSED examples (60% true)
        add_system: Whether to add system prompt prefix
    """
    # 1. Select PASSED payloads (same technique AND same attack type)
    blocked_list = []
    passed_list = []
    
    same_tech_candidates = data_by_tech.get(target_entry['technique'], [])
    # CRITICAL: Filter to same attack type
    same_tech_candidates = [p for p in same_tech_candidates 
                           if p['attack_type'] == target_entry['attack_type']]
    
    # Helpful Scenario: Add 1-2 PASSED examples
    if is_helpful and len(same_tech_candidates) >= 2:
        candidates = [p for p in same_tech_candidates if p['payload'] != target_entry['payload']]
        if candidates:
            num_passed = min(2, len(candidates))
            passed_samples = random.sample(candidates, num_passed)
            passed_list = [sanitize_payload(p['payload']) for p in passed_samples]
        else:
            # DEBUG: This should rarely happen
            pass
            
    # 2. Select BLOCKED payloads (different techniques BUT SAME attack type)
    num_blocked = random.randint(2, 4)
    
    # CRITICAL: Only get payloads from SAME attack type
    same_attack_candidates = data_by_attack.get(target_entry['attack_type'], [])
    different_tech_candidates = [p for p in same_attack_candidates
                                  if p['technique'] != target_entry['technique'] 
                                  and p['payload'] not in passed_list 
                                  and p['payload'] != target_entry['payload']]
    
    if len(different_tech_candidates) >= num_blocked:
        blocked_samples = random.sample(different_tech_candidates, num_blocked)
        blocked_list = [sanitize_payload(p['payload']) for p in blocked_samples]
    else:
        # Fallback: any payloads from SAME attack type
        fallback_candidates = [p for p in same_attack_candidates
                              if p['payload'] not in passed_list 
                              and p['payload'] != target_entry['payload']]
        if fallback_candidates:
            num_to_pick = min(num_blocked, len(fallback_candidates))
            blocked_list = [sanitize_payload(p['payload']) for p in random.sample(fallback_candidates, num_to_pick)]

    # 3. Construct Prompt
    system_prefix = ""
    if add_system:
        system_prefix = "Generate WAF-evasion payloads.\n\n"
    
    # Format observations
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
    print("Building Phase 3 Lightweight Adaptive Dataset (Enhanced)")
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
        
        # Sanitize target payload as well
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
    
    # Summary statistics - count samples with non-empty PASSED list
    helpful_count = 0
    for item in output_data:
        content = item['messages'][0]['content']
        # Check if PASSED has actual content (not empty array)
        if '"PASSED": [' in content and '"PASSED": []' not in content:
            helpful_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total samples: {len(output_data):,}")
    print(f"Samples with PASSED examples: {helpful_count:,} ({helpful_count/len(output_data)*100:.1f}%)")
    print(f"Samples without PASSED: {len(output_data)-helpful_count:,} ({(len(output_data)-helpful_count)/len(output_data)*100:.1f}%)")
    print(f"\nNext steps:")
    print(f"1. Review a few samples: head -n 5 {OUTPUT_FILE}")
    print(f"2. Train model: python scripts/train_red_model.py --config configs/red_phase3_lightweight_enhanced.yaml")
    print(f"3. Evaluate: python scripts/evaluate_remote_adapters.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
