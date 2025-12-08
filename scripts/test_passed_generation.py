#!/usr/bin/env python3
"""Test build_lightweight_prompt directly"""
import sys
sys.path.insert(0, 'scripts')

# Import function and data from build script
from build_phase3_lightweight import load_and_group_data
import random

INPUT_FILE = "data/processed/red_phase1_enriched_v2.jsonl"

# Load data using same function as build script
all_data, data_by_tech, data_by_attack = load_and_group_data(INPUT_FILE)

print(f"Loaded {len(all_data):,} payloads")
print(f"Techniques: {len(data_by_tech)}")
print(f"Attack types: {len(data_by_attack)}")

# Pick random sample
random.seed(42)
target = random.choice(all_data)

print(f"\nTarget sample:")
print(f"  Technique: {target['technique']}")
print(f"  Attack type: {target['attack_type']}")
print(f"  Payload: {target['payload'][:80]}...")

# Test PASSED logic
same_tech_candidates = data_by_tech.get(target['technique'], [])
same_tech_candidates = [p for p in same_tech_candidates 
                       if p['attack_type'] == target['attack_type']]

print(f"\nCandidates with same technique + attack type: {len(same_tech_candidates)}")

is_helpful = True
if is_helpful and len(same_tech_candidates) >= 2:
    candidates = [p for p in same_tech_candidates if p['payload'] != target['payload']]
    print(f"After excluding target: {len(candidates)} candidates")
    
    if candidates:
        num_passed = min(2, len(candidates))
        passed_samples = random.sample(candidates, num_passed)
        print(f"\n✅ SUCCESS: Selected {num_passed} PASSED examples:")
        for p in passed_samples:
            print(f"  - {p['payload'][:60]}...")
    else:
        print("\n❌ FAIL: No candidates after excluding target")
else:
    print(f"\n❌ FAIL: Condition not met")
    print(f"  is_helpful: {is_helpful}")
    print(f"  len(same_tech_candidates): {len(same_tech_candidates)}")
