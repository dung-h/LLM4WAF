#!/usr/bin/env python3
"""Check dataset quality"""

import json

print("=" * 70)
print("Phase 1 Dataset Quality Check")
print("=" * 70)

# Check Phase 1
with open('data/processed/phase1_passed_only_39k.jsonl', 'r') as f:
    phase1_samples = [json.loads(line) for line in f.readlines()[:5]]

print(f"\nTotal samples: {sum(1 for _ in open('data/processed/phase1_passed_only_39k.jsonl'))}")
print("\nFirst 3 samples:\n")

for i, sample in enumerate(phase1_samples[:3], 1):
    print(f"Sample {i}:")
    print(f"  Attack Type: {sample['attack_type']}")
    print(f"  Technique: {sample.get('technique', 'N/A')}")
    print(f"  User Prompt: {sample['messages'][0]['content']}")
    print(f"  Payload: {sample['messages'][1]['content'][:150]}")
    print()

# Check attack type distribution
print("\n" + "=" * 70)
print("Attack Type Distribution (first 1000 samples):")
print("=" * 70)

from collections import Counter
with open('data/processed/phase1_passed_only_39k.jsonl', 'r') as f:
    attack_types = [json.loads(line)['attack_type'] for line in f.readlines()[:1000]]
    
for attack, count in Counter(attack_types).most_common():
    print(f"  {attack}: {count}")

print("\n" + "=" * 70)
print("Phase 2 Dataset (Observations-based Adaptive Learning)")
print("=" * 70)

# Check Phase 2
with open('data/processed/phase2_observations_20k.jsonl', 'r') as f:
    phase2_samples = [json.loads(line) for line in f.readlines()[:2]]

print(f"\nTotal samples: {sum(1 for _ in open('data/processed/phase2_observations_20k.jsonl'))}")
print("\nFirst sample:\n")

sample = phase2_samples[0]
print(f"User Prompt:\n{sample['messages'][0]['content'][:400]}")
print(f"\nPayload: {sample['messages'][1]['content'][:100]}")
