#!/usr/bin/env python3
"""Final verification of dataset quality"""
import json

dataset_file = "data/processed/red_phase3_lightweight.jsonl"

samples = []
with open(dataset_file, 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

# Count samples with PASSED (correct detection)
helpful_count = 0
for sample in samples:
    content = sample['messages'][0]['content']
    # Correct format: "- PASSED: [..."
    if '- PASSED: [' in content and '- PASSED: []' not in content:
        helpful_count += 1

print(f"{'='*60}")
print(f"FINAL DATASET QUALITY REPORT")
print(f"{'='*60}")
print(f"Total samples: {len(samples):,}")
print(f"Samples with PASSED examples: {helpful_count:,} ({helpful_count/len(samples)*100:.1f}%)")
print(f"Samples without PASSED: {len(samples)-helpful_count:,} ({(len(samples)-helpful_count)/len(samples)*100:.1f}%)")
print(f"\nExpected PASSED ratio: 60%")
print(f"Actual PASSED ratio: {helpful_count/len(samples)*100:.1f}%")
print(f"Status: {'✅ GOOD' if 55 <= helpful_count/len(samples)*100 <= 65 else '⚠️ CHECK'}")
print(f"{'='*60}")
