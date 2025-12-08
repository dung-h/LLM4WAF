#!/usr/bin/env python3
"""
Create balanced 10k subset from Phase 1 dataset
Ensures diverse technique representation using stratified sampling
"""

import json
from collections import defaultdict
import random

def main():
    input_file = "data/processed/phase1_passed_only_39k.jsonl"
    output_file = "data/processed/phase1_balanced_10k.jsonl"
    target_samples = 10000
    
    # Group by technique
    technique_groups = defaultdict(list)
    
    print(f"📖 Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            technique = sample.get('technique', 'Unknown')
            technique_groups[technique].append(sample)
    
    total_samples = sum(len(samples) for samples in technique_groups.values())
    num_techniques = len(technique_groups)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Unique techniques: {num_techniques}")
    
    # Stratified sampling: proportional to original distribution
    selected_samples = []
    
    for technique, samples in sorted(technique_groups.items(), key=lambda x: len(x[1]), reverse=True):
        # Calculate proportional sample count
        proportion = len(samples) / total_samples
        technique_target = int(target_samples * proportion)
        
        # Ensure at least 1 sample per technique (for rare ones)
        technique_target = max(1, technique_target)
        
        # Don't exceed available samples
        technique_target = min(technique_target, len(samples))
        
        # Random sample from this technique
        sampled = random.sample(samples, technique_target)
        selected_samples.extend(sampled)
        
        if len(samples) >= 50:  # Only show major techniques
            print(f"  {technique}: {len(samples):,} → {technique_target}")
    
    # If we're under target, randomly sample more from larger groups
    if len(selected_samples) < target_samples:
        remaining = target_samples - len(selected_samples)
        print(f"\n⚖️  Need {remaining} more samples, sampling from large groups...")
        
        # Get samples from groups with >100 samples
        large_groups = [(t, s) for t, s in technique_groups.items() if len(s) > 100]
        extra_pool = []
        for technique, samples in large_groups:
            extra_pool.extend(samples)
        
        # Remove already selected
        selected_set = {json.dumps(s, sort_keys=True) for s in selected_samples}
        extra_pool = [s for s in extra_pool if json.dumps(s, sort_keys=True) not in selected_set]
        
        extra = random.sample(extra_pool, min(remaining, len(extra_pool)))
        selected_samples.extend(extra)
    
    # Shuffle to mix techniques
    random.shuffle(selected_samples)
    
    # Trim to exact target
    selected_samples = selected_samples[:target_samples]
    
    # Write output
    print(f"\n💾 Writing {len(selected_samples):,} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Verify distribution
    technique_count = defaultdict(int)
    for sample in selected_samples:
        technique_count[sample.get('technique', 'Unknown')] += 1
    
    print(f"\n✅ Created balanced subset:")
    print(f"  Total samples: {len(selected_samples):,}")
    print(f"  Unique techniques: {len(technique_count)}")
    print(f"\n  Top 10 techniques:")
    for technique, count in sorted(technique_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {technique}: {count}")

if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
