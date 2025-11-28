#!/usr/bin/env python3
"""
Create 8K subsample from full v29 dataset for local Qwen2.5-3B training
Balanced sampling across attack types and sources
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def create_8k_subsample():
    """Create balanced 8K subsample from 11.6K dataset"""
    
    input_file = Path("data/processed/train_full_qwen_cleaned.jsonl")
    output_file = Path("data/splits/sft_experiment/train_8k_qwen_local.jsonl")
    
    print("📊 Creating 8K subsample from v29 dataset...")
    
    # Load all samples
    samples = []
    attack_type_counts = defaultdict(int)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            samples.append(data)
            
            # Count attack types from instruction
            instruction = data.get('instruction', '').lower()
            if 'xss' in instruction:
                attack_type_counts['xss'] += 1
            elif 'sql' in instruction:
                attack_type_counts['sqli'] += 1
            else:
                attack_type_counts['other'] += 1
    
    print(f"📈 Total samples: {len(samples):,}")
    print(f"📊 Attack type distribution:")
    for attack_type, count in attack_type_counts.items():
        print(f"   {attack_type}: {count:,} ({count/len(samples)*100:.1f}%)")
    
    # Stratified sampling to maintain balance
    random.seed(42)  # Reproducible
    target_size = 8000
    
    # Group samples by attack type
    samples_by_type = defaultdict(list)
    for sample in samples:
        instruction = sample.get('instruction', '').lower()
        if 'xss' in instruction:
            samples_by_type['xss'].append(sample)
        elif 'sql' in instruction:
            samples_by_type['sqli'].append(sample)
        else:
            samples_by_type['other'].append(sample)
    
    # Calculate proportional sampling
    selected_samples = []
    for attack_type, type_samples in samples_by_type.items():
        proportion = len(type_samples) / len(samples)
        target_count = int(target_size * proportion)
        
        # Shuffle and sample
        random.shuffle(type_samples)
        selected = type_samples[:target_count]
        selected_samples.extend(selected)
        
        print(f"🎯 {attack_type}: {len(selected):,}/{len(type_samples):,} samples")
    
    # If we're short, add random samples
    if len(selected_samples) < target_size:
        remaining_samples = [s for s in samples if s not in selected_samples]
        random.shuffle(remaining_samples)
        needed = target_size - len(selected_samples)
        selected_samples.extend(remaining_samples[:needed])
    
    # Final shuffle
    random.shuffle(selected_samples)
    selected_samples = selected_samples[:target_size]
    
    # Save subsample
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Created 8K subsample: {output_file}")
    print(f"📊 Final size: {len(selected_samples):,} samples")
    
    return output_file

if __name__ == "__main__":
    create_8k_subsample()