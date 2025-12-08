#!/usr/bin/env python3
"""
Script to combine Phase 1 and Phase 2 datasets for continual training.
This prevents catastrophic forgetting by mixing both prompt formats.

Usage:
    python scripts/build_phase1_phase2_combined_dataset.py

Output:
    data/processed/red_phase1_phase2_combined.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Configuration
PHASE1_PATH = "data/processed/red_phase1_enriched_v2.jsonl"
PHASE2_PATH = "data/processed/red_v40_phase2_reasoning.jsonl"
OUTPUT_PATH = "data/processed/red_phase1_phase2_combined.jsonl"

# Sampling strategy
PHASE1_SAMPLE_RATIO = 0.4  # 40% from Phase 1 (to remind model of basics)
PHASE2_SAMPLE_RATIO = 1.0  # 100% from Phase 2 (to learn new reasoning)

# Or use absolute numbers
USE_ABSOLUTE_NUMBERS = True
PHASE1_SAMPLES = 17000  # ~40% of 42k
PHASE2_SAMPLES = 5000   # 100% of 5k (all)

RANDOM_SEED = 42


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Save data to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    random.seed(RANDOM_SEED)
    
    print("="*60)
    print("Building Phase1 + Phase2 Combined Dataset")
    print("="*60)
    
    # Load datasets
    print(f"\n[1/4] Loading Phase 1 data from: {PHASE1_PATH}")
    phase1_data = load_jsonl(PHASE1_PATH)
    print(f"  ✓ Loaded {len(phase1_data):,} samples")
    
    print(f"\n[2/4] Loading Phase 2 data from: {PHASE2_PATH}")
    phase2_data = load_jsonl(PHASE2_PATH)
    print(f"  ✓ Loaded {len(phase2_data):,} samples")
    
    # Sample datasets
    print(f"\n[3/4] Sampling datasets...")
    
    if USE_ABSOLUTE_NUMBERS:
        # Use absolute numbers
        phase1_sample_size = min(PHASE1_SAMPLES, len(phase1_data))
        phase2_sample_size = min(PHASE2_SAMPLES, len(phase2_data))
    else:
        # Use ratios
        phase1_sample_size = int(len(phase1_data) * PHASE1_SAMPLE_RATIO)
        phase2_sample_size = int(len(phase2_data) * PHASE2_SAMPLE_RATIO)
    
    phase1_sampled = random.sample(phase1_data, phase1_sample_size)
    phase2_sampled = random.sample(phase2_data, phase2_sample_size) if phase2_sample_size < len(phase2_data) else phase2_data
    
    print(f"  ✓ Phase 1: {len(phase1_sampled):,} samples ({len(phase1_sampled)/len(phase1_data)*100:.1f}%)")
    print(f"  ✓ Phase 2: {len(phase2_sampled):,} samples ({len(phase2_sampled)/len(phase2_data)*100:.1f}%)")
    
    # Combine and shuffle
    combined_data = phase1_sampled + phase2_sampled
    random.shuffle(combined_data)
    
    print(f"\n  Total combined: {len(combined_data):,} samples")
    print(f"  Ratio - Phase1: {len(phase1_sampled)/len(combined_data)*100:.1f}% | Phase2: {len(phase2_sampled)/len(combined_data)*100:.1f}%")
    
    # Save combined dataset
    print(f"\n[4/4] Saving combined dataset to: {OUTPUT_PATH}")
    save_jsonl(combined_data, OUTPUT_PATH)
    print(f"  ✓ Saved {len(combined_data):,} samples")
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATASET READY FOR CONTINUAL TRAINING")
    print("="*60)
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total samples: {len(combined_data):,}")
    print(f"Phase 1 contribution: {len(phase1_sampled):,} ({len(phase1_sampled)/len(combined_data)*100:.1f}%)")
    print(f"Phase 2 contribution: {len(phase2_sampled):,} ({len(phase2_sampled)/len(combined_data)*100:.1f}%)")
    print("\nNext steps:")
    print("1. Create/update training config (e.g., configs/red_phase1_phase2_combined.yaml)")
    print("2. Set train_path: data/processed/red_phase1_phase2_combined.jsonl")
    print("3. Use LOWER learning rate (e.g., 5e-5) to preserve Phase 1 knowledge")
    print("4. Train for 2-3 epochs")
    print("="*60)


if __name__ == "__main__":
    main()
