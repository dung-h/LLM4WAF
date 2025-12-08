#!/usr/bin/env python3
"""
Create Phase 2 dataset with replay buffer from Phase 1
Mixes Phase 2 observations + 20% Phase 1 samples to prevent catastrophic forgetting
"""

import json
import random

def main():
    phase2_file = "data/processed/phase2_observations_20k.jsonl"
    phase1_balanced = "data/processed/phase1_balanced_10k.jsonl"
    output_file = "data/processed/phase2_with_replay_22k.jsonl"
    
    # Read Phase 2 (20k samples)
    print(f"📖 Reading Phase 2: {phase2_file}")
    with open(phase2_file, 'r', encoding='utf-8') as f:
        phase2_samples = [json.loads(line) for line in f]
    
    print(f"  Loaded: {len(phase2_samples):,} Phase 2 samples")
    
    # Read Phase 1 balanced (10k samples)
    print(f"\n📖 Reading Phase 1 balanced: {phase1_balanced}")
    with open(phase1_balanced, 'r', encoding='utf-8') as f:
        phase1_samples = [json.loads(line) for line in f]
    
    print(f"  Loaded: {len(phase1_samples):,} Phase 1 samples")
    
    # Calculate 20% replay buffer
    replay_count = int(len(phase2_samples) * 0.2)
    print(f"\n⚖️  Replay buffer: 20% of Phase 2 = {replay_count:,} samples")
    
    # Random sample from Phase 1
    replay_samples = random.sample(phase1_samples, replay_count)
    
    # Combine: Phase 2 + Phase 1 replay
    combined_samples = phase2_samples + replay_samples
    
    # Shuffle to mix
    random.shuffle(combined_samples)
    
    # Write output
    print(f"\n💾 Writing combined dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in combined_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Created Phase 2 with replay buffer:")
    print(f"  Phase 2 observations: {len(phase2_samples):,}")
    print(f"  Phase 1 replay (20%): {len(replay_samples):,}")
    print(f"  Total: {len(combined_samples):,} samples")

if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
