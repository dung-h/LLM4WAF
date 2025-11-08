#!/usr/bin/env python3
"""
Generate BLUE dataset from replay results + TRUE BYPASSES
Combines:
- 5 TRUE BYPASSES (should_block=1, label=false_negative)
- 45 BLOCKED payloads (should_block=1, label=true_positive)
"""
import json
import pandas as pd
from pathlib import Path


def create_blue_dataset(replay_parquet, true_bypasses_jsonl, output_jsonl):
    """
    Create BLUE training dataset with labels:
    - FALSE NEGATIVES: Bypasses that WAF missed (should_block=1, waf_blocked=0)
    - TRUE POSITIVES: Attacks that WAF caught (should_block=1, waf_blocked=1)
    """
    
    # Load replay results
    df = pd.read_parquet(replay_parquet)
    
    # Load TRUE BYPASSES
    true_bypasses = []
    with open(true_bypasses_jsonl, 'r') as f:
        for line in f:
            true_bypasses.append(json.loads(line))
    
    bypass_payloads = {bp['payload'] for bp in true_bypasses}
    
    print(f"{'='*70}")
    print(f"BLUE DATASET GENERATION")
    print(f"{'='*70}\n")
    print(f"Total payloads: {len(df)}")
    print(f"TRUE BYPASSES (false negatives): {len(bypass_payloads)}")
    print(f"BLOCKED (true positives): {df['blocked'].sum()}")
    
    # Create BLUE dataset
    blue_dataset = []
    
    # Add TRUE BYPASSES (false negatives - critical for training)
    for idx, row in df.iterrows():
        if row['payload'] in bypass_payloads:
            blue_dataset.append({
                "payload": row['payload'],
                "should_block": 1,  # Ground truth: should be blocked
                "waf_blocked": 0,   # Actual: WAF did NOT block
                "label": "false_negative",  # Critical: WAF missed this attack
                "status_code": row['status_code'],
                "resp_len": row['resp_len'],
                "resp_ms": row['resp_ms'],
                "severity": "critical",  # Bypasses are high priority
                "notes": "TRUE BYPASS - SQL executed, returned user data"
            })
    
    # Add BLOCKED payloads (true positives)
    for idx, row in df.iterrows():
        if row['blocked'] == 1:
            blue_dataset.append({
                "payload": row['payload'],
                "should_block": 1,  # Ground truth: should be blocked
                "waf_blocked": 1,   # Actual: WAF blocked it
                "label": "true_positive",  # Good: WAF caught the attack
                "status_code": row['status_code'],
                "resp_len": row['resp_len'],
                "resp_ms": row['resp_ms'],
                "severity": "medium",
                "notes": "Correctly blocked by WAF"
            })
    
    # Save BLUE dataset
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in blue_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Statistics
    false_negatives = sum(1 for x in blue_dataset if x['label'] == 'false_negative')
    true_positives = sum(1 for x in blue_dataset if x['label'] == 'true_positive')
    
    print(f"\n{'='*70}")
    print(f"BLUE DATASET CREATED")
    print(f"{'='*70}\n")
    print(f"Total samples: {len(blue_dataset)}")
    print(f"  FALSE NEGATIVES (bypasses): {false_negatives} ({false_negatives/len(blue_dataset)*100:.1f}%)")
    print(f"  TRUE POSITIVES (blocked): {true_positives} ({true_positives/len(blue_dataset)*100:.1f}%)")
    print(f"\nOutput: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show sample bypasses for review
    print(f"\n{'='*70}")
    print(f"SAMPLE FALSE NEGATIVES (bypasses to train on):")
    print(f"{'='*70}\n")
    
    bypass_samples = [x for x in blue_dataset if x['label'] == 'false_negative']
    for i, sample in enumerate(bypass_samples[:5], 1):
        print(f"{i}. {sample['payload']}")
        print(f"   Severity: {sample['severity']}, Response: {sample['resp_len']} bytes")
        print(f"   Notes: {sample['notes']}\n")
    
    return blue_dataset


if __name__ == "__main__":
    replay_parquet = "artifacts/replays/gemma2_finetuned.parquet"
    true_bypasses_jsonl = "artifacts/replays/true_bypasses.jsonl"
    output_jsonl = "data/processed/blue_gemma2_bypasses.jsonl"
    
    blue_dataset = create_blue_dataset(replay_parquet, true_bypasses_jsonl, output_jsonl)
    
    print("\n✅ BLUE dataset ready for WAF tuning and ML training!")
    print("\nNext steps:")
    print("  1. Train BLUE classifier on this dataset")
    print("  2. Extract WAF features from ModSecurity audit logs")
    print("  3. Tune CRS rules to catch these bypass patterns")
