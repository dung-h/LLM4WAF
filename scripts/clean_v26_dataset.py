#!/usr/bin/env python3
"""
SCRIPT 1: Clean redundant fields from v26 dataset
- Remove: status_code, elapsed, source, waf_result (duplicate)
- Standardize attack_type casing (SQLI->SQLi, xss->XSS)
- Output: red_v26_cleaned.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter

def normalize_attack_type(attack_type):
    """Standardize attack type casing"""
    mapping = {
        'SQLI': 'SQLi',
        'sqli': 'SQLi',
        'xss': 'XSS',
        'XSS': 'XSS',
        'SQLi': 'SQLi',
    }
    return mapping.get(attack_type, attack_type)

def clean_record(record):
    """Remove redundant fields and normalize"""
    
    # Fields to remove
    remove_fields = {'status_code', 'elapsed', 'source', 'waf_result'}
    
    # Clean record
    cleaned = {k: v for k, v in record.items() if k not in remove_fields}
    
    # Normalize attack_type
    if 'attack_type' in cleaned:
        cleaned['attack_type'] = normalize_attack_type(cleaned['attack_type'])
    
    # Keep only essential fields in priority order
    essential_keys = ['payload', 'prompt', 'chosen', 'attack_type', 'instruction']
    optional_keys = ['context', 'reasoning', 'constraints', 'attack_subtype', 'result']
    
    # Rebuild with order
    ordered = {}
    for key in essential_keys:
        if key in cleaned:
            ordered[key] = cleaned[key]
    
    for key in optional_keys:
        if key in cleaned:
            ordered[key] = cleaned[key]
    
    # Add any remaining fields (shouldn't be many)
    for key in cleaned:
        if key not in ordered:
            ordered[key] = cleaned[key]
    
    return ordered

def main():
    input_file = Path("data/processed/red_v26_unified_mysql_xss.jsonl")
    output_file = Path("data/processed/red_v26_cleaned.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print()
    
    total = 0
    written = 0
    attack_types = Counter()
    errors = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                # Clean and normalize
                cleaned = clean_record(record)
                attack_types[cleaned.get('attack_type', 'unknown')] += 1
                
                # Write cleaned record
                fout.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
                written += 1
                
                if written % 1000 == 0:
                    print(f"   ✓ {written:6d} records cleaned...")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 80)
    print("✅ CLEANING COMPLETE")
    print("=" * 80)
    print(f"Total records:      {total:6d}")
    print(f"Cleaned & written:  {written:6d}")
    print(f"Errors:             {errors:6d}")
    print()
    print("Attack Type Distribution (after normalization):")
    for attack_type, count in attack_types.most_common():
        pct = 100 * count / written
        print(f"   • {attack_type:10s}: {count:6d} ({pct:5.1f}%)")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
