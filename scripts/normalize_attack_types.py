#!/usr/bin/env python3
"""
FIX: Standardize attack types in red_v26_with_instructions.jsonl
Then create final Phi-3 ready dataset with proper attack type format
"""

import json
import sys
from pathlib import Path

def normalize_attack_type(attack_type):
    """Comprehensive attack type standardization"""
    if not attack_type:
        return "Unknown"
    
    normalized = attack_type.strip().lower()
    
    # SQLi variants
    if normalized in ['sqli', 'sqli injection', 'sql injection', 'sqlinjection']:
        return 'SQLi'
    
    # XSS variants
    if normalized in ['xss', 'cross-site-scripting', 'xss injection']:
        return 'XSS'
    
    # Default: capitalize first letter
    return attack_type.strip().title()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="data/processed/red_v26_with_instructions.jsonl")
    parser.add_argument('--output', default="data/processed/red_v26_normalized.jsonl")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Normalizing attack types: {input_file}")
    print(f"📤 Output: {output_file}")
    print()
    
    total = 0
    changed = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                old_type = record.get('attack_type', '')
                new_type = normalize_attack_type(old_type)
                
                if old_type != new_type:
                    changed += 1
                    record['attack_type'] = new_type
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if total % 2000 == 0:
                    print(f"   ✓ {total:6d} records normalized...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print()
    print(f"✅ Normalization complete: {total} total, {changed} changed")
    print(f"📊 Output: {output_file}")

if __name__ == "__main__":
    main()
