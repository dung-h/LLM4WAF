#!/usr/bin/env python3
"""
SCRIPT 3: Merge cleaned + instructed records, then format for Phi-3
- Combines: cleaned data + instructions + payload pattern enrichment
- Format: Phi-3 chat format suitable for SFT
- Output: red_v26_phi3_final.jsonl (ready for training)
"""

import json
import sys
from pathlib import Path
from collections import Counter

def format_phi3_chat(record):
    """Format record as Phi-3 chat template"""
    
    instruction = record.get('instruction', f"Generate a {record.get('attack_type')} payload")
    payload = record.get('payload', '')
    attack_type = record.get('attack_type', 'Unknown')
    
    # Build system context from optional fields
    context_parts = []
    if record.get('context'):
        context_parts.append(f"Context: {record['context']}")
    if record.get('constraints'):
        context_parts.append(f"Constraints: {record['constraints']}")
    if record.get('attack_subtype'):
        context_parts.append(f"Technique: {record['attack_subtype']}")
    
    context_str = " ".join(context_parts) if context_parts else f"Attack Type: {attack_type}"
    
    # Build the formatted message
    formatted = {
        "messages": [
            {
                "role": "user",
                "content": f"{instruction}\n\n{context_str}"
            },
            {
                "role": "assistant",
                "content": payload
            }
        ]
    }
    
    # Keep metadata
    formatted["attack_type"] = attack_type
    formatted["result"] = record.get('result', 'unknown')
    
    return formatted

def main():
    input_file = Path("data/processed/red_v26_with_instructions.jsonl")
    clean_file = Path("data/processed/red_v26_cleaned.jsonl")
    output_file = Path("data/processed/red_v26_phi3_final.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Formatting for Phi-3 chat template)")
    print()
    
    total = 0
    written = 0
    errors = 0
    attack_types = Counter()
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                # Format for Phi-3
                formatted = format_phi3_chat(record)
                attack_types[formatted['attack_type']] += 1
                
                # Write formatted record
                fout.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                written += 1
                
                if total % 1000 == 0:
                    print(f"   ✓ {total:6d} records formatted...")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 80)
    print("✅ PHI-3 FORMATTING COMPLETE")
    print("=" * 80)
    print(f"Total records:      {total:6d}")
    print(f"Formatted & written:{written:6d}")
    print(f"Errors:             {errors:6d}")
    print()
    print("Attack Type Distribution:")
    for attack_type, count in attack_types.most_common():
        pct = 100 * count / written
        print(f"   • {attack_type:10s}: {count:6d} ({pct:5.1f}%)")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("✅ Ready for training! Next: sft_phi3_mini_red_v26_final.yaml config")

if __name__ == "__main__":
    main()
