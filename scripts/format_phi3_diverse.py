#!/usr/bin/env python3
"""
Format normalized diverse instructions dataset for Phi-3 training
- Input: red_v26_final_normalized.jsonl (normalized + diverse instructions)
- Output: red_v26_phi3_ready.jsonl (Phi-3 chat format)
"""

import json
import sys
import argparse
from pathlib import Path

def format_phi3_from_diverse(record):
    """Format record with diverse instruction for Phi-3"""
    
    instruction = record.get('instruction', 'Generate a payload')
    payload = record.get('payload', '')
    attack_type = record.get('attack_type', 'Unknown')
    
    # Use instruction as main user content (already diverse & technique-specific)
    user_content = instruction
    
    # Add optional context if available
    context_parts = []
    if record.get('context'):
        context_parts.append(record['context'])
    if record.get('constraints'):
        context_parts.append(f"Constraints: {record['constraints']}")
    if record.get('attack_subtype'):
        context_parts.append(f"Technique: {record['attack_subtype']}")
    
    if context_parts:
        user_content += "\n\n" + "\n".join(context_parts)
    
    formatted = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": payload
            }
        ]
    }
    
    # Metadata
    formatted["attack_type"] = attack_type
    formatted["result"] = record.get('result', 'unknown')
    
    return formatted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="data/processed/red_v26_final_normalized.jsonl")
    parser.add_argument('--output', default="data/processed/red_v26_phi3_ready.jsonl")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Phi-3 format with diverse instructions)")
    print()
    
    total = 0
    written = 0
    errors = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                formatted = format_phi3_from_diverse(record)
                fout.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                written += 1
                
                if total % 1000 == 0:
                    print(f"   ✓ {total:6d} formatted...")
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"❌ Error: {e}")
    
    print()
    print("=" * 80)
    print(f"✅ Phi-3 formatting complete: {written}/{total} records")
    print(f"📊 Output: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    main()
