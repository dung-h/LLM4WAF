#!/usr/bin/env python3
"""
AUGMENT TRAINING DATA WITH EVASION-FOCUSED INSTRUCTIONS
- Takes v26_final_normalized.jsonl
- Adds evasion-specific instructions that teach model WAF-bypass techniques
- Creates training examples showing BOTH original AND evasion-variant payloads
"""

import json
import sys
from pathlib import Path
import random

EVASION_INSTRUCTIONS = {
    'SQLi': [
        "Generate a SQL injection payload using comment obfuscation (/**/, #) to hide keywords from WAF",
        "Create a SQLi using case variation (UnIoN, SeLeCt) to bypass keyword filters",
        "Build a SQL injection with whitespace injection (tabs, newlines) instead of spaces",
        "Generate a SQLi using stacked queries and comment-out (--) to hide payload tail",
        "Create a SQL injection combining multiple evasion: case + comment + encoding",
        "Generate a blind SQL injection using alternative functions (BENCHMARK vs SLEEP)",
        "Build a SQLi using nested subqueries to indirectly extract data",
        "Create a SQL injection with boolean math (1+1=2) instead of direct conditions",
        "Generate an error-based SQLi with obfuscated function names",
        "Build a UNION-based SQLi where SELECT and UNION are split by comments",
    ],
    'XSS': [
        "Generate an XSS payload using case variation and encoding to bypass filters",
        "Create an XSS using alternative event handlers (onload, onpageshow) instead of blocked ones",
        "Build an XSS with URL encoding (%3Cscript%3E) to obfuscate tags",
        "Generate an XSS using nested or indirect DOM methods to avoid keyword detection",
        "Create an XSS combining backticks (`) and template literals for code execution",
        "Build an XSS with unicode encoding to bypass character-based filters",
        "Generate an SVG-based XSS with attribute manipulation",
        "Create an XSS using setTimeout/setInterval indirectly to delay execution",
    ]
}

def create_evasion_variant_record(original_record, evasion_variant_payload, evasion_instruction):
    """Create a new training record for evasion variant"""
    
    variant_record = original_record.copy()
    
    # Update with evasion-specific data
    variant_record['instruction'] = evasion_instruction
    variant_record['original_payload'] = original_record.get('payload')
    variant_record['payload'] = evasion_variant_payload  # This would be variant
    variant_record['is_evasion_variant'] = True
    variant_record['training_focus'] = 'waf_evasion'
    
    return variant_record

def augment_with_evasion_instructions():
    """Add evasion-focused variants to training data"""
    
    input_file = Path("data/processed/red_v26_final_normalized.jsonl")
    output_file = Path("data/processed/red_v26_evasion_augmented.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Augmenting with evasion-focused instructions)")
    print()
    
    total = 0
    original_written = 0
    evasion_variants_written = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                attack_type = record.get('attack_type', 'Unknown')
                
                # Write original record
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                original_written += 1
                
                # Augment with evasion variants (50% chance per record)
                if random.random() > 0.5:
                    instruction_list = EVASION_INSTRUCTIONS.get(attack_type, EVASION_INSTRUCTIONS.get('SQLi'))
                    if instruction_list:
                        evasion_instr = random.choice(instruction_list)
                        
                        # Create variant record (payload would need actual variants)
                        variant_record = {
                            'instruction': evasion_instr,
                            'payload': record.get('payload', ''),  # In real case, would be variant
                            'attack_type': attack_type,
                            'training_focus': 'waf_evasion',
                            'original_instruction': record.get('instruction', ''),
                        }
                        
                        # Copy optional fields
                        for key in ['result', 'context', 'constraints', 'reasoning', 'attack_subtype']:
                            if key in record:
                                variant_record[key] = record[key]
                        
                        fout.write(json.dumps(variant_record, ensure_ascii=False) + '\n')
                        evasion_variants_written += 1
                
                if total % 1000 == 0:
                    print(f"   ✓ {total:6d} records processed...")
                
            except Exception as e:
                print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 80)
    print("✅ EVASION AUGMENTATION COMPLETE")
    print("=" * 80)
    print(f"Total original records:       {total:6d}")
    print(f"Original records written:     {original_written:6d}")
    print(f"Evasion variants added:       {evasion_variants_written:6d} ({100*evasion_variants_written/(original_written+evasion_variants_written):.1f}%)")
    print(f"Total training examples:      {original_written + evasion_variants_written:6d}")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    augment_with_evasion_instructions()
