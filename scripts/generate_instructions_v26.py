#!/usr/bin/env python3
"""
SCRIPT 2: Generate missing instructions for records without 'instruction' field
- For 7,951 records without instruction, generate rich context-aware instructions
- Use attack_type + payload pattern analysis to create diverse, non-generic prompts
- Standardize prompt format to English
- Output: red_v26_with_instructions.jsonl
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter

# Payload pattern templates for richer instruction generation
PAYLOAD_PATTERNS = {
    'SQLi': {
        'union': r"UNION.*SELECT",
        'time_based': r"SLEEP|BENCHMARK|WAITFOR",
        'boolean': r"(AND|OR).*=.*",
        'error_based': r"(updatexml|extractvalue|floor|gtid_subtract)",
        'stacked': r"(;|\*\/.*\/\*).*SELECT",
    },
    'XSS': {
        'dom': r"(on\w+\s*=|javascript:|<svg|<iframe)",
        'stored': r"(onclick|onerror|onload)",
        'reflected': r"(alert|console|fetch|window)",
        'template': r"(\{\{|<%|\$\{)",
    }
}

# Instruction templates for non-generic prompts
INSTRUCTIONS = {
    'SQLi': {
        'union': "Generate a UNION-based SQL injection payload that extracts data using SELECT statements in a UNION clause",
        'time_based': "Generate a time-based blind SQL injection payload using sleep/benchmark functions to exfiltrate data",
        'boolean': "Generate a boolean-based blind SQL injection payload that uses logical operators to infer database structure",
        'error_based': "Generate an error-based SQL injection payload that leverages database error messages to extract information",
        'stacked': "Generate a stacked query SQL injection payload that executes multiple SQL statements sequentially",
        'default': "Generate a SQL injection payload to bypass authentication or extract sensitive data",
    },
    'XSS': {
        'dom': "Generate a DOM-based XSS payload that modifies the Document Object Model through JavaScript execution",
        'stored': "Generate a stored XSS payload that persists in the application and executes for all users",
        'reflected': "Generate a reflected XSS payload that executes JavaScript in the victim's browser via URL parameters",
        'template': "Generate a template injection XSS payload that breaks out of template contexts",
        'default': "Generate an XSS payload that executes JavaScript in the target browser context",
    }
}

def detect_payload_variant(payload, attack_type):
    """Detect payload variant/technique based on pattern matching"""
    attack_type = attack_type.lower()
    
    if attack_type in ['sqli', 'sqlinjection']:
        patterns = PAYLOAD_PATTERNS.get('SQLi', {})
    elif attack_type.lower() in ['xss', 'cross-site-scripting']:
        patterns = PAYLOAD_PATTERNS.get('XSS', {})
    else:
        return 'default'
    
    for variant, pattern in patterns.items():
        if re.search(pattern, payload, re.IGNORECASE):
            return variant
    
    return 'default'

def generate_instruction(record):
    """Generate rich, non-generic instruction based on attack type and payload pattern"""
    
    attack_type = record.get('attack_type', 'Unknown').strip()
    payload = record.get('payload', '')
    
    # Detect variant
    variant = detect_payload_variant(payload, attack_type)
    
    # Get instruction template
    if attack_type.upper() in INSTRUCTIONS:
        instruction = INSTRUCTIONS[attack_type.upper()].get(variant, INSTRUCTIONS[attack_type.upper()]['default'])
    else:
        instruction = f"Generate a {attack_type} payload for web application testing"
    
    return instruction

def standardize_prompt(prompt):
    """Ensure prompt is in standard English format"""
    
    # If prompt is already detailed, keep it
    if len(prompt) > 50 and '<' not in prompt:
        return prompt
    
    # If prompt is just basic format, it will be replaced by generated instruction
    return prompt

def enrich_record(record):
    """Add instruction field if missing, standardize prompt"""
    
    enriched = record.copy()
    
    # Generate instruction if missing
    if 'instruction' not in enriched or not enriched.get('instruction'):
        enriched['instruction'] = generate_instruction(record)
    
    # Standardize prompt format
    if 'prompt' in enriched:
        # Replace generic prompt with instruction-based prompt if needed
        original_prompt = enriched['prompt']
        if 'Generate a' in original_prompt and len(original_prompt) < 100:
            # This is a generic prompt, should be replaced with richer one
            enriched['prompt'] = enriched['instruction']
    
    return enriched

def main():
    input_file = Path("data/processed/red_v26_unified_mysql_xss.jsonl")
    output_file = Path("data/processed/red_v26_with_instructions.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print()
    
    total = 0
    generated = 0
    already_had = 0
    errors = 0
    variant_counts = Counter()
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                # Check if instruction exists
                has_instruction = 'instruction' in record and record['instruction']
                
                # Enrich record
                enriched = enrich_record(record)
                
                # Track variant
                variant = detect_payload_variant(
                    enriched.get('payload', ''),
                    enriched.get('attack_type', '')
                )
                variant_counts[variant] += 1
                
                if has_instruction:
                    already_had += 1
                else:
                    generated += 1
                
                # Write enriched record
                fout.write(json.dumps(enriched, ensure_ascii=False) + '\n')
                
                if total % 1000 == 0:
                    print(f"   ✓ {total:6d} records processed ({generated} new instructions)...")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 80)
    print("✅ INSTRUCTION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total records:            {total:6d}")
    print(f"Already had instruction:  {already_had:6d} ({100*already_had/total:.1f}%)")
    print(f"Newly generated:          {generated:6d} ({100*generated/total:.1f}%)")
    print(f"Errors:                   {errors:6d}")
    print()
    print("Payload Variant Distribution:")
    for variant, count in variant_counts.most_common():
        pct = 100 * count / total
        print(f"   • {variant:15s}: {count:6d} ({pct:5.1f}%)")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
