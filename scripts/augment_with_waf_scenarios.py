#!/usr/bin/env python3
"""
EXPAND EVASION DATASET WITH DIVERSE WAF CONFIGURATIONS
- Generate multiple WAF blocking scenarios
- Create training data for each scenario showing HOW to evade
- Model learns to handle various WAF configurations
"""

import json
import sys
import random
from pathlib import Path
from collections import defaultdict

# Different WAF configurations (realistic keyword blocking patterns)
WAF_SCENARIOS = {
    'aggressive_keyword_blocking': {
        'keywords': ['OR', 'UNION', 'SELECT', 'FROM', 'WHERE', 'AND', 'INSERT', 'UPDATE', 'DELETE'],
        'description': 'Aggressive WAF blocks all common SQL keywords',
        'evasion_focus': ['case_variation', 'comment_obfuscation', 'whitespace_injection'],
    },
    'function_blocking': {
        'keywords': ['SLEEP', 'BENCHMARK', 'EXTRACTVALUE', 'UPDATEXML', 'USER', 'DATABASE', 'VERSION'],
        'description': 'WAF blocks common exfiltration functions',
        'evasion_focus': ['alternative_functions', 'encoding', 'nested_subqueries'],
    },
    'operator_blocking': {
        'keywords': ['OR', 'AND', '=', '<', '>', '!', '/*', '--', '#'],
        'description': 'WAF blocks operators and comment characters',
        'evasion_focus': ['boolean_math', 'stacked_queries', 'encoding'],
    },
    'tag_blocking': {
        'keywords': ['script', 'alert', 'onclick', 'onerror', 'onload', 'fetch', 'eval'],
        'description': 'WAF blocks XSS payload components',
        'evasion_focus': ['case_variation', 'encoding', 'alternative_handlers'],
    },
    'encoding_aware': {
        'keywords': ['%', '0x', '\\x', 'CHAR', 'CONCAT', 'UNHEX'],
        'description': 'WAF detects encoding attempts',
        'evasion_focus': ['comment_obfuscation', 'stacked_queries', 'alternative_functions'],
    },
    'strict_filtering': {
        'keywords': ['OR', 'UNION', 'SELECT', 'SLEEP', 'script', 'alert', '%', '0x', '/*'],
        'description': 'Strict WAF blocks keywords from multiple categories',
        'evasion_focus': ['all_techniques'],
    },
    'syntax_blocking': {
        'keywords': ['(', ')', ';', '\'', '"', '`', ','],
        'description': 'WAF blocks special characters',
        'evasion_focus': ['stacked_queries', 'encoding', 'alternative_syntax'],
    },
    'protocol_blocking': {
        'keywords': ['http://', 'https://', 'ftp://', 'file://', 'javascript:', 'data:'],
        'description': 'WAF blocks protocol handlers',
        'evasion_focus': ['indirect_requests', 'dns_exfiltration'],
    },
}

# Technique descriptions per scenario
TECHNIQUE_INSTRUCTIONS = {
    'case_variation': "Use random case mixing: UnIoN, SeLeCt, oR instead of blocked keywords",
    'comment_obfuscation': "Insert comment sequences /**/, #, %00 between keyword characters",
    'whitespace_injection': "Replace spaces with tabs (\\t), newlines (\\n), or carriage returns (\\r)",
    'encoding': "URL encode (%XX), hex encode (0x...), or unicode escape blocked characters",
    'alternative_functions': "Use BENCHMARK instead of SLEEP, DATABASE() instead of USER()",
    'boolean_math': "Use (1+1)=(1+1) instead of 1=1, or arithmetic for conditions",
    'stacked_queries': "Use semicolon (;) to separate statements, hide with comment (--)",
    'nested_subqueries': "Nest SELECT statements to extract data indirectly",
    'all_techniques': "Combine multiple evasion techniques: case + comment + encoding + stacking",
}

def generate_scenario_instructions(blocked_keywords, scenario_name, technique_focus):
    """Generate diverse instructions for a specific WAF scenario"""
    
    instructions = []
    
    # Base instruction
    blocked_str = ', '.join(blocked_keywords[:4]) + ('...' if len(blocked_keywords) > 4 else '')
    
    instructions.append(
        f"Generate a payload bypassing WAF that blocks: {blocked_str} "
        f"({scenario_name})"
    )
    
    # Technique-specific instructions
    for technique in technique_focus:
        if technique in TECHNIQUE_INSTRUCTIONS:
            instr = TECHNIQUE_INSTRUCTIONS[technique]
            instructions.append(
                f"Bypass WAF ({scenario_name}) using {technique}: {instr}"
            )
    
    # Combination instructions (teach model to combine techniques)
    if len(technique_focus) > 1:
        techniques_str = ' + '.join(technique_focus)
        instructions.append(
            f"Create payload combining {techniques_str} to evade {scenario_name} WAF"
        )
    
    return instructions

def augment_with_waf_scenarios():
    """Augment dataset with different WAF scenario training examples"""
    
    input_file = Path("data/processed/red_v26_final_normalized.jsonl")
    output_file = Path("data/processed/red_v26_evasion_expanded.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Augmenting with diverse WAF scenario training examples)")
    print()
    
    total_records = 0
    total_written = 0
    scenario_stats = defaultdict(int)
    technique_stats = defaultdict(int)
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # Write original record
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_written += 1
                
                # For each WAF scenario, create training variant
                attack_type = record.get('attack_type', 'Unknown').lower()
                
                for scenario_name, scenario_config in WAF_SCENARIOS.items():
                    # Filter scenarios by attack type
                    if 'tag_blocking' in scenario_name or 'protocol_blocking' in scenario_name:
                        if 'xss' not in attack_type:
                            continue
                    elif 'xss' in attack_type:
                        continue
                    
                    blocked_keywords = scenario_config['keywords']
                    technique_focus = scenario_config['evasion_focus']
                    
                    # Generate scenario-specific instructions
                    scenario_instructions = generate_scenario_instructions(
                        blocked_keywords, scenario_name, technique_focus
                    )
                    
                    # Create variant record for each instruction
                    for instruction in scenario_instructions:
                        variant_record = {
                            'instruction': instruction,
                            'payload': record.get('payload', ''),
                            'attack_type': record.get('attack_type', 'Unknown'),
                            'waf_scenario': scenario_name,
                            'blocked_keywords': blocked_keywords,
                            'training_focus': 'waf_scenario_evasion',
                        }
                        
                        # Copy optional fields
                        for key in ['result', 'context', 'constraints', 'reasoning', 'attack_subtype']:
                            if key in record:
                                variant_record[key] = record[key]
                        
                        fout.write(json.dumps(variant_record, ensure_ascii=False) + '\n')
                        total_written += 1
                        scenario_stats[scenario_name] += 1
                        
                        # Track technique usage
                        for technique in technique_focus:
                            technique_stats[technique] += 1
                    
                    total_records += 1
                
                if total_records % 500 == 0:
                    print(f"   ✓ {total_records:6d} original payloads augmented ({total_written} total written)...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print()
    print("=" * 100)
    print("✅ WAF SCENARIO AUGMENTATION COMPLETE")
    print("=" * 100)
    print(f"Original payloads:           {total_records:8d}")
    print(f"Total training examples:     {total_written:8d}")
    print(f"Augmentation factor:         {total_written/total_records:.1f}x")
    print()
    print("WAF Scenario Distribution:")
    for scenario, count in sorted(scenario_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_written
        print(f"   • {scenario:30s}: {count:8d} ({pct:5.1f}%)")
    print()
    print("Evasion Technique Coverage:")
    for technique, count in sorted(technique_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_written
        print(f"   • {technique:30s}: {count:8d} ({pct:5.1f}%)")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("✅ Model now trained on 8 DIFFERENT WAF configurations!")
    print("✅ Each configuration teaches specific evasion techniques!")

if __name__ == "__main__":
    augment_with_waf_scenarios()
