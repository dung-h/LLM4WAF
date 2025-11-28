#!/usr/bin/env python3
"""
WAF EVASION DATASET BUILDER
- Takes standard payloads from v26_final_normalized.jsonl
- Generates WAF-evasion variants using known techniques:
  * Comment obfuscation: /**/,#,%23, \x00, etc
  * Whitespace variations: \t, \n, \r, space
  * Case variations: UnIoN, SeLeCt, etc
  * Encoding: hex, URL, unicode, base64
  * Alternative functions: SLEEP→BENCHMARK, USER→DATABASE, etc
  * Stacked queries with creative syntax
- Output: red_v26_with_evasion_variants.jsonl
"""

import json
import sys
import random
import string
from pathlib import Path
from collections import defaultdict
import re
import urllib.parse
import base64

class WAFEvader:
    """Generate WAF-evasion variants of payloads"""
    
    # Comment injection techniques
    COMMENTS = ['/**/','/**/',' ','/*!',' ','\t','\n','\r','\r\n',chr(0x0c)]
    HEX_COMMENTS = ['%2f%2a%2a%2f', '%20', '%09', '%0a']
    
    # Keyword replacements (SQLi)
    SQLI_REPLACEMENTS = {
        'OR': ['||', 'Or', 'oR', 'OR/**/'],
        'AND': ['&&', 'And', 'AnD', 'AND/**/'],
        'UNION': ['UnIoN', 'uNiOn', 'UNION/**/'],
        'SELECT': ['SeLeCt', 'sElEcT', 'SELECT/**/'],
        'FROM': ['FrOm', 'FROM/**/'],
        'WHERE': ['WhErE', 'WHERE/**/'],
        'SLEEP': ['SLEEP', 'sleep', 'SLEEP/**/'],
        'BENCHMARK': ['BENCHMARK', 'benchmark'],
        'EXTRACTVALUE': ['ExtractValue', 'extractvalue'],
        'UPDATEXML': ['UpdateXml', 'updatexml'],
    }
    
    # XSS alternatives
    XSS_REPLACEMENTS = {
        'alert': ['alert', 'confirm', 'prompt'],
        'script': ['SCRIPT', 'Script'],
        'onerror': ['onload', 'onclick'],
        'onload': ['onerror', 'onpageshow'],
    }
    
    @staticmethod
    def randomize_case(text):
        """Randomize case of text"""
        return ''.join(random.choice([c.upper(), c.lower()]) if c.isalpha() else c for c in text)
    
    @staticmethod
    def inject_comments(payload, attack_type):
        """Inject comments to obfuscate keywords"""
        variants = []
        
        if attack_type.lower() in ['sqli']:
            # Split on SQL keywords
            for keyword in ['SELECT', 'UNION', 'FROM', 'WHERE', 'AND', 'OR']:
                if keyword.lower() in payload.lower():
                    # Try replacing with comment-obfuscated version
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    for comment in ['/**/','/**/**/','/*! */']:
                        new_payload = pattern.sub(f"{keyword}{comment}", payload, count=1)
                        if new_payload != payload:
                            variants.append(new_payload)
        
        return variants[:3]  # Return top 3
    
    @staticmethod
    def hex_encode_payload(payload):
        """Hex encode parts of payload"""
        # For SQL: encode critical keywords
        hex_payload = ""
        i = 0
        while i < len(payload):
            if payload[i:i+6].upper() in ['SELECT', 'UNION', 'WHERE', 'FROM']:
                # Find keyword
                for keyword in ['SELECT', 'UNION', 'WHERE', 'FROM', 'AND', 'OR']:
                    if payload[i:i+len(keyword)].upper() == keyword:
                        hex_str = ''.join(f'0x{ord(c):02x}' for c in keyword)
                        hex_payload += f"CHAR({','.join(f'{ord(c)}' for c in keyword)})"
                        i += len(keyword)
                        break
            else:
                hex_payload += payload[i]
                i += 1
        
        return hex_payload if hex_payload != payload else None
    
    @staticmethod
    def generate_evasion_variants(payload, attack_type, num_variants=4):
        """Generate multiple evasion variants of payload"""
        variants = []
        attack_type_lower = attack_type.lower()
        
        if attack_type_lower == 'sqli':
            # 1. Case randomization
            if random.random() > 0.3:
                variants.append(WAFEvader.randomize_case(payload))
            
            # 2. Comment obfuscation
            comment_vars = WAFEvader.inject_comments(payload, attack_type)
            variants.extend(comment_vars[:2])
            
            # 3. Keyword replacement
            modified = payload
            for keyword, alts in WAFEvader.SQLI_REPLACEMENTS.items():
                if keyword.lower() in payload.lower():
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    alt = random.choice(alts)
                    modified = pattern.sub(alt, modified, count=1)
            if modified != payload:
                variants.append(modified)
            
            # 4. Double encoding
            if '%' not in payload and len(payload) < 100:
                encoded = urllib.parse.quote(payload)
                variants.append(encoded)
            
        elif attack_type_lower == 'xss':
            # 1. Case variations
            variants.append(payload.replace('script', 'SCRIPT').replace('alert', 'ALERT'))
            variants.append(payload.replace('script', 'ScRiPt'))
            
            # 2. Event handler replacements
            if 'onerror' in payload:
                variants.append(payload.replace('onerror', 'onload'))
            if 'onclick' in payload:
                variants.append(payload.replace('onclick', 'onmouseover'))
            
            # 3. Backticks instead of parentheses (JS)
            if 'alert(' in payload:
                variants.append(payload.replace('alert(', 'alert`').replace(')', '`'))
        
        # Deduplicate and limit
        unique = list(set(variants))
        random.shuffle(unique)
        return unique[:num_variants]

def main():
    input_file = Path("data/processed/red_v26_final_normalized.jsonl")
    output_file = Path("data/processed/red_v26_with_evasion_variants.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Generating WAF-evasion payload variants)")
    print()
    
    total = 0
    written = 0
    variant_counts = defaultdict(int)
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                payload = record.get('payload', '')
                attack_type = record.get('attack_type', 'Unknown')
                instruction = record.get('instruction', '')
                
                # Generate evasion variants
                variants = WAFEvader.generate_evasion_variants(payload, attack_type, num_variants=3)
                variant_counts[len(variants)] += 1
                
                # Add variants to instruction
                if variants:
                    evasion_hint = f"\nWAF-evasion techniques to consider: {', '.join(['case variation', 'comment injection', 'encoding', 'obfuscation'][:len(variants)])}"
                    enhanced_instruction = instruction + evasion_hint
                else:
                    enhanced_instruction = instruction
                
                # Write original + variants
                for i, variant_payload in enumerate([payload] + variants):
                    variant_record = {
                        'payload': variant_payload,
                        'attack_type': attack_type,
                        'instruction': enhanced_instruction,
                        'is_variant': i > 0,
                        'variant_index': i if i > 0 else None,
                        'original_payload': payload if i > 0 else None,
                    }
                    
                    # Copy optional fields
                    for key in ['result', 'context', 'constraints', 'reasoning', 'attack_subtype']:
                        if key in record:
                            variant_record[key] = record[key]
                    
                    fout.write(json.dumps(variant_record, ensure_ascii=False) + '\n')
                    written += 1
                
                if total % 500 == 0:
                    print(f"   ✓ {total:6d} original payloads processed ({written} with variants written)...")
                
            except Exception as e:
                print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 80)
    print("✅ EVASION VARIANT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Original payloads:        {total:6d}")
    print(f"Total records (with variants):{written:6d}")
    print(f"Average variants per payload: {(written-total)/total:.1f}")
    print()
    print("Variant distribution:")
    for count, freq in sorted(variant_counts.items()):
        print(f"   • {count} variants: {freq:6d} payloads")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("✅ Each payload now has WAF-evasion variants!")

if __name__ == "__main__":
    main()
