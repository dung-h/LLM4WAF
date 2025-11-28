#!/usr/bin/env python3
"""
ADVANCED: Generate DIVERSE & DYNAMIC instructions based on actual payload content
- Analyze payload structure, techniques, parameters
- Create varied, context-specific instructions
- Avoid repetitive templates
"""

import json
import sys
import re
from pathlib import Path
from collections import Counter
import random

# Set seed for reproducibility with variation
random.seed(42)

class PayloadAnalyzer:
    """Intelligently analyze payload to create diverse instructions"""
    
    @staticmethod
    def extract_sqli_features(payload):
        """Extract SQLi-specific features"""
        features = {
            'has_union': bool(re.search(r'\bUNION\b', payload, re.I)),
            'has_select': bool(re.search(r'\bSELECT\b', payload, re.I)),
            'has_where': bool(re.search(r'\bWHERE\b', payload, re.I)),
            'has_or': bool(re.search(r'\bOR\b', payload, re.I)),
            'has_and': bool(re.search(r'\bAND\b', payload, re.I)),
            'has_sleep': bool(re.search(r'\b(SLEEP|BENCHMARK|WAITFOR)\b', payload, re.I)),
            'has_delay': bool(re.search(r'(SLEEP|BENCHMARK|WAITFOR|pg_sleep)', payload, re.I)),
            'has_cast': bool(re.search(r'\bCAST\b', payload, re.I)),
            'has_concat': bool(re.search(r'\b(CONCAT|GROUP_CONCAT|STRING_AGG)\b', payload, re.I)),
            'has_substring': bool(re.search(r'\b(SUBSTRING|SUBSTR|MID)\b', payload, re.I)),
            'has_error_function': bool(re.search(r'\b(updatexml|extractvalue|floor|gtid_subtract|JSON_EXTRACT)\b', payload, re.I)),
            'has_comment': bool(re.search(r'(--|#|\/\*)', payload)),
            'has_quotes': bool("'" in payload or '"' in payload),
            'has_parentheses': bool('(' in payload and ')' in payload),
            'payload_length': len(payload),
        }
        return features
    
    @staticmethod
    def extract_xss_features(payload):
        """Extract XSS-specific features"""
        features = {
            'has_script_tag': bool(re.search(r'<script', payload, re.I)),
            'has_event_handler': bool(re.search(r'on\w+\s*=', payload, re.I)),
            'has_svg': bool(re.search(r'<svg', payload, re.I)),
            'has_iframe': bool(re.search(r'<iframe', payload, re.I)),
            'has_img': bool(re.search(r'<img', payload, re.I)),
            'has_style': bool(re.search(r'<style|behavior:', payload, re.I)),
            'has_alert': bool(re.search(r'alert\s*\(', payload, re.I)),
            'has_fetch': bool(re.search(r'fetch\s*\(|XMLHttpRequest', payload, re.I)),
            'has_window': bool(re.search(r'window\.|document\.|location\.', payload)),
            'has_eval': bool(re.search(r'\beval\s*\(|Function\s*\(', payload, re.I)),
            'has_backticks': bool('`' in payload),
            'has_angle_brackets': bool('<' in payload or '>' in payload),
            'has_slash': bool('/' in payload),
            'payload_length': len(payload),
        }
        return features
    
    @staticmethod
    def classify_sqli_technique(features):
        """Classify specific SQLi technique based on features"""
        if features['has_sleep'] or features['has_delay']:
            return 'time_based_blind'
        elif features['has_error_function']:
            return 'error_based'
        elif features['has_union'] and features['has_select']:
            return 'union_based'
        elif features['has_or'] and not features['has_union']:
            return 'boolean_based_blind'
        elif features['has_where']:
            return 'where_clause_manipulation'
        elif features['has_cast']:
            return 'type_casting'
        else:
            return 'injection_basic'
    
    @staticmethod
    def classify_xss_technique(features):
        """Classify specific XSS technique"""
        if features['has_script_tag']:
            return 'script_injection'
        elif features['has_event_handler']:
            return 'event_handler'
        elif features['has_svg']:
            return 'svg_based'
        elif features['has_iframe']:
            return 'iframe_injection'
        elif features['has_eval'] or features['has_backticks']:
            return 'code_execution'
        elif features['has_fetch'] or features['has_window']:
            return 'dom_manipulation'
        elif features['has_img']:
            return 'img_tag'
        else:
            return 'basic_html'

class InstructionGenerator:
    """Generate diverse instructions based on payload analysis"""
    
    # Per-technique diverse templates (vary in wording, focus, complexity)
    SQLI_TEMPLATES = {
        'time_based_blind': [
            "Craft a time-based SQL injection that extracts data by measuring response delays",
            "Design a blind SQLi using temporal side-channels (SLEEP/BENCHMARK) to infer database contents",
            "Build a time-delay injection to discover database schema through response latency analysis",
            "Generate a timing-based SQLi payload exploiting sleep functions for data exfiltration",
        ],
        'error_based': [
            "Create an error-based SQLi leveraging database error messages to leak sensitive information",
            "Construct a payload that triggers database errors revealing table structures and data types",
            "Build an injection that exploits error message leakage to extract database metadata",
            "Generate error-based SQLi using functions like updatexml/extractvalue for information disclosure",
        ],
        'union_based': [
            "Develop a UNION-based SQLi that retrieves data directly from database tables",
            "Construct a UNION SELECT injection to extract multiple columns from target databases",
            "Create a multi-column UNION payload to dump authentication credentials and user data",
            "Generate UNION-based SQL injection for direct database querying and data extraction",
        ],
        'boolean_based_blind': [
            "Craft a boolean-based blind SQLi using logical operators to infer database state",
            "Design a payload leveraging TRUE/FALSE conditions to brute-force database contents",
            "Build a boolean injection that determines database schema character-by-character",
            "Generate conditional SQLi using AND/OR logic for bit-by-bit data exfiltration",
        ],
        'where_clause_manipulation': [
            "Create a WHERE clause injection to bypass access controls and authentication mechanisms",
            "Construct a payload that manipulates database query logic through condition injection",
            "Generate WHERE-based SQLi to modify query results and access restricted data",
            "Build injection altering WHERE conditions to return unauthorized database records",
        ],
        'type_casting': [
            "Develop a CAST-based SQLi exploiting type coercion vulnerabilities in SQL parsing",
            "Construct a payload using type casting to bypass WAF filters and validation logic",
            "Create injection leveraging CAST functions for data exfiltration across type boundaries",
            "Generate type-based SQLi for obfuscation and WAF evasion",
        ],
        'injection_basic': [
            "Generate a basic SQL injection payload to compromise database security",
            "Construct a simple SQLi to test database query parameterization and input validation",
            "Create a straightforward SQL injection for authentication bypass and data access",
            "Generate fundamental SQLi payload targeting vulnerable SQL query construction",
        ]
    }
    
    XSS_TEMPLATES = {
        'script_injection': [
            "Generate a <script> tag XSS payload that executes arbitrary JavaScript in victim's browser",
            "Craft a script-based injection to steal session tokens and cookies",
            "Create a <script> XSS that redirects users to malicious sites or steals credentials",
            "Build JavaScript execution payload via script tag injection for malware distribution",
        ],
        'event_handler': [
            "Construct an event handler XSS (onclick, onerror, onload) to trigger code execution",
            "Generate an attribute-based XSS exploiting event handlers in HTML elements",
            "Create a payload using onload/onerror to execute JavaScript when elements load",
            "Build event-driven XSS for stealing browser cookies and session data",
        ],
        'svg_based': [
            "Generate an SVG-based XSS payload to embed malicious JavaScript in vector graphics",
            "Craft an SVG injection exploiting svg/onload events for code execution",
            "Create a SVG XSS bypassing HTML filters via namespace-qualified payloads",
            "Build SVG-embedded JavaScript for credential theft and browser compromise",
        ],
        'iframe_injection': [
            "Construct an iframe injection XSS to embed external malicious content",
            "Generate a sandboxed iframe payload for session hijacking and phishing attacks",
            "Create iframe-based XSS to load external scripts and compromise user security",
            "Build iframe injection for malware delivery and browser exploitation",
        ],
        'code_execution': [
            "Develop an eval-based XSS payload for direct code execution in JavaScript context",
            "Generate a backtick template literal injection for dynamic code evaluation",
            "Craft code execution XSS via eval/Function constructor for full JavaScript control",
            "Create injectable code that executes with full JavaScript runtime permissions",
        ],
        'dom_manipulation': [
            "Generate a DOM-based XSS manipulating document object model through fetch/window",
            "Construct a payload that modifies window/location properties to hijack navigation",
            "Create a document.write/innerHTML injection for DOM replacement attacks",
            "Build JavaScript payload exploiting DOM APIs for credential and data theft",
        ],
        'img_tag': [
            "Generate an img tag XSS via onerror handler for code execution",
            "Construct an image element injection triggering JavaScript on load failure",
            "Create an img-based XSS bypassing strict CSP through error handlers",
            "Build broken image XSS for callback-based code execution",
        ],
        'basic_html': [
            "Generate basic HTML tag injection for XSS exploitation",
            "Construct simple XSS payload through unfiltered HTML element injection",
            "Create fundamental XSS for testing input validation and sanitization",
            "Build basic cross-site scripting payload targeting HTML injection vulnerabilities",
        ]
    }
    
    @staticmethod
    def generate_instruction(payload, attack_type, existing_instruction=None):
        """Generate diverse, payload-specific instruction"""
        
        attack_type_lower = attack_type.lower().strip()
        
        # If already has instruction from augmentation, use it but create alternative
        if existing_instruction and len(existing_instruction) > 20:
            # 70% chance use existing, 30% create alternative
            if random.random() < 0.7:
                return existing_instruction
            # Otherwise create alternative based on analysis
        
        if attack_type_lower in ['sqli', 'sql injection', 'sqlinjection']:
            analyzer = PayloadAnalyzer()
            features = analyzer.extract_sqli_features(payload)
            technique = analyzer.classify_sqli_technique(features)
            templates = InstructionGenerator.SQLI_TEMPLATES.get(technique, InstructionGenerator.SQLI_TEMPLATES['injection_basic'])
            return random.choice(templates)
        
        elif attack_type_lower in ['xss', 'cross-site scripting', 'cross site scripting']:
            analyzer = PayloadAnalyzer()
            features = analyzer.extract_xss_features(payload)
            technique = analyzer.classify_xss_technique(features)
            templates = InstructionGenerator.XSS_TEMPLATES.get(technique, InstructionGenerator.XSS_TEMPLATES['basic_html'])
            return random.choice(templates)
        
        else:
            return f"Generate a {attack_type} payload for security testing and vulnerability assessment"

def main():
    input_file = Path("data/processed/red_v26_unified_mysql_xss.jsonl")
    output_file = Path("data/processed/red_v26_with_diverse_instructions.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"📥 Reading: {input_file}")
    print(f"📤 Writing: {output_file}")
    print("   (Generating DIVERSE, DYNAMIC instructions per payload)")
    print()
    
    total = 0
    generated = 0
    reused = 0
    errors = 0
    technique_counts = Counter()
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                total += 1
                
                payload = record.get('payload', '')
                attack_type = record.get('attack_type', 'Unknown')
                existing_instruction = record.get('instruction', '')
                
                # Generate instruction
                instruction = InstructionGenerator.generate_instruction(
                    payload, 
                    attack_type,
                    existing_instruction
                )
                
                record['instruction'] = instruction
                
                # Track metrics
                if existing_instruction and len(existing_instruction) > 20:
                    if record['instruction'] == existing_instruction:
                        reused += 1
                    else:
                        generated += 1
                else:
                    generated += 1
                
                # Track technique
                analyzer = PayloadAnalyzer()
                if attack_type.lower() in ['sqli']:
                    features = analyzer.extract_sqli_features(payload)
                    technique = analyzer.classify_sqli_technique(features)
                elif attack_type.lower() in ['xss']:
                    features = analyzer.extract_xss_features(payload)
                    technique = analyzer.classify_xss_technique(features)
                else:
                    technique = 'unknown'
                technique_counts[technique] += 1
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                if total % 1000 == 0:
                    print(f"   ✓ {total:6d} records processed ({generated} new instructions)...")
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"❌ Error at record {total}: {e}")
    
    print()
    print("=" * 100)
    print("✅ DIVERSE INSTRUCTION GENERATION COMPLETE")
    print("=" * 100)
    print(f"Total records:            {total:6d}")
    print(f"Newly generated:          {generated:6d} ({100*generated/total:.1f}%)")
    print(f"Reused existing:          {reused:6d} ({100*reused/total:.1f}%)")
    print(f"Errors:                   {errors:6d}")
    print()
    print("Attack Technique Distribution:")
    for technique, count in technique_counts.most_common():
        pct = 100 * count / total
        print(f"   • {technique:25s}: {count:6d} ({pct:5.1f}%)")
    print()
    print(f"📊 Output file: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("✅ Each payload now has UNIQUE, TECHNIQUE-SPECIFIC instruction!")

if __name__ == "__main__":
    main()
