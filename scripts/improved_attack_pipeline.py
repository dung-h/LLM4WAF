#!/usr/bin/env python3
"""
IMPROVED ATTACK PIPELINE WITH DYNAMIC WAF-AWARE PROMPTING
- Uses probe results to build SPECIFIC evasion prompts
- Leverages WAF-aware prompt builder for dynamic generation
"""

import sys
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from waf_aware_prompt_builder import WAFAwarePromptBuilder

def run_generation_waf_aware(probe_results, attack_history, dbms_type='MySQL'):
    """Generate payloads using WAF-aware prompting"""
    
    blocked_keywords = probe_results.get("blocked_sqli_keywords", [])
    num_payloads = 3
    
    print("[IMPROVED] Building WAF-aware prompt with evasion techniques...")
    print(f"[IMPROVED] Blocked keywords detected: {blocked_keywords}")
    
    # Build dynamic evasion-focused prompt
    prompt = WAFAwarePromptBuilder.build_evasion_focused_prompt(
        blocked_keywords,
        attack_type='sqli',
        num_payloads=num_payloads
    )
    
    print(f"[IMPROVED] Generated WAF-aware prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    return prompt

def run_generation_encoding_focused(probe_results, attack_history):
    """Generate payloads specifically using encoding techniques"""
    
    blocked_keywords = probe_results.get("blocked_sqli_keywords", [])
    
    print("[ENCODING] Building encoding/obfuscation focused prompt...")
    
    prompt = WAFAwarePromptBuilder.build_encoding_hint_prompt(
        blocked_keywords,
        num_payloads=3
    )
    
    print(f"[ENCODING] Generated prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    return prompt

if __name__ == "__main__":
    # Test with example probing results
    probe_results = {
        "blocked_sqli_keywords": ["OR", "UNION", "SELECT", "SLEEP"],
        "blocked_xss_tags": ["script", "alert"]
    }
    
    print("TEST 1: WAF-Aware Prompt Generation")
    prompt1 = run_generation_waf_aware(probe_results, [], 'MySQL')
    
    print("\n" + "=" * 80 + "\n")
    
    print("TEST 2: Encoding-Focused Prompt Generation")
    prompt2 = run_generation_encoding_focused(probe_results, [])
