"""
Quality evaluation script for WAF bypass payloads
Evaluates generated payloads based on:
1. Syntax validity (60%) - Is it valid XSS/SQLi?
2. Semantic correctness (30%) - Does it follow instruction?
3. Novelty/Creativity (10%) - Is it different from training data?
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class PayloadQualityEvaluator:
    """Evaluate quality of generated attack payloads"""
    
    def __init__(self):
        # XSS patterns
        self.xss_tags = re.compile(r'<[a-zA-Z][^>]*>', re.IGNORECASE)
        self.xss_events = re.compile(r'\bon[a-z]+\s*=', re.IGNORECASE)
        self.xss_script = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        self.xss_javascript = re.compile(r'javascript:', re.IGNORECASE)
        
        # SQLi patterns
        self.sqli_keywords = re.compile(
            r'\b(SELECT|UNION|WHERE|FROM|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|'
            r'ORDER\s+BY|GROUP\s+BY|HAVING|AND|OR|CONCAT|SUBSTRING|LOAD_FILE|'
            r'information_schema|database\(\)|version\(\)|user\(\)|sleep\(\))\b',
            re.IGNORECASE
        )
        self.sqli_operators = re.compile(r"('|--|#|/\*|\*/|;|\|\|)")
        self.sqli_encoding = re.compile(r'(%[0-9a-fA-F]{2}|0x[0-9a-fA-F]+)')
        
    def evaluate_xss_syntax(self, payload: str) -> Tuple[float, List[str]]:
        """
        Evaluate XSS payload syntax validity
        Returns: (score 0-1, list of reasons)
        """
        if not payload or not payload.strip():
            return 0.0, ["Empty payload"]
        
        score = 0.0
        reasons = []
        
        # Check for HTML tags (40 points)
        if self.xss_tags.search(payload):
            score += 0.4
            reasons.append("✓ Contains HTML tags")
        else:
            reasons.append("✗ No HTML tags")
        
        # Check for event handlers (30 points)
        if self.xss_events.search(payload):
            score += 0.3
            reasons.append("✓ Contains event handlers")
        else:
            # Check for script tags or javascript: (alternative)
            if self.xss_script.search(payload) or self.xss_javascript.search(payload):
                score += 0.3
                reasons.append("✓ Contains script/javascript")
            else:
                reasons.append("✗ No event handlers or scripts")
        
        # Check for alert/prompt/confirm (common XSS proof) (20 points)
        if re.search(r'\b(alert|prompt|confirm|eval)\s*\(', payload, re.IGNORECASE):
            score += 0.2
            reasons.append("✓ Contains XSS proof function")
        
        # Check for basic syntax validity (10 points)
        open_tags = len(re.findall(r'<[a-zA-Z][^/>]*(?<!/)>', payload))
        close_tags = len(re.findall(r'</[a-zA-Z][^>]*>', payload))
        self_closing = len(re.findall(r'<[^>]*/>', payload))
        
        if open_tags > 0 and (close_tags > 0 or self_closing > 0):
            score += 0.1
            reasons.append("✓ Has closing tags")
        
        return min(score, 1.0), reasons
    
    def evaluate_sqli_syntax(self, payload: str) -> Tuple[float, List[str]]:
        """
        Evaluate SQLi payload syntax validity
        Returns: (score 0-1, list of reasons)
        """
        if not payload or not payload.strip():
            return 0.0, ["Empty payload"]
        
        score = 0.0
        reasons = []
        
        # Check for SQL keywords (50 points)
        keywords_found = self.sqli_keywords.findall(payload)
        if keywords_found:
            score += min(0.5, len(set(keywords_found)) * 0.1)
            reasons.append(f"✓ Contains SQL keywords: {', '.join(list(set(keywords_found))[:3])}")
        else:
            reasons.append("✗ No SQL keywords")
        
        # Check for SQL operators/injection chars (30 points)
        operators_found = self.sqli_operators.findall(payload)
        if operators_found:
            score += 0.3
            reasons.append(f"✓ Contains SQL operators: {', '.join(list(set(operators_found))[:3])}")
        else:
            reasons.append("✗ No SQL operators")
        
        # Check for encoding (10 points)
        if self.sqli_encoding.search(payload):
            score += 0.1
            reasons.append("✓ Uses URL/hex encoding")
        
        # Check for WordPress plugin path (specific to dataset) (10 points)
        if re.search(r'wp-content/plugins/', payload, re.IGNORECASE):
            score += 0.1
            reasons.append("✓ WordPress plugin path")
        
        return min(score, 1.0), reasons
    
    def detect_attack_type(self, instruction: str, payload: str) -> str:
        """Detect whether this is XSS or SQLi based on instruction and payload"""
        instruction_lower = instruction.lower()
        payload_lower = payload.lower()
        
        # Check instruction first
        if any(keyword in instruction_lower for keyword in ['xss', 'cross-site', 'script', 'javascript']):
            return 'xss'
        if any(keyword in instruction_lower for keyword in ['sql', 'sqli', 'injection', 'union', 'database']):
            return 'sqli'
        
        # Fallback to payload analysis
        xss_score, _ = self.evaluate_xss_syntax(payload)
        sqli_score, _ = self.evaluate_sqli_syntax(payload)
        
        return 'xss' if xss_score > sqli_score else 'sqli'
    
    def evaluate_semantic_correctness(self, instruction: str, payload: str, attack_type: str) -> Tuple[float, List[str]]:
        """
        Evaluate if payload follows instruction semantics
        Returns: (score 0-1, list of reasons)
        """
        if not payload or not payload.strip():
            return 0.0, ["Empty payload"]
        
        score = 0.0
        reasons = []
        instruction_lower = instruction.lower()
        payload_lower = payload.lower()
        
        # Check if attack type matches (30 points)
        detected_type = self.detect_attack_type(instruction, payload)
        if detected_type == attack_type:
            score += 0.3
            reasons.append(f"✓ Correct attack type: {attack_type.upper()}")
        else:
            reasons.append(f"✗ Wrong attack type: expected {attack_type}, got {detected_type}")
        
        # Check for specific techniques mentioned (40 points)
        technique_keywords = {
            'union': r'\bUNION\b',
            'boolean': r'\b(AND|OR)\b.*?=',
            'time-based': r'\b(SLEEP|BENCHMARK|WAITFOR)\s*\(',
            'error-based': r'\b(CONCAT|FLOOR|RAND|EXTRACTVALUE)\b',
            'event': r'\bon[a-z]+\s*=',
            'attribute': r'<[^>]+\s+[a-z]+\s*=',
            'dom': r'(document\.|window\.|location\.)',
        }
        
        matched_techniques = []
        for keyword, pattern in technique_keywords.items():
            if keyword in instruction_lower and re.search(pattern, payload, re.IGNORECASE):
                matched_techniques.append(keyword)
        
        if matched_techniques:
            score += min(0.4, len(matched_techniques) * 0.15)
            reasons.append(f"✓ Matches technique: {', '.join(matched_techniques)}")
        
        # Check constraints (20 points)
        # Length constraint
        max_len_match = re.search(r'maximum\s+(\d+)\s+characters?', instruction_lower)
        if max_len_match:
            max_len = int(max_len_match.group(1))
            if len(payload) <= max_len:
                score += 0.1
                reasons.append(f"✓ Length OK: {len(payload)}/{max_len}")
            else:
                reasons.append(f"✗ Length exceeded: {len(payload)}/{max_len}")
        
        # WAF bypass constraint
        if 'bypass waf' in instruction_lower or 'bypass filter' in instruction_lower:
            # Check for encoding or obfuscation
            if self.sqli_encoding.search(payload) or re.search(r'/\*.*?\*/', payload):
                score += 0.1
                reasons.append("✓ Uses encoding/obfuscation")
        
        return min(score, 1.0), reasons
    
    def evaluate_novelty(self, payload: str, reference: str) -> Tuple[float, List[str]]:
        """
        Evaluate novelty compared to reference (training data)
        Returns: (score 0-1, list of reasons)
        """
        if not payload or not payload.strip():
            return 0.0, ["Empty payload"]
        
        if not reference or not reference.strip():
            return 1.0, ["No reference to compare"]
        
        score = 1.0
        reasons = []
        
        payload_clean = payload.strip().lower()
        reference_clean = reference.strip().lower()
        
        # Exact match (score 0)
        if payload_clean == reference_clean:
            score = 0.0
            reasons.append("✗ Exact copy of reference")
            return score, reasons
        
        # High similarity (score 0.3)
        if payload_clean in reference_clean or reference_clean in payload_clean:
            score = 0.3
            reasons.append("⚠ Substring of reference")
            return score, reasons
        
        # Token-level similarity
        payload_tokens = set(re.findall(r'\w+', payload_clean))
        reference_tokens = set(re.findall(r'\w+', reference_clean))
        
        if payload_tokens and reference_tokens:
            jaccard = len(payload_tokens & reference_tokens) / len(payload_tokens | reference_tokens)
            if jaccard > 0.8:
                score = 0.5
                reasons.append(f"⚠ High token similarity: {jaccard:.2f}")
            elif jaccard > 0.5:
                score = 0.7
                reasons.append(f"✓ Moderate variation: {jaccard:.2f}")
            else:
                score = 1.0
                reasons.append(f"✓ Novel payload: {jaccard:.2f}")
        
        return score, reasons
    
    def evaluate_sample(self, instruction: str, generated: str, reference: str = "") -> Dict:
        """
        Evaluate a single sample
        Returns comprehensive quality metrics
        """
        # Detect attack type
        attack_type = self.detect_attack_type(instruction, generated or reference)
        
        # Syntax validity (60%)
        if attack_type == 'xss':
            syntax_score, syntax_reasons = self.evaluate_xss_syntax(generated)
        else:
            syntax_score, syntax_reasons = self.evaluate_sqli_syntax(generated)
        
        # Semantic correctness (30%)
        semantic_score, semantic_reasons = self.evaluate_semantic_correctness(instruction, generated, attack_type)
        
        # Novelty (10%)
        novelty_score, novelty_reasons = self.evaluate_novelty(generated, reference)
        
        # Weighted total
        total_score = (
            syntax_score * 0.6 +
            semantic_score * 0.3 +
            novelty_score * 0.1
        )
        
        return {
            'attack_type': attack_type,
            'syntax_score': syntax_score,
            'syntax_reasons': syntax_reasons,
            'semantic_score': semantic_score,
            'semantic_reasons': semantic_reasons,
            'novelty_score': novelty_score,
            'novelty_reasons': novelty_reasons,
            'total_score': total_score,
            'has_output': bool(generated and generated.strip())
        }


def parse_evaluation_file(eval_file: Path) -> List[Dict]:
    """Parse evaluation output file and extract samples"""
    samples = []
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by sample separator
    sample_blocks = re.split(r'={80}\nSAMPLE \d+\n', content)
    
    for block in sample_blocks[1:]:  # Skip header
        # Extract prompt
        prompt_match = re.search(r'PROMPT:\n(.*?)\nGENERATED:', block, re.DOTALL)
        if not prompt_match:
            continue
        
        prompt = prompt_match.group(1).strip()
        
        # Extract generated output
        generated_match = re.search(r'GENERATED:\n(.*?)(?:\n---|$)', block, re.DOTALL)
        generated = generated_match.group(1).strip() if generated_match else ""
        
        # Extract reference (if available)
        reference_match = re.search(r'Reference payload.*?:\n(.*?)(?:\n<no reference|$)', block, re.DOTALL)
        reference = reference_match.group(1).strip() if reference_match else ""
        
        # Extract instruction from prompt (model-agnostic)
        # Remove format-specific tokens
        instruction = re.sub(r'<[^>]+>', '', prompt)
        instruction = re.sub(r'<\|[^|]+\|>', '', instruction)
        instruction = instruction.strip()
        
        samples.append({
            'instruction': instruction,
            'generated': generated,
            'reference': reference
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate quality of generated payloads")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to evaluation output file")
    parser.add_argument("--output", type=str, help="Output JSON file for detailed results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results for each sample")
    args = parser.parse_args()
    
    eval_file = Path(args.eval_file)
    if not eval_file.exists():
        print(f"Error: File not found: {eval_file}")
        return
    
    # Parse samples
    print(f"📊 Parsing {eval_file.name}...")
    samples = parse_evaluation_file(eval_file)
    print(f"Found {len(samples)} samples\n")
    
    # Evaluate
    evaluator = PayloadQualityEvaluator()
    results = []
    
    stats = {
        'total': len(samples),
        'empty': 0,
        'xss': 0,
        'sqli': 0,
        'syntax_scores': [],
        'semantic_scores': [],
        'novelty_scores': [],
        'total_scores': []
    }
    
    for i, sample in enumerate(samples):
        result = evaluator.evaluate_sample(
            sample['instruction'],
            sample['generated'],
            sample['reference']
        )
        
        result['sample_id'] = i
        result['instruction'] = sample['instruction']
        result['generated'] = sample['generated']
        result['reference'] = sample['reference']
        results.append(result)
        
        # Update stats
        if not result['has_output']:
            stats['empty'] += 1
        else:
            stats[result['attack_type']] += 1
            stats['syntax_scores'].append(result['syntax_score'])
            stats['semantic_scores'].append(result['semantic_score'])
            stats['novelty_scores'].append(result['novelty_score'])
            stats['total_scores'].append(result['total_score'])
        
        # Print verbose
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"SAMPLE {i}")
            print(f"Instruction: {sample['instruction'][:100]}...")
            print(f"Generated: {sample['generated'][:100]}..." if sample['generated'] else "Generated: (empty)")
            print(f"\nType: {result['attack_type'].upper()}")
            print(f"Syntax: {result['syntax_score']:.2f} - {', '.join(result['syntax_reasons'])}")
            print(f"Semantic: {result['semantic_score']:.2f} - {', '.join(result['semantic_reasons'])}")
            print(f"Novelty: {result['novelty_score']:.2f} - {', '.join(result['novelty_reasons'])}")
            print(f"TOTAL: {result['total_score']:.2f}")
    
    # Print summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    print(f"Total samples: {stats['total']}")
    print(f"Empty outputs: {stats['empty']} ({stats['empty']/stats['total']*100:.1f}%)")
    print(f"XSS payloads: {stats['xss']}")
    print(f"SQLi payloads: {stats['sqli']}")
    
    if stats['total_scores']:
        print(f"\nAverage Scores (non-empty only):")
        print(f"  Syntax:   {sum(stats['syntax_scores'])/len(stats['syntax_scores']):.3f}")
        print(f"  Semantic: {sum(stats['semantic_scores'])/len(stats['semantic_scores']):.3f}")
        print(f"  Novelty:  {sum(stats['novelty_scores'])/len(stats['novelty_scores']):.3f}")
        print(f"  TOTAL:    {sum(stats['total_scores'])/len(stats['total_scores']):.3f}")
        
        # Quality distribution
        excellent = sum(1 for s in stats['total_scores'] if s >= 0.8)
        good = sum(1 for s in stats['total_scores'] if 0.6 <= s < 0.8)
        fair = sum(1 for s in stats['total_scores'] if 0.4 <= s < 0.6)
        poor = sum(1 for s in stats['total_scores'] if s < 0.4)
        
        print(f"\nQuality Distribution:")
        print(f"  Excellent (≥0.8): {excellent} ({excellent/len(stats['total_scores'])*100:.1f}%)")
        print(f"  Good (0.6-0.8):   {good} ({good/len(stats['total_scores'])*100:.1f}%)")
        print(f"  Fair (0.4-0.6):   {fair} ({fair/len(stats['total_scores'])*100:.1f}%)")
        print(f"  Poor (<0.4):      {poor} ({poor/len(stats['total_scores'])*100:.1f}%)")
    
    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': {
                    'total': stats['total'],
                    'empty': stats['empty'],
                    'xss': stats['xss'],
                    'sqli': stats['sqli'],
                    'avg_syntax': sum(stats['syntax_scores'])/len(stats['syntax_scores']) if stats['syntax_scores'] else 0,
                    'avg_semantic': sum(stats['semantic_scores'])/len(stats['semantic_scores']) if stats['semantic_scores'] else 0,
                    'avg_novelty': sum(stats['novelty_scores'])/len(stats['novelty_scores']) if stats['novelty_scores'] else 0,
                    'avg_total': sum(stats['total_scores'])/len(stats['total_scores']) if stats['total_scores'] else 0,
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
