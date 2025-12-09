"""
Synthetic Data Quality Analysis

Purpose: Analyze quality metrics of synthetic LLM-generated payloads for thesis
- Diversity: Unique payloads, technique coverage
- Validity: Successful bypass rate, WAF interaction success
- Bias: Attack type distribution balance
- Complexity: Payload length, obfuscation depth
- Comparison with baseline/heuristic payloads

Outputs:
- reports/dataset_analysis/synthetic_quality_analysis.json
- reports/dataset_analysis/figures/quality_metrics.png
"""

import json
import os
from collections import Counter
from typing import Dict, List, Set
import statistics

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False

# Paths
PHASE1_BALANCED = "data/processed/phase1_balanced_10k.jsonl"
PHASE2_FINAL = "data/processed/phase2_with_replay_22k.jsonl"
OUTPUT_DIR = "reports/dataset_analysis"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_quality_analysis.json")

def load_dataset(filepath: str) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def extract_technique_from_prompt(item: Dict) -> str:
    """Extract technique from user message if not in top-level field."""
    if 'technique' in item:
        return item['technique']
    
    if 'messages' in item:
        for msg in item['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                for line in content.split('\n'):
                    if line.startswith('Technique:'):
                        return line.replace('Technique:', '').strip()
    return 'Unknown'

def infer_attack_type(technique: str, item: Dict) -> str:
    """Infer attack_type from prompt or technique."""
    if 'attack_type' in item:
        return item['attack_type']
    
    if 'messages' in item:
        for msg in item['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                for line in content.split('\n'):
                    if line.startswith('Target:'):
                        target = line.upper()
                        if 'SQLI' in target or 'SQL' in target:
                            return 'SQLI'
                        elif 'XSS' in target:
                            return 'XSS'
                        elif 'OS_INJECTION' in target:
                            return 'OS_INJECTION'
    
    technique_upper = technique.upper()
    if 'SQLI' in technique_upper or 'SQL' in technique_upper:
        return 'SQLI'
    elif 'XSS' in technique_upper:
        return 'XSS'
    elif 'OS_INJECTION' in technique_upper or 'COMMAND' in technique_upper:
        return 'OS_INJECTION'
    return 'Unknown'

def extract_payload(item: Dict) -> str:
    """Extract payload from item."""
    if 'messages' in item:
        for msg in item['messages']:
            if msg.get('role') == 'assistant':
                return msg.get('content', '')
    return item.get('payload', '')

def analyze_diversity(data: List[Dict], name: str) -> Dict:
    """Analyze payload diversity metrics."""
    print(f"\n  Analyzing diversity for {name}...")
    
    payloads = [extract_payload(item) for item in data]
    techniques = [extract_technique_from_prompt(item) for item in data]
    attack_types = [infer_attack_type(t, item) for item, t in zip(data, techniques)]
    
    # Unique payloads
    unique_payloads = len(set(payloads))
    uniqueness_rate = unique_payloads / len(payloads) * 100 if payloads else 0
    
    # Technique coverage
    unique_techniques = len(set(techniques))
    
    # Character set diversity (unique characters used)
    all_chars = set(''.join(payloads))
    
    # Payload length variance
    lengths = [len(p) for p in payloads if p]
    length_variance = statistics.variance(lengths) if len(lengths) > 1 else 0
    
    return {
        'total_samples': len(data),
        'unique_payloads': unique_payloads,
        'uniqueness_rate': round(uniqueness_rate, 2),
        'unique_techniques': unique_techniques,
        'unique_characters': len(all_chars),
        'length_variance': round(length_variance, 2),
        'attack_type_distribution': dict(Counter(attack_types))
    }

def analyze_validity(data: List[Dict], name: str) -> Dict:
    """Analyze payload validity (WAF bypass success)."""
    print(f"\n  Analyzing validity for {name}...")
    
    # Count PASSED vs BLOCKED
    results = []
    for item in data:
        if 'result' in item:
            results.append(item['result'])
        elif 'status' in item:
            results.append(item['status'])
    
    result_counts = Counter(results)
    
    # Bypass success rate
    total_with_result = len(results)
    passed_count = result_counts.get('passed', 0) + result_counts.get('PASSED', 0)
    bypass_rate = passed_count / total_with_result * 100 if total_with_result > 0 else 0
    
    return {
        'total_evaluated': total_with_result,
        'passed': passed_count,
        'blocked': result_counts.get('blocked', 0) + result_counts.get('BLOCKED', 0),
        'bypass_success_rate': round(bypass_rate, 2),
        'result_distribution': dict(result_counts)
    }

def analyze_complexity(data: List[Dict], name: str) -> Dict:
    """Analyze payload complexity metrics."""
    print(f"\n  Analyzing complexity for {name}...")
    
    payloads = [extract_payload(item) for item in data]
    
    # Length statistics
    lengths = [len(p) for p in payloads if p]
    
    # Encoding depth (count of % characters as proxy for URL encoding layers)
    encoding_depths = [p.count('%') for p in payloads if p]
    
    # Special character density
    special_chars = set('%<>"\';(){}[]|&$#@!*+-=/')
    special_densities = [
        sum(1 for c in p if c in special_chars) / len(p) * 100 if len(p) > 0 else 0
        for p in payloads
    ]
    
    return {
        'avg_payload_length': round(statistics.mean(lengths), 2) if lengths else 0,
        'median_payload_length': round(statistics.median(lengths), 2) if lengths else 0,
        'max_payload_length': max(lengths) if lengths else 0,
        'min_payload_length': min(lengths) if lengths else 0,
        'avg_encoding_depth': round(statistics.mean(encoding_depths), 2) if encoding_depths else 0,
        'avg_special_char_density': round(statistics.mean(special_densities), 2) if special_densities else 0,
    }

def analyze_bias(data: List[Dict], name: str) -> Dict:
    """Analyze potential biases in synthetic data."""
    print(f"\n  Analyzing bias for {name}...")
    
    techniques = [extract_technique_from_prompt(item) for item in data]
    attack_types = [infer_attack_type(t, item) for item, t in zip(data, techniques)]
    
    # Attack type balance
    attack_counts = Counter(attack_types)
    total = sum(attack_counts.values())
    attack_balance = {
        k: {
            'count': v,
            'percentage': round(v / total * 100, 2)
        }
        for k, v in attack_counts.items()
    }
    
    # Technique concentration (top techniques dominating)
    technique_counts = Counter(techniques)
    top_10_count = sum(count for _, count in technique_counts.most_common(10))
    concentration_rate = top_10_count / total * 100 if total > 0 else 0
    
    # Entropy (Shannon entropy for technique distribution)
    import math
    entropy = 0
    for count in technique_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return {
        'attack_type_balance': attack_balance,
        'top_10_concentration_rate': round(concentration_rate, 2),
        'technique_entropy': round(entropy, 2),
        'max_entropy_possible': round(math.log2(len(technique_counts)), 2) if len(technique_counts) > 0 else 0
    }

def plot_quality_metrics(phase1_metrics: Dict, phase2_metrics: Dict):
    """Create quality comparison visualization."""
    if not VISUALIZATION_ENABLED:
        return
    
    print("\n  Generating quality metrics visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Synthetic Data Quality Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 1. Diversity metrics
    ax = axes[0, 0]
    metrics = ['Uniqueness\nRate (%)', 'Unique\nTechniques', 'Technique\nEntropy']
    phase1_vals = [
        phase1_metrics['diversity']['uniqueness_rate'],
        phase1_metrics['diversity']['unique_techniques'],
        phase1_metrics['bias']['technique_entropy']
    ]
    phase2_vals = [
        phase2_metrics['diversity']['uniqueness_rate'],
        phase2_metrics['diversity']['unique_techniques'],
        phase2_metrics['bias']['technique_entropy']
    ]
    
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], phase1_vals, width, label='Phase 1', alpha=0.8, color='#3498db')
    ax.bar([i + width/2 for i in x], phase2_vals, width, label='Phase 2', alpha=0.8, color='#e74c3c')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Diversity Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Validity (Bypass success rate)
    ax = axes[0, 1]
    categories = ['Phase 1', 'Phase 2']
    bypass_rates = [
        phase1_metrics['validity']['bypass_success_rate'],
        phase2_metrics['validity']['bypass_success_rate']
    ]
    colors = ['#2ecc71', '#f39c12']
    bars = ax.bar(categories, bypass_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Bypass Success Rate (%)', fontweight='bold')
    ax.set_title('WAF Bypass Success', fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, bypass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Complexity metrics
    ax = axes[1, 0]
    metrics = ['Avg Length', 'Avg Encoding\nDepth', 'Special Char\nDensity (%)']
    phase1_vals = [
        phase1_metrics['complexity']['avg_payload_length'],
        phase1_metrics['complexity']['avg_encoding_depth'],
        phase1_metrics['complexity']['avg_special_char_density']
    ]
    phase2_vals = [
        phase2_metrics['complexity']['avg_payload_length'],
        phase2_metrics['complexity']['avg_encoding_depth'],
        phase2_metrics['complexity']['avg_special_char_density']
    ]
    
    x = range(len(metrics))
    ax.bar([i - width/2 for i in x], phase1_vals, width, label='Phase 1', alpha=0.8, color='#9b59b6')
    ax.bar([i + width/2 for i in x], phase2_vals, width, label='Phase 2', alpha=0.8, color='#1abc9c')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Complexity Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Attack type balance
    ax = axes[1, 1]
    attack_types = ['SQLI', 'XSS', 'OS_INJECTION']
    phase1_percentages = [
        phase1_metrics['bias']['attack_type_balance'].get(at, {}).get('percentage', 0)
        for at in attack_types
    ]
    phase2_percentages = [
        phase2_metrics['bias']['attack_type_balance'].get(at, {}).get('percentage', 0)
        for at in attack_types
    ]
    
    x = range(len(attack_types))
    ax.bar([i - width/2 for i in x], phase1_percentages, width, label='Phase 1', alpha=0.8, color='#34495e')
    ax.bar([i + width/2 for i in x], phase2_percentages, width, label='Phase 2', alpha=0.8, color='#e67e22')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Attack Type Distribution', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, "quality_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [SUCCESS] Saved: {output_path}")

def main():
    """Main analysis function."""
    print("="*80)
    print("SYNTHETIC DATA QUALITY ANALYSIS")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    phase1_data = load_dataset(PHASE1_BALANCED)
    phase2_data = load_dataset(PHASE2_FINAL)
    print(f"  Phase 1: {len(phase1_data):,} samples")
    print(f"  Phase 2: {len(phase2_data):,} samples")
    
    # Analyze Phase 1
    print("\n" + "="*80)
    print("PHASE 1 QUALITY ANALYSIS")
    print("="*80)
    
    phase1_diversity = analyze_diversity(phase1_data, "Phase 1")
    phase1_validity = analyze_validity(phase1_data, "Phase 1")
    phase1_complexity = analyze_complexity(phase1_data, "Phase 1")
    phase1_bias = analyze_bias(phase1_data, "Phase 1")
    
    # Analyze Phase 2
    print("\n" + "="*80)
    print("PHASE 2 QUALITY ANALYSIS")
    print("="*80)
    
    phase2_diversity = analyze_diversity(phase2_data, "Phase 2")
    phase2_validity = analyze_validity(phase2_data, "Phase 2")
    phase2_complexity = analyze_complexity(phase2_data, "Phase 2")
    phase2_bias = analyze_bias(phase2_data, "Phase 2")
    
    # Compile results
    results = {
        'phase1': {
            'diversity': phase1_diversity,
            'validity': phase1_validity,
            'complexity': phase1_complexity,
            'bias': phase1_bias
        },
        'phase2': {
            'diversity': phase2_diversity,
            'validity': phase2_validity,
            'complexity': phase2_complexity,
            'bias': phase2_bias
        }
    }
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Analysis saved to: {OUTPUT_FILE}")
    
    # Generate visualization
    if VISUALIZATION_ENABLED:
        plot_quality_metrics(results['phase1'], results['phase2'])
    else:
        print("\n[WARNING] Matplotlib not available. Install with: pip install matplotlib")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nPhase 1 Quality Metrics:")
    print(f"  Uniqueness Rate: {phase1_diversity['uniqueness_rate']:.2f}%")
    print(f"  Unique Techniques: {phase1_diversity['unique_techniques']}")
    print(f"  Bypass Success: {phase1_validity['bypass_success_rate']:.2f}%")
    print(f"  Avg Payload Length: {phase1_complexity['avg_payload_length']:.2f} chars")
    print(f"  Technique Entropy: {phase1_bias['technique_entropy']:.2f}")
    
    print("\nPhase 2 Quality Metrics:")
    print(f"  Uniqueness Rate: {phase2_diversity['uniqueness_rate']:.2f}%")
    print(f"  Unique Techniques: {phase2_diversity['unique_techniques']}")
    print(f"  Bypass Success: {phase2_validity['bypass_success_rate']:.2f}%")
    print(f"  Avg Payload Length: {phase2_complexity['avg_payload_length']:.2f} chars")
    print(f"  Technique Entropy: {phase2_bias['technique_entropy']:.2f}")
    
    print("\n" + "="*80)
    print("[SUCCESS] Quality analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
