"""
Phase 1 Dataset Creation Analysis

Purpose: Analyze Phase 1 dataset creation characteristics for thesis report
- Total samples, techniques, attack types statistics
- Payload length distribution
- Technique frequency analysis
- Character/encoding distribution

Output: reports/dataset_analysis/phase1_analysis.json
"""

import json
import os
from collections import Counter
from typing import Dict, List
import statistics

# Paths
PHASE1_FULL = "data/processed/phase1_passed_only_39k.jsonl"
PHASE1_BALANCED = "data/processed/phase1_balanced_10k.jsonl"
OUTPUT_DIR = "reports/dataset_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase1_analysis.json")

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

def analyze_dataset(data: List[Dict], name: str) -> Dict:
    """Analyze dataset characteristics."""
    print(f"\nAnalyzing {name}...")
    
    # Basic stats
    total_samples = len(data)
    
    # Extract fields
    techniques = [item.get('technique', 'Unknown') for item in data]
    attack_types = [item.get('attack_type', 'Unknown') for item in data]
    
    # Extract payloads from messages format
    payloads = []
    for item in data:
        if 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') == 'assistant':
                    payloads.append(msg.get('content', ''))
        elif 'payload' in item:
            payloads.append(item['payload'])
    
    # Technique distribution
    technique_counts = Counter(techniques)
    unique_techniques = len([t for t in technique_counts if t != 'Unknown'])
    
    # Attack type distribution
    attack_type_counts = Counter(attack_types)
    
    # Payload length stats
    payload_lengths = [len(p) for p in payloads if p]
    
    length_stats = {
        'mean': statistics.mean(payload_lengths) if payload_lengths else 0,
        'median': statistics.median(payload_lengths) if payload_lengths else 0,
        'min': min(payload_lengths) if payload_lengths else 0,
        'max': max(payload_lengths) if payload_lengths else 0,
        'p25': statistics.quantiles(payload_lengths, n=4)[0] if len(payload_lengths) > 1 else 0,
        'p75': statistics.quantiles(payload_lengths, n=4)[2] if len(payload_lengths) > 1 else 0,
        'p95': statistics.quantiles(payload_lengths, n=20)[18] if len(payload_lengths) > 1 else 0,
        'p99': statistics.quantiles(payload_lengths, n=100)[98] if len(payload_lengths) > 1 else 0,
    }
    
    # Top 20 techniques
    top_20_techniques = dict(technique_counts.most_common(20))
    
    analysis = {
        'dataset_name': name,
        'total_samples': total_samples,
        'unique_techniques': unique_techniques,
        'attack_type_distribution': dict(attack_type_counts),
        'payload_length_stats': length_stats,
        'top_20_techniques': top_20_techniques,
        'technique_coverage': {
            'total_unique': unique_techniques,
            'total_occurrences': sum(technique_counts.values()),
        }
    }
    
    return analysis

def main():
    """Main analysis function."""
    print("="*80)
    print("PHASE 1 DATASET CREATION ANALYSIS")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    phase1_full = load_dataset(PHASE1_FULL)
    phase1_balanced = load_dataset(PHASE1_BALANCED)
    
    # Analyze both
    full_analysis = analyze_dataset(phase1_full, "Phase 1 Full (39k)")
    balanced_analysis = analyze_dataset(phase1_balanced, "Phase 1 Balanced (10k)")
    
    # Combine results
    results = {
        'phase1_full': full_analysis,
        'phase1_balanced': balanced_analysis,
        'comparison': {
            'technique_coverage_full': full_analysis['unique_techniques'],
            'technique_coverage_balanced': balanced_analysis['unique_techniques'],
            'coverage_maintained': full_analysis['unique_techniques'] == balanced_analysis['unique_techniques']
        }
    }
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Analysis saved to: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nPhase 1 Full:")
    print(f"  Samples: {full_analysis['total_samples']:,}")
    print(f"  Techniques: {full_analysis['unique_techniques']}")
    print(f"  Attack Types: {full_analysis['attack_type_distribution']}")
    print(f"  Avg Payload Length: {full_analysis['payload_length_stats']['mean']:.1f} chars")
    
    print(f"\nPhase 1 Balanced:")
    print(f"  Samples: {balanced_analysis['total_samples']:,}")
    print(f"  Techniques: {balanced_analysis['unique_techniques']}")
    print(f"  Coverage Maintained: {results['comparison']['coverage_maintained']}")
    
    print(f"\nTop 5 Techniques:")
    for i, (tech, count) in enumerate(list(full_analysis['top_20_techniques'].items())[:5], 1):
        print(f"  {i}. {tech}: {count:,} samples")

if __name__ == "__main__":
    main()
