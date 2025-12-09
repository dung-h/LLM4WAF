"""
Phase 2 Dataset Creation Analysis

Purpose: Analyze Phase 2 dataset construction process for thesis report
- Composition analysis (observations + replay buffer)
- Technique evolution (Phase 1 → Phase 2)
- Payload complexity comparison
- Format structure analysis

Output: reports/dataset_analysis/phase2_analysis.json
"""

import json
import os
from collections import Counter
from typing import Dict, List
import statistics

# Paths
PHASE1_BALANCED = "data/processed/phase1_balanced_10k.jsonl"
PHASE2_OBSERVATIONS = "data/processed/phase2_observations_20k.jsonl"
PHASE2_FINAL = "data/processed/phase2_with_replay_22k.jsonl"
OUTPUT_DIR = "reports/dataset_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_analysis.json")

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
    
    # Extract from user message (Phase 2 observations format)
    if 'messages' in item:
        for msg in item['messages']:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Look for "Technique: XXX" pattern
                for line in content.split('\n'):
                    if line.startswith('Technique:'):
                        return line.replace('Technique:', '').strip()
    
    return 'Unknown'

def infer_attack_type(technique: str, item: Dict) -> str:
    """Infer attack_type from technique name if not present."""
    # Check if attack_type already exists
    if 'attack_type' in item:
        return item['attack_type']
    
    # Check user message for "Target: SQLI/XSS/..." pattern
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
    
    # Fallback: infer from technique name
    technique_upper = technique.upper()
    if 'SQLI' in technique_upper or 'SQL' in technique_upper:
        return 'SQLI'
    elif 'XSS' in technique_upper:
        return 'XSS'
    elif 'OS_INJECTION' in technique_upper or 'COMMAND' in technique_upper:
        return 'OS_INJECTION'
    else:
        return 'Unknown'

def analyze_dataset(data: List[Dict], name: str) -> Dict:
    """Analyze dataset characteristics."""
    print(f"\nAnalyzing {name}...")
    
    total_samples = len(data)
    
    # Extract fields with technique/attack_type inference
    techniques = [extract_technique_from_prompt(item) for item in data]
    attack_types = [infer_attack_type(extract_technique_from_prompt(item), item) for item in data]
    
    # Extract payloads
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
        'p95': statistics.quantiles(payload_lengths, n=20)[18] if len(payload_lengths) > 1 else 0,
        'p99': statistics.quantiles(payload_lengths, n=100)[98] if len(payload_lengths) > 1 else 0,
    }
    
    # Top 20 techniques
    top_20_techniques = dict(technique_counts.most_common(20))
    
    return {
        'dataset_name': name,
        'total_samples': total_samples,
        'unique_techniques': unique_techniques,
        'attack_type_distribution': dict(attack_type_counts),
        'payload_length_stats': length_stats,
        'top_20_techniques': top_20_techniques
    }

def main():
    """Main analysis function."""
    print("="*80)
    print("PHASE 2 DATASET CREATION ANALYSIS")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    phase1_balanced = load_dataset(PHASE1_BALANCED)
    phase2_obs = load_dataset(PHASE2_OBSERVATIONS)
    phase2_final = load_dataset(PHASE2_FINAL)
    
    # Analyze
    phase1_analysis = analyze_dataset(phase1_balanced, "Phase 1 Balanced (10k)")
    phase2_obs_analysis = analyze_dataset(phase2_obs, "Phase 2 Observations (20k)")
    phase2_final_analysis = analyze_dataset(phase2_final, "Phase 2 Final (22k)")
    
    # Calculate technique evolution
    phase1_techniques = phase1_analysis['unique_techniques']
    phase2_techniques = phase2_final_analysis['unique_techniques']
    new_techniques = phase2_techniques - phase1_techniques
    
    # Payload complexity comparison
    phase1_avg_len = phase1_analysis['payload_length_stats']['mean']
    phase2_avg_len = phase2_final_analysis['payload_length_stats']['mean']
    complexity_increase = ((phase2_avg_len - phase1_avg_len) / phase1_avg_len) * 100 if phase1_avg_len > 0 else 0
    
    # Combine results
    results = {
        'phase1_balanced': phase1_analysis,
        'phase2_observations': phase2_obs_analysis,
        'phase2_final': phase2_final_analysis,
        'composition': {
            'phase2_observations_count': phase2_obs_analysis['total_samples'],
            'phase1_replay_count': phase2_final_analysis['total_samples'] - phase2_obs_analysis['total_samples'],
            'replay_percentage': ((phase2_final_analysis['total_samples'] - phase2_obs_analysis['total_samples']) / phase2_final_analysis['total_samples']) * 100
        },
        'evolution': {
            'phase1_techniques': phase1_techniques,
            'phase2_techniques': phase2_techniques,
            'new_techniques_added': new_techniques,
            'technique_growth': f"{new_techniques} new techniques ({(new_techniques/phase1_techniques)*100:.1f}% increase)"
        },
        'complexity': {
            'phase1_avg_payload_length': phase1_avg_len,
            'phase2_avg_payload_length': phase2_avg_len,
            'complexity_increase_percent': complexity_increase
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
    
    print(f"\nDataset Composition:")
    print(f"  Phase 2 Observations: {results['composition']['phase2_observations_count']:,} samples")
    print(f"  Phase 1 Replay: {results['composition']['phase1_replay_count']:,} samples")
    print(f"  Replay Rate: {results['composition']['replay_percentage']:.1f}%")
    
    print(f"\nTechnique Evolution:")
    print(f"  Phase 1: {results['evolution']['phase1_techniques']} unique techniques")
    print(f"  Phase 2: {results['evolution']['phase2_techniques']} unique techniques")
    print(f"  New Techniques: +{results['evolution']['new_techniques_added']}")
    
    print(f"\nPayload Complexity:")
    print(f"  Phase 1 Avg Length: {results['complexity']['phase1_avg_payload_length']:.1f} chars")
    print(f"  Phase 2 Avg Length: {results['complexity']['phase2_avg_payload_length']:.1f} chars")
    print(f"  Increase: +{results['complexity']['complexity_increase_percent']:.1f}%")
    
    print(f"\nAttack Type Distribution (Phase 2):")
    for attack_type, count in phase2_final_analysis['attack_type_distribution'].items():
        percentage = (count / phase2_final_analysis['total_samples']) * 100
        print(f"  {attack_type}: {count:,} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
