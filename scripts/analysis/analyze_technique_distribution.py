"""
Technique Distribution Analysis

Purpose: Comprehensive technique distribution analysis with visualizations for thesis
- Technique frequency distribution (Phase 1 & 2)
- Top 20 techniques bar charts
- Attack type breakdown (pie charts)
- Payload length histograms
- Comparison visualizations

Outputs:
- reports/dataset_analysis/technique_distribution.csv
- reports/dataset_analysis/figures/technique_dist_phase1.png
- reports/dataset_analysis/figures/technique_dist_phase2.png
- reports/dataset_analysis/figures/attack_type_breakdown.png
- reports/dataset_analysis/figures/payload_length_histogram.png
"""

import json
import os
from collections import Counter
from typing import Dict, List
import csv

# Optional: Visualization libraries (install if needed)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    VISUALIZATION_ENABLED = True
except ImportError:
    print("Warning: matplotlib not installed. Visualizations disabled.")
    VISUALIZATION_ENABLED = False

# Paths
PHASE1_BALANCED = "data/processed/phase1_balanced_10k.jsonl"
PHASE2_FINAL = "data/processed/phase2_with_replay_22k.jsonl"
OUTPUT_DIR = "reports/dataset_analysis"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

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

def extract_techniques_and_payloads(data: List[Dict]) -> tuple:
    """Extract techniques, attack types, and payload lengths."""
    techniques = []
    attack_types = []
    payload_lengths = []
    
    for item in data:
        techniques.append(item.get('technique', 'Unknown'))
        attack_types.append(item.get('attack_type', 'Unknown'))
        
        # Extract payload
        payload = ''
        if 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') == 'assistant':
                    payload = msg.get('content', '')
                    break
        elif 'payload' in item:
            payload = item['payload']
        
        if payload:
            payload_lengths.append(len(payload))
    
    return techniques, attack_types, payload_lengths

def save_distribution_csv(technique_counts: Counter, filename: str):
    """Save technique distribution to CSV."""
    csv_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Technique', 'Count', 'Percentage'])
        
        total = sum(technique_counts.values())
        for technique, count in technique_counts.most_common():
            percentage = (count / total) * 100
            writer.writerow([technique, count, f"{percentage:.2f}%"])
    
    print(f"  ✅ Saved: {csv_path}")

def plot_top_techniques(technique_counts: Counter, phase_name: str, filename: str, top_n: int = 20):
    """Plot top N techniques bar chart."""
    if not VISUALIZATION_ENABLED:
        return
    
    top_techniques = dict(technique_counts.most_common(top_n))
    
    plt.figure(figsize=(12, 8))
    plt.barh(list(top_techniques.keys())[::-1], list(top_techniques.values())[::-1])
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Technique', fontsize=12)
    plt.title(f'Top {top_n} Techniques - {phase_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")

def plot_attack_type_breakdown(attack_types: List[str], phase_name: str, filename: str):
    """Plot attack type distribution pie chart."""
    if not VISUALIZATION_ENABLED:
        return
    
    attack_type_counts = Counter(attack_types)
    
    plt.figure(figsize=(10, 8))
    plt.pie(attack_type_counts.values(), labels=attack_type_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title(f'Attack Type Distribution - {phase_name}', fontsize=14, fontweight='bold')
    plt.axis('equal')
    
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")

def plot_payload_length_histogram(lengths: List[int], phase_name: str, filename: str):
    """Plot payload length distribution histogram."""
    if not VISUALIZATION_ENABLED:
        return
    
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Payload Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Payload Length Distribution - {phase_name}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {output_path}")

def main():
    """Main analysis function."""
    print("="*80)
    print("TECHNIQUE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    phase1_data = load_dataset(PHASE1_BALANCED)
    phase2_data = load_dataset(PHASE2_FINAL)
    
    print(f"  Phase 1: {len(phase1_data):,} samples")
    print(f"  Phase 2: {len(phase2_data):,} samples")
    
    # Extract data
    print("\nExtracting techniques and payloads...")
    phase1_techniques, phase1_attacks, phase1_lengths = extract_techniques_and_payloads(phase1_data)
    phase2_techniques, phase2_attacks, phase2_lengths = extract_techniques_and_payloads(phase2_data)
    
    # Count techniques
    phase1_counts = Counter(phase1_techniques)
    phase2_counts = Counter(phase2_techniques)
    
    print(f"\n  Phase 1 unique techniques: {len([t for t in phase1_counts if t != 'Unknown'])}")
    print(f"  Phase 2 unique techniques: {len([t for t in phase2_counts if t != 'Unknown'])}")
    
    # Save CSVs
    print("\nSaving distribution data...")
    save_distribution_csv(phase1_counts, "technique_distribution_phase1.csv")
    save_distribution_csv(phase2_counts, "technique_distribution_phase2.csv")
    
    # Generate visualizations
    if VISUALIZATION_ENABLED:
        print("\nGenerating visualizations...")
        
        # Top 20 techniques
        plot_top_techniques(phase1_counts, "Phase 1 (10k)", "technique_dist_phase1.png")
        plot_top_techniques(phase2_counts, "Phase 2 (22k)", "technique_dist_phase2.png")
        
        # Attack type breakdown
        plot_attack_type_breakdown(phase1_attacks, "Phase 1", "attack_type_phase1.png")
        plot_attack_type_breakdown(phase2_attacks, "Phase 2", "attack_type_phase2.png")
        
        # Payload length histograms
        plot_payload_length_histogram(phase1_lengths, "Phase 1", "payload_length_phase1.png")
        plot_payload_length_histogram(phase2_lengths, "Phase 2", "payload_length_phase2.png")
    else:
        print("\n⚠️  Matplotlib not available. Install with: pip install matplotlib")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nPhase 1 Top 10 Techniques:")
    for i, (tech, count) in enumerate(phase1_counts.most_common(10), 1):
        percentage = (count / len(phase1_techniques)) * 100
        print(f"  {i:2d}. {tech:50s} {count:5d} ({percentage:5.2f}%)")
    
    print(f"\nPhase 2 Top 10 Techniques:")
    for i, (tech, count) in enumerate(phase2_counts.most_common(10), 1):
        percentage = (count / len(phase2_techniques)) * 100
        print(f"  {i:2d}. {tech:50s} {count:5d} ({percentage:5.2f}%)")
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
