import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import seaborn as sns
import glob

# Setup style
plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = "reports/figures"

# --- 1. Performance Comparison (Bar Chart) ---
def plot_performance():
    print("Plotting Performance Comparison...")
    # Data from remote_performance.md
    data = {
        'Model': ['Qwen 7B', 'Qwen 7B', 'Phi-3 Mini', 'Phi-3 Mini', 'Gemma 2 2B', 'Gemma 2 2B', 'Gemma 2 2B'],
        'Phase': ['Phase 1 SFT', 'Phase 2 Reasoning', 'Phase 1 SFT', 'Phase 2 Reasoning', 'Phase 1 SFT', 'Phase 2 Reasoning', 'Phase 3 RL'],
        'Pass Rate (%)': [80.0, 55.0, 70.0, 65.0, 50.0, 65.0, 70.0]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    sns.barplot(data=df, x='Model', y='Pass Rate (%)', hue='Phase', palette='viridis')
    
    plt.title('WAF Evasion Success Rate by Model & Phase', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    plt.ylabel('Pass Rate (%)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Baseline (Random)')
    plt.legend(title='Training Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

# --- 2. Training Loss Curve (Line Chart) ---
def plot_training_loss():
    print("Plotting Training Loss Curves...")
    
    # Paths to trainer_state.json
    paths = {
        "Qwen 7B (Phase 1)": "experiments/remote_adapters/experiments_remote_optimized/phase1_sft_qwen/checkpoint-2668/trainer_state.json",
        "Phi-3 Mini (Phase 2)": "experiments/remote_adapters/experiments_remote_optimized/phase2_reasoning_phi3/checkpoint-626/trainer_state.json",
        "Gemma 2 2B (Phase 2)": "experiments/phase2_gemma2_2b_reasoning/checkpoint-314/trainer_state.json"
    }
    
    plt.figure(figsize=(12, 7))
    
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        history = data['log_history']
        steps = [entry['step'] for entry in history if 'loss' in entry]
        loss = [entry['loss'] for entry in history if 'loss' in entry]
        
        # Normalize steps to % of training completion for comparison
        max_step = max(steps)
        normalized_steps = [s/max_step * 100 for s in steps]
        
        plt.plot(normalized_steps, loss, label=label, linewidth=2)

    plt.title('Training Loss Convergence (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Training Progress (%)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, 'training_loss_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

# --- 3. Technique Effectiveness Heatmap ---
def plot_technique_heatmap():
    print("Plotting Technique Heatmap...")
    
    # Load payload logs
    log_files = glob.glob("eval/payload_details/*.jsonl")
    all_data = []
    
    for file in log_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Normalize model names for cleaner chart
                    model_name = entry['model_name'].replace(" - Phase ", "\nPhase ")
                    
                    all_data.append({
                        'Model': model_name,
                        'Technique': entry['technique'],
                        'Result': 1 if entry['waf_result'] == 'passed' else 0
                    })
                except:
                    continue
                    
    if not all_data:
        print("No payload data found.")
        return

    df = pd.DataFrame(all_data)
    
    # Pivot table: Technique vs Model, values = Pass Rate
    pivot_df = df.pivot_table(index='Technique', columns='Model', values='Result', aggfunc='mean')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt='.0%', vmin=0, vmax=1, linewidths=.5)
    
    plt.title('Attack Technique Success Rate by Model', fontsize=16, fontweight='bold')
    plt.ylabel('Technique', fontsize=12)
    plt.xlabel('Model Variation', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'technique_heatmap.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_performance()
    plot_training_loss()
    plot_technique_heatmap()
