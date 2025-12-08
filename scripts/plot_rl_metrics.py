"""
Plot RL training metrics (baseline and loss) to check convergence.
"""
import matplotlib.pyplot as plt
import numpy as np

# Read metrics
with open('training_metrics.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

baselines = []
losses = []
for line in lines:
    line = line.strip()
    if ',' in line:
        b, l = line.split(',')
        baselines.append(float(b))
        losses.append(float(l))

epochs = list(range(1, len(baselines) + 1))

print(f"Total epochs: {len(epochs)}")
print(f"Baseline range: [{min(baselines):.4f}, {max(baselines):.4f}]")
print(f"Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
print(f"\nFirst 10 baselines: {baselines[:10]}")
print(f"Last 10 baselines: {baselines[-10:]}")
print(f"\nFirst 10 losses: {losses[:10]}")
print(f"Last 10 losses: {losses[-10:]}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Baseline plot
ax1.plot(epochs, baselines, marker='o', markersize=3, linewidth=1, alpha=0.7)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero baseline')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Baseline (Moving Average Reward)')
ax1.set_title('RL Training: Baseline (Reward)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Loss plot
ax2.plot(epochs, losses, marker='o', markersize=3, linewidth=1, alpha=0.7, color='orange')
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Policy Gradient Loss')
ax2.set_title('RL Training: Policy Loss')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('rl_training_metrics.png', dpi=150)
print(f"\n✅ Plot saved to rl_training_metrics.png")

# Calculate statistics
baseline_mean = np.mean(baselines)
baseline_std = np.std(baselines)
baseline_trend_first_half = np.mean(baselines[:len(baselines)//2])
baseline_trend_second_half = np.mean(baselines[len(baselines)//2:])

print(f"\n📊 Statistics:")
print(f"Baseline - Mean: {baseline_mean:.4f}, Std: {baseline_std:.4f}")
print(f"Baseline - First half avg: {baseline_trend_first_half:.4f}")
print(f"Baseline - Second half avg: {baseline_trend_second_half:.4f}")
print(f"Baseline improvement: {baseline_trend_second_half - baseline_trend_first_half:.4f}")

# Check convergence
if baseline_std < 0.1 and abs(baseline_trend_second_half - baseline_trend_first_half) < 0.05:
    print("\n⚠️  Model appears CONVERGED but baseline is LOW - may need more training")
elif baseline_trend_second_half > baseline_trend_first_half + 0.1:
    print("\n✅ Model is IMPROVING - baseline increasing")
elif baseline_trend_second_half < baseline_trend_first_half - 0.1:
    print("\n❌ Model is DEGRADING - baseline decreasing")
else:
    print("\n🤔 Model NOT CONVERGED - high variance or oscillating")
