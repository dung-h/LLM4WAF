"""
Analyze RL training metrics without matplotlib.
"""
import re

# Read training log
with open('training.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract baseline and loss
baselines = []
losses = []
for line in lines:
    if 'Baseline:' in line and 'Loss:' in line:
        match = re.search(r'Baseline: ([-\d.]+) \| Loss: ([-\d.]+)', line)
        if match:
            baselines.append(float(match.group(1)))
            losses.append(float(match.group(2)))

print(f"Total epochs: {len(baselines)}")
print(f"\nBaseline statistics:")
print(f"  Min: {min(baselines):.4f}")
print(f"  Max: {max(baselines):.4f}")
print(f"  Mean: {sum(baselines)/len(baselines):.4f}")
print(f"  First 10 avg: {sum(baselines[:10])/10:.4f}")
print(f"  Last 10 avg: {sum(baselines[-10:])/10:.4f}")
print(f"  Improvement: {sum(baselines[-10:])/10 - sum(baselines[:10])/10:.4f}")

print(f"\nLoss statistics:")
print(f"  Min: {min(losses):.4f}")
print(f"  Max: {max(losses):.4f}")
print(f"  Mean: {sum(losses)/len(losses):.4f}")
print(f"  First 10 avg: {sum(losses[:10])/10:.4f}")
print(f"  Last 10 avg: {sum(losses[-10:])/10:.4f}")

print(f"\nFirst 10 baselines: {[round(b, 4) for b in baselines[:10]]}")
print(f"Last 10 baselines:  {[round(b, 4) for b in baselines[-10:]]}")

print(f"\nFirst 10 losses: {[round(l, 4) for l in losses[:10]]}")
print(f"Last 10 losses:  {[round(l, 4) for l in losses[-10:]]}")

# Check convergence
baseline_first_half = sum(baselines[:len(baselines)//2]) / (len(baselines)//2)
baseline_second_half = sum(baselines[len(baselines)//2:]) / (len(baselines) - len(baselines)//2)

print(f"\n{'='*60}")
print("Convergence Analysis:")
print(f"{'='*60}")
print(f"First half average baseline:  {baseline_first_half:.4f}")
print(f"Second half average baseline: {baseline_second_half:.4f}")
print(f"Improvement:                  {baseline_second_half - baseline_first_half:+.4f}")

if baseline_second_half > baseline_first_half + 0.05:
    print("\n✅ Model is IMPROVING - baseline increasing")
elif baseline_second_half < baseline_first_half - 0.05:
    print("\n❌ Model is DEGRADING - baseline decreasing")
else:
    print("\n⚠️  Model appears STABLE but NOT strongly converging")
    print("   (baseline oscillates around 0.1-0.2, may need longer training)")
