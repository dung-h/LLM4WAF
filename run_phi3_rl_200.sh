#!/bin/bash
# Train Phi-3 Mini Phase 4 RL - 200 epochs

cd /mnt/d/AI_in_cyber/LLM_in_Cyber
source .venv/bin/activate

echo "Starting Phi-3 Mini RL Training (200 epochs)..."
echo "Config: configs/phi3_mini_phase4_rl_200epochs.yaml"
echo "Output: experiments/phi3_mini_phase4_rl_200epochs/"
echo ""

python scripts/train_rl_reinforce.py \
    --config configs/phi3_mini_phase4_rl_200epochs.yaml \
    2>&1 | tee training_phi3_rl_200epochs.log

echo ""
echo "✅ Phi-3 Mini RL training complete!"
echo "Log: training_phi3_rl_200epochs.log"
