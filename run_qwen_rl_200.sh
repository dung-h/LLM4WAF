#!/bin/bash
# Train Qwen 2.5 3B Phase 4 RL - 200 epochs

cd /mnt/d/AI_in_cyber/LLM_in_Cyber
source .venv/bin/activate

echo "Starting Qwen 2.5 3B RL Training (200 epochs)..."
echo "Config: configs/qwen_3b_phase4_rl_200epochs.yaml"
echo "Output: experiments/qwen_3b_phase4_rl_200epochs/"
echo ""

python scripts/train_rl_reinforce.py \
    --config configs/qwen_3b_phase4_rl_200epochs.yaml \
    2>&1 | tee training_qwen_rl_200epochs.log

echo ""
echo "✅ Qwen 2.5 3B RL training complete!"
echo "Log: training_qwen_rl_200epochs.log"
