#!/bin/bash
# Parallel training launcher with optimized batch sizes and proper logging

# Kill any existing training processes
pkill -f "train_red.py"
sleep 2

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Export HuggingFace token (replace with your actual token)
export HF_TOKEN="your_huggingface_token_here"

# Create logs directory
mkdir -p logs

echo "🚀 Starting parallel training on 4 GPUs..."
echo "📊 GPU 0: LLaMA-3-8B"
echo "📊 GPU 1: Qwen2-7B" 
echo "📊 GPU 2: DeepSeek-7B"
echo "📊 GPU 3: Phi-3-14B"

# GPU 0: LLaMA-3-8B (batch_size=4, should use ~14-15GB)
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_red.py \
    --config configs/phase3_llama_8b_fixed.yaml \
    --gpu 0 --log-suffix "_llama8b" \
    > logs/llama8b_gpu0.out 2>&1 &
GPU0_PID=$!
echo "🎯 GPU 0 (LLaMA-8B): PID $GPU0_PID"

# Wait a bit between launches to avoid resource conflicts
sleep 30

# GPU 1: Qwen2-7B (batch_size=6, should use ~13-14GB)
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_red.py \
    --config configs/phase3_qwen_7b_fixed.yaml \
    --gpu 1 --log-suffix "_qwen7b" \
    > logs/qwen7b_gpu1.out 2>&1 &
GPU1_PID=$!
echo "🎯 GPU 1 (Qwen2-7B): PID $GPU1_PID"

sleep 30

# GPU 2: DeepSeek-7B (batch_size=6, should use ~13-14GB)
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_red.py \
    --config configs/phase3_deepseek_7b_fixed.yaml \
    --gpu 2 --log-suffix "_deepseek7b" \
    > logs/deepseek7b_gpu2.out 2>&1 &
GPU2_PID=$!
echo "🎯 GPU 2 (DeepSeek-7B): PID $GPU2_PID"

sleep 30

# GPU 3: Phi-3-14B (batch_size=3, should use ~15GB - largest model)
CUDA_VISIBLE_DEVICES=3 nohup python scripts/train_red.py \
    --config configs/phase3_phi3_14b_fixed.yaml \
    --gpu 3 --log-suffix "_phi14b" \
    > logs/phi14b_gpu3.out 2>&1 &
GPU3_PID=$!
echo "🎯 GPU 3 (Phi-3-14B): PID $GPU3_PID"

echo ""
echo "✅ All 4 training processes started!"
echo "📋 Process IDs:"
echo "   GPU 0 (LLaMA-8B):  $GPU0_PID"
echo "   GPU 1 (Qwen2-7B):   $GPU1_PID" 
echo "   GPU 2 (DeepSeek-7B): $GPU2_PID"
echo "   GPU 3 (Phi-3-14B):   $GPU3_PID"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f logs/llama8b_gpu0.out"
echo "   tail -f logs/qwen7b_gpu1.out"
echo "   tail -f logs/deepseek7b_gpu2.out"
echo "   tail -f logs/phi14b_gpu3.out"
echo ""
echo "🔧 Check GPU usage: watch -n 2 nvidia-smi"
echo "🛑 Kill all: pkill -f train_red.py"

# Save PIDs to file for easy management
echo "$GPU0_PID $GPU1_PID $GPU2_PID $GPU3_PID" > logs/training_pids.txt
echo "💾 PIDs saved to logs/training_pids.txt"