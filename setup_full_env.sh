#!/bin/bash
set -e

echo "🚀 Initializing LLM4WAF Remote Environment (GPU Optimized)..."

# 1. System dependencies
echo "📦 Installing system packages..."
if command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    SUDO=""
fi
$SUDO apt-get update && $SUDO apt-get install -y git python3-pip python3-venv unzip curl wget build-essential

# 2. Python Virtual Environment
echo "🐍 Setting up Python venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install Python Libraries
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
# Force CUDA 12.1 for better FlashAttn compatibility if driver supports, else default
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets trl accelerate scipy tensorboard pyyaml httpx tqdm pandas matplotlib seaborn gdown

echo "⚡ Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# 4. Directory Structure
echo "Tk Creating project directories..."
mkdir -p experiments/remote_adapters
mkdir -p data/processed
mkdir -p logs

echo "----------------------------------------------------------------"
echo "✅ Environment Ready!"
echo "👉 Run: source .venv/bin/activate"
echo "👉 To download adapters (if not uploaded): check GEMINI-FAKE.md for gdown links."
echo "👉 To start training:"
echo "   nohup python scripts/train_red.py --config configs/red_phase3_adaptive_gemma.yaml > logs/train_gemma.log 2>&1 &"
echo "   nohup python scripts/train_red.py --config configs/red_phase3_adaptive_phi3.yaml > logs/train_phi3.log 2>&1 &"
echo "   nohup python scripts/train_red.py --config configs/red_phase3_adaptive_qwen.yaml > logs/train_qwen.log 2>&1 &"
echo "----------------------------------------------------------------"
