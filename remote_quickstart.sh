#!/bin/bash
# Quick start script for remote RTX 4090 training

set -e

echo "🚀 LLM4WAF Remote Training - Quick Start"
echo "========================================"
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected!"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Setup environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "🔧 Setting up environment..."
    bash setup_full_env.sh
else
    echo ""
    echo "✅ Environment already set up"
fi

# Activate venv
source .venv/bin/activate

# Prepare datasets
if [ ! -f "data/processed/red_v40_phase1_10k.jsonl" ]; then
    echo ""
    echo "📦 Preparing datasets..."
    bash prepare_datasets.sh
else
    echo ""
    echo "✅ Datasets ready"
fi

# Show available configs
echo ""
echo "📋 Available training configs:"
ls -1 configs/remote_*.yaml

echo ""
echo "🎯 Example training commands:"
echo ""
echo "# Gemma 2 2B Phase 1:"
echo "  python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml"
echo ""
echo "# Phi-3 Mini Phase 1:"
echo "  python scripts/train_red.py --config configs/remote_phi3_mini_phase1.yaml"
echo ""
echo "# Qwen 2.5 7B Phase 1:"
echo "  python scripts/train_red.py --config configs/remote_qwen_7b_phase1.yaml"
echo ""
echo "Monitor with: nvidia-smi -l 1"
echo ""
