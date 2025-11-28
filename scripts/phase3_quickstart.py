#!/usr/bin/env python3
"""
Phase 3 QuickStart Guide - First steps after renting GPU
Progressive training strategy with evaluation checkpoints
"""

import json
import os
import time
from pathlib import Path

def get_training_samples():
    """Check available training samples"""
    data_dir = Path("data/splits/sft_experiment")
    
    print("📊 Training Data Overview:")
    for file in data_dir.glob("train_full_*_cleaned.jsonl"):
        with open(file, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        model_name = file.stem.replace("train_full_", "").replace("_cleaned", "")
        print(f"  {model_name}: {lines:,} samples")
    
    print("\n🧪 Test Data:")
    for file in data_dir.glob("test_200_*.jsonl"):
        model_name = file.stem.replace("test_200_", "")
        print(f"  {model_name}: 200 samples")

def progressive_training_plan():
    """Progressive training strategy"""
    
    print("\n🚀 PHASE 3 PROGRESSIVE TRAINING STRATEGY")
    print("=" * 50)
    
    strategy = {
        "Stage 1 - Quick Test (1K samples)": {
            "samples": "1,000",
            "time": "30-45 minutes", 
            "purpose": "GPU setup validation, memory check",
            "command": "python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 100"
        },
        
        "Stage 2 - Small Scale (5K samples)": {
            "samples": "5,000",
            "time": "2-3 hours",
            "purpose": "Quality baseline, convergence test",
            "command": "python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 500"
        },
        
        "Stage 3 - Full Training (11.6K samples)": {
            "samples": "11,589",
            "time": "6-8 hours", 
            "purpose": "Production model training",
            "command": "python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml"
        }
    }
    
    for stage, details in strategy.items():
        print(f"\n🎯 {stage}")
        print(f"   Samples: {details['samples']}")
        print(f"   Duration: {details['time']}")
        print(f"   Purpose: {details['purpose']}")
        print(f"   Command: {details['command']}")

def quickstart_checklist():
    """Post-GPU rental checklist"""
    
    print("\n✅ QUICKSTART CHECKLIST - Run in Order:")
    print("=" * 50)
    
    steps = [
        {
            "step": "1. Environment Setup",
            "commands": [
                "git clone https://github.com/dung-h/LLM4WAF.git",
                "cd LLM4WAF",
                "git checkout v11",
                "pip install -r requirements.txt",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                "# CRITICAL: Set HuggingFace token",
                "export HF_TOKEN='your_hf_token_here'",
                "# Alternative: huggingface-cli login"
            ],
            "time": "10-15 minutes"
        },
        
        {
            "step": "2. GPU Validation",
            "commands": [
                "nvidia-smi",
                "python -c \"import torch; print(f'GPUs: {torch.cuda.device_count()}')\""
            ],
            "time": "1 minute"
        },
        
        {
            "step": "3. Data Verification", 
            "commands": [
                "python scripts/phase3_quickstart.py --check-data",
                "ls -la data/splits/sft_experiment/"
            ],
            "time": "1 minute"
        },
        
        {
            "step": "4. Quick Test Training (CRITICAL FIRST)",
            "commands": [
                "python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 10",
                "# If this fails → batch size too high, reduce per_device_train_batch_size"
            ],
            "time": "5 minutes"
        },
        
        {
            "step": "5. Stage 1 Training (1K samples)",
            "commands": [
                "python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 100",
                "python scripts/evaluate_model.py --model_path experiments/qwen_7b_4gpu_phase3 --test_data data/splits/sft_experiment/test_200_qwen.jsonl"
            ],
            "time": "45 minutes"
        },
        
        {
            "step": "6. Quality Check & Decision",
            "commands": [
                "python scripts/evaluate_quality.py --input_file results/qwen_outputs.jsonl",
                "# If quality >0.4 → proceed to Stage 2",
                "# If quality <0.4 → debug or try different model"
            ],
            "time": "5 minutes"
        }
    ]
    
    for step_info in steps:
        print(f"\n🔧 {step_info['step']} ({step_info['time']})")
        for cmd in step_info['commands']:
            print(f"   {cmd}")

def evaluation_strategy():
    """Progressive evaluation strategy"""
    
    print("\n📊 EVALUATION STRATEGY:")
    print("=" * 30)
    
    checkpoints = [
        {"after": "100 steps", "action": "Quick quality check (50 samples)", "threshold": "Quality >0.3"},
        {"after": "500 steps", "action": "Medium evaluation (200 samples)", "threshold": "Quality >0.4"},
        {"after": "Full training", "action": "Complete evaluation + WAF test", "threshold": "Quality >0.55, WAF >70%"}
    ]
    
    for checkpoint in checkpoints:
        print(f"📍 {checkpoint['after']}: {checkpoint['action']}")
        print(f"   Success: {checkpoint['threshold']}")

def generate_quick_commands():
    """Generate quick command reference"""
    
    commands_file = Path("QUICK_COMMANDS.md")
    
    with open(commands_file, 'w') as f:
        f.write("""# 🚀 Phase 3 Quick Commands Reference

## Initial Setup (Run Once)
```bash
# GPU check
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# CRITICAL: HuggingFace Token Setup
export HF_TOKEN='your_hf_token_here'
# OR: huggingface-cli login

# Data check
python scripts/phase3_quickstart.py --check-data
```

## Training Commands (Progressive)
```bash
# Stage 1: Quick test (5 minutes)
python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 10

# Stage 2: Small scale (45 minutes)  
python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml --max_steps 100

# Stage 3: Full training (6-8 hours)
python scripts/run_sft.py --config configs/phase3_qwen_7b_4x3090.yaml
```

## Evaluation Commands
```bash
# Quick evaluation
python scripts/evaluate_model.py --model_path experiments/qwen_7b_4gpu_phase3 --test_data data/splits/sft_experiment/test_200_qwen.jsonl --output_file results/qwen_quick.jsonl

# Quality assessment
python scripts/evaluate_quality.py --input_file results/qwen_quick.jsonl --output_file results/qwen_quality.json

# WAF testing
python scripts/enhanced_waf_test.py results/qwen_quick.jsonl results/qwen_waf.json
```

## Emergency Commands
```bash
# If OOM error → reduce batch size
sed -i 's/per_device_train_batch_size: 10/per_device_train_batch_size: 6/' configs/phase3_qwen_7b_4x3090.yaml

# Monitor GPU usage
watch -n 1 nvidia-smi

# Kill training if needed
pkill -f "python scripts/run_sft.py"
```
""")
    
    print(f"\n📝 Quick commands saved: {commands_file}")

def main():
    """Main quickstart guide"""
    
    print("🎯 PHASE 3 QUICKSTART GUIDE")
    print("After renting 4x RTX 3090 GPUs")
    print("=" * 40)
    
    get_training_samples()
    progressive_training_plan()
    quickstart_checklist() 
    evaluation_strategy()
    generate_quick_commands()
    
    print(f"\n🏁 PRIORITY ORDER AFTER GPU RENTAL:")
    print("1️⃣ Run Steps 1-4 (Environment + Quick Test)")
    print("2️⃣ If Step 4 succeeds → Stage 1 training") 
    print("3️⃣ Evaluate Stage 1 → decide next model")
    print("4️⃣ Full training pipeline")
    
    print(f"\n💡 CRITICAL: Always start with max_steps=10 to test GPU setup!")

if __name__ == "__main__":
    import sys
    
    if "--check-data" in sys.argv:
        get_training_samples()
    else:
        main()