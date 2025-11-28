# 🚀 LLM4WAF - Vast.AI GPU Setup Guide

## 📋 Overview
Complete setup guide for training WAF evasion models on Vast.AI GPU instances. This workspace is optimized for **Phase 3: Qwen-7B scaling** with cleaned 11.6K dataset.

---

## 🏗️ 1. VAST.AI INSTANCE SETUP

### Recommended Configuration
```bash
# GPU Requirements (4-GPU setup preferred)
- GPU: 4x RTX 4090 / A6000 / H100 (32GB+ VRAM)
- RAM: 128GB+ system memory  
- Storage: 500GB+ NVMe SSD
- CUDA: 12.1+
- PyTorch: 2.1.0+
```

### Instance Launch
```bash
# Search for instance
vast search offers 'reliability > 0.98 gpu_name=RTX_4090 num_gpus=4 disk_space>=500'

# Launch instance  
vast create instance <INSTANCE_ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Connect to instance
vast ssh <INSTANCE_ID>
```

---

## 🔧 2. ENVIRONMENT SETUP

### System Dependencies
```bash
# Update system
apt update && apt upgrade -y

# Install essential packages
apt install -y git wget curl htop nvtop tree zip unzip

# Install Python dependencies
pip install --upgrade pip setuptools wheel

# Install training framework
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes wandb
pip install deepspeed flash-attn xformers

# Install evaluation tools
pip install scikit-learn numpy pandas matplotlib seaborn tqdm
```

### Environment Variables
```bash
# Add to ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
echo 'export HF_HOME=/workspace/huggingface_cache' >> ~/.bashrc  
echo 'export WANDB_PROJECT=llm4waf-phase3' >> ~/.bashrc
source ~/.bashrc

# Login to services
wandb login <YOUR_WANDB_KEY>
huggingface-cli login <YOUR_HF_TOKEN>
```

---

## 📂 3. WORKSPACE DEPLOYMENT

### Clone Repository
```bash
# Create workspace directory
mkdir -p /workspace && cd /workspace

# Clone repository
git clone https://github.com/dung-h/LLM4WAF.git
cd LLM4WAF

# Switch to production branch
git checkout v11
```

### Verify Data Structure
```bash
# Check essential training data
ls -la data/processed/
# Should contain:
# - red_v29_cleaned.jsonl (11,692 → 11,589 samples)
# - train_full_qwen_cleaned.jsonl (11,589 samples for Qwen-7B)
# - train_full_phi3_cleaned.jsonl (11,589 samples)
# - train_full_gemma_cleaned.jsonl (11,589 samples)

# Check test data
ls -la data/splits/sft_experiment/
# Should contain:
# - test_200_qwen.jsonl (test set)
# - train_4k_*_cleaned.jsonl (Phase 2 comparison)
```

### Verify Configs
```bash
# Check essential configs
ls -la configs/
# Key files:
# - red_qwen2_7b_dora_8gb_smoke.yaml (Qwen-7B config)
# - red_phi3_mini_lora.yaml (Phi-3 config) 
# - red_deepseek_7b_coder_lora.yaml (DeepSeek config)
```

---

## 🎯 4. TRAINING PIPELINE

### Phase 3: Qwen-7B Training (Recommended)
```bash
# Multi-GPU training with DeepSpeed
python scripts/train_model.py \
    --config configs/red_qwen2_7b_dora_8gb_smoke.yaml \
    --train_data data/processed/train_full_qwen_cleaned.jsonl \
    --test_data data/splits/sft_experiment/test_200_qwen.jsonl \
    --output_dir experiments/qwen_7b_phase3 \
    --num_gpus 4 \
    --batch_size_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 100 \
    --deepspeed_config configs/deepspeed_stage2.json
```

### Alternative: Phi-3 Mini Training
```bash
# Single GPU training
python scripts/train_model.py \
    --config configs/red_phi3_mini_lora.yaml \
    --train_data data/processed/train_full_phi3_cleaned.jsonl \
    --test_data data/splits/sft_experiment/test_200_phi3.jsonl \
    --output_dir experiments/phi3_mini_phase3 \
    --num_gpus 1 \
    --batch_size_per_gpu 4 \
    --gradient_accumulation_steps 4
```

---

## 📊 5. EVALUATION PIPELINE

### Model Evaluation
```bash
# Run comprehensive evaluation
python scripts/evaluate_model.py \
    --model_path experiments/qwen_7b_phase3 \
    --test_data data/splits/sft_experiment/test_200_qwen.jsonl \
    --output_dir results/qwen_7b_evaluation \
    --model_format qwen

# Quality assessment
python scripts/evaluate_quality.py \
    --input_file results/qwen_7b_evaluation/model_outputs.json \
    --output_file results/qwen_7b_evaluation/quality_report.json
```

### Expected Results (Phase 3 Targets)
```bash
# Quality Metrics:
# - Syntax Score: 0.70+ (vs 0.60 Phase 2)
# - Semantic Score: 0.45+ (vs 0.35 Phase 2) 
# - Novelty Score: 0.60+ (vs 0.50 Phase 2)
# - Overall Quality: 0.55+ (vs 0.463 Phase 2)

# Output Quality:
# - Empty outputs: 0% (fixed in Phase 2)
# - Conversational responses: 0% (cleaned in scaling)
# - Valid payloads: 100% target
```

---

## 🛠️ 6. TROUBLESHOOTING

### Common Issues
```bash
# CUDA Memory Issues
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# DeepSpeed Issues
pip install deepspeed --no-cache-dir
ds_report  # Check DeepSpeed installation

# Model Loading Issues
python -c "from transformers import AutoTokenizer; print('OK')"

# Data Loading Issues  
python scripts/verify_data.py --input data/processed/train_full_qwen_cleaned.jsonl
```

### Performance Optimization
```bash
# Enable Flash Attention (A100/H100)
export FLASH_ATTENTION=1

# Mixed Precision Training
export MIXED_PRECISION=fp16

# Gradient Checkpointing
export GRADIENT_CHECKPOINTING=1
```

### Monitoring
```bash
# GPU Usage
watch -n 1 nvidia-smi

# Training Progress
tail -f experiments/qwen_7b_phase3/training.log

# WandB Dashboard
# Visit: https://wandb.ai/your-username/llm4waf-phase3
```

---

## 📈 7. SCALING STRATEGY

### Data Scaling (Current: Phase 3)
```bash
Training Data Evolution:
├── Phase 1: 2K samples → Proof of concept
├── Phase 2: 4K samples → 0.463 quality score  
└── Phase 3: 11.6K samples → Target 0.55+ quality (CURRENT)

Future Scaling:
└── Phase 4: 50K+ samples → Production readiness
```

### Model Scaling Path
```bash
Model Evolution:
├── Phase 2: Gemma-2B, Phi-3-Mini, Qwen-3B (tested)
├── Phase 3: Qwen-7B (current target)
└── Phase 4: Qwen-14B, Llama-7B (future)
```

---

## 🚀 8. DEPLOYMENT

### Model Export
```bash
# Export trained model
python scripts/export_model.py \
    --model_path experiments/qwen_7b_phase3 \
    --output_dir models/qwen_7b_waf_evasion_v3 \
    --format huggingface

# Test exported model
python scripts/test_exported.py \
    --model_path models/qwen_7b_waf_evasion_v3 \
    --test_prompts "Generate XSS payload for input validation bypass"
```

### Production Pipeline
```bash
# API Server (optional)
python scripts/serve_model.py \
    --model_path models/qwen_7b_waf_evasion_v3 \
    --host 0.0.0.0 \
    --port 8000 \
    --max_length 512
```

---

## 📋 9. PHASE 3 CHECKLIST

### Pre-Training
- [ ] Vast.AI instance configured (4x RTX 4090)
- [ ] All dependencies installed
- [ ] Repository cloned and branch v11 active  
- [ ] Training data verified (11,589 clean samples)
- [ ] Configs validated for Qwen-7B
- [ ] WandB/HF authentication complete

### During Training
- [ ] Multi-GPU training launched
- [ ] DeepSpeed working correctly
- [ ] Training metrics monitoring
- [ ] Checkpoints saving properly
- [ ] No CUDA memory issues

### Post-Training  
- [ ] Model evaluation complete
- [ ] Quality metrics >= 0.55 target
- [ ] Zero empty/conversational outputs
- [ ] Model exported for deployment
- [ ] Results documented

---

## 📞 SUPPORT

### Key Files Reference
```bash
Training Pipeline:
├── scripts/train_model.py          # Main training script
├── scripts/evaluate_model.py       # Evaluation pipeline  
├── scripts/evaluate_quality.py     # Quality assessment
└── scripts/clean_training_data.py  # Data cleaning

Data Files:
├── data/processed/train_full_qwen_cleaned.jsonl  # 11.6K training
├── data/splits/sft_experiment/test_200_qwen.jsonl # Test set
└── configs/red_qwen2_7b_dora_8gb_smoke.yaml      # Training config
```

### Performance Targets
```bash
Phase 2 Results (4K samples):
- Qwen: 0.463 quality score (best)
- Phi-3: 0.452 quality score  
- Gemma: 0.443 quality score

Phase 3 Targets (11.6K samples):
- Quality Score: 0.55+ (20% improvement)
- Empty Outputs: 0% (maintained)
- Conversational: 0% (cleaned)
- Training Time: ~6-8 hours (4x RTX 4090)
```

---

**🎯 Ready for Phase 3 scaling on Vast.AI! Let's achieve 0.55+ quality score!** 🚀