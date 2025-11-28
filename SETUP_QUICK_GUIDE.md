# KINH NGHIỆM SETUP GPU TRAINING NHANH CHO RTX 4080/4090

## 1. ENVIRONMENT SETUP CHUẨN (5 phút)
```bash
# Python packages tương thích CHẮC CHẮN
pip install transformers==4.35.2
pip install trl==0.7.11  
pip install peft==0.6.0
pip install bitsandbytes==0.41.3
pip install accelerate==0.24.1
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. MODELS TƯƠNG THÍCH 100% (không cần gated access)
```yaml
✅ LLaMA-3-8B: meta-llama/Meta-Llama-3-8B-Instruct
✅ DeepSeek-7B: deepseek-ai/deepseek-coder-7b-instruct-v1.5  
✅ Mistral-7B: mistralai/Mistral-7B-Instruct-v0.2
✅ CodeLlama-7B: codellama/CodeLlama-7b-Instruct-hf
✅ LLaMA-2-7B: NousResearch/Llama-2-7b-chat-hf

❌ TRÁNH: Qwen2 (cần transformers 4.37+), Phi-3 (cần transformers 4.36+), Gemma2 (gated)
```

## 3. CONFIG TEMPLATE CHUẨN (RTX 4080 16GB)
```yaml
model_name: [MODEL_NAME]
use_auth_token_env: HF_TOKEN

# Quantization
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: float16

# PEFT (KHÔNG DÙNG DORA với version cũ)
use_dora: false
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training (batch size tối ưu cho 16GB VRAM)
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
seq_length: 2048
```

## 4. PARALLEL TRAINING SCRIPT
```bash
#!/bin/bash
cd /workspace/LLM4WAF

# GPU 0: LLaMA-3-8B
HF_TOKEN="your_token" CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_red.py --config configs/llama3_8b.yaml --gpu 0 --log-suffix "_llama8b" > logs/llama8b_gpu0.out 2>&1 &

# GPU 1: Mistral-7B  
HF_TOKEN="your_token" CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_red.py --config configs/mistral_7b.yaml --gpu 1 --log-suffix "_mistral7b" > logs/mistral7b_gpu1.out 2>&1 &

# GPU 2: DeepSeek-7B
HF_TOKEN="your_token" CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_red.py --config configs/deepseek_7b.yaml --gpu 2 --log-suffix "_deepseek7b" > logs/deepseek7b_gpu2.out 2>&1 &

# GPU 3: CodeLlama-7B
HF_TOKEN="your_token" CUDA_VISIBLE_DEVICES=3 nohup python scripts/train_red.py --config configs/codellama_7b.yaml --gpu 3 --log-suffix "_codellama7b" > logs/codellama7b_gpu3.out 2>&1 &
```

## 5. CRITICAL FIXES ĐÃ BIẾT
```python
# train_red.py - FIX DataParallel issue
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # ĐẶT TRƯỚC import torch

# SFTTrainer fix cho TRL 0.7.11
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,  # KHÔNG DÙNG tokenizer parameter
    train_dataset=train_dataset,
    # LOẠI BỎ use_dora=True
    peft_config=peft_config,
)
```

## 6. MONITORING COMMANDS
```bash
# Check GPU usage
nvidia-smi

# Check processes  
ps aux | grep "train_red.py"

# Check logs
tail -f logs/[model]_gpu[X].out
```

## 7. EXPECTED VRAM USAGE (4-bit quantized)
- **7B models**: ~13-14GB VRAM
- **8B models**: ~15-16GB VRAM  
- **Target utilization**: 85-95% VRAM, 100% GPU util

## 8. TIME ESTIMATE
- **Setup**: 5 phút
- **Model download**: 10-15 phút mỗi model
- **Training start**: 2-3 phút mỗi GPU
- **Full 4-GPU setup**: ~30 phút total