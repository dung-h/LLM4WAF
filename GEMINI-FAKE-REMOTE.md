# Gemini Agent Instructions - REMOTE SERVER (LLM4WAF)

This document is for the Agent operating on the **Remote GPU Server** to execute Phase 3 (Adaptive SFT) Training.

## 1. Initial Setup
1.  **Run Setup Script:** `bash setup_full_env.sh`
2.  **Activate Env:** `source .venv/bin/activate`
3.  **Set Token:** `export HF_TOKEN=hf_...` (Get from user)

## 2. Data & Model Preparation
*   **Dataset:** Ensure `data/processed/red_phase3_lightweight.jsonl` exists (20k samples).
    *   If not found, regenerate it: `python scripts/build_phase3_lightweight.py`
*   **Adapters:** Phase 3 Lightweight is trained from **Base Model** directly, no need for Phase 1/2 adapters.

## 3. Execution Tasks (Training Phase 3 Lightweight - 20k Enhanced Dataset)

**IMPORTANT:** Choose based on your server GPU setup:

### Option A: Single GPU (16GB+ VRAM)
```bash
# 1. Gemma 2B (fastest, recommended for 8-16GB)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml --gpu 0 > logs/train_gemma.log 2>&1 &

# 2. Phi-3 Mini 4k (3.8B params, good balance)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_phi3.yaml --gpu 0 > logs/train_phi3.log 2>&1 &

# 3. Qwen 2.5 3B (newest architecture)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_qwen3b.yaml --gpu 0 > logs/train_qwen3b.log 2>&1 &

# 4. Qwen 2.5 7B (best quality, requires 24GB+ VRAM single GPU)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_qwen7b.yaml --gpu 0 > logs/train_qwen7b.log 2>&1 &
```

### Option B: Multi-GPU (2x GPU - 2x faster training)
**Recommended for running multiple models in parallel:**
```bash
# Train Gemma on GPU 0,1 (2x speedup)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml --gpu 0,1 > logs/train_gemma_multi.log 2>&1 &

# Train Phi-3 on GPU 0,1
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_phi3.yaml --gpu 0,1 > logs/train_phi3_multi.log 2>&1 &

# Train Qwen 3B on GPU 0,1
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_qwen3b.yaml --gpu 0,1 > logs/train_qwen3b_multi.log 2>&1 &

# Train Qwen 7B on GPU 0,1 (recommended for 7B)
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_qwen7b.yaml --gpu 0,1 > logs/train_qwen7b_multi.log 2>&1 &
```

### Option C: Parallel Training (if you have 2+ GPUs)
**Train different models simultaneously on different GPUs:**
```bash
# GPU 0: Gemma 2B
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml --gpu 0 > logs/train_gemma_gpu0.log 2>&1 &

# GPU 1: Phi-3 Mini
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_phi3.yaml --gpu 1 > logs/train_phi3_gpu1.log 2>&1 &
```

**Monitor Training:**
```bash
# Check loss progression (any model)
tail -f SFT_*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## 4. Post-Training
1.  **Verify Output:** Check `experiments/red_phase3_lightweight_enhanced_*` directories.
2.  **Check Logs:** Review `SFT_*.log` files for final loss values:
    - `SFT_gemma-2-2b_*.log`
    - `SFT_Phi-3-mini-4k_*.log`
    - `SFT_Qwen2.5-3B-Instruct_*.log`
    - `SFT_Qwen2.5-7B-Instruct_*.log`
3.  **Pack & Download:**
    ```bash
    tar -czvf adapters_phase3_lightweight_all.tar.gz experiments/red_phase3_lightweight_enhanced_* SFT_*.log
    ```
4.  **Upload:** Upload to Google Drive or Transfer.sh and provide link.

## Model Size & Speed Reference
| Model | Params | VRAM (4bit) | Training Time (20k×3, Single GPU) | Multi-GPU Speedup |
|-------|--------|-------------|----------------------------------|-------------------|
| Gemma 2B | 2B | ~8GB | ~4 hours | 2x (2 hours) |
| Phi-3 Mini | 3.8B | ~12GB | ~5 hours | 2x (2.5 hours) |
| Qwen 2.5 3B | 3B | ~10GB | ~4.5 hours | 2x (2.25 hours) |
| Qwen 2.5 7B | 7B | ~16GB | ~7 hours | 2x (3.5 hours) |
