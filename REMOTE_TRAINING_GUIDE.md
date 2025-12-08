# Training Strategy & Setup for 16GB VRAM GPU

## ⚡ Quick Reference

```bash
# Phase 1 - Basic SFT
python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml

# Phase 2 - Adaptive (continue from Phase 1)
python scripts/train_red.py --config configs/remote_gemma2_2b_phase2.yaml

# Phase 3 - RL (continue from Phase 2, 150 epochs)
python scripts/train_rl_reinforce.py --config configs/gemma2_2b_phase3_rl.yaml

# Test adapter
python scripts/runner_phase2_eval.py --adapter_path experiments/remote_gemma2_2b_phase2 --test_size 20
```

**Training Strategy: Sequential (One model at a time)**

1. Complete Gemma 2B: Phase 1 → Phase 2 → Phase 3 (all 3 phases)
2. Then start Phi-3 Mini: Phase 1 → Phase 2 → Phase 3
3. Finally Qwen 7B (if time allows)

**Datasets:**

- Phase 1: `phase1_balanced_10k.jsonl` → **10k samples, 509 techniques** (stratified sampling)
- Phase 2: `phase2_with_replay_22k.jsonl` → **24k samples** (20k observations + 4k Phase 1 replay)
- Phase 3: Live WAF at `http://modsec.llmshield.click` (150 episodes ~900 tests)

---

## 📊 Training Strategy Recommendation

### ✅ Sequential Training (RECOMMENDED)

**Complete one model fully before starting the next**

**Why:**

- ✅ Can evaluate full pipeline (Phase 1→2→3) for each model
- ✅ Easier to debug if issues occur
- ✅ Clear checkpoints for thesis results
- ✅ Can stop after Gemma if time runs out (still have complete results)

**Timeline per model:**

- Gemma 2B: ~6-9 hours total (2h Phase1 + 2h Phase2 + 3-5h RL)
- Phi-3 Mini: ~6-9 hours total
- Qwen 7B: ~10-15 hours total (larger model)

**Alternative:** Could parallelize Phase 1 training for multiple models if GPU allows, but **NOT RECOMMENDED** due to VRAM constraints (RTX 4090 24GB).

### Option 1: Continual Learning (RECOMMENDED for your case)

**Phase 1 → Phase 2 (continue) → Phase 3 RL**

**Pros:**

- ✅ Saves VRAM (single adapter at a time)
- ✅ Faster training per phase
- ✅ Natural progression (basic → reasoning → RL)
- ✅ Can stop at any phase if needed

**Cons:**

- ⚠️ Risk of catastrophic forgetting (Phase 1 knowledge may degrade)
- Need to test after each phase to ensure no regression

**Strategy:**

```
Phase 1 (Basic SFT)
  ↓ save adapter to experiments/remote_gemma2_2b_phase1
Phase 2 (Continue from Phase 1 adapter)
  adapter_path: experiments/remote_gemma2_2b_phase1
  ↓ weights update directly, save to experiments/remote_gemma2_2b_phase2
Phase 3 (RL from Phase 2 adapter)
  adapter_path: experiments/remote_gemma2_2b_phase2
  ↓ RL updates same adapter, save to experiments/remote_gemma2_2b_phase3_rl
```

**Key**: Each phase loads previous adapter with `is_trainable=True` → Continual learning on SAME adapter (not merging multiple adapters)

### Option 2: Mixed Dataset SFT (Slower convergence)

**Train all phases together in one SFT run**

**Pros:**

- ✅ No catastrophic forgetting
- ✅ Model sees all data patterns

**Cons:**

- ❌ Slower convergence (need more epochs)
- ❌ Harder to debug which phase is working
- ❌ Can't isolate phase improvements

**NOT RECOMMENDED** for thesis - harder to explain incremental improvements

---

## 🎯 Final Recommendation: **Continual + Replay Buffer**

**Best of both worlds:**

1. **Phase 1**: Train on basic dataset (10k samples)
2. **Phase 2**: Continue adapter + mix 20% Phase 1 samples (replay buffer) → prevents forgetting
3. **Phase 3**: RL training (no forgetting risk as it's trial-and-error)

---

## 📦 Dataset Strategy

### Phase 1: Basic SFT

- **Dataset**: `data/processed/phase1_balanced_10k.jsonl`
- **Size**: 10,000 samples covering **509 techniques**
- **Sampling**: Stratified to ensure diverse coverage (vs simple head -n 10000)
- **Format**: Simple instruction → payload, no observations

### Phase 2: Observations-based with Replay Buffer

- **Dataset**: `data/processed/phase2_with_replay_22k.jsonl`
- **Size**: 24,000 samples (20k observations + 4k Phase 1 replay)
- **Format**: Instruction + [Observations] BLOCKED/PASSED history → payload
- **Replay**: 20% Phase 1 samples to prevent catastrophic forgetting
- **Continues**: From Phase 1 adapter with `adapter_path` + `is_trainable: true`

### Phase 3: RL

- **No static dataset** - live interaction with **http://modsec.llmshield.click**
- **Episodes**: 150 (increased from 50 for better convergence)
- **Authentication**: Handled automatically by WAFEnv (admin/password)
- **Algorithm**: REINFORCE (policy gradient with baseline)
  - **NOT PPO**: Too complex for thesis, REINFORCE is sufficient
  - **NOT DPO**: Wrong use case (DPO for human preference, we have binary WAF feedback)
- **Start from**: Phase 2 checkpoint (continual learning)
- **Training**: Adapter weights update directly (not merged multiple times)
- **Expected**: 150 episodes × ~6 scenarios = ~900 WAF interactions
- **Duration**: 3-5 hours on RTX 4090

---

## 🔧 Training Configs for 16GB VRAM

### Gemma 2 2B - Phase 1

```yaml
# configs/remote_gemma2_2b_phase1.yaml
model_name: "google/gemma-2-2b-it"
train_path: "data/processed/phase1_balanced_10k.jsonl" # 509 techniques
output_dir: "experiments/remote_gemma2_2b_phase1"

num_train_epochs: 3
per_device_train_batch_size: 1 # Conservative for 16GB
gradient_accumulation_steps: 16 # Effective batch = 16
learning_rate: 2.0e-4
max_seq_length: 1024 # Increased from 512

# 4-bit quantization
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "float16"

# LoRA
lora_r: 16 # Increased from 8 for better capacity
lora_alpha: 32
target_modules:
  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout: 0.05

# Optimization
optim: "paged_adamw_8bit"
gradient_checkpointing: true
fp16: true
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

# Data loading (RTX 4090 has more RAM)
dataloader_num_workers: 8
dataloader_prefetch_factor: 4
dataloader_pin_memory: true

# Logging
logging_steps: 10
save_steps: 500
save_total_limit: 3
```

### Phi-3 Mini - Phase 1 (WITH DynamicCache FIX)

```yaml
# configs/remote_phi3_mini_phase1.yaml
model_name: "microsoft/Phi-3-mini-4k-instruct"
train_path: "data/processed/phase1_balanced_10k.jsonl" # 509 techniques
output_dir: "experiments/remote_phi3_mini_phase1"

num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
max_seq_length: 1024

# Same 4-bit + LoRA settings as Gemma
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "float16"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# CRITICAL: Phi-3 specific settings
gradient_checkpointing: true # Must enable
use_cache: false # Fix DynamicCache error (add to train_red.py)

optim: "paged_adamw_8bit"
fp16: true
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

dataloader_num_workers: 8
dataloader_prefetch_factor: 4

logging_steps: 10
save_steps: 500
save_total_limit: 3
```

### Qwen 2.5 7B - Phase 1

```yaml
# configs/remote_qwen_7b_phase1.yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
train_path: "data/processed/phase1_balanced_10k.jsonl" # 509 techniques
output_dir: "experiments/remote_qwen_7b_phase1"

num_train_epochs: 3
per_device_train_batch_size: 1 # 7B model, reduce batch size
gradient_accumulation_steps: 16 # Keep effective batch = 16
learning_rate: 2.0e-4
max_seq_length: 1024

# 4-bit essential for 7B on 16GB
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "float16"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

gradient_checkpointing: true
optim: "paged_adamw_8bit"
fp16: true
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

dataloader_num_workers: 8
dataloader_prefetch_factor: 4

logging_steps: 10
save_steps: 500
save_total_limit: 3
```

---

## ⚙️ Max Token Settings: Training vs Inference

### Question: Different max_length in training vs inference?

**Answer**: YES, it affects performance but not critically

**Training max_length: 1024-1536**

- Determines context window model sees during training
- If you train with 512 but data has 800 tokens → **TRUNCATION** → model never learns long contexts
- **Recommendation**: 1024 for Phase 1, 1536 for Phase 2 (observations add length)

**Inference max_new_tokens: 64-128**

- Only controls OUTPUT length (not input)
- Payload usually short (20-100 chars)
- Can be different from training max_length without issues

**Best Practice:**

```yaml
# Training
max_seq_length: 1024 # or max_length: 1536 for observations

# Inference (generation)
max_new_tokens: 128 # Just for output
max_length: null # Don't confuse with training param
```

**⚠️ Warning**: If you train with `max_seq_length: 512` and your prompts are 800 tokens:

- Training: Truncates to 512 → model never sees full context
- Inference: You pass 800 tokens → model confused (never saw this during training)
- **Result**: Poor quality

**Solution**: Train with **1024-1536** to be safe!

---

## 🧹 Cleanup Strategy

### Files to KEEP (important for training)

```bash
# Core code
scripts/train_red.py
scripts/train_rl_reinforce.py
rl/waf_env.py

# Datasets
data/processed/red_v40_balanced_final_v13.jsonl
data/processed/red_phase3_lightweight.jsonl

# Configs (create new ones for remote)
configs/remote_gemma2_2b_phase1.yaml
configs/remote_phi3_mini_phase1.yaml
configs/remote_qwen_7b_phase1.yaml

# Setup scripts
setup_full_env.sh
requirements.txt
README.md
```

### Files to DELETE (before upload)

```bash
# Eval results (can regenerate)
rm -rf eval/
rm -rf reports/

# Old experiments (keep only best checkpoints if needed)
# For clean start, delete all:
rm -rf experiments/*

# Logs
rm *.log
rm SFT_*.log
rm rl_training_*.log

# Temp files
rm -rf .git_temp
rm -rf __pycache__
rm -rf **/__pycache__

# Large files not needed for training
rm -rf data/archive_*
rm -rf data/raw/*  # Keep only if needed
rm -rf archive_legacy/

# VSCode/IDE
rm -rf .vscode/
rm -rf .idea/

# Jupyter notebooks (if not needed)
rm -rf notebooks/
```

### Cleanup script

```bash
#!/bin/bash
# cleanup_for_remote.sh

echo "🧹 Cleaning workspace for remote training..."

# Remove eval results
rm -rf eval/
rm -rf reports/

# Remove old experiments (CAREFUL!)
read -p "⚠️  Delete ALL experiments? (y/N): " confirm
if [ "$confirm" = "y" ]; then
    rm -rf experiments/*
    echo "✅ Experiments deleted"
fi

# Remove logs
rm -f *.log SFT_*.log rl_training_*.log training_*.log

# Remove temp/cache
rm -rf .git_temp __pycache__ **/__pycache__ .pytest_cache

# Remove archives
rm -rf data/archive_* archive_legacy/

# Remove IDE files
rm -rf .vscode/ .idea/ .DS_Store

# Check size
echo ""
echo "📊 Directory size after cleanup:"
du -sh .

echo "✅ Cleanup complete!"
```

---

## 🚀 Remote Training Workflow

### 1. Prepare datasets

```bash
# Create Phase 1 dataset (10k samples)
# Create Phase 1 balanced dataset
bash prepare_datasets.sh

# Verify
head -n 10000 data/processed/phase1_balanced_10k.jsonl | jq -r '.technique' | sort -u | wc -l
# Should output: 509

# Create Phase 2 dataset (Phase 1 + observations)
# TODO: Create script to add observations to 20% of Phase 1 samples
```

### 2. Update setup_full_env.sh for CUDA 12.1 (RTX 4090)

```bash
# Change line 42:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Upload to remote

```bash
# Compress (exclude unnecessary files)
tar -czf llm4waf.tar.gz \
  --exclude='eval/*' \
  --exclude='experiments/*' \
  --exclude='*.log' \
  --exclude='__pycache__' \
  --exclude='.git' \
  .

# Upload
scp llm4waf.tar.gz user@remote-server:~/
```

### 4. On remote server

```bash
tar -xzf llm4waf.tar.gz
cd LLM_in_Cyber
bash setup_full_env.sh

# Set environment variables for WAF access
export DVWA_USERNAME="admin"
export DVWA_PASSWORD="password"
export HF_TOKEN="your_huggingface_token"

# Train Phase 1
source .venv/bin/activate
python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml

# After Phase 1 finishes, continue to Phase 2...
```

---

## 🔐 Environment Variables

Required for training:

```bash
# Hugging Face token (for model download)
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# WAF credentials (for Phase 3 RL)
export DVWA_USERNAME="admin"
export DVWA_PASSWORD="password"

# Optional: Override WAF URL
# export WAF_BASE_URL="http://modsec.llmshield.click"
```

Add to `~/.bashrc` for persistence:

```bash
echo 'export HF_TOKEN="hf_xxxxx"' >> ~/.bashrc
echo 'export DVWA_USERNAME="admin"' >> ~/.bashrc
echo 'export DVWA_PASSWORD="password"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🔥 Phi-3 DynamicCache Fix

Add to `scripts/train_red.py` after line 147 (where gradient checkpointing is set):

```python
if cfg.get("gradient_checkpointing", True):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Already there

    # ADD THIS for Phi-3:
    if "phi" in model_name.lower():
        logger.info("🔧 Applying Phi-3 DynamicCache workaround...")
        # Force legacy cache implementation
        if hasattr(model.config, "cache_implementation"):
            model.config.cache_implementation = None
```

---

## 🚀 Step-by-Step Training Commands

### Phase 1: Basic SFT (Stratified Dataset - 509 Techniques)

```bash
# Activate environment
source .venv/bin/activate

# Train Gemma 2B Phase 1
python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml

# Monitor training (in another terminal)
tail -f logs/SFT_*.log
watch -n 1 nvidia-smi

# Expected output:
# - Dataset: phase1_balanced_10k.jsonl (10k samples, 509 techniques)
# - Loss should start ~2.5 and drop to ~0.8-1.2
# - Training time: ~2-3 hours
# - GPU usage: 18-22GB VRAM
# - Checkpoint saved: experiments/remote_gemma2_2b_phase1/
```

### Phase 2: Adaptive Learning (Observations-based with Replay)

```bash
# Phase 2 config already created (continual learning from Phase 1)
# configs/remote_gemma2_2b_phase2.yaml:
#   adapter_path: "experiments/remote_gemma2_2b_phase1"
#   train_path: "data/processed/phase2_with_replay_22k.jsonl"  # 20k obs + 4k replay
#   is_trainable: true  # Continual learning enabled
#   output_dir: "experiments/remote_gemma2_2b_phase2"

# Train Phase 2
python scripts/train_red.py --config configs/remote_gemma2_2b_phase2.yaml

# Monitor: Loss should start low (~0.8) and drop further (~0.5-0.7)
# Time: ~2-3 hours

# Test adapter after Phase 2
python scripts/runner_phase2_eval.py \
  --adapter_path experiments/remote_gemma2_2b_phase2 \
  --test_size 20 \
  --waf_url http://modsec.llmshield.click
```

### Phase 3: RL Training (150 episodes)

```bash
# Ensure WAF credentials are set
export DVWA_USERNAME="admin"
export DVWA_PASSWORD="password"

# Train RL (continue from Phase 2, connects to llmshield.click)
python scripts/train_rl_reinforce.py --config configs/gemma2_2b_phase3_rl.yaml

# Monitor RL training
tail -f rl_training_*.log

# Expected output:
# - Bypass rate should increase from ~60% to 80%+
# - Training time: 3-5 hours (150 epochs)
# - Checkpoints saved every 5 epochs
# - Final checkpoint: experiments/gemma2_2b_phase3_rl/
```

**Important:** Phase 3 RL uses `waf_url: "http://modsec.llmshield.click"` from config.
The WAFEnv will automatically:

1. Connect to remote DVWA
2. Login with credentials from environment variables
3. Test payloads against ModSecurity PL1
4. Return PASSED/BLOCKED feedback for RL training

```
# - Checkpoints: experiments/remote_gemma2_2b_phase3_rl/
```

### Multi-Model Training (Parallel)

```bash
# Run in tmux/screen for each model
tmux new -s gemma_training
python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml
# Ctrl+B, D to detach

tmux new -s phi3_training
python scripts/train_red.py --config configs/remote_phi3_mini_phase1.yaml
# Ctrl+B, D to detach

# Reattach with:
tmux attach -t gemma_training
```

---

## 📝 Training Order for Thesis

### Timeline suggestion:

1. **Gemma 2 2B** (fastest, 2-3 hours per phase)

   - Phase 1 → Phase 2 → Phase 3 RL
   - Use for main results

2. **Phi-3 Mini** (similar speed, 2-3 hours per phase)

   - Phase 1 → Phase 2 → Phase 3 RL
   - Compare with Gemma

3. **Qwen 2.5 7B** (slower, 4-5 hours per phase)
   - Only if time allows
   - Shows scaling effects

### Total time estimate:

- **Minimum** (Gemma + Phi-3): ~18-24 hours
- **Full** (+ Qwen 7B): ~30-36 hours

---

## ✅ Verification Checklist

Before uploading to remote:

- [ ] Created Phase 1 dataset (10k samples)
- [ ] Created remote configs with correct paths
- [ ] Updated setup_full_env.sh for CUDA 12.1
- [ ] Added Phi-3 DynamicCache fix to train_red.py
- [ ] Tested configs locally (at least 10 steps)
- [ ] Cleaned up unnecessary files
- [ ] Compressed workspace (<1GB if possible)

After remote setup:

- [ ] Run setup_full_env.sh successfully
- [ ] Test 1 epoch of Phase 1 training
- [ ] Monitor GPU usage (should be 18-22GB for 2B, 22-23GB for 7B)
- [ ] Check loss is decreasing
- [ ] Verify checkpoints are being saved

---

**Questions answered:**
✅ Continual vs mixed: **Continual with 20% replay buffer**
✅ Max token: **Train 1024-1536, infer 128 is fine**
✅ Phi-3 cache: **Fix included above**
✅ 16GB VRAM configs: **batch_size=1 for all models**
✅ Dataset check: **Need to create Phase 1 10k subset**
