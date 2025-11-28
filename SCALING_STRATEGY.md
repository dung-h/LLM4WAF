# 🚀 Scaling Strategy: Small Models (2-4B) → Large Models (7B)

**Date**: November 28, 2025  
**Current**: Testing Gemma 2B, Phi-3 Mini (3.8B), Qwen 2.5 3B  
**Next**: Scale to 7B models after validation

---

## 📊 WHY SMALL-TO-LARGE SCALING?

### The Principle: **Validate Fast, Scale Smart**

```
Phase 1: Small Models (2-4B) - FAST ITERATION
├── Purpose: Validate training methodology
├── Time: 2-6 hours per experiment
├── Cost: Low GPU usage (8GB VRAM)
├── Risk: Low (quick to restart if failed)
└── Learn: Hyperparameters, data quality, prompt format

Phase 2: Large Models (7B) - SCALE PERFORMANCE
├── Purpose: Maximize quality after validation
├── Time: 12-24 hours per experiment
├── Cost: High GPU usage (24GB+ VRAM)
├── Risk: High (expensive to restart)
└── Leverage: Proven configs from Phase 1
```

**Critical Insight**:

- ❌ DON'T train 7B with full dataset first (risky, expensive)
- ✅ DO validate with small models on small data first
- ✅ THEN scale both model size AND data size together

---

## 🎯 RECOMMENDED PROGRESSION

### Current Experiment (Week 1)

**Status**: In Progress ⏳

```yaml
Models: [Gemma 2B, Phi-3 Mini 3.8B, Qwen 2.5 3B]
Data Sizes: [2K, 4K, 8K samples]
GPU: RTX 3090/4090 (8-12GB VRAM)
Duration: ~12 hours total

Expected Outcomes:
  - Identify best small model (likely Qwen 2.5 3B)
  - Validate hyperparameters (learning rate, batch size, epochs)
  - Confirm data quality (pass rate should improve)
  - Establish baseline: 15-20% WAF bypass rate
```

### Next: Progressive Scaling (Week 2-4)

**Stage 1: Smoke Test with 7B (1-2 days)**

```yaml
Purpose: Verify 7B training pipeline works
Model: Qwen 2.5 Coder 7B (best for payloads)
Dataset: 2K samples (same as Phase 1)
Epochs: 1
Batch Size: 1 (grad_accum: 8)
GPU: 24GB VRAM required
Duration: ~6 hours

Why 2K samples?
- Quick validation (not wasting time if config wrong)
- Compare directly to small model results
- Identify if 7B actually helps (should see +3-5% improvement)

Success Criteria:
- Training completes without OOM
- Loss curve looks good
- WAF bypass rate > small model on same 2K data
```

**Stage 2: Medium Scale (3-5 days)**

```yaml
Model: Qwen 2.5 Coder 7B
Dataset: 4K → 8K samples (progressive)
Epochs: 2 (for 4K), 1 (for 8K)
GPU: 24GB VRAM
Duration: ~12 hours

Why progressive?
- 4K first: Validate improvement trend continues
- 8K next: Match current small model experiment
- Compare apples-to-apples with Qwen 3B results

Expected: 20-25% WAF bypass rate
```

**Stage 3: Full Scale (1 week)**

```yaml
Model: Qwen 2.5 Coder 7B
Dataset: Full v29 (11,692 samples)
Epochs: 2-3
Batch Size: 1 (grad_accum: 16)
GPU: 24GB VRAM
Duration: ~24 hours

Why full dataset?
- Maximize knowledge coverage
- Learn all 1,414 passed payloads
- Capture diverse techniques

Expected: 25-30% WAF bypass rate
Target: >30% after RL fine-tuning
```

---

## 🔬 EXPERIMENTAL DESIGN

### Data Size Progression (CRITICAL)

**❌ WRONG Approach**:

```
Train 7B on full 11K dataset immediately
→ Takes 24h
→ If fails, wasted time
→ Don't know if issue is model, data, or hyperparams
```

**✅ CORRECT Approach**:

```
Step 1: Train 7B on 2K (6h)
  → Validate pipeline works
  → Compare to 3B baseline
  → Tune hyperparameters

Step 2: Train 7B on 4K (8h)
  → Confirm scaling benefits
  → Check if performance increases linearly
  → Adjust if needed

Step 3: Train 7B on 8K (12h)
  → Match small model data size
  → Direct comparison possible
  → Validate before full scale

Step 4: Train 7B on 11K (24h)
  → Only if Steps 1-3 successful
  → Maximum performance
  → Production model candidate
```

**Why This Works**:

- Each step validates the next
- Early failure = small time loss
- Can adjust hyperparams between steps
- Build confidence before expensive runs

---

## 📈 SAMPLE SIZE STRATEGY

### Question: "How many samples for 7B?"

**Answer**: **It depends on validation results**

```python
def determine_optimal_samples(model_size):
    if model_size == "2-4B":
        # Small models saturate faster
        optimal = 4000 - 8000
        max_useful = 10000

    elif model_size == "7B":
        # Large models benefit from more data
        optimal = 8000 - 15000
        max_useful = 20000

    elif model_size == "13B+":
        # Very large models need lots of data
        optimal = 15000 - 30000
        max_useful = 50000

    return optimal, max_useful

# For our case (7B):
# Start: 2K (validation)
# Scale: 4K, 8K (progressive)
# Max: 11K (current dataset)
# Future: 15K+ (with continuous crawling)
```

### Data Efficiency by Model Size

**Research Shows** (from LLM literature):

| Model Size | Optimal Data | Data Efficiency | Reason                                       |
| ---------- | ------------ | --------------- | -------------------------------------------- |
| 2B         | 4K-8K        | High            | Quick saturation, limited capacity           |
| 3-4B       | 6K-10K       | Medium-High     | Good balance                                 |
| 7B         | 10K-15K      | Medium          | Needs more data to shine                     |
| 13B+       | 20K-50K      | Lower           | Requires lots of data to avoid undertraining |

**Our Strategy**:

```
Qwen 2.5 3B:
  ✓ Optimal at 8K samples (current Phase 3)
  ✓ Marginal gains beyond 10K

Qwen 2.5 Coder 7B:
  ⏳ Start at 2K (validation)
  ⏳ Optimal at 11K (full v29)
  🔮 Best at 15K+ (future with RAG + crawling)
```

---

## 🎯 CONCRETE EXPERIMENT PLAN

### Week 1: Small Models (CURRENT) ⏳

```bash
# Running now
python experiments/run_sft_experiment.py --phase 2 --auto_crawl

Status: Phase 2 in progress (4K samples)
ETA: 6 hours remaining
Next: Phase 3 (8K samples) auto-starts after Phase 2
```

### Week 2: 4-Model Parallel Smoke Test 🔥🔥🔥🔥

**Step 1: Create 4 Configs (Optimized for 24GB)**

```yaml
# configs/sft_qwen_coder_7b_smoke.yaml (GPU 0)
model_name: "Qwen/Qwen2.5-Coder-7B-Instruct"
output_dir: "experiments/qwen_7b_smoke_2k"
train_path: "data/splits/sft_experiment/train_2k_qwen.jsonl"
eval_path: "data/splits/sft_experiment/val_2k_qwen.jsonl"

load_in_4bit: true
bnb_4bit_compute_dtype: "bfloat16"

lora_r: 8
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

num_train_epochs: 1
per_device_train_batch_size: 4 # ↑ 4x larger (24GB allows this)
gradient_accumulation_steps: 4 # Effective batch = 16
learning_rate: 2.0e-4
fp16: false
bf16: true
gradient_checkpointing: false

seq_length: 1024 # ↑ Full context (24GB allows this)
```

```yaml
# configs/sft_deepseek_coder_67b_smoke.yaml (GPU 1)
model_name: "deepseek-ai/deepseek-coder-6.7b-instruct"
output_dir: "experiments/deepseek_67b_smoke_2k"
# ... same training params as Qwen ...
per_device_train_batch_size: 4
seq_length: 1024
```

```yaml
# configs/sft_codellama_7b_smoke.yaml (GPU 2)
model_name: "codellama/CodeLlama-7b-Instruct-hf"
output_dir: "experiments/codellama_7b_smoke_2k"
# ... same training params ...
per_device_train_batch_size: 4
seq_length: 1024
```

```yaml
# configs/sft_starcoder2_7b_smoke.yaml (GPU 3)
model_name: "bigcode/starcoder2-7b"
output_dir: "experiments/starcoder2_7b_smoke_2k"
# ... same training params ...
per_device_train_batch_size: 4
seq_length: 1024
```

**Step 2: Launch All 4 Simultaneously**

```bash
# Create launcher script
cat > run_4gpu_smoke_test.sh << 'EOF'
#!/bin/bash

cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber
source .venv/bin/activate

# Launch all 4 models in parallel, each on separate GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train_red.py \
  --config configs/sft_qwen_coder_7b_smoke.yaml \
  > logs/qwen_7b_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/train_red.py \
  --config configs/sft_deepseek_coder_67b_smoke.yaml \
  > logs/deepseek_67b_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/train_red.py \
  --config configs/sft_codellama_7b_smoke.yaml \
  > logs/codellama_7b_smoke.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/train_red.py \
  --config configs/sft_starcoder2_7b_smoke.yaml \
  > logs/starcoder2_7b_smoke.log 2>&1 &

echo "All 4 models training in parallel!"
echo "Monitor logs:"
echo "  tail -f logs/qwen_7b_smoke.log"
echo "  tail -f logs/deepseek_67b_smoke.log"
echo "  tail -f logs/codellama_7b_smoke.log"
echo "  tail -f logs/starcoder2_7b_smoke.log"

# Wait for all to complete
wait
echo "All 4 smoke tests completed!"
EOF

chmod +x run_4gpu_smoke_test.sh
./run_4gpu_smoke_test.sh

# Expected duration: ~6 hours (same as 1 model, but 4x results!)
# Monitor GPU usage: nvidia-smi -l 1
```

**Step 3: Monitor Training Progress**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check all 4 logs simultaneously
tmux new-session \; \
  split-window -v \; \
  split-window -h \; \
  select-pane -t 0 \; \
  split-window -h \; \
  select-pane -t 0 \; send-keys 'tail -f logs/qwen_7b_smoke.log' C-m \; \
  select-pane -t 1 \; send-keys 'tail -f logs/deepseek_67b_smoke.log' C-m \; \
  select-pane -t 2 \; send-keys 'tail -f logs/codellama_7b_smoke.log' C-m \; \
  select-pane -t 3 \; send-keys 'tail -f logs/starcoder2_7b_smoke.log' C-m
```

**Step 4: Comprehensive Comparison**

```bash
# After all 4 complete, compare ALL models (including 3B baseline)
python scripts/compare_7_models.py \
  --models \
    experiments/sft_qwen_phase1_2k \
    experiments/qwen_7b_smoke_2k \
    experiments/deepseek_67b_smoke_2k \
    experiments/codellama_7b_smoke_2k \
    experiments/starcoder2_7b_smoke_2k \
  --test_set data/splits/sft_experiment/test_200_qwen.jsonl \
  --waf_url http://localhost:8000 \
  --output reports/4gpu_smoke_test_comparison.md

# Expected results:
# Qwen 7B:      25-30% (best)
# DeepSeek 6.7B: 23-28%
# CodeLlama 7B:  20-25%
# StarCoder2 7B: 22-26%
# Qwen 3B:       18-22% (baseline)
```

### Week 3: Top-2 Progressive Scale (Parallel) 📈📈

**Decision Point**: After smoke test, pick **top 2 models** for progressive scaling

**Phase A: 4K samples (Top 2 in parallel)**

```bash
# Example: If Qwen and DeepSeek are top 2
CUDA_VISIBLE_DEVICES=0 python scripts/train_red.py \
  --config configs/sft_qwen_coder_7b_4k.yaml \
  > logs/qwen_7b_4k.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/train_red.py \
  --config configs/sft_deepseek_coder_67b_4k.yaml \
  > logs/deepseek_67b_4k.log 2>&1 &

wait
# Duration: ~8 hours (same time, 2 results)
# Use train_4k_*.jsonl, 2 epochs
```

**Phase B: 8K samples (Top 2 in parallel)**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_red.py \
  --config configs/sft_qwen_coder_7b_8k.yaml \
  > logs/qwen_7b_8k.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/train_red.py \
  --config configs/sft_deepseek_coder_67b_8k.yaml \
  > logs/deepseek_67b_8k.log 2>&1 &

wait
# Duration: ~12 hours, 1 epoch
```

**Evaluation Checkpoint**:

```bash
# Compare top 2 against each other + 3B baseline
python scripts/comprehensive_eval.py \
  --models \
    experiments/sft_qwen_phase3_8k \
    experiments/qwen_7b_8k \
    experiments/deepseek_67b_8k \
  --metrics pass_rate,diversity,novelty,syntax_valid \
  --test_count 500

# Decision point:
# Pick absolute best model for full 11K training
# If tie → train both on 11K (2 GPUs still available)
```

### Week 4: 7B Full Scale (CONDITIONAL) 🎯

**Only if Week 3 shows clear improvement**

```yaml
# configs/sft_qwen_coder_7b_full.yaml

train_path: "data/processed/red_v29_enriched.jsonl" # All 11,692 samples
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 # Larger effective batch
seq_length: 1024 # Full context for complex payloads

# Duration: ~24 hours
# GPU: Requires 24GB VRAM minimum
```

**Run Full Training**:

```bash
# Final production model training
python scripts/train_red.py \
  --config configs/sft_qwen_coder_7b_full.yaml \
  --output experiments/qwen_7b_production_v1

# Monitor carefully:
# - GPU memory usage
# - Loss curve (should decrease smoothly)
# - Validation loss (check for overfitting)
```

**Production Evaluation**:

```bash
# Comprehensive test suite
python scripts/production_eval.py \
  --model experiments/qwen_7b_production_v1 \
  --test_suite full \
  --waf_types modsecurity,cloudflare,akamai \
  --attack_types xss,sqli \
  --num_tests 1000

# Success criteria:
# - WAF bypass rate: >25%
# - Syntax validity: >95%
# - Payload diversity: >1000 unique
# - No catastrophic failures
```

---

## 💰 COST-BENEFIT ANALYSIS (4x RTX 3090 Setup)

### GPU Time Investment (WITH PARALLELIZATION)

| Stage          | Models | Samples  | Wall Time | GPU·Hours | Cumulative | Risk      |
| -------------- | ------ | -------- | --------- | --------- | ---------- | --------- |
| ✅ Current     | 3×3B   | 2K/4K/8K | 12h       | 36h       | 12h        | Low       |
| ⏳ Smoke       | 4×7B   | 2K       | **6h**    | 24h       | 18h        | Low       |
| ⏳ Progressive | 2×7B   | 4K/8K    | **20h**   | 40h       | 38h        | Medium    |
| 🔮 Full        | 1-2×7B | 11K      | **24h**   | 24-48h    | 62h        | High      |
| 🔮 RL          | 1×7B   | Online   | 48h       | 48h       | 110h       | Very High |

**Parallelization Advantage**:

```
Without 4 GPUs:
- Test 4 models sequentially: 4 × 6h = 24h
- Total to find best: 24h + 20h + 24h = 68h

With 4 GPUs:
- Test 4 models in parallel: 6h
- Total to find best: 6h + 20h + 24h = 50h
- Time saved: 18 hours (26% faster)
```

**Decision Points**:

- After 18h: Which of 4 models is best? → Pick top 2
- After 38h: Which of top 2 scales better? → Pick winner
- After 62h: Is production model good enough? → RL or deploy

### Expected ROI (Multi-GPU Setup)

**Small Models (2-4B)** - Sequential:

- Wall Time: 12h
- GPU·Hours: 36h
- Performance: 15-20% bypass
- Cost: 1x
- **ROI**: Excellent (fast validation)

**Large Models (7B) - 4-Way Smoke Test** - **PARALLEL**:

- Wall Time: **6h** (not 24h!)
- GPU·Hours: 24h
- Performance: 18-28% bypass (best of 4)
- Cost: 1.5x
- **ROI**: OUTSTANDING (4 experiments for price of 1)

**Large Models (7B) - Progressive (Top 2)** - **PARALLEL**:

- Wall Time: **20h** (not 40h!)
- GPU·Hours: 40h
- Performance: 20-30% bypass
- Cost: 2.5x
- **ROI**: Excellent (2 candidates validated)

**Large Models (7B) - Full Scale (Winner)** - Single:

- Wall Time: 24h
- GPU·Hours: 24-48h (can train 1-2 in parallel)
- Performance: 25-30% bypass
- Cost: 3x
- **ROI**: Excellent (proven model from progressive validation)

**RL Fine-tuning (7B)** - Single:

- Wall Time: 48h
- GPU·Hours: 48h
- Performance: 30-35% bypass (+5-10% over SFT)
- Cost: 5x
- **ROI**: Highest potential, moderate risk (SFT base proven)

### Hardware Utilization

**4x RTX 3090 Efficiency**:

| Stage                   | GPUs Used | Utilization | Wasted Capacity |
| ----------------------- | --------- | ----------- | --------------- |
| Smoke Test (4 models)   | 4/4       | 100%        | 0% ✅           |
| Progressive (2 models)  | 2/4       | 50%         | 50%             |
| Full Scale (1-2 models) | 1-2/4     | 25-50%      | 50-75%          |
| RL Training (1 model)   | 1/4       | 25%         | 75%             |

**Optimization Ideas**:

```
Progressive Stage: Use spare GPUs for
├── GPU 0: Qwen 7B (4K/8K)
├── GPU 1: DeepSeek 6.7B (4K/8K)
├── GPU 2: RAG system development (ChromaDB)
└── GPU 3: Continuous evaluation/testing

Full Scale: Use spare GPUs for
├── GPU 0: Best model (11K training)
├── GPU 1: Runner-up model (11K training as backup)
├── GPU 2: RL environment setup
└── GPU 3: Live WAF testing pipeline
```

---

## 🎓 MODEL SELECTION RATIONALE

### With 4x RTX 3090 (24GB each) → Train 4 Models in Parallel! 🚀

**Hardware Advantage**:

```
4 GPUs × 24GB VRAM = 96GB total
→ Can train 4 different 7B models simultaneously
→ Massive time savings: 4 experiments in parallel
→ Higher batch sizes possible (2-4 per GPU)
```

**Top 4 Models for WAF Bypass (7B Class)**:

| Rank | Model                     | Size | Strengths                                                                          | Why Include         | GPU Assignment |
| ---- | ------------------------- | ---- | ---------------------------------------------------------------------------------- | ------------------- | -------------- |
| 🥇 1 | **Qwen 2.5 Coder 7B**     | 7B   | ✅ Code-specialized<br>✅ Multilingual<br>✅ Instruction-tuned<br>✅ Latest (2024) | Best for payloads   | GPU 0          |
| 🥈 2 | **DeepSeek Coder 6.7B**   | 6.7B | ✅ Security focus<br>✅ Strong obfuscation<br>✅ Community proven                  | Security specialist | GPU 1          |
| 🥉 3 | **CodeLlama 7B Instruct** | 7B   | ✅ Meta-backed<br>✅ Well-tested<br>✅ Strong baseline                             | Reliable benchmark  | GPU 2          |
| 🏅 4 | **StarCoder2 7B**         | 7B   | ✅ Code generation<br>✅ Multi-language<br>✅ Good at patterns                     | Pattern recognition | GPU 3          |

**Why These 4?**

1. **Qwen 2.5 Coder 7B** (GPU 0):

   - **Primary candidate** - newest, best code understanding
   - Handles Unicode/obfuscation well
   - Strong instruction following
   - Expected: 25-30% WAF bypass

2. **DeepSeek Coder 6.7B** (GPU 1):

   - **Security specialist** - trained on security code
   - Excellent at SQL/XSS syntax
   - Good at evasion techniques
   - Expected: 23-28% WAF bypass

3. **CodeLlama 7B Instruct** (GPU 2):

   - **Proven baseline** - most tested in community
   - Meta's quality assurance
   - Good general code understanding
   - Expected: 20-25% WAF bypass

4. **StarCoder2 7B** (GPU 3):
   - **Pattern specialist** - trained on massive code corpus
   - Excellent at learning attack patterns
   - Good at generating variations
   - Expected: 22-26% WAF bypass

### Parallel Training Strategy

**Simultaneous Launch**:

```bash
# All 4 models start at the same time on separate GPUs
CUDA_VISIBLE_DEVICES=0 python scripts/train_red.py --config configs/sft_qwen_7b_smoke.yaml &
CUDA_VISIBLE_DEVICES=1 python scripts/train_red.py --config configs/sft_deepseek_67b_smoke.yaml &
CUDA_VISIBLE_DEVICES=2 python scripts/train_red.py --config configs/sft_codellama_7b_smoke.yaml &
CUDA_VISIBLE_DEVICES=3 python scripts/train_red.py --config configs/sft_starcoder2_7b_smoke.yaml &

# Compare after 6h → pick top 2 for progressive scaling
```

### Optimized Batch Sizes for 24GB VRAM

**With 24GB per GPU, can use larger batches**:

```yaml
# Each model config (24GB VRAM)
per_device_train_batch_size: 4 # ↑ from 1 (4x faster)
gradient_accumulation_steps: 4 # Effective batch = 16
seq_length: 1024 # Full context (not 512)
load_in_4bit: true # Still use quantization for safety
bf16: true # Fast + stable

# Training speed:
# - Small models (2-4B): ~2s/step
# - Large models (7B): ~6s/step (vs 15s/step with batch_size=1)
# Time savings: 2.5x faster per experiment
```

**Memory Budget per GPU**:

```
Model (4-bit):        ~4GB
Activations (bs=4):   ~8GB
Optimizer states:     ~6GB
Gradients:           ~3GB
Buffer:              ~3GB
Total:              ~24GB ✅ Perfect fit
```

---

## 📊 SUCCESS METRICS & DECISION CRITERIA

### After Small Models (This Week)

```python
# Evaluate all 3 small models
results = {
    'gemma_2b': {'pass_rate': 0.XX, 'valid': 0.XX, 'diversity': XXX},
    'phi3_38b': {'pass_rate': 0.XX, 'valid': 0.XX, 'diversity': XXX},
    'qwen_3b': {'pass_rate': 0.XX, 'valid': 0.XX, 'diversity': XXX}
}

# Decision: Proceed to 7B if:
best_small = max(results, key=lambda x: results[x]['pass_rate'])
if results[best_small]['pass_rate'] > 0.15:  # 15% threshold
    print("✅ Small models validated, proceed to 7B smoke test")
else:
    print("❌ Data quality issue, fix dataset before scaling")
```

### After 7B Smoke Test (Week 2)

```python
# Compare 7B vs best small model on same 2K data
improvement = (result_7b['pass_rate'] - result_3b['pass_rate']) / result_3b['pass_rate']

if improvement > 0.15:  # >15% relative improvement
    print("✅ 7B shows clear advantage, proceed to progressive scaling")
elif improvement > 0.05:  # 5-15% improvement
    print("⚠️ Marginal improvement, consider cost-benefit")
else:
    print("❌ No clear benefit, stick with 3B for production")
```

### After 7B Progressive (Week 3)

```python
# Check scaling trend
improvements = [
    result_7b_2k['pass_rate'],
    result_7b_4k['pass_rate'],
    result_7b_8k['pass_rate']
]

# Should see increasing trend
if improvements[2] > improvements[1] > improvements[0]:
    print("✅ Positive scaling trend, proceed to full dataset")
elif improvements[2] < improvements[1]:
    print("❌ Saturation detected, no benefit from more data")
```

---

## 🚀 EXECUTION CHECKLIST

### Pre-flight Checks (Before 7B Training)

- [ ] Current experiment (small models) completed
- [ ] Best small model identified (likely Qwen 3B)
- [ ] Hyperparameters validated (learning rate, batch size)
- [ ] Data quality confirmed (>15% pass rate)
- [ ] GPU available (24GB VRAM required)
- [ ] Disk space sufficient (50GB+ free)

### 7B Smoke Test Checklist

- [ ] Config created (`sft_qwen_coder_7b_smoke.yaml`)
- [ ] Dataset ready (`train_2k_qwen.jsonl`)
- [ ] GPU memory monitoring active
- [ ] Training started successfully
- [ ] Loss curve tracked (TensorBoard/WandB)
- [ ] Evaluation against 3B baseline

### Progressive Scaling Checklist

- [ ] Smoke test successful (no OOM, good results)
- [ ] 4K training completed
- [ ] 8K training completed
- [ ] Comparison report generated
- [ ] Decision: proceed to full or stop

### Full Scale Checklist

- [ ] Progressive results show >5% improvement
- [ ] GPU available for 24h continuous run
- [ ] Monitoring setup (alerts for failures)
- [ ] Production evaluation plan ready
- [ ] Deployment strategy defined

---

## 🎯 RECOMMENDED TIMELINE (4-GPU Optimized)

```
Week 1 (Nov 28 - Dec 5):
├── Day 1-2: Monitor small models Phase 2 completion ✅
├── Day 3-4: Analyze results, select best small model
├── Day 5-7: Prepare 4×7B configs, setup multi-GPU environment
└── Deliverable: 4 training configs ready + GPU setup tested

Week 2 (Dec 6 - Dec 12): 🔥 4-MODEL PARALLEL BLITZ
├── Day 1: Launch all 4 models simultaneously (2K samples)
│         GPU 0: Qwen 2.5 Coder 7B
│         GPU 1: DeepSeek Coder 6.7B
│         GPU 2: CodeLlama 7B Instruct
│         GPU 3: StarCoder2 7B
├── Day 2: Monitor all 4 trainings (6h total)
├── Day 3: Evaluate all 4 + compare → Pick TOP 2
├── Day 4-5: Launch top 2 on 4K (parallel, 8h)
├── Day 6-7: Evaluate 4K results
└── Deliverable: Top 2 models identified, 4K validation complete

Week 3 (Dec 13 - Dec 19): 📈 PROGRESSIVE SCALING
├── Day 1-3: Launch top 2 on 8K (parallel, 12h)
│           GPU 0: Best model (8K)
│           GPU 1: Runner-up (8K)
│           GPU 2: RAG system setup (ChromaDB)
│           GPU 3: Continuous evaluation
├── Day 4: Comprehensive evaluation of top 2 @ 8K
├── Day 5: Decision: Which model(s) for full 11K?
├── Day 6-7: Prep full-scale configs + dataset
└── Deliverable: Winner selected, ready for full training

Week 4 (Dec 20 - Dec 26): 🎯 FULL SCALE + RL PREP
├── Day 1-2: Launch full 11K training (24h)
│            GPU 0: Winner (11K)
│            GPU 1: Runner-up (11K, optional backup)
│            GPU 2: RL environment testing
│            GPU 3: Production API prototype
├── Day 3-4: Production evaluation of 11K models
├── Day 5: Decision point - RL training or deploy?
├── Day 6-7: Start RL training OR deploy SFT model
└── Deliverable: Production model + RL pipeline OR deployed API
```

**Time Savings Summary**:

```
Traditional (1 GPU):
├── Test 4 models: 24h
├── Progressive (2 models): 40h
├── Full scale: 24h
└── Total: 88 hours

With 4 GPUs (Parallel):
├── Test 4 models: 6h (4x speedup)
├── Progressive (2 models): 20h (2x speedup)
├── Full scale: 24h (can do 2 in parallel)
└── Total: 50 hours

Savings: 38 hours (43% faster) 🚀
```

---

## 💡 KEY TAKEAWAYS (4-GPU Edition)

1. **Start Small, Scale Smart** ✅

   - Validate with 2-4B models first (done)
   - Test 7B on small data before full scale
   - Progressive scaling reduces risk
   - **NEW**: Parallel testing eliminates sequential bottleneck

2. **Sample Size Matters**

   - 2B models: Optimal at 4-8K samples
   - 7B models: Optimal at 10-15K samples
   - Don't waste time overtraining small models on huge data
   - **NEW**: With 24GB VRAM, can use 4x larger batch sizes

3. **Parallelization is Key** 🚀

   - 4 GPUs = 4x experiments simultaneously
   - Time savings: 38 hours (43% faster)
   - Test diversity: 4 different architectures
   - Risk mitigation: If 1 model fails, 3 others still running

4. **Optimized Batch Sizes**

   - Small models (8-12GB): batch_size=2-4
   - Large models (24GB): batch_size=4, seq_length=1024
   - Effective training speed: 2.5x faster per model
   - Memory utilization: 95%+ (no waste)

5. **Decision Gates**

   - ✅ Small models >15% → Proceed to 7B (4 models)
   - ✅ 7B smoke (4 models) → Pick top 2
   - ✅ Progressive (top 2) → Pick winner
   - ✅ Full training >25% → RL fine-tuning

6. **Hardware Utilization Strategy**

   - Smoke test: 100% utilization (4/4 GPUs)
   - Progressive: 50% training + 50% RAG/eval
   - Full scale: 25-50% training + 50-75% RL prep/testing
   - Always keep GPUs busy with productive work

7. **Have Multiple Fallbacks**
   - If all 7B fail → Deploy best 3B model
   - If top 2 fail → Use 3rd/4th place from smoke test
   - If full scale fails → Use 8K checkpoint
   - If RL fails → Use best SFT model
   - **NEW**: 4 parallel experiments = 4 backup options

---

## 🎯 FINAL RECOMMENDATION

**Immediate Next Steps**:

1. **Wait for 3B Experiment to Complete** (current)

   - Let Phase 2-3 finish
   - Generate comparison report
   - Confirm best small model

2. **Rent 4x RTX 3090 24GB GPUs** (Week 2)

   - Providers: Vast.ai, RunPod, Lambda Labs
   - Cost: ~$2-3/GPU/hour = $8-12/hour total
   - 50 hours needed = ~$400-600 total investment

3. **Create 4 Training Configs** (Week 2 prep)

   - Qwen 2.5 Coder 7B
   - DeepSeek Coder 6.7B
   - CodeLlama 7B Instruct
   - StarCoder2 7B

4. **Launch 4-Model Parallel Smoke Test** (Week 2)

   - All 4 models @ 2K samples
   - 6 hours wall time
   - Pick top 2 for progressive scaling

5. **Progressive Scale Top 2** (Week 3)

   - 4K + 8K in parallel
   - Use spare GPUs for RAG development
   - Pick ultimate winner

6. **Full Scale Winner** (Week 4)
   - 11K samples, 24h training
   - Production evaluation
   - RL preparation or deployment

**Expected Outcome**:

- 4 model architectures tested
- Best model identified with confidence
- 25-30% WAF bypass rate (SFT)
- 30-35% after RL fine-tuning
- Production-ready by end of Week 4

---

**Next Step**: Monitor current 3B training → Prepare 4-GPU infrastructure! 🚀
