# Dataset Unification Summary

## 📋 Final Dataset Naming Convention

### Phase 1: Basic SFT - Stratified Sampling

- **File**: `data/processed/phase1_balanced_10k.jsonl`
- **Samples**: 10,000
- **Techniques**: 509 (all techniques represented proportionally)
- **Creation**: `scripts/create_balanced_phase1_10k.py`
- **Source**: `data/processed/phase1_passed_only_39k.jsonl` (stratified sampling)
- **Purpose**: Ensure comprehensive technique coverage in training subset

**Critical Fix**: Previous simple `head -n 10000` only captured 3 techniques (XSS Keyword, Triple URL, SQLI Keyword). Stratified sampling ensures all 509 techniques are represented proportionally.

### Phase 2: Adaptive Learning - Observations + Replay Buffer

- **File**: `data/processed/phase2_with_replay_22k.jsonl`
- **Total Samples**: 22,000
  - 20,000 from Phase 2 observations
  - 2,000 from Phase 1 replay buffer (10% replay rate)
- **Techniques**: 517 (extracted from observations)
- **Creation**: `scripts/create_phase2_with_replay.py`
- **Sources**:
  - `data/processed/phase2_observations_20k.jsonl`
  - `data/processed/phase1_balanced_10k.jsonl` (random 2k sample)
- **Purpose**: Prevent catastrophic forgetting during continual learning

### Phase 3: RL Training - Live WAF Testing

- **No static dataset** (live interaction with WAF)
- **WAF URL**: `http://modsec.llmshield.click`
- **Episodes**: 150
- **Estimated Tests**: ~900 (6 tests per episode)
- **Authentication**: Environment variables (`DVWA_USERNAME`, `DVWA_PASSWORD`)

---

## 🔄 Training Pipeline

### Sequential Training Strategy (One Model at a Time)

```
Model Training Flow:
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Basic SFT                                          │
│ Dataset: phase1_balanced_10k.jsonl (509 techniques)         │
│ Time: ~2-3 hours                                            │
│ Output: experiments/remote_<model>_phase1/                  │
└────────────────────┬────────────────────────────────────────┘
                     │ Load adapter
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Adaptive Learning (Continual)                      │
│ Dataset: phase2_with_replay_22k.jsonl (20k + 2k replay)     │
│ Config: adapter_path = Phase 1, is_trainable = true         │
│ Time: ~2-3 hours                                            │
│ Output: experiments/remote_<model>_phase2/                  │
└────────────────────┬────────────────────────────────────────┘
                     │ Load adapter
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: RL Training (REINFORCE)                            │
│ WAF: http://modsec.llmshield.click (150 episodes)           │
│ Config: adapter_path = Phase 2                              │
│ Time: ~3-5 hours                                            │
│ Output: experiments/remote_<model>_phase3_rl/               │
└─────────────────────────────────────────────────────────────┘
```

### Model Training Order

1. **Gemma 2B** (Train first - fastest iteration)
   - Total time: ~6-9 hours
   - VRAM: 18-22GB
2. **Phi-3 Mini** (Train second)
   - Total time: ~6-9 hours
   - VRAM: 18-22GB
   - Note: Requires DynamicCache fix in `train_red.py`
3. **Qwen 7B** (Train last - if time allows)
   - Total time: ~10-15 hours
   - VRAM: 22-24GB
   - Config: `batch_size: 1`, `gradient_accumulation_steps: 16`

---

## 📁 Config Files Mapping

### Phase 1 Configs

- `configs/remote_gemma2_2b_phase1.yaml`
- `configs/remote_phi3_mini_phase1.yaml`
- `configs/remote_qwen_7b_phase1.yaml`

**Common settings:**

- `train_path: "data/processed/phase1_balanced_10k.jsonl"`
- `epochs: 3`
- `max_seq_length: 1024`

### Phase 2 Configs

- `configs/remote_gemma2_2b_phase2.yaml`
- `configs/remote_phi3_mini_phase2.yaml`
- `configs/remote_qwen_7b_phase2.yaml`

**Common settings:**

- `train_path: "data/processed/phase2_with_replay_22k.jsonl"`
- `adapter_path: "experiments/remote_<model>_phase1"`
- `is_trainable: true` (continual learning enabled)
- `epochs: 2`
- `max_seq_length: 1536` (longer for observation reasoning)

### Phase 3 Configs (RL)

- `configs/gemma2_2b_phase3_rl.yaml`
- `configs/phi3_mini_phase4_rl_200epochs.yaml`
- `configs/qwen_3b_phase4_rl_200epochs.yaml`

**Common settings:**

- `adapter_path: "experiments/remote_<model>_phase2"`
- `waf_url: "http://modsec.llmshield.click"`
- `episodes: 150`
- `algorithm: "reinforce"`

---

## 🛠️ Scripts Overview

### Dataset Generation Scripts

**`scripts/create_balanced_phase1_10k.py`**

- **Purpose**: Create stratified 10k subset from 39k Phase 1 dataset
- **Algorithm**: Proportional sampling per technique
- **Input**: `data/processed/phase1_passed_only_39k.jsonl`
- **Output**: `data/processed/phase1_balanced_10k.jsonl`
- **Key Feature**: Ensures all 509 techniques represented

**`scripts/create_phase2_with_replay.py`**

- **Purpose**: Combine Phase 2 observations with Phase 1 replay buffer
- **Algorithm**:
  1. Load 20k Phase 2 observations
  2. Random sample 2k from Phase 1 (10% replay rate)
  3. Shuffle combined dataset
- **Inputs**:
  - `data/processed/phase2_observations_20k.jsonl`
  - `data/processed/phase1_balanced_10k.jsonl`
- **Output**: `data/processed/phase2_with_replay_22k.jsonl`
- **Purpose**: Prevent catastrophic forgetting

**`prepare_datasets.sh`**

- **Purpose**: Auto-generate both datasets with one command
- **Usage**: `bash prepare_datasets.sh`
- **Actions**:
  1. Creates Phase 1 balanced dataset (if missing)
  2. Creates Phase 2 with replay dataset (if missing)
  3. Shows dataset summary and sample

### Training Scripts

**`scripts/train_red.py`**

- Phase 1 and Phase 2 SFT training
- Usage: `python scripts/train_red.py --config <config_path>`

**`scripts/train_rl_reinforce.py`**

- Phase 3 RL training with REINFORCE algorithm
- Connects to remote WAF via `waf_url` config parameter
- Usage: `python scripts/train_rl_reinforce.py --config <config_path>`

---

## ✅ Verification Checklist

Before remote training, ensure:

- [ ] Both datasets generated: `bash prepare_datasets.sh`
- [ ] Phase 1 dataset has 10,000 samples
- [ ] Phase 2 dataset has 24,000 samples
- [ ] All config files point to correct datasets (verified ✅)
- [ ] Environment variables set: `HF_TOKEN`, `DVWA_USERNAME`, `DVWA_PASSWORD`
- [ ] Git repository clean and committed
- [ ] WAF accessible: `curl http://modsec.llmshield.click`

---

## 📊 Expected Training Metrics

### Phase 1 (Basic SFT)

- **Initial Loss**: ~2.5
- **Final Loss**: ~0.8-1.2
- **Bypass Rate**: 60-70% (baseline)
- **GPU Usage**: 18-22GB VRAM

### Phase 2 (Adaptive Learning)

- **Initial Loss**: ~0.8 (continues from Phase 1)
- **Final Loss**: ~0.5-0.7
- **Bypass Rate**: 70-80%
- **GPU Usage**: 18-22GB VRAM

### Phase 3 (RL Training)

- **Initial Bypass Rate**: ~70-80% (from Phase 2)
- **Final Bypass Rate**: 80-90%+ (target)
- **Episodes**: 150 (6 tests each = 900 total tests)
- **GPU Usage**: 18-22GB VRAM

---

## 🚨 Critical Issues Fixed

### Issue 1: Technique Diversity in Phase 1

**Problem**: Simple `head -n 10000` only captured 3 techniques out of 509

- XSS Keyword
- Triple URL Encoding
- SQLI Keyword

**Solution**: Stratified sampling (`create_balanced_phase1_10k.py`)

- Analyzes all 39k samples to extract technique distribution
- Samples proportionally from each technique
- **Result**: All 509 techniques represented in 10k samples

### Issue 2: Catastrophic Forgetting in Phase 2

**Problem**: Phase 2 observations-only training could forget Phase 1 knowledge

**Solution**: Replay buffer (20% of Phase 2 size)

- Mixes 20k Phase 2 observations + 4k random Phase 1 samples
- Shuffles to prevent ordering bias
- **Result**: Maintains Phase 1 technique diversity during Phase 2 training

### Issue 3: Inconsistent Naming

**Problem**: Multiple naming schemes (red_v40, phase1_10k, etc.)

**Solution**: Unified naming convention

- `phase1_balanced_10k.jsonl`
- `phase2_with_replay_22k.jsonl`
- All configs updated to reference correct files

---

## 📖 References

- **Training Guide**: `REMOTE_TRAINING_GUIDE.md`
- **Dataset Preparation**: `prepare_datasets.sh`
- **Phase 1 Script**: `scripts/create_balanced_phase1_10k.py`
- **Phase 2 Script**: `scripts/create_phase2_with_replay.py`
- **RL Environment**: `rl/waf_env.py` (supports remote WAF URL + auth)

---

**Last Updated**: 2024-12-07  
**Status**: ✅ Ready for remote RTX 4090 training
