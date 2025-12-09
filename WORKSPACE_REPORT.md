# Workspace Organization Report

**Generated:** December 9, 2025  
**Purpose:** Comprehensive workspace structure for thesis/report writing

---

## 📊 Executive Summary

### Workspace Statistics

| Category          | Count                         | Total Size | Status                              |
| ----------------- | ----------------------------- | ---------- | ----------------------------------- |
| **Datasets**      | 18 JSONL files                | 695 MB     | ✅ Production ready                 |
| **Configs**       | 34 YAML files                 | -          | ✅ 9 active (remote\_\*), 25 legacy |
| **Scripts**       | 40+ active Python             | -          | ✅ Cleaned & organized              |
| **Documentation** | 6 Markdown files              | 82 KB      | ✅ Current & relevant               |
| **Archived**      | 16 scripts + 6 docs + 12 logs | 366 KB     | 📦 Preserved for reference          |

---

## 1️⃣ DATASETS (`data/processed/`)

### 🎯 Active Training Datasets (3 files - 45 MB)

#### **Phase 1: Basic SFT**

```
phase1_balanced_10k.jsonl                    4.51 MB
├─ Samples: 10,000
├─ Techniques: 509 (stratified sampling)
├─ Source: phase1_passed_only_39k.jsonl
├─ Purpose: Comprehensive technique coverage
└─ Used by: remote_*_phase1.yaml configs
```

#### **Phase 2: Adaptive Learning**

```
phase2_with_replay_22k.jsonl                22.71 MB
├─ Samples: 22,000 (20k observations + 2k replay)
├─ Techniques: 517
├─ Sources:
│   ├─ phase2_observations_20k.jsonl (20k)
│   └─ phase1_balanced_10k.jsonl (2k random sample)
├─ Purpose: Continual learning with catastrophic forgetting prevention
└─ Used by: remote_*_phase2.yaml configs
```

#### **Phase 3: RL Training**

```
No static dataset (live WAF interaction)
├─ WAF URL: http://modsec.llmshield.click
├─ Episodes: 150
├─ Tests per episode: ~6
├─ Total interactions: ~900
└─ Used by: remote_*_phase3_rl.yaml configs
```

### 📚 Source Datasets (3 files - 48.3 MB)

| File                            | Size     | Samples | Purpose                               |
| ------------------------------- | -------- | ------- | ------------------------------------- |
| `phase1_passed_only_39k.jsonl`  | 17.74 MB | 39,155  | Full Phase 1 dataset (509 techniques) |
| `phase2_observations_20k.jsonl` | 20.91 MB | 20,000  | Phase 2 observations from RL          |
| `phase2_observations_10k.jsonl` | 10.4 MB  | 10,000  | Subset for testing                    |

### 🧪 Legacy/Experimental Datasets (12 files - 69 MB)

**RED Agent Experiments:**

- `red_phase1_enriched_v2.jsonl` (18.53 MB) - Phase 1 with enrichment
- `red_phase2_rag_sft_candidates.jsonl` (0.05 MB) - RAG-aware SFT candidates
- `red_phase3_adaptive_sft.jsonl` (18.55 MB) - Adaptive SFT dataset
- `red_v40_phase2_reasoning.jsonl` (11.78 MB) - Reasoning dataset
- `red_v40_phase2_reasoning_ready.jsonl` (7.21 MB) - Preprocessed reasoning
- `red_v40_phase2_phi3_1500.jsonl` (2.14 MB) - Phi-3 specific dataset

**Evaluation Datasets:**

- `red_v40_phase2_eval_100.jsonl` (0.14 MB)
- `red_v40_phase2_eval_test.jsonl` (0.24 MB)
- `red_v40_test_200.jsonl` (0.06 MB)
- `red_v40_test_200_converted.jsonl` (0.09 MB)

**Other:**

- `phase1_failed_mixed_20k.jsonl` (6.07 MB) - Failed payloads dataset
- `phase2_rag_old.jsonl` (9.81 MB) - Old RAG dataset
- `red_v40_quick_test_1k.jsonl` (0.41 MB) - Quick test subset

### 💾 Archived Datasets (`data/archive_*`)

| Location                 | Files | Size     | Contents                         |
| ------------------------ | ----- | -------- | -------------------------------- |
| `data/archive_20251129/` | 67    | 326.3 MB | Old datasets from Nov 29 cleanup |
| `data/archive_legacy/`   | 12    | 11.1 MB  | Legacy datasets                  |

---

## 2️⃣ CONFIGURATIONS (`configs/`)

### ✅ Active Remote Training Configs (9 files)

**Phase 1 SFT (3 configs):**

```yaml
remote_gemma2_2b_phase1.yaml
├─ Model: google/gemma-2-2b-it
├─ Dataset: phase1_balanced_10k.jsonl
├─ Epochs: 3
├─ Batch size: 1, Gradient accumulation: 16
├─ Max seq length: 1024
└─ Output: experiments/remote_gemma2_2b_phase1/

remote_phi3_mini_phase1.yaml
├─ Model: microsoft/Phi-3-mini-4k-instruct
├─ Dataset: phase1_balanced_10k.jsonl
├─ Epochs: 3
├─ Max seq length: 1024
└─ Output: experiments/remote_phi3_mini_phase1/

remote_qwen_7b_phase1.yaml
├─ Model: Qwen/Qwen2.5-7B-Instruct
├─ Dataset: phase1_balanced_10k.jsonl
├─ Epochs: 3
├─ Batch size: 1 (7B model)
└─ Output: experiments/remote_qwen_7b_phase1/
```

**Phase 2 Adaptive (3 configs):**

```yaml
remote_gemma2_2b_phase2.yaml
├─ Dataset: phase2_with_replay_22k.jsonl
├─ Adapter: experiments/remote_gemma2_2b_phase1
├─ is_trainable: true (continual learning)
├─ Epochs: 3
└─ Max seq length: 1536

remote_phi3_mini_phase2.yaml
remote_qwen_7b_phase2.yaml
```

**Phase 3 RL (3 configs):**

```yaml
remote_gemma2_2b_phase3_rl.yaml
├─ Adapter: experiments/remote_gemma2_2b_phase2
├─ WAF URL: http://modsec.llmshield.click
├─ Episodes: 150
├─ Max context length: 1024
├─ Gradient checkpointing: true
└─ VRAM optimizations enabled

remote_phi3_mini_phase3_rl.yaml
remote_qwen_7b_phase3_rl.yaml
```

### 📦 Legacy Configs (25 files)

**Old Phase Naming (deprecated):**

- `phase2_gemma2_2b_reasoning.yaml` - Old Phase 2
- `phase2_phi3_mini_reasoning.yaml` - Old Phase 2
- `phase3_gemma2_2b_v38_5k.yaml` - Old Phase 3
- `phase3_phi3_mini_v38_10k.yaml` - Old Phase 3
- `phi3_mini_phase3_rl.yaml` - Old RL config
- `gemma2_2b_phase3_rl.yaml` - Old RL config
- `phi3_mini_phase4_rl_200epochs.yaml` - Misnamed (should be Phase 3)
- `qwen_3b_phase4_rl_200epochs.yaml` - Misnamed (should be Phase 3)

**RED Agent Experiments:**

- `red_phase1_phase2_combined_phi3.yaml`
- `red_phase2_rag_sft.yaml`
- `red_phase3_adaptive_*.yaml` (4 files)
- `red_phase3_lightweight_*.yaml` (6 files)
- `red_gemma2_2b_lora_v2.yaml`
- `red_phi3_mini_lora.yaml`
- `red_qwen2_7b_dora_8gb_smoke.yaml`

**Test Configs:**

- `gemma2_2b_phase1_quick_test_1k.yaml`
- `phase3_qwen_7b_fixed.yaml`

**Action:** Consider moving to `configs/archive/` after remote training completes.

---

## 3️⃣ SCRIPTS (`scripts/`)

### ✅ Active Core Scripts (10 files)

**Training:**

- `train_red.py` - Main SFT training script (Phases 1 & 2)
- `train_rl_reinforce.py` - RL training script (Phase 3)

**Dataset Creation:**

- `create_balanced_phase1_10k.py` - Stratified Phase 1 sampling
- `create_phase2_with_replay.py` - Phase 2 with replay buffer
- `analyze_phase2_seq_length.py` - Dataset sequence length analysis

**Testing & Evaluation:**

- `test_adapter_universal.py` - Universal adapter testing
- `test_remote_waf_connection.py` - WAF connectivity test
- `test_remote_waf_outbound.py` - Outbound requests test
- `test_training_payloads_strict_waf.py` - Payload validation

**Reporting:**

- `evaluate_all_adapters_for_report.py` - Comprehensive evaluation
- `evaluate_thesis_report_final.py` - Thesis final report
- `check_dataset_quality.py` - Dataset quality validation

### 📁 Script Subdirectories

| Directory             | Files | Purpose                 | Status               |
| --------------------- | ----- | ----------------------- | -------------------- |
| `etl/`                | 22    | ETL pipelines           | Active               |
| `rag/`                | 5     | RAG system scripts      | Active (RED-RAG v2)  |
| `replay/`             | 16    | Replay buffer utilities | Active               |
| `util/`               | 2     | Utility functions       | Active               |
| `icl/`                | 1     | In-context learning     | Experimental         |
| `infer/`              | 1     | Inference utilities     | Active               |
| `loop/`               | 1     | Training loops          | Active               |
| `archive_deprecated/` | 16    | Deprecated scripts      | 📦 Archived (Dec 9)  |
| `archive_20251129/`   | 41    | Old scripts             | 📦 Archived (Nov 29) |
| `archive_cleanup/`    | 19    | Cleanup artifacts       | 📦 Archived          |
| `archive_legacy/`     | 38    | Legacy scripts          | 📦 Archived          |
| `archive/`            | 11    | General archive         | 📦 Archived          |

### 🔴 RED-RAG v2 Scripts (9 files)

**Note:** Separate experimental research track, NOT part of main Phase 1→2→3 pipeline.

- `red_rag_build_index.py` - Build RAG index
- `red_rag_merge_internal_corpus.py` - Merge corpus
- `red_rag_merge_with_writeups.py` - Add writeups
- `red_rag_extract_internal_patterns.py` - Extract patterns
- `red_rag_generate_strategies_from_patterns.py` - Generate strategies
- `red_rag_generate_sft_dataset.py` - Create RAG-aware SFT dataset
- `red_rag_eval_multiwaf_extended.py` - Evaluate on multiple WAFs
- `red_rag_analyze_doc_impact.py` - Analyze document impact
- `red_rag_report_v2.py` - Generate overall report
- `build_red_rag_sft_dataset.py` - Build RAG SFT dataset

---

## 4️⃣ BASH SCRIPTS (Root)

### ✅ Active Scripts (6 files)

| Script                  | Size   | Purpose                     | Status                       |
| ----------------------- | ------ | --------------------------- | ---------------------------- |
| `setup_full_env.sh`     | 2.5 KB | Full environment setup      | ✅ For remote deployment     |
| `remote_quickstart.sh`  | 1.4 KB | Quick remote training start | ✅ Main entry point          |
| `cleanup_for_remote.sh` | 1.8 KB | Cleanup before deployment   | ✅ Pre-deployment            |
| `run_gemma_phase3.sh`   | 0.5 KB | Run Gemma RL training       | ✅ Phase 3 launcher          |
| `run_phi3_rl_200.sh`    | 0.5 KB | Run Phi-3 RL training       | ⚠️ Old (200 epochs, now 150) |
| `run_qwen_rl_200.sh`    | 0.5 KB | Run Qwen RL training        | ⚠️ Old (200 epochs, now 150) |

**PowerShell equivalent:**

- `run_qwen_rl_200.ps1` - Windows version

**Recommendation:** Update `run_*_rl_200.sh` scripts to use 150 epochs and remote configs.

---

## 5️⃣ DOCUMENTATION (Root)

### ✅ Active Documentation (6 files)

| File                       | Size    | Purpose                      | Audience           |
| -------------------------- | ------- | ---------------------------- | ------------------ |
| `README.md`                | 30.5 KB | Main project documentation   | General            |
| `DATASET_UNIFICATION.md`   | 9.3 KB  | Dataset naming & pipeline    | **Report writing** |
| `REMOTE_TRAINING_GUIDE.md` | 17.6 KB | Remote training instructions | Deployment         |
| `REMOTE_CHECKLIST.md`      | 4.7 KB  | Deployment checklist         | Operations         |
| `GEMINI.md`                | 10 KB   | RED-RAG v2 instructions      | Research           |
| `ARCHIVE_INDEX.md`         | NEW     | Archived files tracking      | Reference          |

### 📦 Archived Documentation (`docs/archive/`)

| File                         | Size   | Reason                       |
| ---------------------------- | ------ | ---------------------------- |
| `ANALYSIS_PHASE_STRATEGY.md` | 8.5 KB | Old strategy (superseded)    |
| `FUTURE_WORK_NOTES.md`       | 1.7 KB | Hypotheses (now implemented) |
| 4 `eval_*.json` files        | -      | Old evaluation results       |

---

## 6️⃣ DATA DIRECTORY DEEP DIVE

### Full Directory Structure

```
data/
├── processed/          637 files, 695 MB    ✅ Active datasets
├── archive_20251129/    67 files, 326.3 MB  📦 Nov 29 cleanup
├── archive_legacy/      12 files, 11.1 MB   📦 Old datasets
├── rag/                  9 files, 7.4 MB    ✅ RAG corpus & indexes
├── raw/                 64 files, 7.3 MB    ✅ Raw source data
├── blue/                14 files, 1.6 MB    ✅ Blue team data
├── red/                  6 files, 0.3 MB    ✅ Red team data
├── splits/               8 files, 14.1 MB   ⚠️ Old train/val splits
├── writeups/             6 files, 0.1 MB    ✅ Technique writeups
└── resources/            1 file,  0 MB       ✅ Additional resources
```

### Key Subdirectories

**`data/processed/` (695 MB)**

- **Active:** 3 files (phase1_balanced_10k, phase2_with_replay_22k, phase1_passed_only_39k)
- **Legacy:** 15 files (red\_\*, old phase datasets)
- **Status:** Ready for training

**`data/rag/` (7.4 MB)**

- RAG corpus files (red_corpus_internal_v2.jsonl)
- RAG indexes (red_rag_v2_index.pkl)
- Used by RED-RAG v2 experiments

**`data/raw/` (7.3 MB)**

- Original raw payloads
- Source data for dataset creation

**`data/splits/` (14.1 MB)**

- ⚠️ Old train/val/test splits
- **Recommendation:** Archive after confirming not needed

---

## 7️⃣ CLEANUP & ARCHIVING HISTORY

### Completed Archiving Operations

| Date         | Operation       | Files         | Size         | Location                      |
| ------------ | --------------- | ------------- | ------------ | ----------------------------- |
| Dec 9, 2025  | Scripts cleanup | 16 Python     | -3,846 lines | `scripts/archive_deprecated/` |
| Dec 9, 2025  | Logs cleanup    | 12 log/html   | 366 KB       | `logs/archive/`               |
| Dec 9, 2025  | Docs cleanup    | 2 MD + 4 JSON | -            | `docs/archive/`               |
| Nov 29, 2024 | Scripts archive | 41 Python     | -            | `scripts/archive_20251129/`   |
| Nov 29, 2024 | Data archive    | 67 JSONL      | 326.3 MB     | `data/archive_20251129/`      |

### Archive Tracking

All archived files are documented in `ARCHIVE_INDEX.md` with:

- Reason for archival
- Original purpose
- Retrieval instructions
- Superseded by (if applicable)

---

## 8️⃣ RECOMMENDATIONS FOR REPORT WRITING

### 📝 Dataset Section (Thesis/Report)

**Use these key files as references:**

1. **Dataset Creation:**

   - `scripts/create_balanced_phase1_10k.py` - Stratified sampling methodology
   - `scripts/create_phase2_with_replay.py` - Replay buffer implementation
   - `scripts/analyze_phase2_seq_length.py` - Sequence length analysis

2. **Dataset Statistics:**

   - Phase 1: 10k samples, 509 techniques (stratified)
   - Phase 2: 22k samples (20k + 2k replay), 517 techniques
   - Phase 3: ~900 live interactions

3. **Key Insights:**
   - Simple head sampling only captured 3/509 techniques
   - Stratified sampling ensures proportional representation
   - Replay buffer prevents catastrophic forgetting (10% rate)
   - Optimal max_seq_length: 1024 (100% coverage)

### 🏗️ Methodology Section

**Training Pipeline:**

1. Sequential training (Phase 1→2→3)
2. Continual learning with adapter loading
3. RL fine-tuning with remote WAF

**Refer to:**

- `DATASET_UNIFICATION.md` - Complete pipeline description
- `REMOTE_TRAINING_GUIDE.md` - Technical implementation details
- `configs/remote_*.yaml` - Hyperparameter settings

### 📊 Experiments Section

**Model Configurations:**

- Gemma 2B: Fastest iteration (6-9 hours total)
- Phi-3 Mini: Best quality (6-9 hours total)
- Qwen 7B: Largest model (10-15 hours total)

**VRAM Optimizations:**

- 16GB target (vs original 24GB)
- Batch size: 1, Gradient accumulation: 16
- Dataloader optimizations (workers=8, prefetch=4)
- RL-specific: gradient checkpointing, cache clearing

### 🧹 Workspace Cleanup Actions

**For final thesis submission:**

1. **Archive legacy configs:**

   ```bash
   mkdir configs/archive
   mv configs/red_*.yaml configs/archive/
   mv configs/phase2_*.yaml configs/archive/
   mv configs/phase3_*.yaml configs/archive/
   mv configs/phi3_mini_phase4_*.yaml configs/archive/
   ```

2. **Archive legacy datasets:**

   ```bash
   mkdir data/processed/archive
   mv data/processed/red_*.jsonl data/processed/archive/
   mv data/processed/phase2_rag_old.jsonl data/processed/archive/
   ```

3. **Archive bash scripts (after training):**

   ```bash
   mkdir scripts/archive_bash
   mv *.sh scripts/archive_bash/
   ```

4. **Update ARCHIVE_INDEX.md** after each archiving operation

---

## 9️⃣ CRITICAL FILES FOR REPRODUCTION

### Minimal Reproduction Package

**Core Scripts (4 files):**

1. `scripts/train_red.py`
2. `scripts/train_rl_reinforce.py`
3. `scripts/create_balanced_phase1_10k.py`
4. `scripts/create_phase2_with_replay.py`

**Datasets (3 files - 45 MB):**

1. `data/processed/phase1_balanced_10k.jsonl`
2. `data/processed/phase2_with_replay_22k.jsonl`
3. `data/processed/phase1_passed_only_39k.jsonl` (source)

**Configs (9 files):**

1. `configs/remote_gemma2_2b_phase1.yaml`
2. `configs/remote_gemma2_2b_phase2.yaml`
3. `configs/remote_gemma2_2b_phase3_rl.yaml`
4. `configs/remote_phi3_mini_phase1.yaml`
5. `configs/remote_phi3_mini_phase2.yaml`
6. `configs/remote_phi3_mini_phase3_rl.yaml`
7. `configs/remote_qwen_7b_phase1.yaml`
8. `configs/remote_qwen_7b_phase2.yaml`
9. `configs/remote_qwen_7b_phase3_rl.yaml`

**Documentation (3 files):**

1. `README.md`
2. `DATASET_UNIFICATION.md`
3. `REMOTE_TRAINING_GUIDE.md`

**Total:** ~50 MB (easily shareable)

---

## 🔟 FINAL CHECKLIST FOR THESIS

- [ ] Complete remote training (all 3 models)
- [ ] Generate evaluation reports (`evaluate_all_adapters_for_report.py`)
- [ ] Archive legacy configs to `configs/archive/`
- [ ] Archive legacy datasets to `data/processed/archive/`
- [ ] Archive bash scripts to `scripts/archive_bash/`
- [ ] Update `ARCHIVE_INDEX.md` with final archiving
- [ ] Create reproduction package (scripts + datasets + configs)
- [ ] Write methodology section (use `DATASET_UNIFICATION.md`)
- [ ] Write experiments section (use evaluation reports)
- [ ] Write results section (use final metrics)
- [ ] Prepare supplementary materials (GitHub repo link)

---

**Document Status:** ✅ Complete  
**Last Updated:** December 9, 2025  
**Next Update:** After remote training completion
