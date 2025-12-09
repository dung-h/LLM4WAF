# Archive Index

This file tracks all archived files and their reasons for archival.

## Archive Date: 2025-12-09

### Logs Archive (`logs/archive/`)

**Archived:** 12 files (366 KB total)

**Reason:** Old training/evaluation logs from local experiments. Cleaned workspace root for remote deployment.

| File                            | Size   | Date       | Notes                        |
| ------------------------------- | ------ | ---------- | ---------------------------- |
| eval_gemma_rl_probing.log       | 8 KB   | 2024       | Gemma RL probing evaluation  |
| eval_phi3_modsec.log            | 0 KB   | 2024       | Empty Phi-3 eval log         |
| eval_phi3_modsec_full.log       | 6 KB   | 2024       | Phi-3 ModSecurity evaluation |
| eval_run.log                    | 0.2 KB | 2024       | General evaluation run log   |
| eval_thesis_final.log           | 0 KB   | 2024       | Empty thesis final eval log  |
| login_page.html                 | 1.5 KB | 2024       | WAF connection test output   |
| rl_training_20251208_005406.log | 7.5 KB | 2024-12-08 | RL training session 1        |
| rl_training_20251208_084022.log | 0.6 KB | 2024-12-08 | RL training session 2        |
| training.log                    | 317 KB | 2024       | Main training log (largest)  |
| training_phase1_quick_1k.log    | 0 KB   | 2024       | Empty quick test log         |
| training_phi3_rl_200epochs.log  | 20 KB  | 2024       | Phi-3 200-epoch RL training  |
| training_qwen_rl_200epochs.log  | 1 KB   | 2024       | Qwen 200-epoch RL training   |

### Documentation Archive (`docs/archive/`)

**Archived:** 2 Markdown files + 4 JSON eval results

**Reason:** Deprecated documentation and old evaluation results superseded by new unified approach.

#### Markdown Files

| File                       | Size   | Reason                                                                                                                    |
| -------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------- |
| ANALYSIS_PHASE_STRATEGY.md | 8.5 KB | Old analysis concluding "train Phase 3 only". Superseded by new sequential Phase 1→2→3 strategy in DATASET_UNIFICATION.md |
| FUTURE_WORK_NOTES.md       | 1.7 KB | Hypotheses about sequential training. Now implemented in actual training pipeline. Historical reference only.             |

#### JSON Evaluation Results

| File                                                                 | Size | Date       | Notes                          |
| -------------------------------------------------------------------- | ---- | ---------- | ------------------------------ |
| eval_gemma_phase3_checkpoint_NVIDIA_GeForce_RTX_4060_Laptop_GPU.json | -    | 2024       | Gemma Phase 3 local evaluation |
| eval_gemma_phase3_remote_hidden_waf_20251207_013646.json             | -    | 2024-12-07 | Gemma remote WAF eval          |
| eval_phi3_phase3_remote_hidden_waf_20251207_021649.json              | -    | 2024-12-07 | Phi-3 remote WAF eval          |
| eval_qwen3b_phase3_remote_hidden_waf_20251207_032920.json            | -    | 2024-12-07 | Qwen 3B remote WAF eval        |

**Note:** These JSON files are early Phase 3 evaluation results. Final comprehensive evaluation will be done after new Phase 1→2→3 RL training completes.

### Scripts Archive (`scripts/archive_deprecated/`)

**Archived:** 16 Python scripts (3,846 lines removed)

**Reason:** Duplicate/deprecated scripts consolidated into universal tools.

See `scripts/archive_deprecated/README.md` for detailed list.

---

## Current Active Documentation (Workspace Root)

- `README.md` - Main project documentation
- `DATASET_UNIFICATION.md` - Dataset naming convention and training pipeline
- `REMOTE_TRAINING_GUIDE.md` - Guide for remote training on vast.ai
- `REMOTE_CHECKLIST.md` - Deployment checklist
- `GEMINI.md` - RED-RAG v2 experiment instructions (separate research track)
- `ARCHIVE_INDEX.md` - This file (tracking archived files)

---

## Retrieval Instructions

If you need to restore any archived file:

```bash
# Restore from logs archive
cp logs/archive/<filename> .

# Restore from docs archive
cp docs/archive/<filename> .

# Restore from scripts archive
cp scripts/archive_deprecated/<filename> scripts/
```

---

**Last Updated:** December 9, 2025
