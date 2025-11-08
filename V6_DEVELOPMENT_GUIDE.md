# v6 Multi-Attack Development Guide

## Overview

**Goal**: Expand from SQLi-only (v5_fixed) → Multi-attack system (SQLi + XSS)

**Current Status**: v5_fixed model (83.3% WAF bypass for SQLi)  
**Next Target**: v6 model (80%+ bypass for both SQLi and XSS)

---

## Development Plan

### Phase 1: Data Collection
- Download XSS payloads from GitHub (PayloadsAllTheThings)
- Create seed dataset with 20+ XSS payloads
- Extract reflection, stored, and DOM-based XSS patterns

### Phase 2: Data Enrichment
- Use DeepSeek API with Chain-of-Thought prompting
- Generate reasoning explanations for each payload
- Cost: ~$0.10, Time: ~10 minutes

### Phase 3: Dataset Combination
- Merge existing SQLi dataset (1,136 samples) with XSS data
- Create balanced training set (80% train / 10% val / 10% test)
- Add `attack_type` field to distinguish SQLi vs XSS

### Phase 4: Model Training
- Base model: `google/gemma-2-2b-it`
- Fine-tuning: PEFT LoRA with DoRA optimization
- Training time: 60-90 minutes
- Target loss: <1.0

### Phase 5: Testing & Validation
- Generate both SQLi and XSS payloads
- Test against ModSecurity WAF
- Measure bypass rates separately for each attack type

---

## Configuration

See `.env.example` for required API keys and environment variables.

For detailed implementation instructions, contact the repository maintainer.

---

## Expected Results

| Metric | v5 (SQLi Only) | v6 Target (Multi-Attack) |
|--------|----------------|--------------------------|
| Dataset size | 348 samples | 1,156+ samples |
| Attack types | 1 (SQLi) | 2 (SQLi + XSS) |
| SQLi bypass rate | 83.3% | ≥80% |
| XSS bypass rate | N/A | ≥70% |
| Training time | 38 min | 60-90 min |
| Model size | 82 MB | ~82 MB |

---

## Repository Structure

```
data/
  raw/                  # XSS seed data (to be created)
  processed/            # Enriched datasets
scripts/
  etl/                  # Data processing scripts
  train_red.py          # Training script
configs/
  red_v6_multi_attack.yaml  # v6 training configuration
```

---

## Quick Start (Developers with Access)

Contact maintainer for:
- Complete implementation guide (AGENT_INSTRUCTION.md)
- Quick reference (V6_MULTI_ATTACK_QUICKSTART.md)
- API credentials
- Script templates

---

**Note**: This is a research project for educational purposes only. Use responsibly and only in authorized testing environments.
