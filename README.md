# LLM4WAF - AI-Powered WAF Bypass System

> 🤖 **Continuous learning system for discovering and generating WAF bypass payloads using LLMs**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

Training small language models (Phi-3 Mini, Gemma-2 2B) to generate SQL injection and XSS payloads that bypass WAF with automated continuous learning from security research.

---

## 🎯 Vision

**Xây dựng hệ thống tự động học và cập nhật kỹ thuật WAF bypass mới từ cộng đồng security research**

```
Data Collection → Knowledge Extraction → Model Training → Deployment
     (RAG)              (LLM)                (SFT + RL)      (API)
```

**See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for detailed 3-phase plan**

---

## 📊 Current Status (v29 → v30)

**Dataset v29**: `data/processed/red_v29_enriched.jsonl`

- **Total**: 11,692 records (100% English, deduplicated)
- **Passed WAF**: 1,414 (12.1%)
- **Blocked**: 10,258 (87.7%)
- **Attack Types**: XSS (61%), SQLi (39%)

**WAF Configuration**:

- ModSecurity + OWASP CRS v3.3
- ParanoidLevel 1
- Testing: 40 parallel workers, ~500 payloads/sec

**Repositories Processed**: 23 repos (see `DATASET_EVOLUTION.md`)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repo
git clone https://github.com/dung-h/LLM4WAF.git
cd LLM4WAF

# Create Python environment (WSL required on Windows)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start WAF for Testing

```bash
cd waf
docker-compose up -d
cd ..
# Wait 15s for ModSecurity to initialize
```

### 3. View Dataset Stats

```bash
python show_v29_stats.py
```

---

## 📁 Project Structure

```
LLM_in_Cyber/
├── data/
│   ├── processed/
│   │   ├── red_v29_enriched.jsonl      # Current dataset (11.7k records)
│   │   ├── archive/                     # Old versions (v26-v28, seeds)
│   └── raw/                             # Original payload sources
├── configs/                             # Training configs (SFT, RL, DPO)
├── scripts/                             # Training & evaluation scripts
├── waf/                                 # ModSecurity WAF (Docker)
├── vendor/                              # Payload repositories (23 repos)
├── show_v29_stats.py                    # Quick dataset statistics
├── DATASET_EVOLUTION.md                 # Full dataset history & lessons
└── README.md                            # This file
```

---

## 🎯 Training Models

### Supervised Fine-Tuning (SFT)

```bash
# Phi-3 Mini (3.8B) - Recommended
python scripts/train_sft.py --config configs/sft_phi3_mini_red_v26_full.yaml

# Gemma-2 2B - Smaller, faster
python scripts/train_sft.py --config configs/sft_gemma2_2b_red_v25_smoke.yaml
```

**Output**: Adapter weights in `phi3-mini-sft-advanced-v*/` or `gemma-2-*-sft-v*/`

### Generate Payloads

```bash
python scripts/simple_gen_v5_fixed_clear_cache.py
# Outputs to: results/v5_fixed_payloads_30.txt
```

### Test Against WAF

```bash
python replay/harness.py results/v5_fixed_payloads_30.txt
```

---

## 📚 Key Findings

### Dataset Quality Issues (v26 → v29)

1. **Vietnamese Text**: 52% of v26 had mixed Vietnamese/English → cleaned to pure English
2. **Duplicates**: 139k records → 11k unique (92.6% duplicate rate)
3. **Seed Garbage**: 99.5% of "passed" seeds were plain text, not payloads
4. **New Repos Ineffective**: 16 new repos, 2,389 payloads tested → **0 passed**

### WAF Bypass Challenges

- **Overall Pass Rate**: 1.9% across 74k payloads tested
- **Simple Mutations Failed**: URL encoding, Unicode, comments → 0/2,290 passed
- **Legacy Repos Only**: All 1,414 passed payloads from first 7 repos
- **Modern WAF Too Strong**: OWASP CRS blocks 98%+ of known payloads

### Lessons Learned

- Always validate "passed" labels (manual spot-checks critical)
- Deduplication essential (MD5 hash of payload content)
- New payload repos mostly contain tool code, not payload lists
- Historical datasets (v19-v23) valuable for extracting passed samples

---

## 🔧 Configuration

### Execution Environment (Windows)

⚠️ **ALWAYS run Python commands in WSL** (Windows Subsystem for Linux)

```powershell
# Command pattern
wsl -e bash -lc 'cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber && source .venv/bin/activate && python script.py'

# Never run directly in PowerShell (will fail)
# python script.py  ❌
```

**Why?** Virtual environment, dependencies, and file paths require WSL context.

### API Keys (Pre-configured)

- **Hugging Face**: `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`
- **DeepSeek**: 10 keys in `scripts/augment_dataset.py` (auto-rotation)

Export HF token before training:

```bash
export HF_TOKEN=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
```

### GPU Requirements

- **Phi-3 Mini 4-bit**: 4GB VRAM
- **Gemma-2 2B 4-bit**: 3GB VRAM
- **Training (LoRA)**: 8GB+ VRAM recommended

### Virtual Environment

**DO NOT reinstall packages** unless adding new libraries. Environment is stable with:

- `transformers`, `peft`, `bitsandbytes`, `httpx`, `tqdm`, `trl`

---

## 📖 Documentation

- **`DATASET_EVOLUTION.md`**: Full version history (v18-v30), repo sources, WAF testing stats
- **`show_v29_stats.py`**: Quick statistics script
- **`configs/`**: 40+ training configs for different experiments
- **`data/processed/archive/`**: Historical datasets and seeds

---

## 🎓 Research Context

This project explores:

1. Can small LLMs (<4B params) learn WAF bypass techniques?
2. What training methods work best? (SFT, RL/PPO, DPO)
3. How effective are modern WAFs against LLM-generated payloads?

**Current Focus**: Training Phi-3 Mini on v29 dataset (1,414 passed examples) to generate novel bypasses.

---

## 📊 Repository Sources (23 Total)

### Legacy (7 repos) - ✅ 1,414 passed

- PayloadsAllTheThings, fuzzdb, SecLists, IntruderPayloads
- payloadbox (sql/xss), humblelad, yogsec

### Round 2 (8 repos) - ❌ 0 passed

- WebGoat, PortSwigger, SQLMap, BeEF, XSStrike, Commix, WAFNinja, xsshunter-express

### Round 3 (8 repos) - ❌ 0 passed

- Nettacker, jsql-injection, nuclei-templates, fuzz.txt, etc.

**See `DATASET_EVOLUTION.md` for detailed breakdown.**

---

## 🤝 Contributing

This is a research project. Key areas for improvement:

1. **Dataset expansion** to 2,000+ passed payloads
2. **Advanced mutation techniques** (LLM-based)
3. **Alternative training methods** (RL, DPO, Constitutional AI)
4. **Evaluation metrics** beyond simple pass/fail

---

## ⚠️ Disclaimer

This research is for **educational purposes only**. The tools and techniques are designed to:

- Improve WAF rule development
- Train security professionals
- Advance defensive cybersecurity research

**Do not use against systems without explicit authorization.**

---

## 📝 License

Research project - See institution guidelines.

---

**Last Updated**: November 27, 2025  
**Current Version**: v29 (11,692 records, 1,414 passed)  
**Status**: Ready for Phi-3 Mini SFT training
