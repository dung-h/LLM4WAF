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

## Quick Resume For New Agent (WSL)

- Always run in WSL (GPU enabled) and the project venv:
  - `wsl -e bash -lc "cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate"`
- Ensure Docker Desktop is running, with WSL integration on, then bring up WAF:
  - `cd waf && docker compose up -d && cd ..`
- Login during replay: use harness `--login` (auto CSRF + admin/password session).

Environment variables (set inside WSL before running enrich/loop):

```bash
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
export HF_TOKEN="hf_your-hf-token"
```

If you need to persist them: add to `~/.bashrc` in WSL or a local `.env` you source before work.

---

## Distill From Existing Passes (SQLi) → 1-Cycle Train + Retest

We already generated and tested several batches. Use the latest SQLi batch that passed (encoded/obfuscated):

- Inputs (existing):
  - `results/sqli_batch_encoded_replay.jsonl` (100 tested; 61 pass)
  - `results/sqli_batch_manual_replay.jsonl` (100 tested; 35 pass)

### Step 1 — Extract passed SQLi payloads into loop additions

```bash
wsl -e bash -lc '
set -e; cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate;
python - <<"PY"
import json
from pathlib import Path
root=Path('.')
add = root/"data"/"processed"/"red_train_v6_loop_additions.jsonl"
def extract(p):
  out=[]
  if not p.exists(): return out
  for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    d=json.loads(line)
    if d.get("blocked") is False:
      out.append(d.get("payload",""))
  return out
passed = []
for f in [
  Path("results/sqli_batch_encoded_replay.jsonl"),
  Path("results/sqli_batch_manual_replay.jsonl"),
]:
  if f.exists(): passed += extract(f)
seen=set()
with add.open("a", encoding="utf-8") as w:
  for p in passed:
    p=p.strip()
    if not p or ("SQLi",p) in seen: continue
    seen.add(("SQLi",p))
    row={
      "instruction":"Generate an SQL injection payload to test DVWA SQLi endpoint.",
      "context":"Target: /vulnerabilities/sqli/ param 'id'. Environment: ModSecurity CRS PL1.",
      "constraints":"Keep real SQL syntax; avoid placeholders; concise boolean/union variants.",
      "payload":p,
      "reasoning":"from replay pass",
      "attack_type":"SQLi"
    }
    w.write(json.dumps(row, ensure_ascii=False)+"\n")
print("Appended", len(passed), "SQLi passes →", add)
PY'
```

### Step 2 — Build train file (base small + loop additions) and run 1 cycle

```bash
wsl -e bash -lc '
set -e; cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate;
python scripts/loop/replay_train_loop.py \
  --cycles 1 \
  --limit 200 \
  --base-train data/processed/red_train_v6_small.jsonl \
  --adapter-dir experiments/red_gemma2_v6_multi_clean/adapter \
  --template-cfg configs/red_v6_small_quick.yaml
'
```

### Step 3 — Retest with the new adapter

```bash
wsl -e bash -lc '
set -e; cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate;
python scripts/infer/v6_langchain_generate.py --limit 200 --adapter-dir experiments/red_gemma2_v6_loop_c1/adapter;
python replay/harness.py results/v6_sqli_langchain.txt --output results/loop_c1_sqli.jsonl --base-url http://localhost:8080/vulnerabilities/sqli/ --param-name id --login;
python replay/harness_xss.py results/v6_xss_langchain.txt --output results/loop_c1_xss.jsonl --base-url http://localhost:8080/vulnerabilities/xss_r/ --param-name name --login;
'
```

Review bypass rates in the printed summaries or the CSVs under `results/`.

---

## Practical Tips & Warnings

- Always login during replay: use `--login` (admin/password). Otherwise requests may resolve to login page and skew results.
- Execution-based selection: we removed keyword-based filters. Keep all non-empty generations and let replay decide true “pass”.
- SQLi: ORDER/**/BY N often bypasses; boolean obfuscation with comment‑smuggling and keyword splitting helps; UNION needs correct column count and may benefit from versioned comments.
- XSS: PL1 blocks basic patterns hard. Favor obfuscation (entities + unicode + bracket access + atob/constructor + CSS events + srcdoc + polyglot). Expect lower pass; iterate enrichment and replay‑in‑the‑loop.
- GPU/WSL: Use WSL venv for training/inference. Confirm `nvidia-smi` inside WSL shows the GPU.

---

## Configuration

See `.env.example` for required API keys and environment variables.

For detailed implementation instructions, contact the repository maintainer.

---

## Expected Results

| Metric           | v5 (SQLi Only) | v6 Target (Multi-Attack) |
| ---------------- | -------------- | ------------------------ |
| Dataset size     | 348 samples    | 1,156+ samples           |
| Attack types     | 1 (SQLi)       | 2 (SQLi + XSS)           |
| SQLi bypass rate | 83.3%          | ≥80%                     |
| XSS bypass rate  | N/A            | ≥70%                     |
| Training time    | 38 min         | 60-90 min                |
| Model size       | 82 MB          | ~82 MB                   |

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
