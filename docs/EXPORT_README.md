# AI-based WAF Evaluation & Self-Hardening — Export Snapshot

This bundle contains the core code, configs, and scripts (no data) to resume the Red→Test→Blue pipeline. Use it to brief another assistant and continue work.

## What’s Included (Code Only)
- `configs/` — SFT configs for Red/Blue (DoRA + QLoRA)
- `scripts/` — ETL, training, RAG, indexing, and utilities
- `rag/` — Retrieval components and prompts
- `replay/` — Harness + audit parser skeletons
- `waf/` — Docker testbed (Nginx + ModSecurity + CRS) configs
- `red/`, `blue/` — readme stubs

Excluded on purpose: `data/`, `results/`, `experiments/` (adapters), `.venv/`, `waf/logs/`.

## Environment Summary
- OS: Windows 11 + WSL2 (Ubuntu 22.04) — primary runtime in WSL
- GPU: NVIDIA RTX 4060 (8 GB VRAM)
- CUDA: 12.8 (WSL visible via `nvidia-smi`)
- Python (WSL venv): 3.10
- Key packages (WSL venv)
  - torch 2.5.1+cu121, transformers 4.41.2, accelerate 0.30.1
  - peft 0.11.1 (DoRA), trl 0.9.6, bitsandbytes 0.43.0
  - datasets 2.19.0, pandas, pyarrow, duckdb, rich, pyyaml
  - scikit-learn 1.5.2, joblib 1.4.2
  - optional: flash-attn (not installed by default)
- Tokens/Secrets: not included. HF token required for gated repos and some datasets; Kaggle via `~/.kaggle/kaggle.json`.

## Data & Index Status (local run)
- Ingested: PurpleAILAB SQLi (HF), Payloadbox (XSS/SQLi), Kaggle XSS (positive class). Shengqin HF unresolved.
- Processed counts (local):
  - `red_train.jsonl`: 1334
  - `red_val.jsonl`: 166
  - `red_test.jsonl`: 168
- Payload TF‑IDF index: 17182 docs (XSS ~6590, SQLi ~3270, Mixed Kaggle ~7322).
- CRS rules parsed: 677 (`data/processed/crs_rules.parquet`), TF‑IDF index built.

## Training Status
- Llama 3 8B: access pending (gated). Smoke/full configs ready.
- TinyLlama smoke SFT: success (adapter saved locally, excluded from bundle).
- Qwen2‑7B smoke SFT: success (adapter saved locally, excluded from bundle).
- Key configs:
  - `configs/red_llm_dora_8gb.yaml` (Llama 3 8B; DoRA + QLoRA; seq=2048)
  - `configs/red_llm_dora_8gb_smoke.yaml` (Llama 3 8B smoke)
  - `configs/red_qwen2_7b_dora_8gb_smoke.yaml` (Qwen2‑7B smoke)
  - `configs/blue_llm_dora_8gb.yaml` (Blue SFT skeleton)

## RAG Components
- Index builder: `scripts/rag/build_index.py` → `rag/indexes/*.joblib`
- Retriever: `rag/retrievers.py` (TF‑IDF)
- Red generator: `scripts/rag/query_red_generate.py`
  - Supports fallback (no model) and model mode (4‑bit; --use_model)
  - New flags: `--samples`, `--max_new_tokens`, `--temperature`, `--top_p`, `--attn_impl`
- Prompt: `rag/red_prompt.py`
- Utils (novelty + mutations): `rag/utils.py`

### Retrieval Tips (Quality)
- Prefer SQLi-only retrieval when generating SQLi. If your TF‑IDF index mixes XSS/Mixed seeds, either rebuild the index with only SQLi seeds or add SQLi hints to the query (e.g., `union select -- %27 or 1=1`).
- Keep exemplars small (1–2) to avoid drifting into HTML/XSS patterns.
- Post‑filter: accept only payloads matching a technique regex (e.g., `(?i)\bunion\b.*\bselect\b`).

## WAF Testbed
- `waf/docker-compose.yml`: Nginx + ModSecurity + CRS (PL2) + DVWA
- `waf/modsecurity/modsecurity.conf`, `waf/modsecurity/crs-setup.conf`
- Logs under `waf/logs/` (excluded), patches under `waf/patches/`

## What’s Working Now
- ETL: ingest PurpleAILAB/Kaggle/Payloadbox; normalize; dedup; split.
- Indices: payloads + CRS TF‑IDF indices built.
- Red SFT (smoke): TinyLlama + Qwen2‑7B succeeded in WSL with DoRA + QLoRA.
- Red RAG generation: fallback (fast) and model mode (slow but higher quality) verified.

## Known Issues / Constraints
- Llama 3 8B gated; access required to train/use.
- Model generation in Windows Python 3.13 fails with bitsandbytes validation; WSL works.
- 8 GB VRAM is tight at seq=2048; generation speed is memory‑bound (low watt despite 100% util).
- Shengqin HF dataset ID provided was not found; needs correction.

## How To Resume (WSL recommended)
1) Create/activate venv, install deps
```
cd /path/to/LLM_in_Cyber
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -r requirements.txt scikit-learn joblib
```
2) Rebuild indices
```
python scripts/rag/build_index.py --payload
python scripts/etl/ingest_crs.py
python scripts/rag/build_index.py --crs data/processed/crs_rules.parquet
```
3) Generate Red payloads
- Fallback (fast bulk):
```
python scripts/rag/query_red_generate.py \
  --index rag/indexes/payloads_tfidf.joblib \
  --context "DB=MySQL; sink=query; method=GET; param=q; CRS=PL2" \
  --instruction "Generate union-based SQLi with light obfuscation" \
  --n 200 > results/red_union_batch_fallback.jsonl
```
- Model mode (Qwen2‑7B; reuse loaded model; adjust samples/tokens):
```
python -u scripts/rag/query_red_generate.py \
  --index rag/indexes/payloads_tfidf.joblib --use_model \
  --model_name "Qwen/Qwen2-7B" \
  --adapter experiments/red_qwen2_7b_dora_smoke/adapter \
  --context "DB=MySQL; sink=query; method=GET; param=q; CRS=PL2; hint: union select -- %27 or 1=1 /* */ information_schema" \
  --instruction "Generate union-based SQLi with light obfuscation" \
  --samples 10 --max_new_tokens 32 | tee -a results/red_union_batch.jsonl
```
> If generation drifts to HTML/XSS, rebuild the payload index without XSS/Kaggle seeds or add stronger SQLi hints to the context.
4) Bring up WAF
```
cd waf
docker compose up -d
```
5) Replay (skeleton)
- `replay/harness.py` shows a single‑URL GET probe example; adapt `base_url` and param.
- For batch replay from JSONL, add a script to read payloads and call harness (or request one next).

## Next Steps (Plan)
- Red
  - Generate diversified batches (union/time/boolean) via RAG + Qwen2‑7B adapter.
  - Add batch replay script and parse ModSecurity JSON audit → joined parquet with rule IDs.
- Blue
  - Build Blue dataset from bypasses; train DoRA+QLoRA to emit minimal CRS patches.
  - Lint patches; canary test; measure FP uplift on benign set.
- Evaluation
  - Report: bypass ratio, Blue effectiveness, uplift vs CRS, FP regression, latency impact.
- Infra
  - Optional: install flash‑attention in WSL for faster generation; toggle seq length to 1536/1024 to ease VRAM.
- Llama 3 8B
  - Once approved, run smoke/full SFT in WSL using `configs/red_llm_dora_8gb*.yaml`.

## Security & Ops Notes
- Never log HF tokens; export via env only.
- All activities remain in lab; don’t scan external hosts.
- Windows path may show symlink cache warnings from HF Hub — safe to ignore or enable Developer Mode.

---

If you need anything missing (e.g., batch replay script, Blue trainer wiring, or flash‑attention integration), include that in your prompt to the assistant.
