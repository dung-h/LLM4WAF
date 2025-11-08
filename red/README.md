# Red LLM (Payload Generation)

- Train with `scripts/train_red.py --config configs/red_llm_dora_8gb.yaml`.
- Input data: `data/processed/red_*.jsonl` created by ETL.
- Base: `meta-llama/Meta-Llama-3-8B` with QLoRA 4-bit + DoRA.
