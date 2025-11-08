# Blue LLM (WAF Patching)

- Train with `scripts/train_blue.py --config configs/blue_llm_dora_8gb.yaml`.
- Input data: `data/processed/blue_*.jsonl` produced from replay bypasses.
- Output: candidate ModSecurity/CRS patch rules + rationale.
