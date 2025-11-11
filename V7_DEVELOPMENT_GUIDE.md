# v7 Development Guide — Execution-First, Quiet‑STaR, RAG‑Aided

## Objectives

- Adopt execution-based selection (no keyword filtering). Let replay decide pass/fail.
- Train with Quiet‑STaR style reasoning tags; inference remains payload-only.
- Integrate simple RAG exemplars for SQLi/XSS modes (boolean/union/time; obfuscation XSS).
- Use replay-in-the-loop to distill real passes into the model.
- Run in WSL with GPU; always login DVWA during replay.

---

## Environment (WSL)

- Enter WSL + venv
  - `wsl -e bash -lc "cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate"`
- Tokens inside WSL
  - `export DEEPSEEK_API_KEY="sk-your-deepseek-key"`
  - `export HF_TOKEN="hf_your-hf-token"`
- Start WAF
  - `cd waf && docker compose up -d && cd ..`
- Always login during replay (harness has `--login`)

---

## Quiet‑STaR Reasoning (Train‑time only)

- Wrap reasoning during SFT (training) with static tags; inference stays payload-only.
- Training formatter (concept):
  - Assistant target becomes:
    - `Payload: <payload>\nReasoning: <start-thought><reasoning></end-thought>`
- Inference prompt ends with `Payload:` and must not ask for reasoning.
- Optionally add “Think silently; only output the payload on the first line after ‘Payload:’.”

---

## RAG Exemplars (Optional for v7)

- Use 1–2 short exemplars relevant to the mode:
  - SQLi modes: boolean obfuscated / ORDER BY N / UNION with correct columns.
  - XSS modes: entities/unicode/bracket access/atob/constructor/CSS events/srcdoc.
- Keep context small; do not flood prompts. Prioritize execution.
- Start simple: local TF‑IDF or the existing `rag/index.py` scaffolding to select seeds.

---

## Execution‑Based Selection

- Generation scripts should not keyword-filter. Keep all non-empty outputs and let replay decide by HTTP status.
- We already switched to minimal validation in:
  - `scripts/infer/v6_langchain_generate.py` (post_validate = non-empty)
  - `scripts/etl/enrich_xss_deepseek.py` (accept all non-empty variants)

---

## Replay (with Login)

- SQLi: `replay/harness.py`
  - `python replay/harness.py <payloads.txt> --output results/out.jsonl --base-url http://localhost:8080/vulnerabilities/sqli/ --param-name id --login --username admin --password password`
- XSS: `replay/harness_xss.py`
  - `python replay/harness_xss.py <payloads.txt> --output results/out_xss.jsonl --base-url http://localhost:8080/vulnerabilities/xss_r/ --param-name name --login --username admin --password password`
- Blocked = HTTP 403/406; Passed = not blocked. (Exploit-aware metrics can be added later.)

---

## Distill From Existing Passes → 1‑Cycle Train + Retest

Use the SQLi batches we’ve tested already:

- Inputs:
  - `results/sqli_batch_encoded_replay.jsonl` (100 tested; 61 pass)
  - `results/sqli_batch_manual_replay.jsonl` (100 tested; 35 pass)

Step 1 — Extract SQLi passes into loop additions

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

Step 2 — 1 cycle replay‑in‑the‑loop (small quick)

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

Step 3 — Retest with new adapter

```bash
wsl -e bash -lc '
set -e; cd /mnt/c/Users/HAD/Desktop/AI_in_cyber/LLM_in_Cyber; source .venv/bin/activate;
python scripts/infer/v6_langchain_generate.py --limit 200 --adapter-dir experiments/red_gemma2_v6_loop_c1/adapter;
python replay/harness.py results/v6_sqli_langchain.txt --output results/loop_c1_sqli.jsonl --base-url http://localhost:8080/vulnerabilities/sqli/ --param-name id --login;
python replay/harness_xss.py results/v6_xss_langchain.txt --output results/loop_c1_xss.jsonl --base-url http://localhost:8080/vulnerabilities/xss_r/ --param-name name --login;
'
```

Review CSVs in `results/` for bypass rates.

---

## Encoded Batches (for quick wins)

- Generate SQLi/XSS encoded batches:
  - `python scripts/util/generate_encoded_batches.py`
  - Outputs: `results/sqli_batch_encoded.txt`, `results/xss_batch_encoded.txt`
- Replay with login:
  - `python replay/harness.py results/sqli_batch_encoded.txt --output results/sqli_batch_encoded_replay.jsonl --base-url http://localhost:8080/vulnerabilities/sqli/ --param-name id --login`
  - `python replay/harness_xss.py results/xss_batch_encoded.txt --output results/xss_batch_encoded_replay.jsonl --base-url http://localhost:8080/vulnerabilities/xss_r/ --param-name name --login`

---

## Roadmap / Tips

1) Harden exploit‑aware harness (boolean baseline diff, time‑based latency, error regex; XSS reflect‑regex).
2) Add ModSecurity audit feedback (ruleIds) into generation/mutation prompts.
3) Add mutators (split/case/encode/homoglyph) before replay in the loop.
4) Scale training from small → full cleaned dataset + loop additions; checkpoint and monitor.
5) Expand XSS dataset with advanced obfuscations and real-world vectors.

