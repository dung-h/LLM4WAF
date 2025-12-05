# Gemini Agent Instructions - REMOTE SERVER (LLM4WAF)

This document is for the Agent operating on the **Remote GPU Server** to execute Phase 3 (Adaptive SFT) Training.

## 1. Initial Setup
1.  **Run Setup Script:** `bash setup_full_env.sh`
2.  **Activate Env:** `source .venv/bin/activate`
3.  **Set Token:** `export HF_TOKEN=hf_...` (Get from user)

## 2. Data & Model Preparation
*   **Dataset:** Ensure `data/processed/red_phase3_adaptive_sft.jsonl` exists.
    *   If not found, regenerate it: `python scripts/build_phase3_adaptive_dataset.py`
*   **Adapters:** Phase 3 is trained from **Base Model**, so you do NOT need to download Phase 2 adapters for this specific training task.

## 3. Execution Tasks (Training Phase 3 Adaptive)

Execute these commands sequentially (recommended to avoid OOM) or in parallel if using A100 80GB+.

**Task A: Train Gemma 2 2B (Phase 3)**
```bash
nohup python scripts/train_red.py --config configs/red_phase3_adaptive_gemma.yaml > logs/train_gemma_p3.log 2>&1 &
# Monitor: tail -f logs/train_gemma_p3.log
```

**Task B: Train Phi-3 Mini (Phase 3)**
```bash
nohup python scripts/train_red.py --config configs/red_phase3_adaptive_phi3.yaml > logs/train_phi3_p3.log 2>&1 &
```

**Task C: Train Qwen 7B (Phase 3)**
```bash
nohup python scripts/train_red.py --config configs/red_phase3_adaptive_qwen.yaml > logs/train_qwen_p3.log 2>&1 &
```

## 4. Post-Training
1.  **Verify Output:** Check `experiments/red_phase3_adaptive_*` directories.
2.  **Pack & Download:**
    ```bash
    tar -czvf adapters_phase3.tar.gz experiments/red_phase3_adaptive_*
    ```
3.  **Upload:** Upload `adapters_phase3.tar.gz` to Google Drive or Transfer.sh and provide the link to the Local Agent.
