# Gemini Agent Instructions - REMOTE SERVER (LLM4WAF)

This document is for the Agent operating on the **Remote GPU Server** to execute Phase 3 (Adaptive SFT) Training.

## 1. Initial Setup
1.  **Run Setup Script:** `bash setup_full_env.sh`
2.  **Activate Env:** `source .venv/bin/activate`
3.  **Set Token:** `export HF_TOKEN=hf_...` (Get from user)

## 2. Data & Model Preparation
*   **Dataset:** Ensure `data/processed/red_phase3_lightweight.jsonl` exists (20k samples).
    *   If not found, regenerate it: `python scripts/build_phase3_lightweight.py`
*   **Adapters:** Phase 3 Lightweight is trained from **Base Model** directly, no need for Phase 1/2 adapters.

## 3. Execution Tasks (Training Phase 3 Lightweight - 20k Enhanced Dataset)

Execute these commands sequentially (recommended for 16GB VRAM) or in parallel if using A100 80GB+.

**Task A: Train Gemma 2 2B (Phase 3 Lightweight)**
```bash
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml --gpu 0 > logs/train_gemma_p3_light.log 2>&1 &
# Monitor: tail -f logs/train_gemma_p3_light.log
# Check loss logs: tail -f SFT_gemma-2-2b_*.log
```

**Task B: Train Phi-3 Mini (Phase 3 Lightweight)**
```bash
nohup python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_phi3.yaml --gpu 0 > logs/train_phi3_p3_light.log 2>&1 &
# Monitor: tail -f SFT_Phi-3-mini-4k_*.log
```

## 4. Post-Training
1.  **Verify Output:** Check `experiments/red_phase3_lightweight_enhanced_*` directories.
2.  **Check Logs:** Review `SFT_*.log` files for final loss values.
3.  **Pack & Download:**
    ```bash
    tar -czvf adapters_phase3_lightweight.tar.gz experiments/red_phase3_lightweight_enhanced_* SFT_*.log
    ```
4.  **Upload:** Upload to Google Drive or Transfer.sh and provide link.
