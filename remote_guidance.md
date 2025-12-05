
---

# Session Log (2025-12-03/04) - Scaling & Optimization Report

## 1. Training Execution Summary
Successfully trained three models through Phase 1 (SFT) and Phase 2 (Reasoning) using a highly optimized pipeline on a single 16GB VRAM GPU.

### **Models Trained**
1.  **Qwen/Qwen2.5-7B-Instruct**
    *   Phase 1 Loss: **0.3759**
    *   Phase 2 Loss: **0.3369**
    *   Result: `adapters_qwen_optimized.tar.gz`
2.  **meta-llama/Llama-3.1-8B-Instruct**
    *   Phase 1 Loss: **0.4414**
    *   Phase 2 Loss: **0.3648**
    *   Result: `adapters_llama_optimized.tar.gz`
3.  **microsoft/Phi-3-mini-4k-instruct**
    *   Phase 1 Loss: **0.5273**
    *   Phase 2 Loss: **0.4007**
    *   Result: `adapters_phi3_optimized.tar.gz`

### **Optimized Hyperparameters (Stable on 16GB VRAM)**
These settings achieved 100% GPU utilization and prevented OOM errors.

| Parameter | Value | Reason |
| :--- | :--- | :--- |
| `per_device_train_batch_size` | **2** (4 for Phi-3) | Balance between speed and memory. BS=1 was too slow. |
| `gradient_accumulation_steps` | **8** (4 for Phi-3) | Maintains effective batch size of 16. |
| `max_length` | **1024** (2048 for Phi-3) | Critical for fitting Llama/Qwen into 16GB. |
| `dataloader_num_workers` | **10** | Solved data starvation issue; GPU no longer idle. |
| `group_by_length` | **true** | Significant speedup by grouping similar length samples. |
| `dataloader_pin_memory` | **true** | Faster host-to-device transfer. |

## 2. Static Evaluation
Performed static evaluation using `scripts/evaluate_model.py` on the test split.
*   **Status:** Completed for all 3 models.
*   **Limitation:** WAF testing was skipped because the Docker daemon could not be accessed in the current environment.
*   **Artifact:** `eval_results.tar.gz` (Contains generated payloads in JSONL format).

## 3. Notes & Issues
*   **Docker/RL:** Phase 3 (Reinforcement Learning with WAF) was blocked due to `dial unix /var/run/docker.sock: connect: no such file or directory`. Future runs require a Docker-enabled environment.
*   **Hugging Face Token:** Corrected token handling for gated models (Llama 3.1).
