# Gemini Agent Instructions for LLM4WAF Project

This document provides essential context and instructions for an AI Agent (or human writer) to generate the final project report for **LLM4WAF**.

## 0. General Mandates

*   **Execution Environment:** Always assume execution from the **root of the LLM4WAF repository**.
*   **Data Integrity:** Original dataset files (e.g., `data/processed/red_v40_balanced_final_v13.jsonl`) **MUST NOT BE MODIFIED**. When new datasets are required, **always create new files** (e.g., `data/processed/red_v40_phase1_sft_passed.jsonl`).
*   **Internet Access:** Strictly **NO HTTP requests to the internet** (unless explicitly instructed for tools like `gdown`). Do not use/scrape public payload repositories.
*   **Internal Resources Only:** Utilize only internal datasets, RAG corpus/index, and existing project integrations.
*   **Cleanliness:** Maintain a clean codespace; archive or remove temporary files as instructed.
*   **Logging:** All significant script executions or actions **MUST be logged to `mes.log`** at the repository root in the specified format:
    ```text
    [YYYY-MM-DD HH:MM:SS] CMD="<script or action>" STATUS=OK|FAIL OUTPUT="<main output file or N/A>"
    ```

## 1. Project Structure & Key Directories

The project's key components are organized as follows:

*   **`configs/`**: YAML configuration files for models, training, and evaluation.
    *   **Keep these updated:** `phi3_mini_phase3_rl.yaml`, `red_phase2_rag_sft.yaml` (and other model-specific configs for Qwen, Phi-3, Gemma if restored).
*   **`data/`**: Contains raw, processed, and RAG-related datasets.
    *   `data/processed/red_phase1_enriched_v2.jsonl`: Primary dataset for SFT.
    *   `data/rag/red_corpus_internal_v2.jsonl`: Internal RAG corpus.
    *   `data/rag/index/red_rag_v2_index.pkl`: RAG index.
*   **`docker-compose.multiwaf.yml`**: Defines the multi-WAF testing environment (DVWA, ModSecurity, Coraza).
*   **`eval/`**: Stores evaluation results and detailed payload logs.
    *   `eval/payload_details/*.jsonl`: Detailed logs of generated payloads during evaluation.
    *   `eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl`: RAG evaluation logs.
    *   `eval/red_rag_eval_multiwaf_extended.jsonl`: Multi-WAF evaluation logs.
    *   `eval/archive_legacy/`: Archive for old/legacy evaluation files.
*   **`experiments/`**: Stores trained model adapters (LoRA weights), logs, and checkpoints.
    *   `experiments/remote_adapters/experiments_remote_optimized/`: Location for downloaded/extracted remote adapters.
*   **`red/`**: Core Python modules for the RED agent, including RAG integration.
    *   `red/red_rag_integration.py`: Contains logic for prompt building with RAG.
    *   `red/rag_internal_client.py`: RAG client.
*   **`reports/`**: Stores generated reports and figures.
    *   `reports/figures/`: Contains PNG image files of charts.
    *   `remote_performance.md`: Main performance report.
    *   `training_loss_report.md`: Training loss analysis.
*   **`scripts/`**: Python scripts for various tasks (training, evaluation, data processing).
    *   `scripts/train_red.py`: SFT training script.
    *   `scripts/train_rl_reinforce.py`: RL training script.
    *   `scripts/evaluate_remote_adapters.py`: Main evaluation script.
    *   `scripts/generate_report_charts.py`: Script to generate visualizations.
    *   `scripts/archive_cleanup/`: Archive for old/legacy scripts.
*   **`rl/`**: Reinforcement Learning specific modules.
    *   `rl/waf_env.py`: WAF environment for RL training.
*   **`archive_legacy/`**: Top-level archive for unused repositories (`notebooks`, `XSStrike`, `replay`).
*   **`vendor.zip`**: Archived external dependencies.

## 2. Key Concepts & Techniques in the Project

The project heavily utilizes the following:

*   **PEFT (Parameter-Efficient Fine-Tuning):** Optimizing LLMs without retraining all parameters.
*   **LoRA (Low-Rank Adaptation):** A specific PEFT method.
*   **QLoRA (Quantized LoRA):** Loading models in 4-bit (NF4) for VRAM efficiency.
*   **RAG (Retrieval Augmented Generation):** Enhancing LLM responses with retrieved knowledge from a vector database.
*   **WAF (Web Application Firewall):** ModSecurity and Coraza are used as targets for evasion.
*   **LLMs:** Qwen (7B), Phi-3 Mini (3.8B), Gemma 2 (2B) models.
*   **Training Phases:**
    *   **Phase 1 (SFT):** Supervised Fine-Tuning for basic payload generation.
    *   **Phase 2 (Reasoning):** SFT with structured prompts (e.g., Chain-of-Thought, explicit technique guidance) to improve evasion logic.
    *   **Phase 3 (RL):** Reinforcement Learning to refine payload generation based on WAF interaction feedback.

## 3. Common Tasks for the Agent

An Agent might be asked to perform tasks such as:

1.  **Run Evaluations:** Execute `scripts/evaluate_remote_adapters.py` with specific models or test sets.
2.  **Generate Reports:** Analyze `remote_performance.md`, `training_loss_report.md` and visualizations in `reports/figures/`.
3.  **Perform Training:** Run `scripts/train_red.py` (for SFT) or `scripts/train_rl_reinforce.py` (for RL) with specified configuration files (`configs/*.yaml`).
4.  **Data Preparation:** Use scripts in `scripts/` to build or process datasets.
5.  **Troubleshooting WAF:** Inspect Docker logs, verify DVWA connectivity.
6.  **Cleanup/Archiving:** Maintain an organized workspace by moving old/unused files to `archive_legacy/`, `scripts/archive_cleanup/`, `eval/archive_legacy/`.
7.  **Remote Setup:** Provide instructions for setting up remote training environments (refer to `setup_full_env.sh`).

## 4. Key Considerations for Agent Actions

*   **VRAM Limitations:** Be aware that the local machine likely has 8GB VRAM. This limits the size of models that can be loaded for training/inference, especially for RL. Large models or complex RL tasks may require a remote server.
*   **Prompt Sensitivity:** LLMs in this project are highly sensitive to prompt format. Use the structured prompts (e.g., as seen in `scripts/evaluate_remote_adapters.py`) when generating payloads for Phase 2/3 models.
*   **Reproducibility:** When asked to reproduce results, always refer to the specific model adapters and configurations used in the evaluation reports.
*   **Confirmation:** Always confirm significant file system changes (deletion, archiving) or long-running tasks (training) with the user.
*   **Error Handling:** Be prepared to debug common issues like WAF connectivity (Docker), OOM errors, and library dependencies.

## 5. Dynamic Reporting Strategy (Crucial for Agent)

*   **Initial Report Scope:** The first draft of the report (generated by the Agent) should focus on the **existing evaluation results and insights** as documented in `remote_performance.md`, `training_loss_report.md`, and `reports/figures/`. This currently includes the SFT and Reasoning phases for Qwen, Phi-3, and Gemma. **Gemma 2 2B is the only model that has completed Phase 3 RL locally.**
*   **Future Update:** Explicitly mention that **this report will be updated** with additional findings and analysis once the Reinforcement Learning (RL) training for Phi-3 Mini and Qwen 7B (Phase 3) is successfully completed on the remote server.
*   **Placeholders:** The Agent should indicate where the RL results for Phi-3 Mini and Qwen 7B will be inserted and discussed once available. For example, add a section "Expected RL Results for Phi-3 Mini and Qwen 7B (Future Update)" or similar.

---

This `GEMINI-FAKE.md` file contains the comprehensive context for future tasks related to the LLM4WAF project.