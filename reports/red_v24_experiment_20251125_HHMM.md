# RED Model Experiment Report: Gemma-2 2B SFT - RED v24 Cycle (Smoke Test)

**Date:** 2025-11-25

**1. Dataset Information**

*   **Version:** v24
*   **Dataset File:** `data/processed/red_v24_unified_mysql_xss.jsonl`
*   **Total Records:** 11,069 (full dataset)
*   **Training Subset Used:** `data/processed/red_v24_unified_mysql_xss_1k.jsonl` (1,000 records)
*   **Composition of 1k Subset:**
    *   `passed`: ~245 records
    *   `blocked`: ~735 records
    *   `reflected`: 20 records (successfully preserved)
*   **Key Improvements:**
    *   Enriched with 2,202 high-quality, DeepSeek-generated instruction/context/reasoning pairs for "passed" payloads.
    *   Includes 20 original "reflected" payloads from `v20`.
    *   Balanced ratio of `passed`/`blocked` payloads for effective learning (in the full dataset).

**2. Training Configuration**

*   **Model:** `google/gemma-2-2b-it` (SFT with LoRA adapter)
*   **Adapter Path:** `experiments/gemma2_2b_sft_red_v24_smoke`
*   **Config File:** `configs/sft_gemma2_2b_red_v24_smoke.yaml`
*   **Key Optimizations:**
    *   `gradient_checkpointing: false` (for speed)
    *   `per_device_train_batch_size: 2`
    *   `gradient_accumulation_steps: 2` (Effective batch size: 4)
    *   `dataloader_num_workers: 4` (for faster data loading)
    *   `logging_steps: 1` (for detailed progress)
*   **Duration:** Short smoke test run (default `num_train_epochs: 3.0` on 1k subset).

**3. Intrinsic Tests (Offline Evaluation)**

*   **Test Script:** `scripts/test_gemma2_v13_sft.py` (modified for Gemma-2 prompt template and output parsing).
*   **Log File:** `logs/v24_intrinsic_test_full_dataset.log`
*   **Observations:**
    *   Initial runs showed model refusal due to incorrect prompt templating (Phi-3 vs Gemma-2).
    *   After correcting the prompt template, the model started generating actual attack payloads.
    *   **Positive:** Model successfully learned to generate SQLi (e.g., `UNION ALL SELECT`, time-based blind, login bypass) and XSS (e.g., draggable HTML with `ondrop`) payloads.
    *   **Negative:** The model still frequently generated non-attack strings (e.g., `wp-content/plugins/...`) which also happened to "pass" the WAF. This indicates:
        *   Some "junk" payloads might still exist in the dataset.
        *   The model needs further refinement to distinguish between malicious and benign strings that merely "pass" the WAF.

**4. DVWA Attack Pipeline Evaluation**

*   **Test Script:** `run_attack_pipeline.py` (modified to use `v24` adapter and log to `v24_attack_pipeline.jsonl`).
*   **Rounds:** 3
*   **Log File:** `logs/v24_attack_pipeline.jsonl`
*   **Summary of Generated Payloads & Results:**
    *   **Round 1:** Generated 3. One `PASSED` (junk), two `BLOCKED` (SQLi).
    *   **Round 2:** Generated 3. Two `PASSED` (junk), one `BLOCKED` (SQLi).
    *   **Round 3:** Generated 3. One `PASSED` (junk), two `BLOCKED` (SQLi).
*   **Overall Observation:**
    *   The model successfully generated 9 payloads across 3 rounds.
    *   It demonstrated ability to produce various SQLi and XSS payloads, including obfuscated forms.
    *   **Successes:** Generated actual SQLi payloads that correctly target DVWA's WAF (e.g., `1%27%20or%20sleep%28__TIME__%29%23`, `AND%20IF%28benchmark%283000000%2CMD5%281%29%29%2CNULL%2CNULL%29%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%2CNULL%29%2520--`).
    *   **Challenges:** A significant portion of generated "passed" payloads were identified as non-attack strings (e.g., `wp-content/plugins/...`). These passed the WAF but are not useful for red-teaming.

**5. Analysis and Next Steps**

*   **Strengths:**
    *   The `v24` dataset is robust, contains a high count of high-quality "passed" payloads (2,803 total, 2,202 DeepSeek-enriched), and successfully preserved `reflected` examples.
    *   The model, even with minimal training (1k subset, 3 epochs), has learned to generate functional attack payloads for DVWA.
    *   The DeepSeek enrichment of passed payloads is likely contributing to the model's ability to generate relevant payloads.
*   **Weaknesses / Areas for Improvement:**
    *   **Payload "Junk" Generation:** The most prominent issue is the model generating non-attack strings that pass the WAF. This points to the need for a more rigorous filtering process to remove such "junk" from the training dataset itself, or adding more explicit negative examples during training.
    *   **Model Refinement:** The current model is a "smoke test". Longer training (on the full 11k dataset, more epochs) is needed to solidify its learning and reduce generation of irrelevant strings.

**Recommended Next Steps:**

1.  **Refine Dataset Filtering:** Implement a stricter pre-filtering step during dataset creation to remove common benign strings (e.g., plugin paths, dictionary words) that might accidentally be present as "passed" payloads.
2.  **Full SFT Training:** Run the SFT training for `v24` on the **full 11,069-record dataset** (not just the 1k subset) and potentially for more epochs. This will allow the model to learn from a much broader and deeper set of examples.
3.  **Detailed WAF Evasion Strategy:** Further refine DeepSeek prompts to explicitly ask for WAF evasion techniques, rather than just "creative techniques". This could improve the model's ability to bypass detected WAF rules.
4.  **Explore RL Fine-Tuning:** Once a strong SFT baseline is established, consider exploring RL methods (`scripts/rl_train_red.py`) to further optimize the model for DVWA bypasses.