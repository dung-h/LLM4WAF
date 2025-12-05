# Report Generation Helper: LLM4WAF Project

This document serves as a guide and context repository for an AI Agent (or human writer) to generate the final project report for **LLM4WAF**.

## 1. Core Objective
To evaluate and enhance the capabilities of Open-Source Large Language Models (LLMs) in generating payloads to evade Web Application Firewalls (WAFs) using Parameter-Efficient Fine-Tuning (PEFT) and Reinforcement Learning techniques.

## 2. Directory Mapping (Where to find info)
The agent should look at these files to extract specific data:

*   **Performance Metrics:** `remote_performance.md` (Contains Pass Rates, Diversity Analysis, and Payload Examples).
*   **Training Stats:** `training_loss_report.md` (Contains Loss curves, convergence analysis).
*   **Visualizations:** `reports/figures/` (Contains PNG charts).
*   **Config Details:** `configs/` (Contains hyperparameters).
*   **Code Logic:** `scripts/evaluate_remote_adapters.py` (Contains validation logic and Prompt Templates).
*   **Infrastructure:** `docker-compose.multiwaf.yml` (Contains WAF setup details).

---

## 3. Report Outline & Narrative Instructions

Please structure the final report using the following sections. Adhere strictly to the "Narrative Notes" for each section.

### Section 1: Introduction & Background techniques
*   **LLMs in Cybersecurity:** Brief intro on Offensive AI.
*   **Key Techniques (Must Explain):**
    *   **PEFT (Parameter-Efficient Fine-Tuning):** Why we didn't fine-tune the whole model (cost/resource efficiency).
    *   **LoRA (Low-Rank Adaptation):** The specific PEFT method used. Explain conceptually (injecting rank decomposition matrices).
    *   **QLoRA (Quantized LoRA):** Loading base models in 4-bit (NF4) to fit in consumer GPU memory while training.
*   **WAFs:** Introduce ModSecurity (Rule-based) and Coraza.

### Section 2: Methodology

#### 2.1. Models Selected
*   **Qwen 2.5 7B:** Represents a larger, capable general-purpose model.
*   **Phi-3 Mini (3.8B):** Represents a highly optimized, "reasoning-dense" small model.
*   **Gemma 2 2B:** Represents the edge/mobile class model.

#### 2.2. Data Construction (**Crucial Narrative**)
*   *Instruction:* Frame this as a semi-automated process.
*   **Seed Data:** We manually curated high-quality "seed" payloads targeting specific WAF rules (SQLi, XSS, OS Injection).
*   **Augmentation:** We developed scripts (`scripts/build_phase2_dataset.py` logic) to mutate these seeds, adding obfuscation layers (e.g., Hex encoding, Comment injection) to create a diverse training set (~40k samples).
*   **Phases:**
    *   **Phase 1 (SFT):** Direct payload generation.
    *   **Phase 2 (Reasoning):** Data enriched with "Chain-of-Thought" or structured context (Technique -> Strategy -> Payload).

#### 2.3. Training Pipeline
*   **Environment:** Remote training server (due to local hardware limits).
*   **Configurations:** Used QLoRA (4-bit), AdamW optimizer.
*   **Reference:** See `training_loss_report.md` for specific loss curves.

### Section 3: Validation Strategy (The Local Benchmark)

#### 3.1. The WAF Environment
*   Described in `docker-compose.multiwaf.yml`.
*   **Targets:** DVWA (Damn Vulnerable Web App) protected by:
    *   **ModSecurity (PL1):** Baseline protection.
    *   **ModSecurity (PL4):** Paranoia mode (extreme difficulty).
    *   **Coraza:** Go-based WAF (Modern alternative).

#### 3.2. Prompt Engineering (The "Secret Sauce")
*   *Insight:* The project discovered that prompt structure is critical.
*   **Phase 1 Prompt:** Simple instruction. *"Generate an SQLi payload..."*
*   **Phase 2 Prompt:** Structured Context. *"You are an offensive security assistant... Context: WAF PL1... Technique: Double URL..."*.
*   *Source:* Check `scripts/evaluate_remote_adapters.py` functions `generate_payload_from_model` for exact templates.

### Section 4: Experimental Results

**Use the charts in `reports/figures/` for this section.**

#### 4.1. Quantitative Analysis
*   Refer to the table in `remote_performance.md`.
*   **Key Finding:** **Phi-3 Mini Phase 2** was the best performer (High Pass Rate + High Stability).
*   **Surprise:** Qwen 7B performed worse in Phase 2 than Phase 1 (Over-fitting/Over-thinking).
*   **Progression:** Gemma 2 2B improved steadily (Phase 1 -> 2 -> 3).

#### 4.2. Qualitative Analysis (Payload Diversity)
*   Analyze the specific payloads found in `remote_performance.md`.
*   **Phi-3:** Used advanced nested encoding (Triple URL, Unicode Homoglyphs).
*   **Qwen:** Sometimes generated "hallucinated" Hex strings that were valid but weird.

### Section 5: Limitations & Challenges (The "Realism" Section)

*   **Hardware Constraints:**
    *   **Llama 3.1 8B Failure:** We attempted to evaluate Llama 3.1 8B, but it consistently caused OOM (Out Of Memory) on the local 8GB VRAM GPU, even with 4-bit quantization. This highlights the barrier to entry for larger models.
    *   **RL Training Issues:** Running Reinforcement Learning (PPO/REINFORCE) locally for Phi-3 Mini failed due to memory overhead (gradients + buffer), requiring a move back to SFT or smaller batch sizes.
*   **WAF Login Issues:** Docker container stability issues with ModSecurity (crashing under load) required robust retry logic in scripts.

### Section 6: Conclusion
*   Small, reasoned models (Phi-3) > Larger, standard models (Qwen 7B) for this specific niche task.
*   RAG (Retrieval Augmented Generation) coupled with Reasoning Prompts is the most promising path forward.

---

## 4. Suggested Artifacts to Include
*   **Tables:** Pass rates from `remote_performance.md`.
*   **Charts:** `reports/figures/performance_comparison.png`, `reports/figures/technique_heatmap.png`.
*   **Code Snippets:** The Prompt Template from `scripts/evaluate_remote_adapters.py`.
