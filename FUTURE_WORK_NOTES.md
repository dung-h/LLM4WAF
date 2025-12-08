# Future Work & Hypotheses: Sequential vs. Independent Fine-Tuning

## 1. Current Limitation (The "Independent" Strategy)
Due to local hardware constraints (8GB VRAM), we currently train each Phase (SFT, Reasoning, Adaptive) as a separate LoRA adapter starting from the Base Model.
*   **Pros:** Saves memory, easy to manage parallel experiments.
*   **Cons:** Models in Phase 2/3 do not "inherit" the exact weights/knowledge from Phase 1. They rely solely on the new dataset to relearn basic syntax while learning new logic.

## 2. The "Sequential" Hypothesis (To be tested)
**Hypothesis:** Sequentially fine-tuning the model (Base -> Phase 1 -> Phase 2 -> Phase 3) will produce a superior "Super-Model" capable of handling multiple prompt formats (Basic, Reasoning, Adaptive) and achieving higher success rates.

**Logic:**
*   Phase 1 lays the foundation (Syntax).
*   Phase 2 adds depth (Reasoning).
*   Phase 3 adds adaptability (In-Context Learning).
*   *Catastrophic Forgetting* can be mitigated by mixing a small percentage of previous datasets into the current training set (Replay Buffer).

## 3. Execution Plan (Requires >24GB VRAM)
1.  **Train Phase 1:** Save Adapter P1.
2.  **Merge:** Merge Adapter P1 into Base Model -> Create Base_v1.
3.  **Train Phase 2:** Load Base_v1, train on Reasoning Dataset. Save Adapter P2.
4.  **Merge:** Merge Adapter P2 into Base_v1 -> Create Base_v2.
5.  **Train Phase 3:** Load Base_v2, train on Adaptive Dataset.
6.  **Compare:** Benchmark `Base_v3` vs. the isolated adapters we have now.

## 4. Action Item
Keep the current datasets (`red_phase1_...`, `red_phase2_...`, `red_phase3_...`) safe. They are the key to running this sequential experiment later.
