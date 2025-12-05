# Remote Training Loss Analysis

This report summarizes the training loss progression for the remote adapter models (Qwen 7B, Phi-3 Mini, Gemma 2 2B). The data is extracted from the `trainer_state.json` files within the finalized checkpoints.

## 1. Qwen 7B - Phase 1 SFT
*   **Model Base:** `Qwen/Qwen2.5-7B-Instruct`
*   **Steps:** 2668
*   **Start Loss:** 0.8985
*   **End Loss:** 0.2598
*   **Analysis:**
    *   The model started with a relatively low loss (< 1.0), indicating the base model already had good language understanding.
    *   Loss stabilized quickly around step 100.
    *   Final convergence at ~0.26 suggests solid learning without excessive overfitting (mean token accuracy > 92%).

## 2. Phi-3 Mini - Phase 2 Reasoning
*   **Model Base:** `microsoft/Phi-3-mini-4k-instruct`
*   **Steps:** 199 (Phase 2 often uses fewer steps for alignment/reasoning injection)
*   **Start Loss:** 1.0539
*   **End Loss:** 0.2876
*   **Analysis:**
    *   Rapid convergence.
    *   Entropy decreased consistently, showing increased confidence in the "reasoning" format (Chain-of-Thought style generation).
    *   The final loss is comparable to Qwen's Phase 1, which is impressive for a smaller model in a more complex task (reasoning).

## 3. Gemma 2 2B - Phase 2 Reasoning
*   **Model Base:** `google/gemma-2-2b-it`
*   **Steps:** 314
*   **Start Loss:** 2.864
*   **End Loss:** 0.3381
*   **Analysis:**
    *   **High Initial Loss:** Started much higher than Qwen and Phi-3 (2.86 vs ~1.0). This suggests the initial "Reasoning" prompt format or data distribution was quite different from what the base Gemma model expected.
    *   **Fast Learning Rate:** Despite the high start, it dropped to < 0.7 within just 50 steps. This shows Gemma 2's high plasticity and ability to adapt quickly to new paradigms.
    *   Final loss (0.33) is slightly higher than Phi-3 and Qwen, correlating with its slightly lower performance in benchmarks, but still indicates successful training.

## Summary Table

| Model | Phase | Start Loss | End Loss | Convergence Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 7B** | Phase 1 SFT | 0.8985 | 0.2598 | Steady |
| **Phi-3 Mini** | Phase 2 Reasoning | 1.0539 | 0.2876 | Fast |
| **Gemma 2 2B** | Phase 2 Reasoning | 2.864 | 0.3381 | Very Fast (High Initial Gap) |

**Conclusion:** All three models successfully converged. The higher initial loss for Gemma suggests it benefited the most from the fine-tuning process (learning something new), whereas Qwen and Phi-3 might have been closer to the target distribution initially.
