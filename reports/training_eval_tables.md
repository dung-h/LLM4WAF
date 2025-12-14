# Section 4 Tables (auto-generated)

## Hyperparameters - SFT (Phase 1 & Phase 2)
| Model | Phase | LR | Per-device batch | Grad accum | Effective batch | Epochs | Max seq len | LoRA r | LoRA alpha | 4-bit |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma_2B_Phase1 | Phase 1 | 0.0002 | 1 | 16 | 16 | 3 | 1024 | 16 | 32 | True |
| Gemma_2B_Phase2 | Phase 2 | 0.0002 | 1 | 16 | 16 | 2 | 1024 | 16 | 32 | True |
| Phi3_Mini_Phase1 | Phase 1 | 0.0002 | 1 | 16 | 16 | 3 | 1024 | 16 | 32 | True |
| Phi3_Mini_Phase2 | Phase 2 | 0.0002 | 1 | 16 | 16 | 2 | 1024 | 16 | 32 | True |
| Qwen_3B_Phase1 | Phase 1 | 0.0002 | 2 | 8 | 16 | 3 | 1024 | 16 | 32 | True |
| Qwen_3B_Phase2 | Phase 2 | 0.0002 | 2 | 8 | 16 | 2 | 1024 | 16 | 32 | True |

## Hyperparameters - RL (Phase 3)
| Model | LR | Batch | Epochs | Max new tokens | Context len | Adapter from |
|---|---|---|---|---|---|---|
| Gemma_2B_RL | 1e-6 | 1 | 150 | 256 | 1024 | experiments/remote_gemma2_2b_phase2 |
| Phi3_Mini_RL | 1e-6 | 1 | 150 | 256 | 1024 | experiments/remote_phi3_mini_phase2 |
| Qwen_3B_RL | 1e-6 | 2 | 150 | 256 | 1024 | experiments/remote_qwen_3b_phase2 |

## Pass Rate - ModSecurity (matched dataset per model)
| Model | PL1 Pass % | PL1 Invalid % | PL4 Pass % | PL4 Invalid % |
|---|---|---|---|---|
| Gemma_2B_Phase1 | 60.0 | 0.0 | 70.0 | 0.0 |
| Gemma_2B_Phase2 | 75.0 | 25.0 | 100.0 | 0.0 |
| Gemma_2B_RL | 80.0 | 20.0 | 90.0 | 10.0 |
| Qwen_3B_Phase1 | 30.0 | 0.0 | 40.0 | 0.0 |
| Qwen_3B_Phase2 | 93.8 | 6.2 | 93.8 | 6.2 |
| Qwen_3B_RL | 100.0 | 0.0 | 100.0 | 0.0 |
| Phi3_Mini_Phase1 | 40.0 | 0.0 | 50.0 | 0.0 |
| Phi3_Mini_Phase2 | 18.8 | 75.0 | 25.0 | 68.8 |
| Phi3_Mini_RL | 60.0 | 0.0 | 60.0 | 0.0 |
| Gemma_2B_Pretrained | 40.0 | 50.0 | 40.0 | 50.0 |
| Qwen_3B_Pretrained | 10.0 | 60.0 | 30.0 | 60.0 |
| Phi3_Mini_Pretrained | 20.0 | 70.0 | 40.0 | 60.0 |

## Pass Rate - Coraza (matched dataset per model)
| Model | CORAZA Pass % | CORAZA Invalid % |
|---|---|---|
| Gemma_2B_Phase1 | 50.5 | 0.0 |
| Gemma_2B_Phase2 | 97.0 | 2.0 |
| Gemma_2B_RL | 94.0 | 5.5 |
| Qwen_3B_Phase1 | 37.0 | 5.5 |
| Qwen_3B_Phase2 | 73.0 | 22.0 |
| Qwen_3B_RL | 95.5 | 0.5 |
| Phi3_Mini_Phase1 | 42.0 | 2.0 |
| Phi3_Mini_Phase2 | 25.0 | 59.5 |
| Phi3_Mini_RL | 62.5 | 0.5 |

## Raw CSV references
- reports/eval_modsec_pass_rates.csv
- reports/eval_coraza_pass_rates.csv