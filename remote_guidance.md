# Hướng Dẫn Train Remote: Phase 1-2-3 Cũ (Scale Up 7B/14B)

Tài liệu này hướng dẫn tái tạo quy trình training cũ (SFT -> Reasoning -> RL) trên các model lớn hơn để kiểm chứng hiệu năng.

## 1. Chiến Lược Dữ Liệu & Model

*   **Dataset:** Khuyên dùng **`data/processed/red_phase1_enriched_v2.jsonl`** (~42k samples). Đây là bản đã được làm giàu, cân bằng lại tỉ lệ SQLi/XSS/OS_Injection và bổ sung các kỹ thuật né tránh phức tạp.
*   **Model Size:**
    *   **7B/8B:** Cân bằng tốt giữa hiệu năng và tốc độ.
    *   **14B:** Khả năng học reasoning và generalize tốt nhất, nhưng cần VRAM 24GB+.

## 2. Chuẩn Bị Dữ Liệu (Local -> Remote)

Nén và upload các file sau lên server:
1.  `data/processed/red_phase1_enriched_v2.jsonl` (Phase 1 SFT - **Bản Enriched**)
2.  `data/processed/red_v40_phase2_reasoning.jsonl` (Phase 2 Reasoning cũ)
3.  Codebase: Toàn bộ thư mục `scripts/` (đặc biệt `scripts/train_red.py`), `rl/`, `configs/`.

## 3. Script Training Tự Động (One-Click)

Tạo file `run_remote_optimized.sh` trên server:

```bash
#!/bin/bash

# --- CONFIGURATION ---
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-14B-Instruct" 

# Hugging Face Token
export HF_TOKEN="hf_..."

# Data Paths (Lưu ý: Dùng bản Enriched cho Phase 1)
DATA_P1="data/processed/red_phase1_enriched_v2.jsonl"
DATA_P2="data/processed/red_v40_phase2_reasoning.jsonl"
OUTPUT_ROOT="experiments_remote_optimized"

echo "🚀 Starting Optimized Training Pipeline for $MODEL_NAME..."

# --- HYPERPARAMETERS SETUP ---
# Tự động điều chỉnh tham số dựa trên size model để tránh OOM (Out of Memory)
if [[ "$MODEL_NAME" == *"14B"* ]]; then
    echo "⚙️ Config for 14B Model (High VRAM usage)"
    BATCH_SIZE=1          # Giảm batch size để không tràn VRAM 24GB
    GRAD_ACCUM=16         # Tăng tích lũy để giữ effective batch size ~16
    LORA_R=64             # Rank cao cho model lớn
    LORA_ALPHA=128
    LR="1e-4"             # Learning rate an toàn
elif [[ "$MODEL_NAME" == *"7B"* ]] || [[ "$MODEL_NAME" == *"8B"* ]]; then
    echo "⚙️ Config for 7B/8B Model"
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LORA_R=32
    LORA_ALPHA=64
    LR="2e-4"
else
    # Default / Small models
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LORA_R=16
    LORA_ALPHA=32
    LR="2e-4"
fi

# --- PHASE 1: BASE SFT (Kiến thức nền) ---
echo "--- Phase 1: Base SFT (Enriched Data) ---"
# Epoch = 1 là đủ cho 40k samples để tránh catastrophic forgetting
cat <<EOF > config_p1_opt.yaml
model_name: "$MODEL_NAME"
train_path: "$DATA_P1"
output_dir: "$OUTPUT_ROOT/phase1_sft"
load_in_4bit: true
lora_r: $LORA_R
lora_alpha: $LORA_ALPHA
lora_dropout: 0.05
num_train_epochs: 1
per_device_train_batch_size: $BATCH_SIZE
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: $LR
logging_steps: 10
save_steps: 200
max_length: 2048
use_auth_token_env: "HF_TOKEN"
EOF

python3 scripts/train_red.py --config config_p1_opt.yaml
echo "✅ Phase 1 Complete."

# --- PHASE 2: REASONING (Tư duy) ---
echo "--- Phase 2: Legacy Reasoning ---"
# Phase 2 dataset thường nhỏ hơn, có thể train 2-3 epochs
cat <<EOF > config_p2_opt.yaml
model_name: "$MODEL_NAME"
train_path: "$DATA_P2"
output_dir: "$OUTPUT_ROOT/phase2_reasoning"
load_in_4bit: true
lora_r: $LORA_R
lora_alpha: $LORA_ALPHA
lora_dropout: 0.05
num_train_epochs: 2
per_device_train_batch_size: $BATCH_SIZE
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: $LR
logging_steps: 10
save_steps: 100
max_length: 2048 
use_auth_token_env: "HF_TOKEN"
EOF

python3 scripts/train_red.py --config config_p2_opt.yaml
echo "✅ Phase 2 Complete."

echo "🎉 DONE! Adapters saved in $OUTPUT_ROOT"
tar -czvf adapters_optimized_$(date +%Y%m%d).tar.gz $OUTPUT_ROOT
```

## 4. Lời khuyên vận hành

*   **Flash Attention:** Trên server, nhớ cài `pip install flash-attn --no-build-isolation` để tăng tốc độ train gấp 2-3 lần cho model Qwen/Llama.
*   **Monitoring:** Mở thêm một terminal chạy `watch -n 1 nvidia-smi` để theo dõi VRAM. Nếu thấy gần full (23xxx MiB), hãy stop và giảm `BATCH_SIZE` xuống 1 ngay lập tức.
