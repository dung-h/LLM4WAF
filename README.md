# LLM4WAF: Hệ Thống Red & Blue Teaming Tự Động Cho Web Application Firewalls

Dự án này triển khai một framework khép kín (end-to-end) cho **Tấn công Đối kháng (Red Team)** và **Tinh chỉnh Phòng thủ Thông minh (Blue Team)** sử dụng các Mô hình Ngôn ngữ Lớn (LLMs).

## 🚀 Tổng Quan Dự Án

Mục tiêu là tự động hóa quy trình tìm kiếm lỗ hổng (bypass WAF) và vá chúng:
1.  **Red Agent (Tấn công):** Sử dụng Học tăng cường (Reinforcement Learning - RL) để sinh ra các payload SQL Injection (SQLi) và XSS tinh vi nhằm vượt qua WAF.
2.  **Blue Agent (Phòng thủ):** Phân tích các cuộc tấn công thành công bằng RAG (Retrieval-Augmented Generation) và cơ sở tri thức OWASP Core Rule Set (CRS) để đề xuất cấu hình WAF chính xác.

---

## 🛠️ Cài Đặt Môi Trường

### 1. Yêu cầu tiên quyết
*   **OS:** Linux (Khuyên dùng WSL2 trên Windows).
*   **Python:** 3.10+.
*   **Docker & Docker Compose:** Bắt buộc để chạy các WAF và ứng dụng mục tiêu (DVWA, Juice Shop).
*   **GPU:** NVIDIA GPU (Khuyên dùng 16GB+ VRAM) để train và inference LLM cục bộ.

### 2. Cài đặt
```bash
# 1. Clone repository
git clone <repo_url>
cd LLM_in_Cyber

# 2. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

---

## 🔴 RED Agent Pipeline (Đội Tấn Công)

Red Agent tiến hóa từ một bộ sinh payload cơ bản thành một công cụ né tránh thông minh qua 3 giai đoạn.

### Phase 1: Supervised Fine-Tuning (SFT) - Học Cú Pháp
**Mục tiêu:** Dạy model nắm vững cú pháp của các payload SQLi và XSS hợp lệ đã từng bypass thành công.
*   **Dữ liệu đầu vào:** `data/processed/red_v40_passed_waf_only.jsonl` (Tập hợp các payload bypass thành công).
*   **1. Huấn luyện (Training):**
    ```bash
    python scripts/train_red.py --config configs/red_gemma2_2b_lora_v2.yaml
    ```
    *   *Output:* Checkpoint model tại `experiments/red_gemma2_2b_lora_v2`.
*   **2. Kiểm thử (Testing):**
    Đánh giá khả năng sinh payload hợp lệ trên tập test tĩnh.
    ```bash
    python scripts/evaluate_model.py \
      --base_model google/gemma-2-2b-it \
      --adapter experiments/red_gemma2_2b_lora_v2 \
      --dataset data/processed/red_v40_test_200.jsonl \
      --format gemma
    ```

### Phase 2: Reasoning SFT (Chain-of-Thought) - Học Tư Duy
**Mục tiêu:** Cải thiện khả năng thích ứng bằng cách dạy model *cách suy nghĩ* về việc né tránh (Reasoning Traces).
*   **Dữ liệu đầu vào:** `data/processed/red_v40_phase2_reasoning.jsonl` (Bộ ba: Lịch sử -> Suy luận -> Payload mới).
*   **1. Huấn luyện (Training):**
    ```bash
    # (Optional) Format dataset
    # python scripts/build_phase2_dataset.py 
    
    python scripts/train_red.py --config configs/phase2_gemma2_2b_reasoning.yaml
    ```
    *   *Output:* Checkpoint model tại `experiments/phase2_gemma2_2b_reasoning`.
*   **2. Kiểm thử (Testing):**
    Kiểm tra model có sinh ra chuỗi suy luận (reasoning) hợp lý trước khi tạo payload không.
    ```bash
    python scripts/evaluate_model.py \
      --base_model google/gemma-2-2b-it \
      --adapter experiments/phase2_gemma2_2b_reasoning \
      --dataset data/processed/red_v40_phase2_eval_test.jsonl \
      --format gemma
    ```

### Phase 3: Reinforcement Learning (RL) - Tối Ưu Hóa
**Mục tiêu:** Tối đa hóa tỷ lệ Bypass WAF và Thực thi thành công thông qua tương tác Thử & Sai.
*   **Đầu vào:** 
    *   **Model khởi tạo:** Load từ Phase 2 (`experiments/phase2_gemma2_2b_reasoning`).
    *   **Môi trường:** Docker container cục bộ (`WafEnv`) chạy ModSecurity/Coraza.
*   **1. Huấn luyện (Training):**
    ```bash
    # Khởi động môi trường WAF trước
    docker-compose -f docker-compose.multiwaf.yml up -d
    
    # Chạy RL Training loop
    python scripts/train_rl_reinforce.py --epochs 25 --batch_size 2
    ```
    *   *Output:* Model hoàn thiện tại `experiments/phase3_gemma2_2b_rl`.
*   **2. Kiểm thử (Testing - Tấn công thực tế):**
    Sử dụng model RL để tấn công vào các WAF mục tiêu và đo tỷ lệ bypass.
    ```bash
    python scripts/run_red_eval_profile.py \
      --config configs/eval_phase3_multiwaf_gemma2.yaml \
      --num_samples 50
    ```

---

## 🔵 BLUE Agent Pipeline (Đội Phòng Thủ)

Blue Agent đóng vai trò là một Chuyên gia An ninh AI để tinh chỉnh WAF dựa trên dữ liệu từ Red Team.

### Phase 1: Chuẩn Bị Dữ Liệu & Knowledge Base
**Mục tiêu:** Chuẩn bị dữ liệu cho AI Analyst.
*   **Đầu vào:** Log tấn công từ Red Team (format JSONL).
*   **Quy trình:**
    1.  **Episodes:** Chuyển đổi log thô thành "Episodes" có cấu trúc (Attack + WAF Response + App Response).
    2.  **Knowledge Base:** Index tài liệu OWASP CRS (regex rules, tags) vào vector store.
*   **Lệnh chạy:**
    ```bash
    # Build Episodes
    python scripts/blue_build_phase1_episodes.py
    
    # Build Knowledge Base
    python scripts/blue_build_crs_kb.py
    ```
*   **Dữ liệu đầu ra:** 
    *   `data/blue/blue_phase1_episodes.jsonl`
    *   `data/blue/blue_phase1_crs_kb.jsonl`

### Phase 2: RAG Analysis & Evaluation
**Mục tiêu:** Truy xuất các rule liên quan và kiểm chứng khả năng phân tích của AI trên tập Golden Set.
*   **Đầu vào:** `data/blue/blue_phase1_golden.jsonl` (Các case đã được xác minh).
*   **Lệnh chạy:**
    ```bash
    python blue/runner_phase2_eval.py
    ```
*   **Đầu ra:** `data/blue/blue_phase2_eval_summary.txt` (Báo cáo độ chính xác phân tích).

### Phase 3: Recommendation Generation (Tạo Đề Xuất)
**Mục tiêu:** Sinh ra các thay đổi cấu hình cụ thể (Bản vá).
*   **Đầu vào:** `data/blue/blue_phase1_episodes.jsonl` + RAG Knowledge Base.
*   **Quy trình:** Blue LLM (Sử dụng Gemma 2 Base Model để đảm bảo format JSON chuẩn) phân tích từng cuộc tấn công thành công và đề xuất rule WAF cụ thể.
*   **Lệnh chạy:**
    ```bash
    python blue/runner_phase3_suggest.py
    ```
*   **Đầu ra:** `data/blue/blue_phase3_suggestions.jsonl` (Danh sách JSON các rule được đề xuất).

### Phase 4: WAF Overlay & Evaluation (Áp Dụng & Đánh Giá)
**Mục tiêu:** Áp dụng bản vá và kiểm tra hiệu quả.
*   **Quy trình:** 
    1.  **Generate Config:** Chuyển đổi JSON suggestions thành file config WAF thực tế (`.conf`, `.yaml`).
    2.  **Re-Eval:** Khởi động lại WAF với config mới và cho Red Team tấn công lại.
*   **Lệnh chạy:**
    ```bash
    # 1. Tạo file cấu hình WAF
    python blue/phase3_generate_waf_overlays.py
    
    # 2. Khởi động môi trường Multi-WAF
    docker-compose -f docker-compose.multiwaf.yml up -d --build
    
    # 3. Chạy đánh giá Red Team (Kiểm tra lại khả năng bypass)
    python scripts/run_red_eval_profile.py --config configs/eval_phase3_multiwaf_gemma2.yaml
    ```
*   **Đầu ra:** 
    *   `waf/blue_modsecurity_suggestions.conf`: File chứa rule WAF mới sinh ra.
    *   `eval/red_phase4_overall_summary.json`: Báo cáo so sánh hiệu quả (Base WAF vs. Blue Tuned WAF).

---

## 📊 Kết Quả Chính (Ví dụ)

Đánh giá gần nhất trên DVWA (SQL Injection):

| Profile | WAF Engine | Ruleset | Blocked % | WAF Bypass % |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | ModSecurity | OWASP CRS PL1 | ~5.7% | 94.3% |
| **Blue Tuned** | ModSecurity | PL1 + Blue Overlay | **Đã cải thiện** | **(Mục tiêu: < 90%)** |
| **Strict** | ModSecurity | OWASP CRS PL4 | 0% (Cần check config) | 100% |

*Lưu ý: Hiệu suất của "Blue Tuned" phụ thuộc vào chất lượng suy luận của LLM trong Phase 3.*

---

## 📂 Cấu Trúc Dự Án

*   `blue/`: Source code cho Blue Agent (RAG, LLM client, Prompts).
*   `configs/`: Các file cấu hình YAML cho training và evaluation.
*   `data/`: Dữ liệu (Logs đã xử lý, Episodes, Knowledge Base).
*   `docker-compose.multiwaf.yml`: Định nghĩa môi trường đánh giá nhiều WAF.
*   `rl/`: Môi trường và logic cho Reinforcement Learning.
*   `scripts/`: Các script tiện ích cho training, xử lý dữ liệu và đánh giá.
*   `waf/`: Các file cấu hình WAF overlay được sinh ra tự động.

---

## 🤝 Thực hiện
*   **HAD** - Lead Developer / AI Security Researcher

---

## 🐛 Known Issues / Troubleshooting

### 1. CUDA Out of Memory (OOM) on 8GB GPUs for Gemma 2B Training

*   **Vấn đề:** Khi fine-tune Gemma 2 2B (kể cả với QLoRA 4-bit), GPU 8GB (ví dụ RTX 3050, 3060, 4060) thường gặp lỗi `CUDA Out of Memory` (`torch.OutOfMemoryError`). Điều này xảy ra ngay cả khi `per_device_train_batch_size` đã giảm xuống 1 và `gradient_accumulation_steps` đã tăng.
*   **Nguyên nhân:** Model Gemma 2 2B, dù là 2 tỷ tham số, nhưng có kiến trúc phức tạp và `max_seq_length` lớn (đặc biệt cần cho RAG context) đòi hỏi lượng VRAM đáng kể. Cấu hình mặc định (ví dụ `max_seq_length=1024`) quá lớn đối với 8GB VRAM.
*   **Giải pháp được đề xuất:**
    *   **Tốt nhất:** Sử dụng GPU có VRAM từ **16GB trở lên** (ví dụ: RTX 3090/4090, A10G, A5000/6000).
    *   **Tạm thời (nếu chỉ có 8GB VRAM):**
        *   Giảm `max_seq_length` trong file config (`configs/red_phase2_rag_sft.yaml`) xuống **512 hoặc thậm chí 256**. Tuy nhiên, điều này sẽ làm giảm đáng kể lượng RAG context mà model có thể xử lý, ảnh hưởng đến hiệu quả của RAG.
        *   Đảm bảo `per_device_train_batch_size` là `1` và `gradient_accumulation_steps` được tăng lên để giữ `effective_batch_size` hợp lý.
        *   Thử tắt `bnb_4bit_use_double_quant` trong `BitsAndBytesConfig` (mặc dù script `train_red.py` đã đọc từ config file, cần thêm tùy chọn này vào config file nếu muốn điều chỉnh).
*   **Liên quan đến RAG:** RAG-SFT rất cần `max_seq_length` đủ lớn để chứa RAG context. Việc giảm `max_seq_length` xuống quá thấp sẽ làm giảm hiệu quả của việc fine-tune RAG-SFT.

---

## ⚠️ Critical Findings Regarding RED Agent Performance

Trong quá trình đánh giá (Evaluation) các model RED Agent (Phase 1, 2, 3), đã phát hiện ra một yếu tố cực kỳ quan trọng ảnh hưởng đến hiệu năng:

1.  **Prompt Sensitivity (Độ nhạy với Prompt):**
    *   **Phase 1 (SFT):** Model này ít nhạy cảm với format prompt. Nó có thể hoạt động tốt (~55% bypass rate) với các prompt đơn giản (e.g., "Generate payload for...").
    *   **Phase 2 (Reasoning) & Phase 3 (RL):** Hai model này **YÊU CẦU BẮT BUỘC** phải sử dụng đúng format prompt mà chúng được huấn luyện (bao gồm các trường `Context`, `Payload History`, `Target Technique`).
    *   **Thực nghiệm:**
        *   Sử dụng prompt đơn giản: Phase 2 đạt ~20%, Phase 3 đạt ~10%.
        *   Sử dụng prompt chuẩn (structured): Phase 2 đạt **~85%**, Phase 3 đạt **~90%**.

2.  **Model Size & RAG Compliance (Phân tích chuyên sâu):**
    *   Ban đầu có thể lầm tưởng các model nhỏ như Gemma 2B ít tuân thủ context RAG. Tuy nhiên, các nghiên cứu gần đây (ví dụ: Ghosh et al., EMNLP 2024, Farahani & Johansson, EMNLP 2024 - tham khảo `evidence.txt`) cho thấy **cả Small (như Phi) lẫn Large LLM đều có xu hướng "dựa vào context nhiều hơn parametric" khi context có liên quan.**
    *   Vấn đề thực sự của Small Model không phải là "không muốn" tuân thủ, mà là **thiếu năng lực xử lý** để đọc, lọc nhiễu, xử lý mâu thuẫn giữa context và parametric knowledge, và tuân thủ các instruction phức tạp trong một context RAG dài. Chúng dễ bị "overloaded" và sinh ra output kém chất lượng.
    *   Do đó, việc huấn luyện RAG-SFT (Phase 2.5) là để **tăng cường khả năng xử lý context hiệu quả** cho model, dạy nó cách tích hợp thông tin RAG vào payload một cách chính xác, đúng cú pháp và tuân thủ các ràng buộc.

3.  **Kết luận:**
    *   Khi tích hợp model Phase 2/3 vào hệ thống khác (ví dụ: RAG), **PHẢI** đảm bảo xây dựng prompt đúng cấu trúc như trong `scripts/build_phase2_dataset.py`.
    *   Việc performance thấp đột ngột thường do "Prompt Mismatch" hoặc "Context Overload" chứ không phải do model bị lỗi hay cố tình bỏ qua RAG.
