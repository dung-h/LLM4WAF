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