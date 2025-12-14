# LLM4WAF: Red Team LLM for WAF Evasion

## 🚀 Tổng quan

Framework huấn luyện LLM (3 phase) để sinh payload SQLi/XSS né WAF (ModSecurity/Coraza) và kiểm thử trên DVWA. Báo cáo chi tiết: `demo1_report.pdf`.

## 👥 Nhóm & Cảm ơn

- Lớp/Nhóm: TN01 – Đồ án CO3101. GVHD: TS. Nguyễn An Khương. Trợ giảng: Trần Lê Quốc Khánh (B.Eng.).
- Thành viên: Hồ Anh Dũng, Nguyễn Đăng Hiên, Lê Tiến Đạt.
- Cảm ơn: Đoàn Thế Anh, Nguyễn Ban Hữu Quang, Nguyễn Anh Kiệt hỗ trợ pipeline/môi trường.

## ⚡ Quick Start

```bash
git clone https://github.com/dung-h/LLM4WAF.git
cd LLM4WAF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Adapters (không nằm trong repo):** tải và giải nén vào `experiments/` với các thư mục: `remote_phi3_mini_phase{1,2,3_rl}`, `remote_qwen_3b_phase{1,2,3_rl}`, `remote_gemma2_2b_phase{1,2,3_rl}`. Link: https://drive.google.com/drive/folders/1WBIh6O_NoPfmZg_hMWydStO3sNWf_o2Y?usp=sharing

### WAF environment

```bash
docker-compose -f docker-compose.multiwaf.yml up -d --remove-orphans
```

Endpoints: ModSecurity PL1 `http://localhost:8000/dvwa/`, Blue `:8001`, PL4 `:9008`, Coraza `:9005` (proxy DVWA).

### Train Red Agent (3 phase)

```bash
# Phase 1 SFT
python scripts/train_red.py --config configs/remote_gemma2_2b_phase1.yaml
python scripts/train_red.py --config configs/remote_phi3_mini_phase1.yaml
python scripts/train_red.py --config configs/remote_qwen_3b_phase1.yaml

# Phase 2 Reasoning/Observation SFT
python scripts/train_red.py --config configs/remote_gemma2_2b_phase2.yaml
python scripts/train_red.py --config configs/remote_phi3_mini_phase2.yaml
python scripts/train_red.py --config configs/remote_qwen_3b_phase2.yaml

# Phase 3 RL (adaptive WAF)
python scripts/train_rl_adaptive_pipeline.py --config configs/remote_gemma2_2b_phase3_rl.yaml
python scripts/train_rl_adaptive_pipeline.py --config configs/remote_phi3_mini_phase3_rl.yaml
python scripts/train_rl_adaptive_pipeline.py --config configs/remote_qwen_3b_phase3_rl.yaml
```

### Đánh giá nhanh

```bash
python scripts/test_training_payloads_strict_waf.py      # sanity vs WAF
python scripts/test_rl_checkpoint.py                     # so sánh checkpoint RL
python scripts/analyze_rl_metrics.py                     # phân tích reward
```

### Demo Gradio

```bash
source .venv/bin/activate
HF_TOKEN=<token> python demo/app.py
```

- Local target mặc định `http://localhost:8000/dvwa` (chỉnh được trong UI). Remote: nhập dạng `http://host:port/dvwa` (không slash cuối); app tự nối `/dvwa/login.php` khi verify.

### Attack Pipeline (Headless)

Script `run_attack_pipeline.py` hỗ trợ Phase 1 (direct generation) và Phase 3 (RL adaptive):

**Phase 1 - Direct Generation:**

```bash
# Phi-3 Phase 1: Generate 5 payloads
python scripts/run_attack_pipeline.py --phase 1 --model phi3 --num-payloads 5

# Qwen Phase 1: Generate 10 payloads
python scripts/run_attack_pipeline.py --phase 1 --model qwen --num-payloads 10

# Gemma Phase 1: Generate 3 payloads
python scripts/run_attack_pipeline.py --phase 1 --model gemma --num-payloads 3
```

**Phase 3 - RL Adaptive Attack:**

```bash
# Phi-3 Phase 3 RL (with WAF probing)
python scripts/run_attack_pipeline.py --phase 3 --model phi3

# Qwen Phase 3 RL
python scripts/run_attack_pipeline.py --phase 3 --model qwen

# Gemma Phase 3 RL
python scripts/run_attack_pipeline.py --phase 3 --model gemma
```

Options:

- `--phase`: Training phase (1 = SFT only, 3 = RL adaptive)
- `--model`: Model to use (phi3, qwen, gemma)
- `--num-payloads`: Number of payloads to generate (Phase 1 only)

## 🧠 Pipeline RED (tóm tắt)

- **Phase 1 SFT:** instruction → payload (SQLi/XSS cơ bản). Dataset gốc: `data/processed/phase1_passed_only_39k.jsonl` (seed sinh từ LLM ngoại repo + cân bằng).
- **Phase 2 Reasoning SFT:** prompt có Context + History + Target Technique + Reasoning → payload; dataset: `data/processed/red_phase2_reasoning_combined.jsonl` (kèm replay/observations).
- **Phase 3 RL:** thưởng từ WAF thật (DVWA + ModSecurity/Coraza); env: `rl/waf_env.py`, script `train_rl_adaptive_pipeline.py`.

## 📊 Kết quả chính (reports/training_eval_tables.md)

- **ModSecurity PL1/PL4:** Qwen 3B RL 100% pass; Gemma 2B Phase2 ~75–100%, RL ~80–90%; Phi-3 Mini RL ~60%.
- **Coraza:** Gemma 2B Phase2 ~97% pass; Qwen 3B RL ~95.5%; Phi-3 Mini RL ~62.5%.
- CSV: `reports/eval_modsec_pass_rates.csv`, `reports/eval_coraza_pass_rates.csv`.

## 📚 Dataset scripts (theo báo cáo demo1)

```bash
python scripts/analysis/dataset_construction.py       # Sinh Phase1 (10k) via LLM + WAF filter (cần API key + DVWA)
python scripts/create_phase2_with_replay.py           # Phase2 reasoning + replay observations
python scripts/build_phase3_lightweight.py            # Phase3 lightweight/filtered set
```

## 🧾 Prompt templates (định dạng chính)

- **Phase 1 (instruction → payload, không giải thích)**

```text
Generate WAF-evasion payloads.
Target: SQLI on ModSecurity PL1.
Technique: Double URL Encode
IMPORTANT: Generate ONLY the payload. No explanation.
```

- **Phase 2 (structured + reasoning)**

```text
Context: ModSecurity + OWASP CRS 3.3 (PL1)
Attack Type: SQLI
Injection Point: GET param 'id'
Payload History:
1. ' OR 1=1 -- -> BLOCKED
2. %27%20OR%20%271%27%3D%271 -> PASSED
Target Technique: Comment Obfuscation
Task: Learn from PASSED, avoid BLOCKED, output ONLY payload.
```

- **Phase 3 RL**
  - Prompt/state dựng trong `train_rl_adaptive_pipeline.py` từ probe history; reward +1 (bypass), -1 (block). Không có template cố định, model học qua trial-and-error.

## 🎯 Demo: Attack Pipeline (ngắn gọn)

- Chọn target (Local/Remote) → Verify `/dvwa/login.php`.
- Load model + adapter phase (1/2/3) → chọn attack type/kỹ thuật → Generate & Attack.
- Kết quả hiển thị Live Logs + bảng payload/status/latency.

## 📂 Cấu trúc chính

- `configs/` – YAML cho từng phase/model.
- `scripts/` – huấn luyện, eval, RL, dữ liệu, attack pipeline.
- `demo/` – Gradio app + WAF executor.
- `rl/` – môi trường RL (`waf_env.py`).
- `waf/`, `dvwa-modsecurity-waf/`, `coraza/`, `naxsi/` – cấu hình WAF.
- `experiments/remote_*_phase{1,2,3_rl}` – đặt adapter tải về từ Drive.
- `reports/` – bảng kết quả, mẫu payload.

## 📄 Tham khảo

- demo1_report.pdf – tổng hợp thiết kế, kết quả, giới hạn.
