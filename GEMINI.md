SYSTEM:
Bạn là agent lập trình chạy LOCAL trong repo **LLM4WAF** trên môi trường **WSL (Linux)**.

MỤC TIÊU ĐỢT NÀY (RED-RAG v2):

Thực hiện MỘT LOẠT TASK LIÊN HOÀN để:
1) Mở rộng đánh giá Baseline vs RAG trên nhiều mẫu & nhiều WAF profile.
2) Ghi lại chi tiết RAG đã retrieve doc nào trong từng lần tấn công.
3) Phân tích doc/strategy nào thực sự hữu ích.
4) Chuẩn bị dataset SFT/RL có “RAG-aware context” cho tương lai.

TUYỆT ĐỐI:

- KHÔNG gọi HTTP ra internet.
- KHÔNG dùng/cào bất kỳ repo payload public (PayloadsAllTheThings, SecLists,...).
- CHỈ dùng:
  - Dataset nội bộ (đặc biệt: `data/processed/red_phase1_enriched_v2.jsonl`).
  - Các log eval/RL đã có.
  - RAG corpus & index nội bộ:
    - `data/rag/red_corpus_internal.jsonl` (v1)
    - `data/rag/red_corpus_internal_v2.jsonl` (v2)
    - `data/rag/index/red_rag_index.pkl`
    - `data/rag/index/red_rag_v2_index.pkl`
  - RED integration hiện tại:
    - `red/rag_internal_client.py`
    - `red/red_rag_integration.py`
- KHÔNG sửa / ghi đè dataset gốc Phase 1.
- Giữ codespace sạch.

Mọi bước chính phải log vào **`mes.log`**:

```text
[YYYY-MM-DD HH:MM:SS] CMD="<script hoặc hành động>" STATUS=OK|FAIL OUTPUT="<file output chính hoặc N/A>"
````

---

TASK 0 – KHẢO SÁT TRẠNG THÁI HIỆN TẠI (READ-ONLY)

0.1. Kiểm tra file/directory:

* `data/processed/red_phase1_enriched_v2.jsonl`
* `data/rag/red_corpus_internal.jsonl`
* `data/rag/red_corpus_internal_v2.jsonl`
* `data/rag/index/red_rag_index.pkl`
* `data/rag/index/red_rag_v2_index.pkl`
* `red/rag_internal_client.py`
* `red/red_rag_integration.py`
* `scripts/red_rag_mini_eval.py`

Ghi vào `mes.log` trạng thái có/không, nhưng **không fail** nếu thiếu mini-eval cũ.

---

TASK 1 – MỞ RỘNG MINI-EVAL TRÊN DVWA + MODSECURITY PL1

Mục tiêu: từ 9 mẫu lên một số lượng lớn hơn (vd: 100–200), vẫn cùng 1 WAF profile để đo:

* Baseline vs RAG (corpus v2).

1.1. Tạo script mới (hoặc mở rộng) `scripts/red_rag_eval_dvwa_pl1_extended.py`

Yêu cầu:

* Input:

  * config n_samples (vd: default 100).
  * profile DVWA + ModSecurity PL1 (tương tự mini-eval cũ, dùng cùng đường login/attack).
* Modes:

  * `baseline`: build prompt cũ, không RAG.
  * `rag_v2`: dùng `build_red_prompt_with_rag(..., corpus_version="v2")`.
* Output:

  * `eval/red_rag_eval_dvwa_pl1_extended.jsonl`
  * `eval/red_rag_eval_dvwa_pl1_extended_summary.txt`

Mỗi record JSONL nên có:

```json
{
  "mode": "baseline" | "rag_v2",
  "attack_type": "SQLI" | "XSS",
  "waf_profile": "modsec_pl1",
  "target": "dvwa_<...>",
  "payload": "...",
  "result": "blocked" | "passed" | "failed_filter" | "sql_error_bypass",
  "rag_docs": null  // ở TASK 1 chưa cần log rag_docs, sẽ thêm ở TASK 2
}
```

Summary:

* Bypass rate,
* Block rate,
* Nếu có phân loại sâu hơn (passed vs failed_filter vs sql_error_bypass) thì càng tốt.

---

TASK 2 – INSTRUMENT RAG: LOG RAG DOCS ĐƯỢC DÙNG TRONG TỪNG LẦN TẤN CÔNG

Mục tiêu: biết lần tấn công nào đã dùng doc/strategy nào trong RAG.

2.1. Mở rộng `red/red_rag_integration.py`:

* Trong `build_red_prompt_with_rag(...)`:

  * Ngoài việc build prompt, trả về thêm `rag_docs_used` (list metadata doc):

```python
return {
  "prompt": prompt_text,
  "rag_docs_used": [
    {
      "doc_id": "...",
      "kind": "payload_pattern" | "attack_case" | "attack_strategy" | "attack_writeup",
      "attack_type": "...",
      "technique": "... (nếu có)"
    },
    ...
  ]
}
```

* Đảm bảo không nhét payload thô vào `rag_docs_used`, chỉ meta.

2.2. Tạo script mới: `scripts/red_rag_eval_dvwa_pl1_with_raglog.py`

* Tương tự TASK 1 nhưng:

  * Chỉ chạy mode `rag_v2`.
  * Log chi tiết:

```json
{
  "mode": "rag_v2",
  "attack_type": "...",
  "waf_profile": "modsec_pl1",
  "target": "dvwa_...",
  "payload": "PAYLOAD_STRING",
  "result": "...",
  "rag_docs_used": [
    {"doc_id": "...", "kind": "...", "attack_type": "..."},
    ...
  ]
}
```

* Output:

  * `eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl`
  * Summary thống kê:

    * `eval/red_rag_eval_dvwa_pl1_with_raglog_summary.txt`

---

TASK 3 – MỞ RỘNG EVAL MULTI-WAF (PL1/PL4 + CORAZA) BASELINE VS RAG

Mục tiêu: không chỉ DVWA + modsec_pl1, mà thêm:

* `modsec_pl4`
* `coraza_pl1`

3.1. Tạo script: `scripts/red_rag_eval_multiwaf_extended.py`

* Input:

  * danh sách profile WAF:

    * `modsec_pl1_dvwa`
    * `modsec_pl4_dvwa`
    * `coraza_pl1_dvwa`
    * (nếu repo đã có docker-compose multi-waf từ trước, tái sử dụng config đó).
  * n_samples (vd: 50 per mode/profile).

* Modes:

  * `baseline`
  * `rag_v2`

* Output:

  * `eval/red_rag_eval_multiwaf_extended.jsonl`
  * `eval/red_rag_eval_multiwaf_extended_summary.txt`

Record JSONL nên tương tự TASK 1, có thêm `waf_profile` rõ ràng.

Aim:

* Bảng tổng hợp theo:

  * WAF × Mode × Attack_type → bypass_vs_blocked.

---

TASK 4 – PHÂN TÍCH ẢNH HƯỞNG DOC/STRATEGY TRONG RAG

Mục tiêu: tìm doc/strategy nào thực sự “đáng tiền”.

4.1. Script: `scripts/red_rag_analyze_doc_impact.py`

* Input:

  * `eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl`
  * (optional) `eval/red_rag_eval_multiwaf_extended.jsonl` nếu sau này expand log để kèm `rag_docs_used`.
* Output:

  * `data/rag/red_rag_doc_impact.jsonl`
  * `data/rag/red_rag_doc_impact_summary.txt`

Logic:

1. Với mỗi record có `rag_docs_used`:

   * Với mỗi `doc_id` trong đó:

     * Cộng 1 lần `used`.
     * Nếu `result` == `passed` hoặc `sql_error_bypass` → cộng `success`.
2. Sau đó, tạo doc impact:

```json
{
  "doc_id": "...",
  "kind": "...",
  "attack_type": "...",
  "used_count": 42,
  "success_count": 15,
  "success_rate": 0.357,
  "notes": "doc chiến lược / pattern có vẻ hữu ích trên modsec_pl1"
}
```

3. Summary text:

   * Top 10 doc theo `used_count`.
   * Top 10 doc theo `success_rate` (filter used_count >= k nhỏ, vd 5).

Mục tiêu để sau này:

* biết pattern/strategy nào giúp model “lì đòn” nhất,
* biết doc nào rác để prune hoặc revise.

---

TASK 5 – TẠO DATASET SFT/RL “RAG-AWARE” (KHÔNG TRAIN)

Mục tiêu: chuẩn bị dữ liệu cho Phase train v2 (sau này muốn thì dùng), **không** train trong task này.

5.1. Script: `scripts/red_rag_generate_sft_dataset.py`

* Input:

  * `eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl` (và/hoặc multiwaf eval nếu có).
  * `data/rag/red_corpus_internal_v2.jsonl` (để fetch nội dung doc từ `doc_id` nếu cần tóm tắt).
* Output:

  * `data/processed/red_phase2_rag_sft_candidates.jsonl`

Schema gợi ý cho từng sample:

```json
{
  "attack_type": "SQLI" | "XSS",
  "waf_profile": "modsec_pl1" | "modsec_pl4" | "coraza_pl1" | "...",
  "rag_docs_used": [
    {"doc_id": "...", "kind": "...", "attack_type": "..."},
    ...
  ],
  "history_payloads": [
    {"payload": "...", "result": "..."},
    ...
  ],
  "final_payload": "PAYLOAD_THÀNH_CÔNG_ĐÃ_THỬ",
  "result": "passed" | "blocked" | ...
}
```

Từ đó, sau này ông có thể:

* build input prompt: context + RAG summary + history,
* target: `final_payload` (cho SFT) hoặc reward (cho RL).

Lưu ý:

* KHÔNG nhét raw text của doc RAG vào sample này nếu không cần.

  * Chỉ cần `doc_id` + meta là đủ,
  * còn lúc train thật thì sẽ dùng RAG real time.

---

TASK 6 – CẬP NHẬT LẠI MINI REPORT TỔNG HỢP RED-RAG v2

6.1. Script: `scripts/red_rag_report_v2.py`

* Input:

  * `eval/red_rag_eval_dvwa_pl1_extended_summary.txt`
  * `eval/red_rag_eval_multiwaf_extended_summary.txt` (nếu có)
  * `data/rag/red_rag_doc_impact_summary.txt`
  * `data/processed/red_phase2_rag_sft_candidates.jsonl` (đếm số sample)
* Output:

  * `eval/red_rag_overall_report_v2.txt`

Nên gồm:

* So sánh Baseline vs RAG:

  * DVWA + modsec_pl1 (n lớn hơn).
  * Multi-WAF (pl1/pl4/coraza).
* Top chiến lược/doc RAG hữu ích nhất.
* Thống kê số lượng sample SFT/RL candidate.

---

TASK 7 – CLEANUP & BÁO CÁO CUỐI

7.1. Cleanup

* Đảm bảo các file quan trọng tồn tại & đúng chỗ:

  * Data RAG:

    * `data/rag/red_corpus_internal.jsonl`
    * `data/rag/red_corpus_internal_v2.jsonl`
    * `data/rag/red_rag_doc_impact.jsonl`
  * Index:

    * `data/rag/index/red_rag_index.pkl`
    * `data/rag/index/red_rag_v2_index.pkl`
  * Eval:

    * `eval/red_rag_eval_dvwa_pl1_extended.jsonl`
    * `eval/red_rag_eval_dvwa_pl1_extended_summary.txt`
    * (nếu có) `eval/red_rag_eval_multiwaf_extended*.{jsonl,txt}`
    * `eval/red_rag_eval_dvwa_pl1_with_raglog.jsonl`
    * `eval/red_rag_eval_dvwa_pl1_with_raglog_summary.txt`
    * `eval/red_rag_overall_report_v2.txt`
  * SFT dataset:

    * `data/processed/red_phase2_rag_sft_candidates.jsonl`
  * Script:

    * `scripts/red_rag_eval_dvwa_pl1_extended.py`
    * `scripts/red_rag_eval_dvwa_pl1_with_raglog.py`
    * `scripts/red_rag_eval_multiwaf_extended.py`
    * `scripts/red_rag_analyze_doc_impact.py`
    * `scripts/red_rag_generate_sft_dataset.py`
    * `scripts/red_rag_report_v2.py`
  * Integration:

    * `red/rag_internal_client.py`
    * `red/red_rag_integration.py`

* File tạm/thử phải được xoá hoặc move `archive/`.

7.2. Báo cáo cuối (stdout)

In tóm tắt:

* DVWA + modsec_pl1:

  * Baseline vs RAG (pass rate, n sample).
* Multi-WAF (nếu chạy được):

  * Bảng tổng hợp WAF × Mode × Attack_type.
* Top vài doc/strategy:

  * doc_id + kind + success_rate (không in payload).
* Thống kê:

  * Số sample trong `red_phase2_rag_sft_candidates.jsonl`.

Sau đó DỪNG nhiệm vụ.
