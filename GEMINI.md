SYSTEM:
Bạn là một agent lập trình chạy LOCAL trong repo **LLM4WAF** trên môi trường **WSL (Linux)**.

Bối cảnh:

- RED:
  - Gemma 2 2B Phase 3 RL là RED Agent (tấn công).
- BLUE hiện tại:
  - Phase 1: OK – dữ liệu đã chuẩn hoá:
    - `data/blue/blue_phase1_episodes.jsonl` (~423 episodes)
    - `data/blue/blue_phase1_crs_kb.jsonl` (KB CRS)
    - `data/blue/blue_phase1_golden.jsonl` (~39 golden episodes)
  - Phase 2: OK – RAG layer & prompt:
    - `blue/rag_config.yaml`
    - `blue/rag_index.py`
    - `blue/rag_retriever.py`
    - `blue/prompts.py`
    - `blue/runner_phase2_eval.py`
  - Phase 3: **KHUNG có nhưng NÃO GIẢ**:
    - `data/blue/blue_phase3_suggestions.jsonl` có 119 entries,
      nhưng tất cả đều là stub:
        - `recommended_rules: []`
        - `recommended_actions: ["No specific actions recommended by stub LLM."]`
    - `waf/blue_modsecurity_suggestions.conf` hầu như rỗng, không CRS hữu ích.
    - Nguyên nhân: `blue/llm_client.py` chỉ là **stub**, không gọi LLM thật.

MỤC TIÊU TASK NÀY:

1. **Nâng cấp BLUE từ “dumb stub” thành thật sự dùng LLM:**
   - Sửa `blue/llm_client.py`:
     - Không còn trả stub cứng.
     - Thay bằng gọi tới LLM thật (hoặc LLM nội bộ của repo, nếu có).
   - Đảm bảo output là JSON đúng schema và có `recommended_rules` thực sự.

2. **Re-run BLUE Phase 3 (trên tập đã chọn):**
   - Chạy lại `runner_phase3_suggest.py` (hoặc version nâng cấp),
     nhưng với LLM thật.
   - Sinh lại:
     - `data/blue/blue_phase3_suggestions.jsonl` (bản mới, có CRS rule thực sự).
     - `data/blue/blue_phase3_report.txt` (bản mới).
     - `waf/blue_modsecurity_suggestions.conf` & `waf/blue_coraza_suggestions.yaml` (overlay mới có rule thật).

3. (OPTIONAL, nếu còn thời gian) **Re-run Phase 4 nhanh**:
   - Re-eval RED trên profile base vs profile blue_v1 mới.
   - Chỉ cần chạy nhẹ trên DVWA, log lại diff.

RÀNG BUỘC:

- KHÔNG đụng vào file Phase 1–2 gốc:
  - `data/blue/blue_phase1_*`
  - `data/blue/blue_phase2_*`
- CÓ THỂ ghi đè **Phase 3**:
  - `data/blue/blue_phase3_suggestions.jsonl`
  - `data/blue/blue_phase3_report.txt`
  - `waf/blue_modsecurity_suggestions.conf`
  - `waf/blue_coraza_suggestions.yaml`
- KHÔNG gọi HTTP ra Internet. Nếu gọi LLM:
  - Phải dùng client/infra LLM đã có sẵn trong project hoặc môi trường (ví dụ: wrapper có sẵn, local endpoint), **không** tự thêm call API public ra ngoài.
- Mọi script/command quan trọng phải log vào `mes.log`.
- Script tạm/debug phải được xoá hoặc move `archive/`.

----------------------------------------------------
PHẦN 0 – LOG & CLEANUP

0.1. Ghi log `mes.log`

Mỗi lệnh chính (chạy script, regenerate suggestions, generate overlay, re-eval) phải append:

```text
[YYYY-MM-DD HH:MM:SS] CMD="<lệnh đang chạy>" STATUS=OK|FAIL OUTPUT="<file output chính hoặc N/A>"
````

0.2. Dọn rác

* File debug/temp (vd: `blue/tmp_*.py`, notebook tạm):

  * Nếu còn giá trị → move `archive/`.
  * Nếu không → xoá.

---

PHẦN 1 – SỬA `blue/llm_client.py` (THAY STUB BẰNG LLM THẬT)

1.1. Kiểm tra client LLM chung của repo

* Tìm xem trong repo đã có:

  * Một client dùng cho RED hoặc pipeline khác (vd: `llm/client.py`, `core/llm_client.py`, `models/gemma_client.py`, …).
* Mục tiêu: tái sử dụng hạ tầng LLM có sẵn (API key, endpoint).

1.2. Thay stub trong `blue/llm_client.py`

* Mở `blue/llm_client.py`, xác định rõ:

  * Hàm stub hiện tại (vd: `call_blue_llm(prompt: str) -> dict`).
  * Nội dung stub kiểu:

```python
return {
  "vuln_effect": "no-effect",
  "is_false_negative": False,
  "recommended_rules": [],
  "recommended_actions": ["No specific actions recommended by stub LLM."],
  "notes": "This is a stub response. Connect to a real LLM for actual analysis."
}
```

* Thay bằng logic mới:

  * Dùng client LLM có sẵn trong repo **hoặc** nếu không có:

    * Viết TODO rõ ràng & interface sạch cho user nhét backend, nhưng **tạm thời** bạn vẫn có thể simulate một LLM “thông minh hơn stub” bằng cách:

      * RAG → dùng `rag_retriever` để lấy CRS entry (vd rule 956100),
      * Dùng heuristics đơn giản để tạo `recommended_rules`/`recommended_actions`.
    * Tuy nhiên, nếu môi trường có LLM thật, ưu tiên dùng LLM thật.

* API mới của `call_blue_llm`:

```python
def call_blue_llm(prompt: str) -> dict:
    """
    Nhận prompt (BLUE_PROMPT_TEMPLATE đã fill),
    gọi LLM backend (hoặc local heuristic fallback),
    parse JSON, trả về dict.

    YÊU CẦU:
      - Nếu LLM trả về JSON không hợp lệ:
          + Log lỗi (mes.log hoặc file debug).
          + Có thể retry 1 lần với prompt "JSON ONLY".
      - Nếu vẫn fail:
          + Trả về dict dạng:
              {
                "vuln_effect": "unknown",
                "is_false_negative": False,
                "recommended_rules": [],
                "recommended_actions": ["LLM_ERROR_OR_INVALID_JSON"],
                "notes": "..."
              }
    """
```

1.3. Unit test nhỏ

* Tạo script/test nhỏ (có thể ngay trong `llm_client.py` hoặc file riêng) để:

  * Build 1 prompt đơn giản với fake EPISODE + KB_SNIPPETS.
  * Gọi `call_blue_llm` và in ra kết quả.
* Tối thiểu:

  * Kết quả **không còn** là stub cũ.
  * Có khả năng xuất hiện `recommended_rules` ≠ `[]` trong một số case.

---

PHẦN 2 – TEST LẠI TRÊN GOLDEN SET (PHASE 2 HARNESS)

2.1. Chạy lại `blue/runner_phase2_eval.py` (hoặc chỉnh nhẹ)

* Dùng **golden set**:

  * `data/blue/blue_phase1_golden.jsonl`

* Mục tiêu:

  * Gọi BLUE LLM mới cho từng golden episode.
  * Ghi output vào:

    * `data/blue/blue_phase2_eval_raw.jsonl` (có thể ghi đè bản cũ hoặc tạo file `_v2`).

* Nếu golden có field `expected_crs_rules`:

  * Tính hit rate:

    * recommended_rules của BLUE có chứa rule_id expected hay không.
  * Ghi Summary:

    * `data/blue/blue_phase2_eval_summary.txt` (hoặc `_v2`).

* Log command & output vào `mes.log`.

2.2. Kiểm tra nhanh kết quả

* Lấy ngẫu nhiên 3–5 record trong `blue_phase2_eval_raw.jsonl`:

  * Xem:

    * `recommended_rules` có ID thật không (vd: `"956100"`, `"942100"`, …).
    * `recommended_actions` có nội dung liên quan tới rule/vuln không (không phải stub).

---

PHẦN 3 – RE-RUN BLUE PHASE 3 (SUGGESTIONS + OVERLAY) VỚI LLM MỚI

3.1. Chạy lại `blue/runner_phase3_suggest.py`

* Dùng BLUE LLM mới:

  * Lúc này `call_blue_llm` không còn là stub.
* Input:

  * `data/blue/blue_phase1_episodes.jsonl`
  * `data/blue/blue_phase1_crs_kb.jsonl`
* Output (ghi ĐÈ hoặc tạo version mới – tuỳ bạn nhưng nên GHI ĐÈ cho đơn giản, nhớ log trong `mes.log`):

  * `data/blue/blue_phase3_suggestions.jsonl` – 119 entries với real output.
* Kiểm tra nhanh:

  * 1–2 record phải có:

    * `blue_output.recommended_rules` không rỗng.
    * `blue_output.recommended_actions` mang tính phòng thủ thực sự.

3.2. Chạy lại `blue/phase3_aggregate_report.py`

* Sinh lại:

  * `data/blue/blue_phase3_report.txt`
* Xem có thống kê:

  * Top `rule_id` được recommend.
  * Theo `vuln_effect`.

3.3. Chạy lại `blue/phase3_generate_waf_overlays.py`

* Sinh lại overlay:

  * `waf/blue_modsecurity_suggestions.conf`
  * `waf/blue_coraza_suggestions.yaml`

* Đảm bảo:

  * File này **không rỗng**.
  * Có ít nhất vài `SecRule`/rule block với `id:` riêng (tránh đụng CRS id range nếu có).

* Log từng bước vào `mes.log`.

---

PHẦN 4 – (OPTIONAL) MINI RE-EVAL RED–BLUE (PHASE 4 LITE)

Nếu thời gian cho phép, chạy một vòng Phase 4 nhẹ để chủ nhân thấy BLUE mới khác gì:

4.1. Dùng lại profile:

* `waf/modsecurity_profile_base.conf`
* `waf/modsecurity_profile_blue_v1.conf` (giờ đã include overlay mới)

4.2. Run RED eval nhẹ

* Chạy `scripts/run_red_eval_profile.py` (hoặc script tương đương đã dựng từ Phase 4) cho:

  * profile `modsec_base` (DVWA, SQLI+XSS, ~30–50 request)
  * profile `modsec_blue_v1` (DVWA, cùng số request)
* Output:

  * `eval/red_phase4_modsec_base_*`
  * `eval/red_phase4_modsec_blue_v1_*`

4.3. (Optional) FP check nhẹ

* Chạy `scripts/run_fp_check_profile.py`:

  * profile base vs blue_v1
  * vài chục request benign.

4.4. Joint mini report

* Nếu đã có script `redblue_phase4_joint_report.py`, chạy lại.
* Nếu chưa, ít nhất:

  * In nhanh summary ra stdout:

    * Blocked% trước/sau
    * Full Exec% trước/sau (nếu bạn có logic detect)
    * FP rate trước/sau (nếu có fp_check)

---

PHẦN 5 – CLEANUP & BÁO CÁO

5.1. Dọn dẹp

* Xoá/move mọi script/debug tạm.
* Giữ lại:

  * BLUE:

    * `blue/llm_client.py` (đã dùng LLM thật hoặc heuristic thông minh, KHÔNG còn stub vô nghĩa).
    * `data/blue/blue_phase3_suggestions.jsonl` (bản mới).
    * `data/blue/blue_phase3_report.txt` (bản mới).
    * `waf/blue_modsecurity_suggestions.conf` (overlay mới, có rule).
    * `waf/blue_coraza_suggestions.yaml`.

  * RED–BLUE:

    * Bất kỳ file eval/reports mới bạn tạo trong Phase 4 lite.

5.2. Báo cáo (stdout cho user)

* In tóm tắt:

  * `blue/llm_client.py` hiện trạng: stub đã được thay bằng gì? (LLM thật / heuristic local).
  * Số suggestions trong `blue_phase3_suggestions.jsonl` và ví dụ:

    * 1 entry có recommended_rules ≠ [].
  * Tóm tắt Quick diff (nếu chạy lại Phase 4 nhẹ):

    * Blocked% / FullExec% trước vs sau BLUE overlay mới.

Sau bước này:

* BLUE Agent không còn là khung rỗng nữa mà thực sự sinh recommendation dựa trên LLM/RAG.
* Có thể dùng vào các vòng đánh giá tiếp theo.
