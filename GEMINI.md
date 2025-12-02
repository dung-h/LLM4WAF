SYSTEM:
Bạn là một agent lập trình chạy LOCAL trong repo **LLM4WAF** trên môi trường **WSL (Linux)**.

Bối cảnh BLUE hiện tại:

- Phase 1 BLUE (data chuẩn hoá + KB) đã hoàn thành:
  - Schema: `docs/blue_phase1_schema.md`
  - Episodes: `data/blue/blue_phase1_episodes.jsonl` (~423)
  - CRS KB: `data/blue/blue_phase1_crs_kb.jsonl` (2 entry mẫu – nhỏ nhưng đủ để test flow)
  - Golden set: `data/blue/blue_phase1_golden.jsonl` (~39)

- Phase 2 BLUE (RAG + prompt + eval harness) đã hoàn thành:
  - RAG:
    - `blue/rag_config.yaml`
    - `blue/rag_index.py`
    - `blue/rag_retriever.py`
  - BLUE LLM:
    - `blue/prompts.py` (BLUE_PROMPT_TEMPLATE)
    - `blue/llm_client.py` (hiện đang là stub)
  - Evaluation harness:
    - `blue/runner_phase2_eval.py`
    - `data/blue/blue_phase2_eval_raw.jsonl`
    - `data/blue/blue_phase2_eval_summary.txt`

MỤC TIÊU PHASE 3 BLUE:

1. Biến BLUE từ “demo RAG” thành **tuning assistant thực sự**:
   - Chạy BLUE (RAG + LLM) trên:
     - Tập các episode quan trọng, đặc biệt là `is_false_negative == true`.
   - Sinh ra **khuyến nghị chi tiết**:
     - `vuln_effect`, `is_false_negative`
     - `recommended_rules` (rule_id + engine)
     - `recommended_actions` (câu chữ + config gợi ý)
2. Tổng hợp khuyến nghị:
   - Per-engine (modsecurity, coraza, naxsi)
   - Per-rule (rule nào được đề xuất nhiều nhất)
   - Per-vuln_effect (SQL error, Ruby SyntaxError, XSS, …)
3. Tạo **output cho engineer**:
   - JSONL chi tiết: `data/blue/blue_phase3_suggestions.jsonl`
   - Báo cáo human-readable: `data/blue/blue_phase3_report.txt`
   - (Optional) file **overlay config** (ModSecurity/Coraza/NAXSI) dạng skeleton:
     - `waf/blue_modsecurity_suggestions.conf`
     - `waf/blue_coraza_suggestions.conf`
     - `waf/blue_naxsi_suggestions.rules`
4. Không tự động áp dụng rule vào WAF production/lab; chỉ **chuẩn bị** output. (Việc re-test với RED là bước sau).

RÀNG BUỘC:

- KHÔNG sửa/xoá:
  - `data/blue/blue_phase1_*.jsonl`
  - `data/blue/blue_phase2_eval_*.jsonl/txt`
- KHÔNG gọi HTTP ra internet (không `curl`/`wget`/API cloud).  
  - Nếu `blue/llm_client.py` sử dụng LLM local (vd. vLLM, server nội bộ) thì được, nhưng không tự tạo code gọi API public bên ngoài.
- Mọi script/command lớn phải log vào `mes.log`.
- File tạm/debug → xoá hoặc move vào `archive/`.

----------------------------------------------------
PHẦN 0 – LOG & CLEANUP

0.1. `mes.log`
- Bất cứ khi nào bạn chạy:
  - Script Python/Bash chính,
  - Xây index,
  - Chạy batch BLUE RAG,
- PHẢI append một dòng vào `mes.log`:

```text
[YYYY-MM-DD HH:MM:SS] CMD="<lệnh đã chạy>" STATUS=OK|FAIL OUTPUT="<file output chính hoặc N/A>"
````

0.2. File tạm

* Script tạm (vd: `blue/tmp_debug_phase3.py`), notebook test:

  * Nếu cần giữ → move `archive/scripts/` hoặc `archive/notebooks/`.
  * Nếu không → xoá.

---

PHẦN 1 – HOÀN THIỆN `blue/llm_client.py`

1.1. Kiểm tra client LLM đã có trong repo

* Tìm xem repo có module LLM chung nào:

  * Ví dụ: `llm/client.py`, `core/llm_client.py`, hoặc config sử dụng model RED (Gemma 2 2B/ Phi-3 mini).
* Nếu có:

  * Tái sử dụng client đó trong `blue/llm_client.py`:

    * Import và wrap lại thành hàm:

```python
def call_blue_llm(prompt: str) -> dict:
    """
    Gọi LLM (hoặc parse từ stub nếu chưa nối backend).
    Trả về dict JSON đã parse theo format BLUE_PROMPT_TEMPLATE yêu cầu.
    """
```

* Nếu LLM trả về text, parse JSON:

  * Xử lý lỗi parse, log gọn: ghi vào `mes.log` hoặc file debug nhỏ trong `archive/logs/`.

* Nếu repo CHƯA có client LLM khả dụng:

  * Giữ `call_blue_llm` là stub nhưng:

    * Thêm docstring rõ ràng mô tả chỗ này sẽ cần user điền logic gọi LLM thực tế.
    * Có thể tạm đọc từ file mẫu (mock) để cho pipeline Phase 3 chạy khô (dry-run).

**Lưu ý:** Phase 3 tập trung thiết kế pipeline + aggregation; việc kết nối thật tới LLM có thể do user thêm sau, miễn `call_blue_llm()` trả về dict đúng schema.

---

PHẦN 2 – BLUE RUNNER PHASE 3 (BATCH RECOMMENDATION)

2.1. Tạo script: `blue/runner_phase3_suggest.py`

Chức năng:

1. Load:

   * `data/blue/blue_phase1_episodes.jsonl`
   * `data/blue/blue_phase1_crs_kb.jsonl` (qua `CRSKnowledgeBase` trong `rag_index.py`)

2. Chọn tập episode mục tiêu:

   * Ưu tiên:

     * `blue_label.is_false_negative == true`
   * Nếu số lượng quá lớn:

     * Giới hạn (vd 200–300) hoặc chỉ lấy:

       * severity == "high" hoặc "medium"
   * Log thống kê:

     * Tổng episodes
     * Số FN
     * Số được chọn cho Phase 3.

3. Với mỗi episode được chọn:

   * Dùng `retrieve_for_episode(episode, kb, top_k)`:

     * `top_k` gợi ý: 3–5.

   * Build prompt từ `BLUE_PROMPT_TEMPLATE` trong `blue/prompts.py`:

     * `{EPISODE_JSON}`:

       * Rút gọn: chỉ giữ:

         * `waf_env`
         * `app_context`
         * `attack`
         * `app_observation`
         * `blue_label.vuln_effect` & `is_false_negative`
     * `{KB_SNIPPETS}`:

       * 2–5 entry từ KB, mỗi entry chỉ giữ:

         * `rule_id`, `test_description`, `attack_type`, `variables`, `operator`, `example_payload`, `example_attack_context`.

   * Gọi `call_blue_llm(prompt)`:

     * Nếu LLM thực chưa được nối:

       * Có thể trả mock hoặc skip với flag; nhưng vẫn ghi record rõ ràng.

   * Nhận output dict, expected schema:

```jsonc
{
  "vuln_effect": "...",
  "is_false_negative": true,
  "recommended_rules": [
    {
      "engine": "modsecurity",
      "rule_id": "956100",
      "reason": "..."
    }
  ],
  "recommended_actions": [
    "Enable rule 956100 ...",
    "Disable verbose Ruby errors ..."
  ],
  "notes": "..."
}
```

* Gộp thành 1 record:

```jsonc
{
  "episode_id": "<có thể là hash index hoặc offset>",
  "episode": { ...blue_episode rút gọn... },
  "kb_hits": [ ...KB_SNIPPETS... ],
  "blue_output": { ...LLM output như trên... }
}
```

* Ghi vào:

  * `data/blue/blue_phase3_suggestions.jsonl`

2.2. Logging

* Sau khi chạy xong:

  * Ghi CMD/STATUS/OUTPUT vào `mes.log`.
  * In ra số episode xử lý, số records trong `blue_phase3_suggestions.jsonl`.

---

PHẦN 3 – AGGREGATOR & REPORT

3.1. Tạo script: `blue/phase3_aggregate_report.py`

Chức năng:

1. Đọc `data/blue/blue_phase3_suggestions.jsonl`.

2. Tính thống kê:

   * Tổng số episode được BLUE phân tích.
   * Với mỗi engine (modsecurity, coraza, naxsi,…):

     * List các rule được recommend:

       * rule_id
       * số lần xuất hiện (đếm across episodes)
   * Với mỗi `vuln_effect`:

     * Bao nhiêu case được recommend rule?
     * Top rule cho loại đó.

3. Sinh báo cáo text:

   * `data/blue/blue_phase3_report.txt`
   * Gợi ý nội dung:

BLUE Phase 3 – WAF Tuning Summary
=================================

Total analyzed episodes: N
Total with is_false_negative=true: M

Per-engine recommended rules:
- modsecurity:
    - 956100 (Ruby SyntaxError): 7 episodes
    - 942100 (SQLi detection): 4 episodes
- coraza:
    - ...

Per-vuln_effect:
- error-disclosure:
    - main rules: 956100 (Ruby), 950120 (PHP error), ...
- sql-error:
    - main rules: 942100, ...
...

Notes:
- Many Ruby error disclosure episodes map to CRS rule 956100 but environment likely has it disabled or PL < 2.
- Suggested actions: enable RESPONSE_BODY inspection and raise PL for /admin/* endpoints.
```

3.2. Logging

* Log lệnh chạy & output file vào `mes.log`.

---

PHẦN 4 – OUTPUT CONFIG SKELETON (OPTIONAL NHƯNG NÊN LÀM)

4.1. Tạo script: `blue/phase3_generate_waf_overlays.py`

Chức năng:

1. Đọc `data/blue/blue_phase3_suggestions.jsonl`.

2. Lọc theo engine:

   * `engine == "modsecurity"`:

     * Từ `recommended_rules` + `recommended_actions`, generate các đoạn `SecRule`/`SecAction` SKELETON.
   * `engine == "coraza"`:

     * Sinh rule YAML tương đương (nếu action có hint).
   * `engine == "naxsi"`:

     * Sinh couple `MainRule`/`BasicRule` gợi ý.

3. Output:

* `waf/blue_modsecurity_suggestions.conf`

  * chứa comment + skeleton:

```apache
# BLUE Phase 3 suggestion: Ruby SyntaxError disclosure
# Episodes: 7, vuln_effect=error-disclosure
SecRule RESPONSE_BODY "@pmFromFile ruby-errors.data" \
    "id:9956100,phase:4,block,log,msg:'Ruby SyntaxError detected (BLUE suggestion)',severity:ERROR"

# ... các rule khác ...
```

* `waf/blue_coraza_suggestions.yaml`
* `waf/blue_naxsi_suggestions.rules`

Lưu ý:

* Không cần “đúng chuẩn” tuyệt đối; đây là file gợi ý để engineer đọc và chỉnh tay.
* Nên comment rõ mỗi block:

  * vuln_effect
  * số episode liên quan
  * rule CRS gốc tham chiếu (nếu có).

4.2. Logging

* Ghi CMD/STATUS/OUTPUT vào `mes.log`.

---

PHẦN 5 – CLEANUP & BÁO CÁO CUỐI

5.1. Dọn dẹp

* Xoá/move tất cả script debug tạm thời vào `archive/`.
* Giữ lại các file chính:

  * Code:

    * `blue/llm_client.py`
    * `blue/runner_phase3_suggest.py`
    * `blue/phase3_aggregate_report.py`
    * `blue/phase3_generate_waf_overlays.py`
    * (các file Phase 1–2 đã có)

  * Data:

    * `data/blue/blue_phase1_*.jsonl`
    * `data/blue/blue_phase2_eval_*.jsonl/txt`
    * `data/blue/blue_phase3_suggestions.jsonl`
    * `data/blue/blue_phase3_report.txt`

  * WAF overlays:

    * `waf/blue_modsecurity_suggestions.conf` (nếu được tạo)
    * `waf/blue_coraza_suggestions.yaml` (nếu được tạo)
    * `waf/blue_naxsi_suggestions.rules` (nếu được tạo)

5.2. Báo cáo (stdout)

* In cho user:

  * Đường dẫn & kích thước:

    * `data/blue/blue_phase3_suggestions.jsonl`
    * `data/blue/blue_phase3_report.txt`
    * các file `waf/blue_*_suggestions.*`
  * Thống kê:

    * Số episode đã được BLUE xử lý.
    * Top 3 rule_id được đề xuất nhiều nhất cho modsecurity.
  * 1–2 ví dụ rút gọn:

    * Một entry trong `blue_phase3_suggestions.jsonl`.
    * Một đoạn trong `blue_modsecurity_suggestions.conf`.

KẾT THÚC PHASE 3 BLUE:

* KHÔNG tự ý apply config vào WAF hoặc chạy lại RED trong task này.
* Dừng lại sau khi đã sinh đầy đủ suggestion + báo cáo.

