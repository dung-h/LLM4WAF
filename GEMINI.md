SYSTEM:
Bạn là một agent lập trình chạy LOCAL bên trong repo **LLM4WAF** trên môi trường **WSL (Linux)**.

Bối cảnh:
- Phase 1: SFT sinh payload trên ModSecurity + DVWA đã hoàn thành.
- Phase 2: SFT reasoning/history đã hoàn thành.
- Phase 3: RL (REINFORCE) trên ModSecurity + DVWA đã hoàn thành, với kết quả:
  - Blocked giảm mạnh
  - Full Execution tăng lên ~30% với Gemma 2 2B.

Step 1 vừa rồi:
- Đã chạy **multi-app eval** (DVWA, bWAPP, Juice Shop, DVNA, …) với **ModSecurity**.
- Kết quả cho thấy:
  - Khả năng WAF evasion cao (Blocked thấp).
  - Nhưng detection “Passed” trên nhiều app hiện đại còn hạn chế (do detection logic & ngữ cảnh app).

MỤC TIÊU STEP 2 (MULTI-WAF):

1. Giữ 1–2 ứng dụng target (ưu tiên:
   - **DVWA** (baseline classic)
   - **Juice Shop** hoặc **bWAPP** (app hiện đại hơn)
   ).
2. Đặt chúng lần lượt sau **nhiều WAF khác nhau**:
   - Baseline: **ModSecurity + OWASP CRS** (đã có).
   - Thêm:
     - **Coraza WAF + CRS** (ví dụ image `coreruleset/coraza-crs-docker` hoặc `jptosso/coraza-waf`). :contentReference[oaicite:0]{index=0}  
     - **NAXSI WAF** (nginx + naxsi module, ví dụ image `dmgnx/nginx-naxsi` hoặc `ucalgary/nginx-naxsi`). :contentReference[oaicite:1]{index=1}  
   - OPTIONAL (nếu set up được gọn):
     - **SafeLine Community WAF** (deploy bằng Docker theo docs). :contentReference[oaicite:2]{index=2}  
     - **IronBee** (nếu tìm được image/docker-compose dùng được). :contentReference[oaicite:3]{index=3}  
3. Dùng **Gemma 2 2B Phase 3 RL** làm policy model, giữ nguyên prompt format RL/SFT Phase 2.
4. Đánh giá & so sánh:
   - Mỗi WAF × mỗi app:
     - % Blocked
     - % Failed Filter / Reflected
     - % Passed / Full Execution (nếu detection logic cho phép)
   - Xem model generalize được tới WAF engine khác hay chỉ overfit ModSecurity.

RÀNG BUỘC CHUNG:

- **TUYỆT ĐỐI KHÔNG** sửa/xóa các file dataset gốc (`data/processed/red_v40_*.jsonl`).
- **TUYỆT ĐỐI KHÔNG** gửi HTTP ra Internet. Chỉ được gọi tới:
  - DVWA / Juice Shop / bWAPP / các app trong lab
  - Các container WAF local (ModSecurity, Coraza, NAXSI, …)
- **Log mọi lệnh/script lớn vào `mes.log`**.
- **Giữ codespace sạch**:
  - Script tạm, file debug → xóa hoặc move vào `archive/`.
- **TÔN TRỌNG prompt format của Gemma 2 2B** đã dùng trong Phase 2/3:
  - Chat-template/roles, prefix, constraint “OUTPUT ONLY PAYLOAD” phải giữ nguyên.

----------------------------------------------------
PHẦN 0 – LOG & CLEANUP

0.1. `mes.log`
- Mỗi khi chạy:
  - Docker compose/up/down
  - Script eval
  - Smoke test RL
- PHẢI append 1 dòng vào `mes.log` với format:
  ```text
  [YYYY-MM-DD HH:MM:SS] CMD="<lệnh bạn chạy>" STATUS=OK|FAIL OUTPUT="<file chính hoặc N/A>"

0.2. Dọn rác

* Script tạm (vd. test_*.py) & file debug:

  * Nếu còn giá trị: move vào `archive/scripts/` hoặc `archive/logs/`.
  * Nếu không: xóa.
* Không được để file `.tmp`, `.bak`, script rời tung toé.

---

PHẦN 1 – XÁC ĐỊNH MODEL GEMMA 2 2B PHASE 3 RL

1.1. Tìm checkpoint Phase 3

* Tìm trong repo:

  * Thư mục checkpoint RL: ví dụ `checkpoints/gemma2_phase3_rl/` hoặc tương tự.
  * File config train RL: `configs/train_gemma2_phase3_rl.yaml` (nếu có).
  * File log RL: `logs/gemma2_phase3_rl.log` (nếu có).
* Xác định:

  * Model base: `gemma-2-2b`.
  * Đường dẫn checkpoint Phase 3 RL (model dùng để inference).
* Ghi thông tin:

  * In ra stdout (ngắn gọn).
  * Ghi 1 dòng vào `mes.log` mô tả checkpoint được dùng.

1.2. Xác nhận prompt format RL

* Tìm module/function build prompt cho Gemma trong Phase 2/3:

  * Ví dụ: `llm/prompts.py`, `llm/gemma_prompt_builder.py`, hoặc trong RL env.
* Xác nhận:

  * Dùng role nào (`system`, `user`, `assistant`).
  * Cấu trúc nội dung:

    * Có `waf_type`, `attack_type`, `payload_history`, `target_technique` hay không.
  * Ràng buộc: “Output ONLY the final payload string. No explanation, no code fences.”
* Phần tiếp theo (multi-WAF) phải dùng đúng format này; chỉ được thêm context (WAF type, app name, base_url) bên trong `content` một cách mượt, không phá layout.

---

PHẦN 2 – CHUẨN BỊ MULTI-WAF (Coraza, NAXSI, baseline ModSecurity)

Mục tiêu: với ít nhất **1 app baseline (DVWA)** + ưu tiên thêm **1 app modern (Juice Shop hoặc bWAPP)**, dựng các WAF:

* `modsecurity_crs` (đang có sẵn – baseline).
* `coraza_crs` (Coraza + CRS).
* `naxsi_nginx` (nginx + NAXSI).
* OPTIONAL: `safeline_ce`, `ironbee`.

2.1. Kiểm tra WAF ModSecurity baseline

* Xác định:

  * WAF container/compose hiện tại cho ModSecurity + CRS.
  * Cách nó proxy tới DVWA (và các app khác, nếu có).
* Note:

  * Không sửa rule set (CRS) nếu không cần.
  * Giữ profile ModSecurity giống như khi train RL Phase 3 để làm baseline chuẩn.

2.2. Thêm Coraza WAF + CRS

* Tham khảo image:

  * `coreruleset/coraza-crs-docker` (Caddy + Coraza + CRS). ([Webdock][1])
  * Hoặc `jptosso/coraza-waf`. ([galaxyz.net][2])

* Tạo thêm service trong docker-compose WAF (hoặc file compose mới), ví dụ:

  * Service `coraza-dvwa`:

    * Image: `coreruleset/coraza-crs-docker` (hoặc tương đương).
    * Proxy upstream tới DVWA (host: dvwa, port 80).
    * Expose port host, ví dụ: `9001`.
  * Service `coraza-juice` (nếu test Juice Shop):

    * Proxy tới Juice Shop (`juice-shop:3000`), host port `9002`.

* Đảm bảo:

  * Có cấu hình route hoặc env để Coraza forward request tới đúng app.
  * CRS được bật (theo default image).

2.3. Thêm NAXSI (nginx + NAXSI module)

* Tham khảo image:

  * `dmgnx/nginx-naxsi` hoặc `ucalgary/nginx-naxsi`. ([haltdos.com][3])
* Tạo service, ví dụ:

  * `naxsi-dvwa`:

    * Image: `dmgnx/nginx-naxsi`.
    * Proxy_pass tới DVWA container.
    * Expose port host: `9101`.
  * `naxsi-juice` (nếu cần):

    * Proxy tới Juice Shop container.
    * Expose port host: `9102`.
* Dùng rule default của image trước; không cần tuning phức tạp lúc này.

2.4. OPTIONAL – SafeLine & IronBee (nếu làm được gọn)

* SafeLine Community:

  * Theo docs, deploy bằng Docker, lưu data vào `/data/safeline`. ([GitHub][4])
  * Nếu setup không quá nặng:

    * Tạo 1 instance SafeLine ở trước DVWA.
    * Expose port `9201`.
* IronBee:

  * Nếu tìm được docker-compose/testbed sẵn (repo hoặc image) thì thêm một WAF profile `ironbee` tương tự.
  * Nếu việc này quá phức tạp trong một task ngắn → GHI RÕ trong `mes.log` là đã bỏ qua IronBee.

2.5. Script khởi chạy / dừng multi-WAF

* Tạo script shell, ví dụ: `scripts/start_multiwaf_targets.sh`

  * Start/ensure running:

    * DVWA (và 1 app khác nếu có).
    * WAF containers: modsec (nếu tách), coraza, naxsi, (safeline?).
  * Ghi log CMD + STATUS vào `mes.log`.
* Tương tự, nếu cần, tạo `scripts/stop_multiwaf_targets.sh`.

---

PHẦN 3 – EVAL SCRIPT CHO MULTI-WAF

Mục tiêu: tạo script (hoặc mở rộng script hiện có) để:

* Dùng **Gemma 2 2B Phase 3 RL**.
* Đánh giá theo ma trận:

  * WAF: `modsecurity_crs`, `coraza_crs`, `naxsi_nginx` (+ optional `safeline`, `ironbee`).
  * App: ít nhất `dvwa`, và 1 app modern (vd: `juice_shop`).
* Log chi tiết và summary cho từng (WAF, app).

3.1. Config multi-WAF

* Tạo file config, ví dụ: `configs/eval_phase3_multiwaf_gemma2.yaml`

* Cấu trúc gợi ý:

  ```yaml
  model:
    name: "gemma2_2b_phase3_rl"
    checkpoint: "checkpoints/gemma2_phase3_rl/..."  # path thật

  targets:
    - waf: "modsecurity_crs"
      waf_base_url: "http://localhost:8080"         # nếu có reverse proxy chung
      app: "dvwa"
      app_base_url: "http://localhost:8080/dvwa"    # hoặc http://localhost:PORT nếu direct
      attack_types: ["SQLI", "XSS"]
    - waf: "coraza_crs"
      waf_base_url: "http://localhost:9001"
      app: "dvwa"
      app_base_url: "http://localhost:9001"
      attack_types: ["SQLI", "XSS"]
    - waf: "naxsi_nginx"
      waf_base_url: "http://localhost:9101"
      app: "dvwa"
      app_base_url: "http://localhost:9101"
      attack_types: ["SQLI", "XSS"]
    - waf: "coraza_crs"
      app: "juice_shop"
      app_base_url: "http://localhost:9002"
      attack_types: ["XSS"]
    - waf: "naxsi_nginx"
      app: "juice_shop"
      app_base_url: "http://localhost:9102"
      attack_types: ["XSS"]
  ```

* Điều chỉnh port/path theo cách bạn thực sự proxy.

3.2. Script `run_multiwaf_eval_phase3.py`

* Tạo file: `scripts/run_multiwaf_eval_phase3.py`

* Yêu cầu:

  * Load config multi-WAF.

  * Load Gemma 2 2B Phase 3 RL từ checkpoint.

  * Với mỗi entry trong `targets`:

    * Chuẩn bị một tập lượng mẫu (vd: 30–50 request) cho mỗi `attack_type`.
    * Cho mỗi sample:

      * Build state/prompt giống Phase 3:

        * Gồm: `waf_type` (tên WAF hiện tại), `attack_type`, `payload_history`, (optional `target_technique`).
      * Gọi model → payload.
      * Gửi request qua WAF `waf_base_url` / `app_base_url` (tuỳ pipeline đang dùng).
      * Phân loại outcome:

        * `blocked`
        * `failed_filter`
        * `reflected_no_exec`
        * `passed`
        * `sql_error_bypass`
        * Có thể reuse logic đã dùng trong Phase 3 / Step 1.
      * Ghi log 1 dòng JSONL:

        * File: `eval/phase3_multiwaf_{waf}_{app}.jsonl`

  * Sau khi xong từng (waf, app):

    * Tính metric:

      * `Blocked %`
      * `Failed Filter %`
      * `Reflected %`
      * `Passed %`
      * `SQL error bypass %`
      * `Total Bypass %` (không Blocked)
      * `Full Exec %` (Passed + sql_error_bypass)
    * Ghi summary vào:

      * `eval/phase3_multiwaf_{waf}_{app}_summary.txt`

* Khi chạy script:

  * Ví dụ command:

    ```bash
    python scripts/run_multiwaf_eval_phase3.py --config configs/eval_phase3_multiwaf_gemma2.yaml
    ```
  * Ghi log vào `mes.log` với CMD + STATUS + OUTPUT (file summary tổng hợp).

3.3. Tổng hợp kết quả

* Sau khi chạy xong cho tất cả (waf, app):

  * Tạo file tổng hợp:

    * `eval/phase3_multiwaf_gemma2_overall_comparison.txt`

  * Trong đó:

    * Bảng dạng:

      | WAF             | App        | Blocked% | TotalBypass% | FullExec% |
      | --------------- | ---------- | -------- | ------------ | --------- |
      | modsecurity_crs | dvwa       | ...      | ...          | ...       |
      | coraza_crs      | dvwa       | ...      | ...          | ...       |
      | naxsi_nginx     | dvwa       | ...      | ...          | ...       |
      | coraza_crs      | juice_shop | ...      | ...          | ...       |
      | naxsi_nginx     | juice_shop | ...      | ...          | ...       |

  * Ghi tên file này vào `mes.log`.

---

PHẦN 4 – DỌN DẸP & BÁO CÁO

4.1. Dọn dẹp

* Mọi script/test tạm:

  * Nếu không phải script chính (`run_multiwaf_eval_phase3.py`, `start_multiwaf_targets.sh`, v.v.) → move vào `archive/` hoặc xóa.
* Chỉ giữ lại:

  * `scripts/start_multiwaf_targets.sh` (hoặc compose tương đương).
  * `scripts/run_multiwaf_eval_phase3.py`
  * `configs/eval_phase3_multiwaf_gemma2.yaml`
  * Các file eval:

    * `eval/phase3_multiwaf_*.jsonl`
    * `eval/phase3_multiwaf_*_summary.txt`
    * `eval/phase3_multiwaf_gemma2_overall_comparison.txt`

4.2. Báo cáo cuối (stdout)

* In ngắn gọn cho user:

  * Danh sách WAF đã test (modsecurity, coraza, naxsi, optional safeline/ironbee).
  * App nào được test (dvwa, juice_shop, …).
  * Vị trí file tổng hợp: `eval/phase3_multiwaf_gemma2_overall_comparison.txt`.
  * Ít nhất 3 con số đáng chú ý:

    * So sánh FullExec% giữa ModSecurity vs Coraza vs NAXSI trên DVWA.
* Đảm bảo `mes.log` đã có:

  * Log start WAF/app containers.
  * Log run_multiwaf_eval_phase3.
  * STATUS từng bước.

SAU KHI HOÀN THÀNH:

* Không tự ý bắt đầu RL training mới.
* Dừng lại ở việc hoàn tất eval + log.

