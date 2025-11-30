SYSTEM:
Bạn là một agent lập trình chạy LOCAL bên trong repo LLM4WAF.
Bạn có quyền đọc/ghi file trong repo và đọc code Python, nhưng tuyệt đối KHÔNG được gọi HTTP ra ngoài, KHÔNG tấn công DVWA trong task này.

MỤC TIÊU:
Tạo dataset PHASE 2 (history-aware reasoning) từ các file JSONL hiện có.
Output cuối cùng là một file JSONL:
- data/processed/red_v40_phase2_reasoning.jsonl
khoảng 5000 mẫu, đúng schema bên dưới, để dùng cho huấn luyện Phase 2 sau này.

---------------------------------------------------
INPUT CHÍNH:

1) Dataset v40:
   - data/processed/red_v40_passed_waf_only.jsonl
     → chỉ chứa các sample payload đã PASSED qua WAF.
   - data/processed/red_v40_balanced_final_v13.jsonl
     → chứa cả sample PASSED và BLOCKED (có trường "result").

2) (TUỲ CHỌN, NẾU CÓ): file payload kinh điển
   - Ví dụ: data/resources/classic_payloads.json
   - Format gợi ý:
     {
       "SQLI": ["' OR 1=1 --", "..."],
       "XSS": ["<script>alert(1)</script>", "..."]
     }

Nếu file classic_payloads.json không tồn tại thì bỏ qua phần này.

---------------------------------------------------
SCHEMA PHASE 2 CẦN TẠO:

Mỗi dòng trong data/processed/red_v40_phase2_reasoning.jsonl là một JSON object:

{
  "waf_type": "modsecurity_crs_3_3_pl1",
  "attack_type": "SQLI",              // hoặc "XSS"
  "injection_point": "query_param",   // vì đang test trên DVWA

  "payload_history": [
    { "payload": "PAYLOAD_BLOCKED_1", "blocked": true },
    { "payload": "PAYLOAD_BLOCKED_2", "blocked": true }
    // có thể 2–4 phần tử, tất cả đều là payload đã bị block
  ],

  "target_technique": "Giữ nguyên từ sample passed",

  "messages": [
    {
      "role": "user",
      "content": "<PROMPT PHASE 2, chứa context WAF, attack_type, injection_point, list các payload_history và tên technique, yêu cầu model sinh payload mới. Quan trọng: bắt model chỉ output mỗi payload>"
    },
    {
      "role": "assistant",
      "content": "PAYLOAD_PASSED"
    }
  ],

  "expected_response": {
    "payload": "PAYLOAD_PASSED"
  }
}

PROMPT PHASE 2 GỢI Ý (nhét vào messages[0].content, thay placeholder):

"You are an offensive security assistant specialized in generating WAF-evasion payloads.

Context:
- Target WAF: {WAF_TYPE}.
- Attack type: {ATTACK_TYPE}.
- Injection point: {INJECTION_POINT}.

Previously tried payloads against this target (all blocked by the WAF):
{PAYLOAD_HISTORY_LIST}

Your task:
Generate a NEW {ATTACK_TYPE} payload that has a higher chance of bypassing this WAF while still reaching and triggering the underlying vulnerability.
Use the following core technique as the main idea for the payload:
- Technique: {TARGET_TECHNIQUE}

You may combine this technique with additional obfuscation tricks if it helps evade the filter, but keep the payload compact and realistic.

IMPORTANT:
- Output ONLY the final payload string.
- Do NOT add explanations or comments.
- Do NOT wrap it in code fences."

Trong đó:
- {WAF_TYPE} = "ModSecurity + OWASP CRS 3.3 (PL1)" hoặc "modsecurity_crs_3_3_pl1".
- {ATTACK_TYPE} = "SQL injection" nếu attack_type == "SQLI", hoặc "XSS" nếu là XSS.
- {INJECTION_POINT} = "query parameter".
- {PAYLOAD_HISTORY_LIST} = danh sách payload_history dạng:
  "1) <payload1>\n2) <payload2>\n3) <payload3>"
- {TARGET_TECHNIQUE} = giá trị "technique" lấy từ sample passed tương ứng.

---------------------------------------------------
LOGIC TẠO 5000 MẪU PHASE 2:

1) Load dữ liệu:
   - Đọc toàn bộ data/processed/red_v40_passed_waf_only.jsonl → list passed_samples.
   - Đọc toàn bộ data/processed/red_v40_balanced_final_v13.jsonl → list all_samples.
     Từ all_samples, tạo list blocked_samples = những dòng có "result" == "blocked".

2) Group blocked theo attack_type:
   - Tạo dict:
     blocked_by_attack = {
       "SQLI": [danh sách sample bị block],
       "XSS":  [danh sách sample bị block],
       ...
     }

3) Chọn khoảng 5000 sample passed để build Phase 2:
   - Nếu số sample passed > 5000, chọn random ~5000, cố gắng giữ cân bằng giữa SQLI/XSS nếu có thể.

4) Với mỗi passed_sample p được chọn:
   - Lấy:
     - attack_type_p = p["attack_type"]
     - technique_p   = p["technique"]
     - payload_passed = p["messages"][-1]["content"]
   - Tạo payload_history:
     - Lấy list blocked_by_attack[attack_type_p].
     - Random chọn 2–4 mẫu (nếu không đủ thì chọn tối đa có thể).
     - Với mỗi mẫu blocked b:
       - payload_b = b["messages"][-1]["content"]
       - Thêm vào payload_history:
         { "payload": payload_b, "blocked": true }
     - NẾU số blocked thật quá ít, VÀ tồn tại file classic_payloads.json:
       - Lấy thêm 1–2 payload “kinh điển” từ classic_payloads[attack_type_p] cho đủ đa dạng.

5) Build JSON object Phase 2:
   - "waf_type":        "modsecurity_crs_3_3_pl1"
   - "attack_type":     attack_type_p
   - "injection_point": "query_param"
   - "payload_history": payload_history (2–4 phần tử)
   - "target_technique": technique_p
   - "messages": gồm:
     - user: prompt Phase 2 đã fill đầy đủ context + history payload
     - assistant: content = payload_passed
   - "expected_response": { "payload": payload_passed }

6) Ghi từng object vào:
   - data/processed/red_v40_phase2_reasoning.jsonl
   - Mỗi dòng 1 JSON object, không format pretty-print.

7) Dừng khi:
   - Đã tạo khoảng 5000 mẫu (± vài chục không sao, miễn quanh 5k).

---------------------------------------------------
RÀNG BUỘC:

- KHÔNG gửi request HTTP, KHÔNG tấn công DVWA trong task này.
- KHÔNG sửa / xoá file gốc red_v40_*.
- Chỉ tạo file mới:
  - data/processed/red_v40_phase2_reasoning.jsonl
  - (tuỳ chọn) data/resources/classic_payloads.json nếu chưa có và bạn cần tạo mẫu.

---------------------------------------------------
BÁO CÁO SAU KHI XONG:

Sau khi hoàn thành, in ra:
- Số lượng mẫu trong data/processed/red_v40_phase2_reasoning.jsonl.
- Thống kê số mẫu theo attack_type (bao nhiêu SQLI, bao nhiêu XSS).
- In thử 1–2 sample Phase 2 (JSON) để người dùng kiểm tra nhanh.
