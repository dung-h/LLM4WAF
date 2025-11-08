import argparse
import json
import time
import uuid
from pathlib import Path
import pandas as pd

from dvwa_session import DVWASession


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base URL for WAF e.g. http://localhost:8080")
    p.add_argument("--backend_base", default="", help="Direct DVWA base, e.g. http://localhost:18081 (optional)")
    p.add_argument("--input", required=True, help="JSONL payload file with {payload, reasoning}")
    p.add_argument("--out", required=True, help="Output prefix (without extension)")
    args = p.parse_args()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    backend = args.backend_base or None
    sess = DVWASession(args.base, timeout=20.0, backend_base=backend)
    try:
        login_success = sess.login()
        if not login_success:
            print("[FATAL] DVWA login failed. Cannot proceed without authentication.")
            print("Hint: Ensure DVWA is set up and backend is accessible")
            return 2
        
        print("[OK] Login successful")
        
        if not sess.set_security("low"):
            print("[WARN] Set security=low failed; check DVWA security settings")
        else:
            print("[OK] Security level set to low")

        rows = []
        invalid_count = 0
        with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                payload = obj.get("payload", "").strip()
                if not payload:
                    continue
                rid = str(uuid.uuid4())
                headers = {"X-Replay-ID": rid, "User-Agent": "replay/1.0"}
                t0 = time.perf_counter()
                r = sess.get("/vulnerabilities/sqli/", headers=headers, params={"id": payload, "Submit": "Submit"})
                dt = (time.perf_counter() - t0) * 1000.0
                
                # Determine if blocked or invalid response
                blocked = 0
                valid = True
                resp_len = len(r.content)
                resp_text_lower = r.text.lower()
                
                if r.status_code == 403:
                    blocked = 1
                    valid = False  # HTTP 403 is blocked, not valid
                elif r.status_code >= 400:
                    blocked = 1  # Other errors also count as blocked
                    valid = False  # Error responses are not valid
                elif r.status_code in (301, 302):
                    # Redirect - check if it's to login
                    if "login.php" in str(r.url).lower():
                        valid = False
                        invalid_count += 1
                elif resp_len < 500:
                    # Response too short - likely just error page
                    # But check if it has SQL data markers
                    if "first name" not in resp_text_lower and "surname" not in resp_text_lower:
                        blocked = 1  # Likely blocked/error
                elif "login to your account" in resp_text_lower:
                    # Login page detected
                    valid = False
                    invalid_count += 1
                # Response with reasonable length and no login indicators is valid
                
                rows.append({
                    "replay_id": rid,
                    "payload": payload,
                    "status_code": r.status_code,
                    "blocked": blocked,
                    "valid": valid,
                    "resp_len": resp_len,
                    "resp_ms": round(dt, 2),
                    "url": str(r.url),
                })

        df = pd.DataFrame(rows)
        parquet_path = str(out_prefix.with_suffix(".parquet"))
        jsonl_path = str(out_prefix.with_suffix(".jsonl"))
        df.to_parquet(parquet_path, index=False)
        with open(jsonl_path, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        valid_rows = len([r for r in rows if r.get("valid", True)])
        blocked_rows = len([r for r in rows if r.get("blocked", 0) == 1])
        bypass_rows = valid_rows - blocked_rows
        
        print(f"[OK] wrote {len(rows)} rows to {parquet_path}")
        print(f"  Valid: {valid_rows} | Blocked: {blocked_rows} | Bypass: {bypass_rows} | Invalid: {invalid_count}")
        if invalid_count > 0:
            print(f"  [WARN] {invalid_count} responses were invalid (login page/redirect)")
        
        return 0

    finally:
        sess.close()


if __name__ == "__main__":
    raise SystemExit(main())
