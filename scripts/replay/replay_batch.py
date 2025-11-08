#!/usr/bin/env python3
import argparse, json, time, uuid, urllib.parse, pathlib, re
from typing import List, Dict, Any
import httpx, pandas as pd

def load_jsonl(p: str) -> List[Dict[str, Any]]:
    rows=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                j=json.loads(line)
                # tolerate schema differences
                payload = j.get("payload") or j.get("text") or j.get("query") or ""
                rows.append({"payload": str(payload), "meta": j.get("meta", {})})
            except Exception:
                # allow raw payload per line
                rows.append({"payload": line, "meta": {}})
    return rows

def build_url(base: str, path_tmpl: str, payload: str) -> str:
    # replace {payload} with URL-encoded payload
    return base.rstrip("/") + "/" + path_tmpl.lstrip("/").replace("{payload}", urllib.parse.quote_plus(payload))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file with {payload}")
    ap.add_argument("--out", required=True, help="output prefix (without extension) OR directory; parquet will be written")
    ap.add_argument("--base", default=None, help="Base WAF URL (override env WAF_URL)")
    ap.add_argument("--path", default=None, help="Target path template (override env TARGET_PATH)")
    ap.add_argument("--method", default=None, help="HTTP method (default GET; override env METHOD)")
    ap.add_argument("--verify", default="false", choices=["true","false"], help="TLS verify for HTTPS")
    args=ap.parse_args()

    import os
    base = args.base or os.environ.get("WAF_URL") or "http://localhost:8080"
    path_tmpl = args.path or os.environ.get("TARGET_PATH") or "/vulnerabilities/sqli/?id={payload}&Submit=Submit"
    method = (args.method or os.environ.get("METHOD") or "GET").upper()
    verify = (args.verify.lower()=="true")

    rows = load_jsonl(args.input)
    out_prefix = args.out
    out_path = out_prefix if out_prefix.endswith(".parquet") else (out_prefix.rstrip("/") + ".parquet")

    client = httpx.Client(follow_redirects=True, timeout=15.0, verify=verify)
    out=[]
    for i, r in enumerate(rows, 1):
        payload = r["payload"]
        url = build_url(base, path_tmpl, payload)
        rid = str(uuid.uuid4())
        headers={"X-Replay-ID": rid, "User-Agent": "waf-llm-lab/0.1"}
        t0=time.perf_counter()
        try:
            resp = client.request(method, url, headers=headers)
            ms = int(1000*(time.perf_counter()-t0))
            status = resp.status_code
            blocked = 1 if status==403 else 0
            out.append({
                "replay_id": rid,
                "payload": payload,
                "url": url,
                "method": method,
                "status_code": status,
                "blocked": blocked,
                "resp_ms": ms,
                "when_ts": pd.Timestamp.utcnow(),
                "meta": r.get("meta", {})
            })
        except Exception as e:
            ms = int(1000*(time.perf_counter()-t0))
            out.append({
                "replay_id": rid,
                "payload": payload,
                "url": url,
                "method": method,
                "status_code": 0,
                "blocked": 0,
                "resp_ms": ms,
                "error": str(e),
                "when_ts": pd.Timestamp.utcnow(),
                "meta": r.get("meta", {})
            })

    df = pd.DataFrame(out)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # write parquet
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        # fallback to fastparquet if needed
        df.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Wrote {len(df)} rows to {out_path}")

if __name__=="__main__":
    main()
