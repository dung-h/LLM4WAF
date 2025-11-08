#!/usr/bin/env python3
import sys, json, re

def valid_payload(s: str) -> bool:
    """Check if payload is valid SQLi (not placeholder)"""
    if not s:
        return False
    s_lower = s.lower()
    # Reject placeholders
    if "<payload>" in s_lower or "<insert" in s_lower:
        return False
    # Reject too long
    if len(s.strip()) > 160:
        return False
    # Must have SQL keywords
    sql_keywords = r"(select|union|sleep|benchmark|waitfor|or\s*1\s*=\s*1|and\s*1\s*=\s*1|xor|between|%27|%EF%BC%87|--|/\*\*/|/*!)"
    return bool(re.search(sql_keywords, s, re.I))

n_ok = 0
n_skip = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        p = (obj or {}).get("payload", "")
    except Exception:
        n_skip += 1
        continue
    
    if valid_payload(p):
        print(json.dumps({"payload": p, "meta": {"source": "llm_clean"}}))
        n_ok += 1
    else:
        n_skip += 1

sys.stderr.write(f"[postclean] kept={n_ok}, skipped={n_skip}\n")
