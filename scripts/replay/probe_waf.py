#!/usr/bin/env python3
"""
Probe WAF with known malicious payloads to verify blocking
"""
import requests
import json
import sys
from typing import Dict, Any

def probe_test(url: str, payload: str, replay_id: str) -> Dict[str, Any]:
    """Send probe and capture response"""
    headers = {"X-Replay-ID": replay_id}
    try:
        resp = requests.get(
            f"{url}/?id={payload}",
            headers=headers,
            timeout=10,
            allow_redirects=False
        )
        return {
            "replay_id": replay_id,
            "payload": payload,
            "status_code": resp.status_code,
            "content_length": len(resp.content),
            "blocked": resp.status_code == 403
        }
    except Exception as e:
        return {
            "replay_id": replay_id,
            "payload": payload,
            "error": str(e)
        }

if __name__ == "__main__":
    BASE_URL = "http://localhost:8080"
    
    # Known bad payloads that MUST be blocked
    probes = [
        ("' or '1'='1", "probe-942100-basic"),
        ("1' UNION SELECT 1,2,3-- -", "probe-942100-union"),
        ("';/*!50000sleep*/(2);-- -", "probe-942100-sleep"),
        ("<script>alert(1)</script>", "probe-941100-xss"),
        ("1' AND SLEEP(2)-- -", "probe-942100-time")
    ]
    
    print("=== Step 2: Probe Tests ===\n")
    results = []
    blocked_count = 0
    
    for payload, rid in probes:
        result = probe_test(BASE_URL, payload, rid)
        results.append(result)
        if result.get('blocked', False):
            blocked_count += 1
        status = result.get('status_code', 'ERROR')
        blocked = '✅ BLOCKED' if result.get('blocked') else '❌ NOT BLOCKED'
        print(f"[{rid}]")
        print(f"  Payload: {payload}")
        print(f"  Status: {status} | {blocked}\n")
    
    print(f"\n=== Summary: {blocked_count}/{len(probes)} probes blocked ===")
    
    if blocked_count == 0:
        print("\n⚠️  WARNING: No probes were blocked! WAF may not be in blocking mode.")
        sys.exit(1)
    elif blocked_count < len(probes):
        print(f"\n⚠️  WARNING: Only {blocked_count}/{len(probes)} blocked. WAF may be misconfigured.")
        sys.exit(1)
    else:
        print("\n✅ All probes blocked - WAF is working correctly")
        sys.exit(0)
