from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import re
from urllib.parse import urlsplit

import httpx
import pandas as pd


RESULTS = Path("results")


async def probe(client: httpx.AsyncClient, url: str, params: Dict[str, str]) -> Dict[str, Any]:
    r = await client.get(url, params=params, timeout=10.0)
    return {
        "status": r.status_code,
        "response_size": len(r.content),
        "blocked": r.status_code in (403, 406),
    }


async def main_async(base_url: str, payloads: List[str], param: str = "name", cookies: httpx.Cookies | None = None) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(follow_redirects=True, cookies=cookies) as client:
        for i, p in enumerate(payloads):
            res = await probe(client, base_url, {param: p})
            out.append({"id": i, "payload": p, **res})
    return pd.DataFrame(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WAF Replay Harness (XSS)")
    parser.add_argument("payloads_file", help="File containing XSS payloads (one per line)")
    parser.add_argument("--output", default="results/replay_results_xss.jsonl", help="Output file")
    parser.add_argument("--base-url", default="http://localhost:8080/vulnerabilities/xss_r/", help="Target URL")
    parser.add_argument("--param-name", default="name", help="Parameter name for injection")
    parser.add_argument("--cookie-file", default=None, help="Cookie file (Netscape format)")
    parser.add_argument("--login", action="store_true", help="Login DVWA (admin/password)")
    parser.add_argument("--login-url", default=None, help="Login URL (defaults to <root>/login.php)")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="password")
    args = parser.parse_args()

    with open(args.payloads_file, "r", encoding="utf-8") as f:
        payloads = [line.strip() for line in f if line.strip()]

    print(f"Testing {len(payloads)} XSS payloads against {args.base_url}")
    print(f"Parameter: {args.param_name}")
    print(f"Output: {args.output}")
    print()

    # Prepare cookies
    cookies = httpx.Cookies()
    if args.cookie_file:
        try:
            with open(args.cookie_file, "r", encoding="utf-8", errors="ignore") as cf:
                for line in cf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 7:
                        name = parts[-2]
                        value = parts[-1]
                        cookies.set(name, value)
        except Exception:
            pass
    if args.login:
        parts = urlsplit(args.base_url)
        login_url = args.login_url or f"{parts.scheme}://{parts.netloc}/login.php"
        with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as client:
            r = client.get(login_url)
            m = re.search(r'name=["\']user_token["\']\s+value=["\']([^"\']+)["\']', r.text, re.I)
            token = m.group(1) if m else ""
            data = {"username": args.username, "password": args.password, "user_token": token, "Login": "Login"}
            client.post(login_url, data=data)
            cookies = client.cookies

    df = asyncio.run(main_async(args.base_url, payloads, param=args.param_name, cookies=cookies))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    total = len(df)
    passed = len(df[~df["blocked"]])
    blocked = len(df[df["blocked"]])
    bypass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n{'='*70}")
    print("WAF REPLAY RESULTS (XSS)")
    print(f"{'='*70}")
    print(f"Total payloads:  {total}")
    print(f"Passed WAF:   {passed} ({bypass_rate:.1f}%)")
    print(f"Blocked:      {blocked} ({100-bypass_rate:.1f}%)")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    print(f"   - {out_path}")
    print(f"   - {csv_path}")


if __name__ == "__main__":
    main()
