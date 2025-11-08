"""
Replay harness (skeleton): sends payloads to the testbed app and records results.

Intended to target: Nginx + ModSecurity (CRS PL2) + DVWA/demo app.
Outputs parquet/csv under results/.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

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


async def main_async(base_url: str, payloads: List[str], param: str = "q") -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i, p in enumerate(payloads):
            res = await probe(client, base_url, {param: p})
            out.append({"id": i, "payload": p, **res})
    return pd.DataFrame(out)


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="WAF Replay Harness")
    parser.add_argument("payloads_file", help="File containing payloads (one per line)")
    parser.add_argument("--output", default="results/replay_results.jsonl", help="Output file")
    parser.add_argument("--base-url", default="http://localhost:8080/vulnerabilities/sqli/", help="Target URL")
    parser.add_argument("--param-name", default="id", help="Parameter name for injection")
    args = parser.parse_args()
    
    # Read payloads
    with open(args.payloads_file, "r", encoding="utf-8") as f:
        payloads = [line.strip() for line in f if line.strip()]
    
    print(f"🎯 Testing {len(payloads)} payloads against {args.base_url}")
    print(f"📝 Parameter: {args.param_name}")
    print(f"💾 Output: {args.output}")
    print()
    
    # Run tests
    df = asyncio.run(main_async(args.base_url, payloads, param=args.param_name))
    
    # Save results
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    
    # Also save CSV for easy viewing
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    total = len(df)
    passed = len(df[~df["blocked"]])
    blocked = len(df[df["blocked"]])
    bypass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"📊 WAF REPLAY RESULTS")
    print(f"{'='*70}")
    print(f"Total payloads:  {total}")
    print(f"✅ Passed WAF:   {passed} ({bypass_rate:.1f}%)")
    print(f"🚫 Blocked:      {blocked} ({100-bypass_rate:.1f}%)")
    print(f"{'='*70}")
    print(f"\n💾 Results saved to:")
    print(f"   - {output_path}")
    print(f"   - {csv_path}")


if __name__ == "__main__":
    main()

