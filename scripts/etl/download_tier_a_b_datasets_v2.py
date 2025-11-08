#!/usr/bin/env python3
"""
Download TIER A/B datasets - Fixed version using datasets library
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict

# Paths
RAW_DATA_DIR = Path("data/raw")
WEB_ATTACKS_DIR = RAW_DATA_DIR / "web_attacks"
HTTP_PARAMS_DIR = RAW_DATA_DIR / "http_params"

# Create directories
for directory in [WEB_ATTACKS_DIR, HTTP_PARAMS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def download_web_attacks_with_datasets():
    """Download shengqin/web-attacks using HuggingFace datasets library"""
    print("\n=== Downloading shengqin/web-attacks (TIER A) ===")
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset from HuggingFace (this may take a minute)...")
        dataset = load_dataset("shengqin/web-attacks", split="train")
        
        print(f"  Loaded {len(dataset)} samples")
        
        # Filter SQLi samples
        sqli_samples = []
        xss_samples = []
        
        for sample in dataset:
            payload = sample.get("Payload", "").strip()
            attack_type = sample.get("text_label", "").strip().lower()  # Fixed: use 'text_label'
            
            if not payload:
                continue
            
            data = {
                "payload": payload,
                "attack_type": attack_type
            }
            
            if "sqli" in attack_type or "sql" in attack_type:
                sqli_samples.append(data)
            elif "xss" in attack_type:
                xss_samples.append(data)
        
        # Save SQLi samples
        output_file = WEB_ATTACKS_DIR / "web_attacks_sqli.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in sqli_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"  Saved {len(sqli_samples)} SQLi samples to {output_file}")
        print(f"  (Skipped {len(xss_samples)} XSS samples)")
        
        # Save stats
        stats = {
            "total_downloaded": len(dataset),
            "sqli_samples": len(sqli_samples),
            "xss_samples": len(xss_samples),
            "source": "shengqin/web-attacks",
            "tier": "A"
        }
        
        stats_file = WEB_ATTACKS_DIR / "stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        return len(sqli_samples)
        
    except ImportError:
        print("  ERROR: 'datasets' library not installed")
        print("  Install with: pip install datasets")
        return 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

def download_http_params_alternative():
    """Try alternative sources for SQLi payloads"""
    print("\n=== Downloading SQLi payloads from alternative sources (TIER B) ===")
    
    # Try multiple GitHub sources
    sources = [
        {
            "name": "fuzzdb-sqli",
            "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/GenericBlind.txt",
            "category": "generic_blind"
        },
        {
            "name": "fuzzdb-sqli-auth",
            "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/auth-bypass.txt",
            "category": "auth_bypass"
        },
        {
            "name": "fuzzdb-sqli-mysql",
            "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/MySQL.txt",
            "category": "mysql"
        },
        {
            "name": "fuzzdb-sqli-postgres",
            "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/Postgres.txt",
            "category": "postgres"
        },
        {
            "name": "fuzzdb-sqli-mssql",
            "url": "https://raw.githubusercontent.com/fuzzdb-project/fuzzdb/master/attack/sql-injection/detect/MSSQL.txt",
            "category": "mssql"
        }
    ]
    
    all_payloads = []
    
    for source in sources:
        print(f"\n  Downloading {source['name']}...")
        try:
            response = requests.get(source['url'], timeout=30)
            response.raise_for_status()
            
            payloads = []
            for line in response.text.splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    payloads.append({
                        "payload": line,
                        "source": source['name'],
                        "category": source['category']
                    })
            
            all_payloads.extend(payloads)
            print(f"    Downloaded {len(payloads)} payloads")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    if not all_payloads:
        print("  WARNING: No payloads downloaded from alternative sources")
        return 0
    
    # Save all payloads
    output_file = HTTP_PARAMS_DIR / "fuzzdb_sqli.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for payload in all_payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    
    print(f"\n  Total: {len(all_payloads)} payloads saved to {output_file}")
    
    # Save stats
    stats = {
        "total_payloads": len(all_payloads),
        "source": "fuzzdb-project (alternative)",
        "tier": "B"
    }
    
    stats_file = HTTP_PARAMS_DIR / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return len(all_payloads)

def main():
    print("=" * 60)
    print("TIER A/B Dataset Download Pipeline (Fixed Version)")
    print("=" * 60)
    
    total_samples = 0
    
    # Download TIER A: shengqin/web-attacks
    print("\n[1/2] TIER A: shengqin/web-attacks")
    try:
        count = download_web_attacks_with_datasets()
        total_samples += count
    except Exception as e:
        print(f"Failed: {e}")
    
    # Download TIER B: Alternative SQLi sources
    print("\n[2/2] TIER B: FuzzDB SQLi payloads")
    try:
        count = download_http_params_alternative()
        total_samples += count
    except Exception as e:
        print(f"Failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nTotal SQLi samples downloaded: {total_samples:,}")
    print(f"\nDataset locations:")
    print(f"  - web-attacks (TIER A): {WEB_ATTACKS_DIR}")
    print(f"  - fuzzdb-sqli (TIER B): {HTTP_PARAMS_DIR}")
    
    if total_samples > 0:
        print("\n✅ SUCCESS! Next steps:")
        print("  1. Review downloaded samples")
        print("  2. Merge with PayloadBox (784 + 2,769 evasion variants)")
        print("  3. Rebalance dataset to target distribution")
    else:
        print("\n⚠️  WARNING: No samples downloaded")
        print("  Check internet connection and try again")
        print("  Or proceed with PayloadBox only (3,553 samples available)")

if __name__ == "__main__":
    main()
