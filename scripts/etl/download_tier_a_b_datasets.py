#!/usr/bin/env python3
"""
Download TIER A/B datasets from HuggingFace and GitHub
- shengqin/web-attacks (HuggingFace - 22K samples)
- trendmicro-ailab/Primus-Seed (HuggingFace - 674K samples, filter SQLi only)
- Morzeux/HttpParamsDataset (GitHub - 31K samples)
"""

import os
import json
import requests
import csv
from pathlib import Path
from typing import List, Dict
import zipfile
import io

# Paths
RAW_DATA_DIR = Path("data/raw")
WEB_ATTACKS_DIR = RAW_DATA_DIR / "web_attacks"
PRIMUS_SEED_DIR = RAW_DATA_DIR / "primus_seed"
HTTP_PARAMS_DIR = RAW_DATA_DIR / "http_params"

# Create directories
for directory in [WEB_ATTACKS_DIR, PRIMUS_SEED_DIR, HTTP_PARAMS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def download_web_attacks():
    """Download shengqin/web-attacks from HuggingFace"""
    print("\n=== Downloading shengqin/web-attacks (TIER A) ===")
    
    # HuggingFace dataset viewer API
    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "shengqin/web-attacks",
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100
    }
    
    all_rows = []
    offset = 0
    batch_size = 100
    
    print("Fetching data in batches...")
    while True:
        params["offset"] = offset
        params["length"] = batch_size
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            rows = data.get("rows", [])
            if not rows:
                break
            
            all_rows.extend(rows)
            offset += batch_size
            
            print(f"  Downloaded {len(all_rows)} samples...", end="\r")
            
            # Safety limit (22K samples expected)
            if len(all_rows) >= 25000:
                break
                
        except Exception as e:
            print(f"\n  Error at offset {offset}: {e}")
            break
    
    print(f"\n  Total downloaded: {len(all_rows)} samples")
    
    # Parse and categorize by attack type
    sqli_samples = []
    xss_samples = []
    
    for row in all_rows:
        row_data = row.get("row", {})
        payload = row_data.get("Payload", "").strip()
        attack_type = row_data.get("Attack Type", "").strip().lower()
        
        if not payload:
            continue
        
        sample = {
            "payload": payload,
            "attack_type": attack_type
        }
        
        if "sqli" in attack_type or "sql" in attack_type:
            sqli_samples.append(sample)
        elif "xss" in attack_type:
            xss_samples.append(sample)
    
    # Save SQLi samples (focus on SQLi for this project)
    output_file = WEB_ATTACKS_DIR / "web_attacks_sqli.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in sqli_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"  Saved {len(sqli_samples)} SQLi samples to {output_file}")
    print(f"  (Skipped {len(xss_samples)} XSS samples)")
    
    # Save stats
    stats = {
        "total_downloaded": len(all_rows),
        "sqli_samples": len(sqli_samples),
        "xss_samples": len(xss_samples),
        "source": "shengqin/web-attacks",
        "tier": "A"
    }
    
    stats_file = WEB_ATTACKS_DIR / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return len(sqli_samples)

def download_primus_seed():
    """Download trendmicro-ailab/Primus-Seed (filter SQLi keywords only)"""
    print("\n=== Downloading Primus-Seed (TIER A) ===")
    print("Note: This is a 674K dataset - will filter SQLi-related samples only")
    
    # HuggingFace dataset viewer API
    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "trendmicro-ailab/Primus-Seed",
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100
    }
    
    all_rows = []
    offset = 0
    batch_size = 100
    max_samples = 50000  # Limit to first 50K for filtering
    
    print(f"Fetching first {max_samples} samples for SQLi filtering...")
    while len(all_rows) < max_samples:
        params["offset"] = offset
        params["length"] = batch_size
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            rows = data.get("rows", [])
            if not rows:
                break
            
            all_rows.extend(rows)
            offset += batch_size
            
            print(f"  Downloaded {len(all_rows)} samples...", end="\r")
                
        except Exception as e:
            print(f"\n  Error at offset {offset}: {e}")
            break
    
    print(f"\n  Total downloaded: {len(all_rows)} samples")
    
    # Filter SQLi-related samples
    sqli_keywords = [
        'select', 'union', 'insert', 'update', 'delete', 'drop',
        'sleep', 'benchmark', 'waitfor', 'cast', 'convert',
        'extractvalue', 'updatexml', 'pg_sleep', 'dbms_pipe',
        'or 1=1', 'and 1=1', "' or '", '" or "', '--', '/*', '*/',
        'information_schema', 'sys.', 'xp_cmdshell'
    ]
    
    sqli_samples = []
    
    for row in all_rows:
        row_data = row.get("row", {})
        
        # Check all fields for SQLi patterns
        text_fields = []
        for key, value in row_data.items():
            if isinstance(value, str):
                text_fields.append(value.lower())
        
        combined_text = " ".join(text_fields)
        
        # Check if any SQLi keyword present
        if any(keyword in combined_text for keyword in sqli_keywords):
            sqli_samples.append({
                "payload": str(row_data),  # Keep full context
                "source": "primus-seed",
                "raw_data": row_data
            })
    
    # Save filtered SQLi samples
    output_file = PRIMUS_SEED_DIR / "primus_seed_sqli_filtered.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in sqli_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"  Filtered {len(sqli_samples)} SQLi-related samples from {len(all_rows)}")
    print(f"  Saved to {output_file}")
    
    # Save stats
    stats = {
        "total_downloaded": len(all_rows),
        "sqli_filtered": len(sqli_samples),
        "filter_rate": f"{len(sqli_samples)/len(all_rows)*100:.2f}%",
        "source": "trendmicro-ailab/Primus-Seed",
        "tier": "A",
        "note": "Filtered first 50K samples only due to size"
    }
    
    stats_file = PRIMUS_SEED_DIR / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return len(sqli_samples)

def download_http_params():
    """Download Morzeux/HttpParamsDataset from GitHub"""
    print("\n=== Downloading HttpParamsDataset (TIER B) ===")
    
    # GitHub raw URL for the dataset
    urls = {
        "sqli": "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/sqli.txt",
        "xss": "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/xss.txt",
        "normal": "https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/normal.txt"
    }
    
    sqli_payloads = []
    
    print("Downloading SQLi payloads...")
    try:
        response = requests.get(urls["sqli"], timeout=30)
        response.raise_for_status()
        
        # Parse line by line
        for line in response.text.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                sqli_payloads.append(line)
        
        print(f"  Downloaded {len(sqli_payloads)} SQLi payloads")
        
    except Exception as e:
        print(f"  Error downloading: {e}")
        return 0
    
    # Save payloads
    output_file = HTTP_PARAMS_DIR / "http_params_sqli.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for payload in sqli_payloads:
            f.write(payload + "\n")
    
    print(f"  Saved to {output_file}")
    
    # Convert to JSONL format
    jsonl_file = HTTP_PARAMS_DIR / "http_params_sqli.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for payload in sqli_payloads:
            sample = {
                "payload": payload,
                "source": "sqlmap-generated",
                "category": "sqlmap"
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"  Converted to JSONL: {jsonl_file}")
    
    # Save stats
    stats = {
        "total_sqli": len(sqli_payloads),
        "source": "Morzeux/HttpParamsDataset",
        "tier": "B",
        "note": "sqlmap-generated payloads"
    }
    
    stats_file = HTTP_PARAMS_DIR / "stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    return len(sqli_payloads)

def main():
    print("=" * 60)
    print("TIER A/B Dataset Download Pipeline")
    print("=" * 60)
    
    total_samples = 0
    
    # Download TIER A: shengqin/web-attacks
    try:
        count = download_web_attacks()
        total_samples += count
    except Exception as e:
        print(f"Failed to download web-attacks: {e}")
    
    # Download TIER A: Primus-Seed (filtered)
    try:
        count = download_primus_seed()
        total_samples += count
    except Exception as e:
        print(f"Failed to download Primus-Seed: {e}")
    
    # Download TIER B: HttpParamsDataset
    try:
        count = download_http_params()
        total_samples += count
    except Exception as e:
        print(f"Failed to download HttpParamsDataset: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nTotal SQLi samples downloaded: {total_samples:,}")
    print(f"\nDataset breakdown:")
    print(f"  - web-attacks (TIER A): {WEB_ATTACKS_DIR}")
    print(f"  - primus-seed (TIER A): {PRIMUS_SEED_DIR}")
    print(f"  - http-params (TIER B): {HTTP_PARAMS_DIR}")
    
    print("\nNext steps:")
    print("  1. Review downloaded samples quality")
    print("  2. Merge with PayloadBox (784 + 2,769 evasion variants)")
    print("  3. Rebalance dataset to target distribution")
    print("  4. Create new train/val/test splits")

if __name__ == "__main__":
    main()
