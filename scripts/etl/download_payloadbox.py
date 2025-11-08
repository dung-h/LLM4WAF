"""
Download and parse PayloadBox SQLi payload list
Extracts payloads by category: time-based, error-based, union-based, auth-bypass
"""

import re
import requests
from pathlib import Path
from typing import Dict, List
import json

def download_payloadbox_readme() -> str:
    """Download PayloadBox README.md"""
    url = "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/master/README.md"
    print(f"[*] Downloading PayloadBox from {url}")
    response = requests.get(url)
    response.raise_for_status()
    print(f"[+] Downloaded {len(response.text)} bytes")
    return response.text

def parse_payloads_by_category(readme_text: str) -> Dict[str, List[str]]:
    """Parse README.md and extract payloads by category"""
    categories = {
        'time_based': [],
        'error_based': [],
        'union_based': [],
        'auth_bypass': [],
        'generic': []
    }
    
    # Split by code blocks
    code_blocks = re.findall(r'```(.*?)```', readme_text, re.DOTALL)
    
    current_category = 'generic'
    
    for block in code_blocks:
        lines = block.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Detect category from payload content
            line_lower = line.lower()
            
            # Time-based detection
            if any(kw in line_lower for kw in ['sleep(', 'benchmark(', 'waitfor', 'pg_sleep', 'dbms_pipe.receive_message']):
                categories['time_based'].append(line)
            # Error-based detection
            elif any(kw in line_lower for kw in ['extractvalue', 'updatexml', 'cast(', 'convert(', 'xmltype']):
                categories['error_based'].append(line)
            # Union-based detection
            elif 'union' in line_lower and 'select' in line_lower:
                categories['union_based'].append(line)
            # Auth bypass detection
            elif any(kw in line_lower for kw in ["admin'", "' or '", "' or 1=1", '" or "', 'or true--']):
                categories['auth_bypass'].append(line)
            else:
                categories['generic'].append(line)
    
    return categories

def deduplicate_payloads(categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove duplicates within and across categories"""
    deduped = {}
    seen_global = set()
    
    for category, payloads in categories.items():
        unique_payloads = []
        for payload in payloads:
            # Normalize for comparison
            normalized = payload.strip().lower()
            if normalized not in seen_global and len(normalized) > 2:
                unique_payloads.append(payload)
                seen_global.add(normalized)
        
        deduped[category] = unique_payloads
    
    return deduped

def save_categorized_payloads(categories: Dict[str, List[str]], output_dir: Path):
    """Save payloads to separate files by category"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for category, payloads in categories.items():
        if not payloads:
            continue
        
        output_file = output_dir / f"payloadbox_{category}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for payload in payloads:
                f.write(f"{payload}\n")
        
        stats[category] = len(payloads)
        print(f"[+] Saved {len(payloads)} payloads to {output_file}")
    
    # Save summary stats
    stats_file = output_dir / "payloadbox_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[+] Total payloads by category:")
    for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"    {category:20s}: {count:5d}")
    print(f"    {'TOTAL':20s}: {sum(stats.values()):5d}")

def main():
    """Main entry point"""
    # Download
    readme_text = download_payloadbox_readme()
    
    # Parse
    print("\n[*] Parsing payloads by category...")
    categories = parse_payloads_by_category(readme_text)
    
    # Deduplicate
    print("[*] Deduplicating payloads...")
    categories = deduplicate_payloads(categories)
    
    # Save
    output_dir = Path("data/raw/payloadbox")
    print(f"\n[*] Saving to {output_dir}")
    save_categorized_payloads(categories, output_dir)
    
    print("\n[✓] PayloadBox download complete!")

if __name__ == "__main__":
    main()
