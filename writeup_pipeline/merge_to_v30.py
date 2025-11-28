"""
Merge extracted payloads with existing dataset v29
Creates dataset v30 with specialized crawler payloads
"""

import json
from pathlib import Path
from collections import Counter


def load_extracted_payloads(filepath: str = 'extracted_payloads_specialized.json'):
    """Load extracted payloads"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    payloads = []
    
    for extraction in data['extractions']:
        # XSS payloads
        for payload in extraction['xss_payloads']:
            payloads.append({
                'payload': payload.strip(),
                'type': 'xss',
                'source': extraction['source'],
                'url': extraction['url']
            })
        
        # SQLi payloads
        for payload in extraction['sqli_payloads']:
            payloads.append({
                'payload': payload.strip(),
                'type': 'sqli',
                'source': extraction['source'],
                'url': extraction['url']
            })
    
    return payloads


def load_v29_dataset(data_dir: str = '../data/processed'):
    """Load v29 dataset (JSONL format)"""
    v29_file = Path(data_dir) / 'red_v29_enriched.jsonl'
    
    if not v29_file.exists():
        print(f"⚠ v29 not found: {v29_file}")
        return []
    
    payloads = []
    
    with open(v29_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Extract payload and type from v29 format
                payload_text = data.get('payload', '')
                
                # Determine type from data
                if 'xss' in data.get('type', '').lower():
                    payload_type = 'xss'
                elif 'sqli' in data.get('type', '').lower() or 'sql' in data.get('type', '').lower():
                    payload_type = 'sqli'
                else:
                    payload_type = 'unknown'
                
                payloads.append({
                    'payload': payload_text,
                    'type': payload_type,
                    'source': data.get('source', 'v29')
                })
    
    return payloads


def deduplicate_payloads(old_payloads: list, new_payloads: list):
    """Deduplicate payloads"""
    # Build set of existing payloads
    existing = set()
    for p in old_payloads:
        payload_str = p.get('payload', '').strip().lower()
        existing.add(payload_str)
    
    # Filter new payloads
    unique_new = []
    for p in new_payloads:
        payload_str = p.get('payload', '').strip().lower()
        if payload_str not in existing:
            unique_new.append(p)
            existing.add(payload_str)
    
    return unique_new


def save_v30_dataset(payloads: list, output_dir: str = '../data/processed'):
    """Save v30 dataset (JSONL format for consistency)"""
    output_file = Path(output_dir) / 'red_v30_specialized.jsonl'
    
    # Count stats
    types = Counter(p['type'] for p in payloads)
    sources = Counter(p.get('source', 'unknown') for p in payloads)
    
    # Save as JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for payload in payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    
    print(f"\n💾 Saved v30 to: {output_file}")
    print(f"  Total: {len(payloads)} payloads")
    print(f"  XSS: {types.get('xss', 0)}")
    print(f"  SQLi: {types.get('sqli', 0)}")
    
    return output_file


def main():
    print("=" * 80)
    print("🔀 MERGE DATASETS - v29 + Specialized Crawlers → v30")
    print("=" * 80)
    
    # Load v29
    print("\n📂 Loading v29 dataset...")
    v29_payloads = load_v29_dataset()
    
    if not v29_payloads:
        print("❌ v29 dataset not found. Aborting.")
        return
    
    print(f"✓ Loaded v29: {len(v29_payloads)} payloads")
    
    v29_types = Counter(p.get('type', 'unknown') for p in v29_payloads)
    print(f"  - XSS: {v29_types.get('xss', 0)}")
    print(f"  - SQLi: {v29_types.get('sqli', 0)}")
    
    # Load extracted payloads
    print("\n📂 Loading extracted payloads...")
    new_payloads = load_extracted_payloads()
    print(f"✓ Loaded: {len(new_payloads)} new payloads")
    
    new_types = Counter(p.get('type') for p in new_payloads)
    print(f"  - XSS: {new_types.get('xss', 0)}")
    print(f"  - SQLi: {new_types.get('sqli', 0)}")
    
    # Deduplicate
    print("\n🔍 Deduplicating...")
    unique_new = deduplicate_payloads(v29_payloads, new_payloads)
    print(f"✓ Unique new payloads: {len(unique_new)}")
    print(f"  (Removed {len(new_payloads) - len(unique_new)} duplicates)")
    
    # Merge
    print("\n🔀 Merging...")
    v30_payloads = v29_payloads + unique_new
    print(f"✓ Total v30 payloads: {len(v30_payloads)}")
    
    v30_types = Counter(p.get('type', 'unknown') for p in v30_payloads)
    print(f"  - XSS: {v30_types.get('xss', 0)}")
    print(f"  - SQLi: {v30_types.get('sqli', 0)}")
    
    # Save v30
    print("\n💾 Saving v30 dataset...")
    output_file = save_v30_dataset(v30_payloads)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    print(f"\nv29 → v30 Growth:")
    print(f"  {len(v29_payloads)} → {len(v30_payloads)} (+{len(unique_new)})")
    print(f"  Growth rate: {len(unique_new) / len(v29_payloads) * 100:.1f}%")
    
    print("\nNew payloads by source:")
    sources = Counter(p.get('source', 'unknown') for p in unique_new)
    for source, count in sources.most_common():
        print(f"  - {source}: {count}")
    
    print(f"\n✨ SUCCESS! Dataset v30 created")


if __name__ == "__main__":
    main()
