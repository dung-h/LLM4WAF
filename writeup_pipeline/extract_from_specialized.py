"""
Extract XSS/SQLi payloads from specialized crawler writeups
Uses simple regex patterns to find payloads in collected content
"""

import json
import re
from typing import List, Dict, Set
from pathlib import Path


class PayloadExtractor:
    """Extract XSS and SQLi payloads from writeup content"""
    
    def __init__(self):
        self.seen_payloads = set()
    
    def extract_xss_payloads(self, content: str) -> List[str]:
        """Extract XSS payloads using regex patterns"""
        payloads = []
        
        # Pattern 1: Script tags
        script_pattern = r'<script[^>]*>.*?</script>'
        for match in re.finditer(script_pattern, content, re.IGNORECASE | re.DOTALL):
            payload = match.group(0)
            if len(payload) < 500 and payload not in self.seen_payloads:
                payloads.append(payload)
                self.seen_payloads.add(payload)
        
        # Pattern 2: Event handlers
        event_patterns = [
            r'<[^>]+on\w+\s*=\s*["\'][^"\']+["\'][^>]*>',
            r'<img[^>]+onerror\s*=\s*["\'][^"\']+["\']',
            r'<svg[^>]+onload\s*=\s*["\'][^"\']+["\']',
        ]
        
        for pattern in event_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                payload = match.group(0)
                if len(payload) < 300 and payload not in self.seen_payloads:
                    payloads.append(payload)
                    self.seen_payloads.add(payload)
        
        # Pattern 3: JavaScript expressions in code blocks
        js_pattern = r'`[^`]*(?:alert|prompt|confirm|fetch|document\.cookie)[^`]*`'
        for match in re.finditer(js_pattern, content):
            payload = match.group(0).strip('`')
            if len(payload) < 200 and payload not in self.seen_payloads:
                payloads.append(payload)
                self.seen_payloads.add(payload)
        
        return payloads
    
    def extract_sqli_payloads(self, content: str) -> List[str]:
        """Extract SQLi payloads using regex patterns"""
        payloads = []
        
        # Pattern 1: UNION SELECT
        union_pattern = r"(?:union|UNION)\s+(?:all\s+)?(?:select|SELECT)\s+[^;'\"\n]{10,200}"
        for match in re.finditer(union_pattern, content):
            payload = match.group(0)
            if payload not in self.seen_payloads:
                payloads.append(payload)
                self.seen_payloads.add(payload)
        
        # Pattern 2: OR conditions
        or_patterns = [
            r"'\s*(?:or|OR)\s+['\"]*1\s*=\s*1",
            r"'\s*(?:or|OR)\s+['\"]*\d+\s*=\s*\d+",
            r"admin'\s*--",
            r"'\s*;\s*DROP\s+TABLE",
        ]
        
        for pattern in or_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                payload = match.group(0)
                if len(payload) < 100 and payload not in self.seen_payloads:
                    payloads.append(payload)
                    self.seen_payloads.add(payload)
        
        # Pattern 3: SQL injection in URLs/parameters
        url_sqli = r'[?&]\w+=(?:[^&\s]+)?(?:union|select|or|and|\'|--|;)[^&\s]{5,150}'
        for match in re.finditer(url_sqli, content, re.IGNORECASE):
            payload = match.group(0)
            if payload not in self.seen_payloads:
                payloads.append(payload)
                self.seen_payloads.add(payload)
        
        return payloads
    
    def extract_from_writeup(self, writeup: Dict) -> Dict:
        """Extract all payloads from a single writeup"""
        content = writeup.get('content', '')
        
        xss_payloads = self.extract_xss_payloads(content)
        sqli_payloads = self.extract_sqli_payloads(content)
        
        return {
            'source': writeup.get('source', 'unknown'),
            'title': writeup.get('title', 'No title'),
            'url': writeup.get('url', ''),
            'xss_count': len(xss_payloads),
            'sqli_count': len(sqli_payloads),
            'xss_payloads': xss_payloads,
            'sqli_payloads': sqli_payloads,
            'total': len(xss_payloads) + len(sqli_payloads)
        }


def load_writeups(filepath: str) -> List[Dict]:
    """Load writeups from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('writeups', [])


def save_payloads(results: List[Dict], output_file: str):
    """Save extracted payloads to file"""
    # Prepare output
    output = {
        'metadata': {
            'total_writeups': len(results),
            'total_xss': sum(r['xss_count'] for r in results),
            'total_sqli': sum(r['sqli_count'] for r in results),
            'total_payloads': sum(r['total'] for r in results),
        },
        'extractions': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: {output_file}")


def main():
    print("=" * 80)
    print("🔍 PAYLOAD EXTRACTION - Specialized Writeups")
    print("=" * 80)
    
    # Load writeups
    input_file = 'specialized_writeups.json'
    print(f"\n📂 Loading writeups from: {input_file}")
    
    try:
        writeups = load_writeups(input_file)
        print(f"✓ Loaded {len(writeups)} writeups")
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return
    
    # Extract payloads
    extractor = PayloadExtractor()
    results = []
    
    print(f"\n🔬 Extracting payloads...\n")
    
    for i, writeup in enumerate(writeups, 1):
        print(f"[{i}/{len(writeups)}] Processing: {writeup.get('title', 'Unknown')[:60]}...")
        result = extractor.extract_from_writeup(writeup)
        results.append(result)
        
        if result['total'] > 0:
            print(f"    ✓ Found {result['xss_count']} XSS + {result['sqli_count']} SQLi = {result['total']} payloads")
        else:
            print(f"    ⊘ No payloads found")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 80)
    
    total_xss = sum(r['xss_count'] for r in results)
    total_sqli = sum(r['sqli_count'] for r in results)
    total = total_xss + total_sqli
    
    print(f"\nTotal XSS payloads:  {total_xss}")
    print(f"Total SQLi payloads: {total_sqli}")
    print(f"Grand total:         {total}")
    
    print("\nBy source:")
    for result in results:
        if result['total'] > 0:
            print(f"  - {result['source']}: {result['total']} payloads")
    
    # Save results
    output_file = 'extracted_payloads_specialized.json'
    save_payloads(results, output_file)
    
    # Sample payloads
    if total > 0:
        print("\n" + "=" * 80)
        print("📝 SAMPLE PAYLOADS (First 5)")
        print("=" * 80)
        
        count = 0
        for result in results:
            if count >= 5:
                break
            
            if result['xss_payloads']:
                print(f"\n🔴 XSS from {result['source']}:")
                for payload in result['xss_payloads'][:2]:
                    print(f"  {payload[:100]}...")
                    count += 1
                    if count >= 5:
                        break
            
            if count < 5 and result['sqli_payloads']:
                print(f"\n🔵 SQLi from {result['source']}:")
                for payload in result['sqli_payloads'][:2]:
                    print(f"  {payload[:100]}...")
                    count += 1
                    if count >= 5:
                        break
    
    print("\n✨ Extraction complete!")


if __name__ == "__main__":
    main()
