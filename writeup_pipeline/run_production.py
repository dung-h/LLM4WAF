"""
Production Specialized Crawlers v2 - OPTIMIZED
- PortSwigger: Faster (RSS only, no full fetch)
- Wayback: Extended (2018-2023, limit 100)
- Orange GitHub: Same (works perfectly)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writeup_pipeline.crawlers.specialized import (
    PortSwiggerResearchCrawler,
    OrangeTsaiGitHubCrawler,
    WaybackMachineCrawler,
)
import json
from datetime import datetime
from collections import Counter


def save_results(all_writeups: list, filename: str = "specialized_writeups_v2.json"):
    """Save all writeups to JSON"""
    output = {
        'metadata': {
            'total_writeups': len(all_writeups),
            'collected_at': datetime.now().isoformat(),
            'version': 'v2_optimized',
        },
        'writeups': all_writeups
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(all_writeups)} writeups to: {filename}")


def main():
    print("=" * 80)
    print("🚀 SPECIALIZED CRAWLERS - PRODUCTION RUN v2 (OPTIMIZED)")
    print("=" * 80)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_writeups = []
    
    # 1. PortSwigger (OPTIMIZED: RSS only, faster)
    print("\n" + "=" * 80)
    print("🔬 PortSwigger Research (OPTIMIZED - RSS Only)")
    print("=" * 80)
    
    try:
        portswigger = PortSwiggerResearchCrawler(
            rate_limit=1.5,
            fetch_full=False  # Skip full fetch for speed
        )
        ps_writeups = portswigger.crawl(limit=20)
        all_writeups.extend(ps_writeups)
        print(f"✅ Collected: {len(ps_writeups)} writeups")
    except Exception as e:
        print(f"❌ PortSwigger Error: {e}")
    
    # 2. Orange Tsai GitHub (Same - works perfectly)
    print("\n" + "=" * 80)
    print("🍊 Orange Tsai GitHub")
    print("=" * 80)
    
    try:
        orange = OrangeTsaiGitHubCrawler(rate_limit=2.0)
        orange_writeups = orange.crawl(limit=20)
        all_writeups.extend(orange_writeups)
        print(f"✅ Collected: {len(orange_writeups)} writeups")
    except Exception as e:
        print(f"❌ Orange GitHub Error: {e}")
    
    # 3. Wayback Machine (OPTIMIZED: 2018-2023, limit 100)
    print("\n" + "=" * 80)
    print("📚 Wayback Machine (OPTIMIZED: 2018-2023)")
    print("=" * 80)
    
    try:
        wayback = WaybackMachineCrawler(rate_limit=3.0)
        wayback_writeups = wayback.crawl(limit=100)
        all_writeups.extend(wayback_writeups)
        print(f"✅ Collected: {len(wayback_writeups)} writeups")
        
        # Show year distribution
        years = [w.get('date', '')[:4] for w in wayback_writeups]
        year_counts = Counter(years)
        print("\n📊 Distribution by year:")
        for year, count in sorted(year_counts.items()):
            print(f"    {year}: {count} writeups")
    
    except Exception as e:
        print(f"❌ Wayback Error: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS v2")
    print("=" * 80)
    
    print(f"\nTotal writeups: {len(all_writeups)}")
    
    # By source
    sources = Counter(w.get('source', 'unknown') for w in all_writeups)
    print("\nBy source:")
    for source, count in sources.most_common():
        print(f"  - {source}: {count} writeups")
    
    # By year
    years = Counter(w.get('date', '')[:4] for w in all_writeups)
    print("\nBy year:")
    for year, count in sorted(years.items(), reverse=True):
        if year:
            print(f"  - {year}: {count} writeups")
    
    # Save
    if all_writeups:
        save_results(all_writeups)
        print(f"\n✨ SUCCESS! {len(all_writeups)} high-quality writeups collected")
    else:
        print("\n⚠ WARNING: No writeups collected")
    
    print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
