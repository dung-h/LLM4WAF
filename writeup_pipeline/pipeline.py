"""
Pipeline orchestrator - Run all crawlers and manage the full workflow.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import yaml

from crawlers.ctftime import CTFtimeCrawler
from crawlers.medium import MediumRSSCrawler, DevToCrawler
from crawlers.github import GitHubWriteupCrawler, GitHubRepoWriteupCrawler
from crawlers.security_blogs import (
    HackerOneCrawler,
    PortSwiggerCrawler,
    PentesterLabCrawler,
    BugBountyBlogsCrawler
)
from crawlers.security_researchers import (
    SecurityResearchersCrawler,
    HistoricalCTFCrawler
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WriteupPipeline:
    """Orchestrate the full writeup collection and extraction pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup paths
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "writeups"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize crawlers
        self.crawlers = {
            'ctftime': CTFtimeCrawler(rate_limit=2.0),
            'medium': MediumRSSCrawler(rate_limit=3.0),
            'devto': DevToCrawler(rate_limit=2.0),
            'github_search': GitHubWriteupCrawler(rate_limit=3.0),
            'github_repos': GitHubRepoWriteupCrawler(rate_limit=2.0),
            'hackerone': HackerOneCrawler(rate_limit=3.0),
            'portswigger': PortSwiggerCrawler(rate_limit=2.0),
            'pentesterlab': PentesterLabCrawler(rate_limit=3.0),
            'bugbounty_blogs': BugBountyBlogsCrawler(rate_limit=3.0),
            'security_researchers': SecurityResearchersCrawler(rate_limit=3.0),
            'historical_ctf': HistoricalCTFCrawler(rate_limit=3.0)
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # Convert crawler limits from config structure
                if 'crawlers' in config:
                    targets = {}
                    for crawler_name, crawler_config in config['crawlers'].items():
                        if isinstance(crawler_config, dict) and crawler_config.get('enabled', True):
                            targets[crawler_name] = crawler_config.get('limit', 50)
                    
                    # Map crawler names to our crawler instances
                    crawler_targets = {}
                    if 'ctftime' in targets:
                        crawler_targets['ctftime'] = targets['ctftime']
                    if 'medium' in targets:
                        crawler_targets['medium'] = targets['medium']
                        crawler_targets['devto'] = 20  # Default for devto
                    if 'github' in targets:
                        crawler_targets['github_search'] = targets['github'] // 2
                        crawler_targets['github_repos'] = targets['github'] // 2
                    
                    return {'crawler': {'targets': crawler_targets}}
                
                return config
                
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'crawler': {
                    'targets': {
                        'ctftime': 50,
                        'medium': 30,
                        'devto': 20,
                        'github_search': 50,
                        'github_repos': 50
                    }
                }
            }
    
    def run_crawlers(self, targets: Dict[str, int] = None) -> Dict[str, List[Dict]]:
        """
        Run all crawlers.
        
        Args:
            targets: Dict of crawler_name -> limit (overrides config)
            
        Returns:
            Dict of crawler_name -> writeups
        """
        if targets is None:
            targets = self.config['crawler']['targets']
        
        logger.info("\n" + "="*70)
        logger.info("🚀 Starting Writeup Collection Pipeline")
        logger.info("="*70)
        
        all_results = {}
        total_writeups = 0
        
        for crawler_name, limit in targets.items():
            if crawler_name not in self.crawlers:
                logger.warning(f"⚠️ Unknown crawler: {crawler_name}")
                continue
            
            logger.info(f"\n{'─'*70}")
            logger.info(f"📡 Running {crawler_name.upper()} (target: {limit})")
            logger.info(f"{'─'*70}")
            
            try:
                crawler = self.crawlers[crawler_name]
                writeups = crawler.crawl(limit=limit)
                
                all_results[crawler_name] = writeups
                total_writeups += len(writeups)
                
                # Save intermediate results
                self._save_crawler_results(crawler_name, writeups)
                
                logger.info(f"✅ {crawler_name}: {len(writeups)}/{limit} writeups")
                
            except Exception as e:
                logger.error(f"❌ {crawler_name} failed: {e}")
                all_results[crawler_name] = []
        
        logger.info("\n" + "="*70)
        logger.info(f"🎉 Pipeline Complete: {total_writeups} total writeups")
        logger.info("="*70)
        
        # Save combined results
        self._save_combined_results(all_results)
        
        return all_results
    
    def _save_crawler_results(self, crawler_name: str, writeups: List[Dict]):
        """Save results from a single crawler."""
        output_file = self.raw_dir / f"{crawler_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for writeup in writeups:
                    f.write(json.dumps(writeup, ensure_ascii=False) + '\n')
            
            logger.info(f"   💾 Saved to {output_file}")
            
        except Exception as e:
            logger.error(f"   Failed to save {crawler_name} results: {e}")
    
    def _save_combined_results(self, all_results: Dict[str, List[Dict]]):
        """Save combined results from all crawlers."""
        output_file = self.raw_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        try:
            total = 0
            with open(output_file, 'w', encoding='utf-8') as f:
                for crawler_name, writeups in all_results.items():
                    for writeup in writeups:
                        f.write(json.dumps(writeup, ensure_ascii=False) + '\n')
                        total += 1
            
            logger.info(f"\n💾 Combined results saved to {output_file}")
            logger.info(f"   Total writeups: {total}")
            
            # Generate summary
            self._generate_summary(all_results, output_file)
            
        except Exception as e:
            logger.error(f"Failed to save combined results: {e}")
    
    def _generate_summary(self, all_results: Dict[str, List[Dict]], data_file: Path):
        """Generate summary statistics."""
        summary_file = self.raw_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file),
            'total_writeups': sum(len(w) for w in all_results.values()),
            'by_source': {name: len(writeups) for name, writeups in all_results.items()},
            'statistics': {}
        }
        
        # Calculate statistics
        all_writeups = []
        for writeups in all_results.values():
            all_writeups.extend(writeups)
        
        if all_writeups:
            lengths = [len(w['content']) for w in all_writeups]
            summary['statistics'] = {
                'avg_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'total_chars': sum(lengths)
            }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n📊 Summary saved to {summary_file}")
            
            # Print statistics
            logger.info("\n" + "="*70)
            logger.info("📈 COLLECTION STATISTICS")
            logger.info("="*70)
            for source, count in summary['by_source'].items():
                logger.info(f"   {source:20s}: {count:4d} writeups")
            logger.info(f"   {'─'*30}")
            logger.info(f"   {'TOTAL':20s}: {summary['total_writeups']:4d} writeups")
            
            if summary['statistics']:
                logger.info(f"\n   Avg length: {summary['statistics']['avg_length']:.0f} chars")
                logger.info(f"   Total size: {summary['statistics']['total_chars']:,} chars")
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")


def main():
    """Run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run writeup collection pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--ctftime', type=int, help='CTFtime limit')
    parser.add_argument('--medium', type=int, help='Medium RSS limit')
    parser.add_argument('--devto', type=int, help='Dev.to limit')
    parser.add_argument('--github-search', type=int, help='GitHub search limit')
    parser.add_argument('--github-repos', type=int, help='GitHub repos limit')
    parser.add_argument('--hackerone', type=int, help='HackerOne reports limit')
    parser.add_argument('--portswigger', type=int, help='PortSwigger articles limit')
    parser.add_argument('--pentesterlab', type=int, help='PentesterLab posts limit')
    parser.add_argument('--bugbounty-blogs', type=int, help='Bug bounty blogs limit')
    parser.add_argument('--security-researchers', type=int, help='Security researcher blogs limit (2020-2022)')
    parser.add_argument('--historical-ctf', type=int, help='Historical CTF writeups limit (2020-2022)')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 each)')
    parser.add_argument('--full', action='store_true', help='Full crawl (all sources)')
    parser.add_argument('--historical', action='store_true', help='Focus on 2020-2022 historical sources')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WriteupPipeline(config_path=args.config)
    
    # Override targets if specified
    targets = {}
    
    if args.quick:
        targets = {
            'ctftime': 10,
            'medium': 10,
            'devto': 10,
            'github_search': 10,
            'github_repos': 10,
            'hackerone': 5,
            'portswigger': 5,
            'pentesterlab': 5,
            'bugbounty_blogs': 10,
            'security_researchers': 10,
            'historical_ctf': 10
        }
    elif args.full:
        targets = {
            'ctftime': 100,
            'hackerone': 30,
            'portswigger': 20,
            'pentesterlab': 15,
            'bugbounty_blogs': 25,
            'github_repos': 30,
            'devto': 10,
            'medium': 10,
            'github_search': 10,
            'security_researchers': 50,
            'historical_ctf': 50
        }
    elif args.historical:
        # Focus on 2020-2022 sources
        targets = {
            'security_researchers': 100,  # Brute Logic, terjanq, etc.
            'historical_ctf': 100,         # CTFtime 2020-2022
            'portswigger': 30,             # Research articles
            'hackerone': 50                # Disclosed reports
        }
    else:
        if args.ctftime:
            targets['ctftime'] = args.ctftime
        if args.medium:
            targets['medium'] = args.medium
        if args.devto:
            targets['devto'] = args.devto
        if args.github_search:
            targets['github_search'] = args.github_search
        if args.github_repos:
            targets['github_repos'] = args.github_repos
        if args.hackerone:
            targets['hackerone'] = args.hackerone
        if args.portswigger:
            targets['portswigger'] = args.portswigger
        if args.pentesterlab:
            targets['pentesterlab'] = args.pentesterlab
        if args.bugbounty_blogs:
            targets['bugbounty_blogs'] = args.bugbounty_blogs
        if args.security_researchers:
            targets['security_researchers'] = args.security_researchers
        if args.historical_ctf:
            targets['historical_ctf'] = args.historical_ctf
    
    # Run pipeline
    results = pipeline.run_crawlers(targets if targets else None)
    
    # Print final summary
    print("\n" + "="*70)
    print("✨ PIPELINE COMPLETE ✨")
    print("="*70)
    print(f"Total writeups collected: {sum(len(w) for w in results.values())}")
    print(f"Data saved to: {pipeline.raw_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
