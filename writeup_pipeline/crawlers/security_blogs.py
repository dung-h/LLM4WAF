"""
Security blog crawlers - Crawl from trusted cybersecurity blogs and platforms.
Focuses on bug bounty reports, penetration testing writeups, and security research.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .base import BaseCrawler

logger = logging.getLogger(__name__)


class HackerOneCrawler(BaseCrawler):
    """Crawl public bug bounty reports from HackerOne."""
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.base_url = "https://hackerone.com"
        self.reports_url = f"{self.base_url}/hacktivity"
        
    def crawl(self, limit: int = 50) -> List[Dict]:
        """
        Crawl public bug bounty reports from HackerOne.
        Focus on XSS and SQLi vulnerabilities.
        """
        logger.info(f"🎯 HackerOne: Starting crawl (target: {limit} reports)")
        
        reports = []
        page = 1
        
        # Target specific vulnerability types
        vuln_types = ['xss', 'sql-injection']
        
        for vuln_type in vuln_types:
            if len(reports) >= limit:
                break
                
            logger.info(f"   🔍 Fetching {vuln_type} reports...")
            
            try:
                # HackerOne hacktivity with filters
                params = {
                    'querystring': vuln_type,
                    'filter': 'type:public',
                    'page': page
                }
                
                response = requests.get(
                    self.reports_url,
                    params=params,
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=15
                )
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract report links
                report_links = soup.find_all('a', href=lambda x: x and '/reports/' in x)
                
                for link in report_links[:limit - len(reports)]:
                    report_url = self.base_url + link['href']
                    report = self._fetch_report(report_url)
                    
                    if report:
                        valid, reason = self.validator.validate(report)
                        if valid and not self.dedup.is_duplicate(report['content']):
                            reports.append(report)
                            logger.debug(f"      ✅ {report['title'][:50]}...")
                    
                    time.sleep(self.rate_limit)
                
                logger.info(f"      ✅ {vuln_type}: +{len([r for r in reports if vuln_type in r.get('metadata', {}).get('vuln_type', '')])} reports")
                
            except Exception as e:
                logger.error(f"      ❌ Error fetching {vuln_type}: {e}")
        
        logger.info(f"   🎉 HackerOne complete: {len(reports)} reports")
        return reports
    
    def _fetch_report(self, url: str) -> Optional[Dict]:
        """Fetch individual report."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else 'Untitled'
            
            # Extract report content
            content_elem = soup.find('div', class_='spec-report-content')
            if not content_elem:
                return None
            
            content = content_elem.get_text(separator='\n', strip=True)
            
            # Extract metadata
            vuln_type_elem = soup.find('span', text=lambda x: x and 'Weakness' in x)
            vuln_type = vuln_type_elem.find_next('span').text.strip() if vuln_type_elem else 'Unknown'
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'source': 'hackerone',
                'crawled_at': datetime.now().isoformat(),
                'metadata': {
                    'vuln_type': vuln_type.lower(),
                    'platform': 'bug_bounty'
                }
            }
            
        except Exception as e:
            logger.debug(f"      Error fetching report {url}: {e}")
            return None


class PortSwiggerCrawler(BaseCrawler):
    """Crawl PortSwigger Web Security Academy labs and research."""
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.base_url = "https://portswigger.net"
        self.research_rss = f"{self.base_url}/research/rss"
        self.labs_url = f"{self.base_url}/web-security/all-labs"
        
    def crawl(self, limit: int = 30) -> List[Dict]:
        """Crawl PortSwigger research and lab solutions."""
        logger.info(f"🎯 PortSwigger: Starting crawl (target: {limit} articles)")
        
        writeups = []
        
        # 1. Crawl research blog
        logger.info("   📰 Fetching research articles...")
        research = self._crawl_research(limit // 2)
        writeups.extend(research)
        
        # 2. Crawl lab solutions from community
        logger.info("   🧪 Fetching lab solutions...")
        labs = self._crawl_lab_solutions(limit - len(writeups))
        writeups.extend(labs)
        
        logger.info(f"   🎉 PortSwigger complete: {len(writeups)} articles")
        return writeups
    
    def _crawl_research(self, limit: int) -> List[Dict]:
        """Crawl research blog via RSS."""
        articles = []
        
        try:
            feed = feedparser.parse(self.research_rss)
            
            for entry in feed.entries[:limit * 2]:
                # Filter for XSS/SQLi topics
                title = entry.title.lower()
                if not any(kw in title for kw in ['xss', 'sql', 'injection', 'bypass', 'waf']):
                    continue
                
                article_url = entry.link
                
                # Fetch full content
                response = requests.get(article_url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                content_elem = soup.find('article')
                if not content_elem:
                    continue
                
                content = content_elem.get_text(separator='\n', strip=True)
                
                writeup = {
                    'url': article_url,
                    'title': entry.title,
                    'content': content,
                    'source': 'portswigger_research',
                    'crawled_at': datetime.now().isoformat(),
                    'metadata': {
                        'published': entry.get('published', ''),
                        'author': entry.get('author', 'PortSwigger')
                    }
                }
                
                valid, reason = self.validator.validate(writeup)
                if valid and not self.dedup.is_duplicate(content):
                    articles.append(writeup)
                    logger.debug(f"      ✅ {writeup['title'][:50]}...")
                
                time.sleep(self.rate_limit)
                
                if len(articles) >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"      ❌ Error crawling research: {e}")
        
        return articles
    
    def _crawl_lab_solutions(self, limit: int) -> List[Dict]:
        """Search GitHub for PortSwigger lab solutions."""
        solutions = []
        
        # Community repos with PortSwigger solutions
        repos = [
            'rkhal101/Web-Security-Academy-Series',
            'botesjuan/Burp-Suite-Certified-Practitioner-Exam-Study',
            'DingyShark/BurpSuiteCertifiedPractitioner'
        ]
        
        for repo in repos:
            if len(solutions) >= limit:
                break
            
            try:
                # Get repo file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    files = response.json().get('tree', [])
                    
                    # Find XSS/SQLi lab files
                    for file in files:
                        if len(solutions) >= limit:
                            break
                        
                        path = file['path']
                        if not path.endswith('.md'):
                            continue
                        
                        if not any(kw in path.lower() for kw in ['xss', 'sql', 'injection']):
                            continue
                        
                        # Fetch file content
                        raw_url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
                        content_response = requests.get(raw_url, timeout=10)
                        
                        if content_response.status_code == 200:
                            content = content_response.text
                            
                            writeup = {
                                'url': f"https://github.com/{repo}/blob/main/{path}",
                                'title': path.split('/')[-1].replace('.md', ''),
                                'content': content,
                                'source': 'portswigger_labs',
                                'crawled_at': datetime.now().isoformat(),
                                'metadata': {
                                    'repo': repo,
                                    'filepath': path
                                }
                            }
                            
                            valid, reason = self.validator.validate(writeup)
                            if valid and not self.dedup.is_duplicate(content):
                                solutions.append(writeup)
                                logger.debug(f"      ✅ {writeup['title'][:50]}...")
                        
                        time.sleep(self.rate_limit)
                        
            except Exception as e:
                logger.error(f"      ❌ Error crawling {repo}: {e}")
        
        return solutions


class PentesterLabCrawler(BaseCrawler):
    """Crawl PentesterLab blog and exercises."""
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.blog_rss = "https://blog.pentesterlab.com/feed"
        
    def crawl(self, limit: int = 20) -> List[Dict]:
        """Crawl PentesterLab blog."""
        logger.info(f"🎯 PentesterLab: Starting crawl (target: {limit} posts)")
        
        posts = []
        
        try:
            feed = feedparser.parse(self.blog_rss)
            
            for entry in feed.entries[:limit * 2]:
                title = entry.title.lower()
                summary = entry.get('summary', '').lower()
                
                # Filter for web security
                if not any(kw in title + summary for kw in ['xss', 'sql', 'injection', 'web', 'bypass']):
                    continue
                
                # Fetch full post
                response = requests.get(entry.link, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                content_elem = soup.find('article') or soup.find('div', class_='post-content')
                if not content_elem:
                    continue
                
                content = content_elem.get_text(separator='\n', strip=True)
                
                writeup = {
                    'url': entry.link,
                    'title': entry.title,
                    'content': content,
                    'source': 'pentesterlab',
                    'crawled_at': datetime.now().isoformat(),
                    'metadata': {
                        'published': entry.get('published', ''),
                        'tags': entry.get('tags', [])
                    }
                }
                
                valid, reason = self.validator.validate(writeup)
                if valid and not self.dedup.is_duplicate(content):
                    posts.append(writeup)
                    logger.debug(f"      ✅ {writeup['title'][:50]}...")
                
                time.sleep(self.rate_limit)
                
                if len(posts) >= limit:
                    break
                    
        except Exception as e:
            logger.error(f"      ❌ Error crawling PentesterLab: {e}")
        
        logger.info(f"   🎉 PentesterLab complete: {len(posts)} posts")
        return posts


class BugBountyBlogsCrawler(BaseCrawler):
    """Crawl popular bug bounty hunter blogs."""
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        
        # Trusted bug bounty blogs
        self.blogs = {
            'nahamsec': 'https://nahamsec.com/feed/',
            'zseano': 'https://blog.zseano.com/feed',
            'stok': 'https://www.stokfredrik.com/feed',
            'jhaddix': 'https://jhaddix.com/feed/'
        }
    
    def crawl(self, limit: int = 30) -> List[Dict]:
        """Crawl bug bounty blogs."""
        logger.info(f"🎯 Bug Bounty Blogs: Starting crawl (target: {limit} posts)")
        
        all_posts = []
        
        for blog_name, feed_url in self.blogs.items():
            if len(all_posts) >= limit:
                break
            
            logger.info(f"   📝 Blog: {blog_name}")
            
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:limit]:
                    if len(all_posts) >= limit:
                        break
                    
                    title = entry.title.lower()
                    summary = entry.get('summary', '').lower()
                    
                    # Filter for relevant content
                    if not any(kw in title + summary for kw in ['xss', 'sql', 'injection', 'bypass', 'payload']):
                        continue
                    
                    try:
                        # Fetch full content
                        response = requests.get(entry.link, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Try different content selectors
                        content_elem = (
                            soup.find('article') or
                            soup.find('div', class_='post-content') or
                            soup.find('div', class_='entry-content') or
                            soup.find('main')
                        )
                        
                        if not content_elem:
                            continue
                        
                        content = content_elem.get_text(separator='\n', strip=True)
                        
                        writeup = {
                            'url': entry.link,
                            'title': entry.title,
                            'content': content,
                            'source': f'bugbounty_{blog_name}',
                            'crawled_at': datetime.now().isoformat(),
                            'metadata': {
                                'author': blog_name,
                                'published': entry.get('published', ''),
                                'blog': blog_name
                            }
                        }
                        
                        valid, reason = self.validator.validate(writeup)
                        if valid and not self.dedup.is_duplicate(content):
                            all_posts.append(writeup)
                            logger.debug(f"      ✅ {writeup['title'][:50]}...")
                        
                    except Exception as e:
                        logger.debug(f"      Error fetching {entry.link}: {e}")
                    
                    time.sleep(self.rate_limit)
                
                logger.info(f"      ✅ {blog_name}: +{len([p for p in all_posts if blog_name in p.get('metadata', {}).get('blog', '')])} posts")
                
            except Exception as e:
                logger.error(f"      ❌ Error crawling {blog_name}: {e}")
        
        logger.info(f"   🎉 Bug Bounty Blogs complete: {len(all_posts)} posts")
        return all_posts


def main():
    """Test security blog crawlers."""
    logging.basicConfig(level=logging.INFO)
    
    # Test HackerOne
    print("\n" + "="*60)
    print("Testing HackerOne Crawler")
    print("="*60)
    h1_crawler = HackerOneCrawler(rate_limit=3.0)
    h1_reports = h1_crawler.crawl(limit=5)
    print(f"\nCrawled {len(h1_reports)} reports from HackerOne")
    
    # Test PortSwigger
    print("\n" + "="*60)
    print("Testing PortSwigger Crawler")
    print("="*60)
    ps_crawler = PortSwiggerCrawler(rate_limit=2.0)
    ps_articles = ps_crawler.crawl(limit=5)
    print(f"\nCrawled {len(ps_articles)} articles from PortSwigger")
    
    # Test PentesterLab
    print("\n" + "="*60)
    print("Testing PentesterLab Crawler")
    print("="*60)
    ptl_crawler = PentesterLabCrawler(rate_limit=3.0)
    ptl_posts = ptl_crawler.crawl(limit=5)
    print(f"\nCrawled {len(ptl_posts)} posts from PentesterLab")
    
    # Test Bug Bounty Blogs
    print("\n" + "="*60)
    print("Testing Bug Bounty Blogs Crawler")
    print("="*60)
    bb_crawler = BugBountyBlogsCrawler(rate_limit=3.0)
    bb_posts = bb_crawler.crawl(limit=10)
    print(f"\nCrawled {len(bb_posts)} posts from Bug Bounty Blogs")
    
    # Show samples
    print("\n" + "="*60)
    print("Sample Writeups")
    print("="*60)
    
    all_writeups = h1_reports + ps_articles + ptl_posts + bb_posts
    for i, w in enumerate(all_writeups[:5]):
        print(f"\n{i+1}. [{w['source']}] {w['title']}")
        print(f"   URL: {w['url']}")
        print(f"   Length: {len(w['content'])} chars")


if __name__ == "__main__":
    main()
