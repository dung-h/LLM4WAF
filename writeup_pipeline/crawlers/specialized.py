"""
Specialized Crawlers - Site-specific strategies for high-value sources
Each crawler is tailored to the specific structure of its target site
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
import time
import re
from typing import List, Dict
import json

from .base import BaseCrawler


class PortSwiggerResearchCrawler(BaseCrawler):
    """
    Specialized crawler for PortSwigger Research
    Strategy: RSS (40 entries) + Pagination + Direct article scraping
    OPTIMIZED: Faster timeout, caching, optional full fetch
    """
    
    def __init__(self, rate_limit: float = 1.5, fetch_full: bool = False):
        super().__init__(rate_limit)
        self.rss_url = "https://portswigger.net/research/rss"
        self.base_url = "https://portswigger.net/research"
        self.fetch_full = fetch_full  # Skip full article fetch for speed
        self.cache = {}  # Cache fetched articles
        self.timeout = 5  # Fast timeout
    
    def crawl(self, limit: int = 50) -> List[Dict]:
        """Crawl PortSwigger research articles"""
        print(f"\n🔬 Crawling PortSwigger Research (limit: {limit})")
        
        writeups = []
        
        # Strategy 1: RSS Feed (gets 40 most recent)
        print("  📰 Fetching RSS feed...")
        writeups.extend(self._crawl_rss(limit))
        
        # Strategy 2: Pagination for older articles
        if len(writeups) < limit:
            print(f"  📄 Crawling pagination (need {limit - len(writeups)} more)...")
            writeups.extend(self._crawl_pagination(limit - len(writeups)))
        
        print(f"  ✅ Collected: {len(writeups)} articles")
        return writeups[:limit]
    
    def _crawl_rss(self, limit: int) -> List[Dict]:
        """Crawl via RSS feed (OPTIMIZED)"""
        writeups = []
        
        try:
            feed = feedparser.parse(self.rss_url)
            
            for entry in feed.entries[:limit]:
                # Parse date
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                
                # Get content
                content = entry.get('summary', '')
                
                # Only fetch full article if enabled (for speed)
                if self.fetch_full:
                    full_content = self._fetch_full_article(entry.link)
                    if full_content:
                        content = full_content
                else:
                    # Use summary only (faster)
                    content = entry.get('summary', '') + '\n\n' + entry.get('description', '')
                
                writeup = {
                    'title': entry.get('title', 'No Title'),
                    'url': entry.get('link', ''),
                    'date': pub_date.strftime('%Y-%m-%d'),
                    'content': content,
                    'author': 'PortSwigger Research',
                    'source': 'portswigger_research',
                    'tags': ['web', 'research', str(pub_date.year)]
                }
                
                # Validate
                is_valid, reason = self.validator.validate(writeup)
                if is_valid:
                    writeups.append(writeup)
                    print(f"    ✓ {writeup['title'][:60]}... ({pub_date.year})")
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"    ❌ RSS Error: {e}")
        
        return writeups
    
    def _crawl_pagination(self, limit: int) -> List[Dict]:
        """Crawl older articles via pagination"""
        writeups = []
        
        # PortSwigger uses ?page= parameter
        for page in range(2, 10):  # Start from page 2 (page 1 is in RSS)
            if len(writeups) >= limit:
                break
            
            url = f"{self.base_url}?page={page}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                articles = soup.find_all('a', href=re.compile(r'/research/[^/]+$'))
                
                if not articles:
                    break
                
                for article in articles[:limit - len(writeups)]:
                    article_url = f"https://portswigger.net{article['href']}"
                    title = article.get_text(strip=True)
                    
                    # Fetch full article
                    content = self._fetch_full_article(article_url)
                    
                    if content:
                        writeup = {
                            'title': title,
                            'url': article_url,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'content': content,
                            'author': 'PortSwigger Research',
                            'source': 'portswigger_research',
                            'tags': ['web', 'research']
                        }
                        
                        is_valid, reason = self.validator.validate(writeup)
                        if is_valid:
                            writeups.append(writeup)
                            print(f"    ✓ Page {page}: {title[:50]}...")
                    
                    time.sleep(self.rate_limit)
            
            except Exception as e:
                print(f"    ⚠ Page {page} error: {e}")
                break
        
        return writeups
    
    def _fetch_full_article(self, url: str, retry: int = 2) -> str:
        """Fetch full article content (OPTIMIZED with caching + retry)"""
        # Check cache first
        if url in self.cache:
            return self.cache[url]
        
        for attempt in range(retry):
            try:
                response = requests.get(url, timeout=self.timeout)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article content
                article = soup.find('article') or soup.find('div', class_=re.compile(r'article|content'))
                
                if article:
                    # Remove script tags, style, etc.
                    for tag in article.find_all(['script', 'style', 'nav', 'footer']):
                        tag.decompose()
                    
                    content = article.get_text(separator='\n', strip=True)
                    # Cache result
                    self.cache[url] = content
                    return content
            
            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(1)  # Wait before retry
                    continue
        
        return ""


class OrangeTsaiGitHubCrawler(BaseCrawler):
    """
    Specialized crawler for Orange Tsai's GitHub repositories
    Strategy: GitHub API + Direct file reading
    """
    
    REPOS = [
        'orangetw/My-CTF-Web-Challenges',
        'orangetw/awesome-jenkins-rce-2019',
    ]
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
    
    def crawl(self, limit: int = 30) -> List[Dict]:
        """Crawl Orange Tsai's GitHub writeups"""
        print(f"\n🍊 Crawling Orange Tsai GitHub (limit: {limit})")
        
        writeups = []
        
        for repo in self.REPOS:
            if len(writeups) >= limit:
                break
            
            print(f"  📁 Repository: {repo}")
            writeups.extend(self._crawl_repo(repo, limit - len(writeups)))
            time.sleep(self.rate_limit)
        
        print(f"  ✅ Collected: {len(writeups)} writeups")
        return writeups[:limit]
    
    def _crawl_repo(self, repo: str, limit: int) -> List[Dict]:
        """Crawl a single repository"""
        writeups = []
        
        # GitHub API
        api_url = f"https://api.github.com/repos/{repo}/contents"
        
        try:
            response = requests.get(api_url, timeout=10, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Mozilla/5.0'
            })
            
            if response.status_code != 200:
                print(f"    ❌ API Error: {response.status_code}")
                return writeups
            
            contents = response.json()
            
            # Look for README or writeup files
            for item in contents:
                if len(writeups) >= limit:
                    break
                
                # Check if it's a markdown file or directory
                if item['type'] == 'file' and item['name'].lower().endswith(('.md', '.txt', '.html')):
                    writeup = self._fetch_file_content(item['download_url'], item['name'], repo)
                    if writeup:
                        # Quick check for XSS/SQLi content before validator
                        content_lower = writeup['content'].lower()
                        has_keywords = any(kw in content_lower for kw in ['xss', 'sqli', 'sql injection', '<script', 'payload', 'alert('])
                        
                        if has_keywords:
                            is_valid, reason = self.validator.validate(writeup)
                            if is_valid:
                                writeups.append(writeup)
                                print(f"    ✓ {item['name']}")
                            else:
                                print(f"    ⊘ {item['name']}: {reason}")
                
                elif item['type'] == 'dir':
                    # Recursively check directories (limit depth)
                    dir_writeups = self._crawl_directory(item['url'], repo, limit - len(writeups))
                    writeups.extend(dir_writeups)
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"    ❌ Repository error: {e}")
        
        return writeups
    
    def _crawl_directory(self, api_url: str, repo: str, limit: int) -> List[Dict]:
        """Recursively crawl a directory"""
        writeups = []
        
        try:
            response = requests.get(api_url, timeout=10, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Mozilla/5.0'
            })
            
            if response.status_code == 200:
                contents = response.json()
                
                for item in contents:
                    if len(writeups) >= limit:
                        break
                    
                    if item['type'] == 'file' and item['name'].lower().endswith(('.md', '.txt')):
                        writeup = self._fetch_file_content(item['download_url'], item['name'], repo)
                        if writeup:
                            is_valid, reason = self.validator.validate(writeup)
                            if is_valid:
                                writeups.append(writeup)
                                print(f"    ✓ {item['path']}")
                    
                    time.sleep(self.rate_limit)
        
        except Exception as e:
            pass
        
        return writeups
    
    def _fetch_file_content(self, download_url: str, filename: str, repo: str) -> Dict:
        """Fetch file content from GitHub"""
        try:
            response = requests.get(download_url, timeout=10)
            if response.status_code != 200:
                return None
            
            content = response.text
            
            # Skip if too short
            if len(content) < 100:
                return None
            
            return {
                'title': filename.replace('.md', '').replace('.txt', '').replace('.html', '').replace('-', ' '),
                'url': download_url,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'content': content,
                'author': 'Orange Tsai',
                'source': f'github_{repo.replace("/", "_")}',
                'tags': ['ctf', 'web', 'github', 'orange']
            }
        
        except Exception as e:
            return None


class WaybackMachineCrawler(BaseCrawler):
    """
    Specialized crawler using Wayback Machine for historical content
    Strategy: Archive.org CDX API for 2020-2022 snapshots
    """
    
    TARGET_SITES = {
        'brutelogic': 'https://brutelogic.com.br/blog/',
        'orange': 'https://blog.orange.tw/',
        'liveoverflow': 'https://liveoverflow.com/',
    }
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        self.start_year = 2018  # Extended from 2020
        self.end_year = 2023  # Extended to 2023
        self.timeout = 10  # Faster timeout
    
    def crawl(self, limit: int = 100) -> List[Dict]:
        """Crawl historical snapshots via Wayback Machine (INCREASED limit)"""
        print(f"\n📚 Wayback Machine ({self.start_year}-{self.end_year}, limit: {limit})")
        
        writeups = []
        
        for site_name, url in self.TARGET_SITES.items():
            if len(writeups) >= limit:
                break
            
            print(f"\n  🌐 {site_name}: {url}")
            site_writeups = self._crawl_site_history(url, site_name, limit - len(writeups))
            writeups.extend(site_writeups)
            time.sleep(self.rate_limit)
        
        print(f"\n  ✅ Total collected: {len(writeups)} historical articles")
        return writeups[:limit]
    
    def _crawl_site_history(self, url: str, site_name: str, limit: int) -> List[Dict]:
        """Crawl historical snapshots for a site"""
        writeups = []
        
        # Wayback CDX API for getting snapshot list
        cdx_url = f"http://web.archive.org/cdx/search/cdx"
        params = {
            'url': url,
            'matchType': 'prefix',
            'from': f"{self.start_year}0101",
            'to': f"{self.end_year}1231",
            'output': 'json',
            'limit': limit * 2,  # Get more to filter
            'filter': 'statuscode:200',
        }
        
        try:
            response = requests.get(cdx_url, params=params, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"    ❌ CDX API Error: {response.status_code}")
                return writeups
            
            snapshots = response.json()
            
            if len(snapshots) <= 1:  # First row is header
                print(f"    ⊘ No snapshots found")
                return writeups
            
            print(f"    📸 Found {len(snapshots) - 1} snapshots")
            
            # Process snapshots (skip header)
            seen_urls = set()
            for snapshot in snapshots[1:limit + 1]:
                if len(writeups) >= limit:
                    break
                
                # CDX format: urlkey, timestamp, original, mimetype, statuscode, digest, length
                timestamp = snapshot[1]
                original_url = snapshot[2]
                
                # Skip duplicates
                if original_url in seen_urls:
                    continue
                seen_urls.add(original_url)
                
                # Construct Wayback URL
                wayback_url = f"http://web.archive.org/web/{timestamp}/{original_url}"
                
                # Fetch snapshot content
                writeup = self._fetch_snapshot(wayback_url, timestamp, site_name)
                if writeup:
                    is_valid, reason = self.validator.validate(writeup)
                    if is_valid:
                        writeups.append(writeup)
                        year = timestamp[:4]
                        print(f"    ✓ [{year}] {writeup['title'][:50]}...")
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"    ❌ Wayback error: {e}")
        
        return writeups
    
    def _fetch_snapshot(self, wayback_url: str, timestamp: str, site_name: str) -> Dict:
        """Fetch content from Wayback snapshot (OPTIMIZED timeout)"""
        try:
            response = requests.get(wayback_url, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove Wayback Machine toolbar
            for toolbar in soup.find_all(id='wm-ipp-base'):
                toolbar.decompose()
            
            # Find article content (Wayback-specific selectors)
            article = soup.find('div', class_='post') or soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|entry'))
            
            if not article:
                return None
            
            # Get title
            title_elem = soup.find('h1') or soup.find('h2') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else f"{site_name} snapshot"
            
            # Get content
            content = article.get_text(separator='\n', strip=True)
            
            if len(content) < 200:  # Too short (lowered from 500)
                return None
            
            # Parse date from timestamp
            date = datetime.strptime(timestamp[:8], '%Y%m%d').strftime('%Y-%m-%d')
            
            return {
                'title': title,
                'url': wayback_url,
                'date': date,
                'content': content,
                'author': site_name,
                'source': f'wayback_{site_name}',
                'tags': ['historical', 'web', timestamp[:4]]
            }
        
        except Exception as e:
            return None


class CTFtimeHistoricalCrawler(BaseCrawler):
    """
    Specialized crawler for CTFtime historical writeups (2020-2022)
    Strategy: Year-based filtering + Pagination
    """
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.base_url = "https://ctftime.org"
    
    def crawl(self, limit: int = 50) -> List[Dict]:
        """Crawl CTFtime writeups from 2020-2022"""
        print(f"\n🏆 Crawling CTFtime Historical (2020-2022, limit: {limit})")
        
        writeups = []
        
        for year in [2020, 2021, 2022]:
            if len(writeups) >= limit:
                break
            
            print(f"\n  📅 Year {year}")
            year_writeups = self._crawl_year(year, (limit - len(writeups)) // (2023 - year))
            writeups.extend(year_writeups)
            time.sleep(self.rate_limit)
        
        print(f"\n  ✅ Total collected: {len(writeups)} writeups")
        return writeups[:limit]
    
    def _crawl_year(self, year: int, limit: int) -> List[Dict]:
        """Crawl writeups for specific year"""
        writeups = []
        
        # CTFtime writeup listing by year
        url = f"{self.base_url}/writeups?year={year}"
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find writeup rows
            rows = soup.find_all('tr')
            
            for row in rows[:limit]:
                # Find writeup link
                link = row.find('a', href=re.compile(r'/writeup/\d+'))
                if not link:
                    continue
                
                writeup_url = f"{self.base_url}{link['href']}"
                title = link.get_text(strip=True)
                
                # Fetch full writeup
                content = self._fetch_writeup(writeup_url)
                
                if content:
                    writeup = {
                        'title': title,
                        'url': writeup_url,
                        'date': f"{year}-01-01",
                        'content': content,
                        'author': 'CTFtime Community',
                        'source': f'ctftime_{year}',
                        'tags': ['ctf', 'web', str(year)]
                    }
                    
                    is_valid, reason = self.validator.validate(writeup)
                    if is_valid:
                        writeups.append(writeup)
                        print(f"    ✓ {title[:60]}...")
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            print(f"    ❌ Year {year} error: {e}")
        
        return writeups
    
    def _fetch_writeup(self, url: str) -> str:
        """Fetch writeup content"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # CTFtime stores writeups in specific div
            content_div = soup.find('div', id='writeup')
            
            if content_div:
                return content_div.get_text(separator='\n', strip=True)
        
        except Exception as e:
            pass
        
        return ""


# Export specialized crawlers
__all__ = [
    'PortSwiggerResearchCrawler',
    'OrangeTsaiGitHubCrawler',
    'WaybackMachineCrawler',
    'CTFtimeHistoricalCrawler',
]
