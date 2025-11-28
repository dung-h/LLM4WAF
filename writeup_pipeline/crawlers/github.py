"""
GitHub writeup crawler - Search for CTF writeups and bug bounty reports.
Uses GitHub's web search (no API token required) + RAW file access.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .base import BaseCrawler

logger = logging.getLogger(__name__)


class GitHubWriteupCrawler(BaseCrawler):
    """Crawl CTF writeups from GitHub using search + RAW access."""
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        
        self.search_url = "https://github.com/search"
        self.raw_url = "https://raw.githubusercontent.com"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # Search queries for writeups
        self.search_queries = [
            'CTF writeup XSS',
            'CTF writeup web exploitation',
            'bug bounty writeup XSS',
            'CTF writeup SQLi',
            'web security writeup payload',
            'CTF writeup CSRF',
            'penetration testing writeup',
            'bug bounty report XSS'
        ]
        
        # File patterns to look for
        self.file_patterns = [
            r'writeup\.md$',
            r'README\.md$',
            r'solution\.md$',
            r'exploit\.md$',
            r'report\.md$',
            r'.*writeup.*\.md$'
        ]
    
    def crawl(self, limit: int = 200) -> List[Dict]:
        """
        Crawl GitHub writeups using search.
        
        Strategy:
        1. Search GitHub for writeup keywords
        2. Extract repository/file links
        3. Fetch RAW Markdown content
        4. Filter for web security topics
        
        Args:
            limit: Maximum writeups
            
        Returns:
            List of writeup dictionaries
        """
        logger.info(f"🎯 GitHub: Starting crawl (target: {limit} writeups)")
        
        all_writeups = []
        
        for query in self.search_queries:
            if len(all_writeups) >= limit:
                break
            
            logger.info(f"   🔍 Query: '{query}'")
            writeups = self._search_writeups(query, limit - len(all_writeups))
            all_writeups.extend(writeups)
            logger.info(f"      ✅ +{len(writeups)} writeups (total: {len(all_writeups)})")
            
            time.sleep(self.rate_limit)
        
        logger.info(f"   🎉 GitHub crawl complete: {len(all_writeups)} writeups")
        return all_writeups[:limit]
    
    def _search_writeups(self, query: str, remaining: int) -> List[Dict]:
        """Search GitHub for writeups."""
        writeups = []
        
        try:
            # Search GitHub
            params = {
                'q': query,
                'type': 'code',
                'l': 'Markdown'
            }
            
            response = requests.get(
                self.search_url,
                headers=self.headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract file links
            file_links = self._extract_file_links(soup)
            logger.debug(f"         Found {len(file_links)} file links")
            
            # Fetch each file
            for link in file_links[:remaining * 2]:  # Get more to filter
                if len(writeups) >= remaining:
                    break
                
                writeup = self._fetch_writeup_file(link)
                
                if writeup:
                    valid, reason = self.validator.validate(writeup)
                    
                    if valid and not self.dedup.is_duplicate(writeup['content']):
                        writeups.append(writeup)
                        logger.debug(f"         ✅ {writeup['title'][:50]}...")
                    else:
                        logger.debug(f"         ⏭️ Skipped: {reason}")
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            logger.error(f"         ❌ Search error for '{query}': {e}")
        
        return writeups
    
    def _extract_file_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract file URLs from search results."""
        links = []
        
        try:
            # GitHub search results: <a class="Link--primary" href="/user/repo/blob/...">
            for link_tag in soup.find_all('a', class_=re.compile(r'Link.*primary|search-title')):
                href = link_tag.get('href', '')
                
                # Look for blob URLs
                if '/blob/' in href and any(re.search(pat, href) for pat in self.file_patterns):
                    # Convert to full URL
                    if not href.startswith('http'):
                        href = f"https://github.com{href}"
                    
                    links.append(href)
            
            # Deduplicate
            links = list(set(links))
            
        except Exception as e:
            logger.debug(f"         Error extracting links: {e}")
        
        return links
    
    def _fetch_writeup_file(self, github_url: str) -> Optional[Dict]:
        """Fetch writeup content from GitHub."""
        try:
            # Convert blob URL to raw URL
            # https://github.com/user/repo/blob/main/writeup.md
            # -> https://raw.githubusercontent.com/user/repo/main/writeup.md
            raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            
            logger.debug(f"         Fetching {raw_url}")
            
            response = requests.get(raw_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Extract title from URL or content
            title = self._extract_title(content, github_url)
            
            # Extract metadata
            metadata = self._extract_github_metadata(github_url)
            
            return {
                'url': github_url,
                'title': title,
                'content': content,
                'source': 'github',
                'crawled_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.debug(f"         Error fetching {github_url}: {e}")
            return None
    
    def _extract_title(self, content: str, url: str) -> str:
        """Extract title from Markdown content or URL."""
        # Try to find first # heading
        lines = content.split('\n')
        for line in lines[:10]:
            if line.startswith('# '):
                return line.replace('# ', '').strip()
        
        # Fallback to filename
        match = re.search(r'/([^/]+)\.md$', url)
        if match:
            return match.group(1).replace('-', ' ').replace('_', ' ').title()
        
        return 'Untitled'
    
    def _extract_github_metadata(self, url: str) -> Dict:
        """Extract repo/file info from GitHub URL."""
        try:
            # Parse URL: github.com/user/repo/blob/branch/path
            match = re.search(r'github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)', url)
            
            if match:
                owner, repo, branch, filepath = match.groups()
                
                return {
                    'owner': owner,
                    'repo': repo,
                    'branch': branch,
                    'filepath': filepath,
                    'repo_url': f'https://github.com/{owner}/{repo}'
                }
        except:
            pass
        
        return {}


class GitHubRepoWriteupCrawler(BaseCrawler):
    """Crawl known CTF writeup repositories."""
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        
        # Known CTF writeup repos
        self.repos = [
            'p4-team/ctf',
            'ctfs/write-ups-2023',
            'ctfs/write-ups-2024',
            'ret2jazzy/ctf-writeups',
            'sajjadium/ctf-writeups',
            'Dvd848/CTFs'
        ]
        
        self.api_url = "https://api.github.com/repos"
        self.raw_url = "https://raw.githubusercontent.com"
    
    def crawl(self, limit: int = 100) -> List[Dict]:
        """
        Crawl known CTF writeup repositories.
        
        No API token needed - uses RAW file access.
        """
        logger.info(f"🎯 GitHub Repos: Starting crawl (target: {limit} writeups)")
        
        all_writeups = []
        
        for repo in self.repos:
            if len(all_writeups) >= limit:
                break
            
            logger.info(f"   📦 Repo: {repo}")
            writeups = self._crawl_repo(repo, limit - len(all_writeups))
            all_writeups.extend(writeups)
            logger.info(f"      ✅ {repo}: +{len(writeups)} writeups (total: {len(all_writeups)})")
        
        logger.info(f"   🎉 Repo crawl complete: {len(all_writeups)} writeups")
        return all_writeups
    
    def _crawl_repo(self, repo: str, remaining: int) -> List[Dict]:
        """Crawl a single repository."""
        writeups = []
        
        try:
            # Get directory listing (using web interface to avoid API limits)
            repo_url = f"https://github.com/{repo}"
            
            response = requests.get(repo_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find .md files
            md_files = self._find_md_files(soup, repo)
            logger.debug(f"         Found {len(md_files)} .md files")
            
            # Fetch each file
            for file_path in md_files[:remaining]:
                if len(writeups) >= remaining:
                    break
                
                raw_url = f"{self.raw_url}/{repo}/main/{file_path}"
                
                try:
                    response = requests.get(raw_url, timeout=10)
                    
                    # Try master if main fails
                    if response.status_code == 404:
                        raw_url = raw_url.replace('/main/', '/master/')
                        response = requests.get(raw_url, timeout=10)
                    
                    response.raise_for_status()
                    
                    content = response.text
                    
                    writeup = {
                        'url': f"https://github.com/{repo}/blob/main/{file_path}",
                        'title': file_path.split('/')[-1].replace('.md', ''),
                        'content': content,
                        'source': 'github_repo',
                        'crawled_at': datetime.now().isoformat(),
                        'metadata': {
                            'repo': repo,
                            'filepath': file_path
                        }
                    }
                    
                    valid, reason = self.validator.validate(writeup)
                    
                    if valid and not self.dedup.is_duplicate(content):
                        writeups.append(writeup)
                        logger.debug(f"         ✅ {file_path}")
                
                except Exception as e:
                    logger.debug(f"         Error fetching {file_path}: {e}")
                
                time.sleep(self.rate_limit)
        
        except Exception as e:
            logger.error(f"         ❌ Error crawling {repo}: {e}")
        
        return writeups
    
    def _find_md_files(self, soup: BeautifulSoup, repo: str) -> List[str]:
        """Find Markdown files in repo listing."""
        files = []
        
        # GitHub file listing: <a href="/user/repo/blob/main/path.md">
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            if f'/{repo}/blob/' in href and href.endswith('.md'):
                # Extract path after /blob/branch/
                match = re.search(rf'/{repo}/blob/[^/]+/(.*\.md)', href)
                if match:
                    files.append(match.group(1))
        
        return files


def main():
    """Test GitHub crawlers."""
    logging.basicConfig(level=logging.INFO)
    
    # Test search
    print("\n" + "="*60)
    print("Testing GitHub Search Crawler")
    print("="*60)
    search_crawler = GitHubWriteupCrawler(rate_limit=3.0)
    search_writeups = search_crawler.crawl(limit=10)
    print(f"\nCrawled {len(search_writeups)} writeups from search")
    
    # Test known repos
    print("\n" + "="*60)
    print("Testing GitHub Repo Crawler")
    print("="*60)
    repo_crawler = GitHubRepoWriteupCrawler(rate_limit=2.0)
    repo_writeups = repo_crawler.crawl(limit=10)
    print(f"\nCrawled {len(repo_writeups)} writeups from repos")
    
    # Show samples
    print("\n" + "="*60)
    print("Sample Writeups")
    print("="*60)
    
    all_writeups = search_writeups + repo_writeups
    for i, w in enumerate(all_writeups[:5]):
        print(f"\n{i+1}. [{w['source']}] {w['title']}")
        print(f"   URL: {w['url']}")
        print(f"   Repo: {w.get('metadata', {}).get('repo', 'N/A')}")
        print(f"   Length: {len(w['content'])} chars")


if __name__ == "__main__":
    main()
