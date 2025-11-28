"""
CTFtime crawler - Extract writeups from CTFtime.org
Based on CyberLLMInstruct methodology but focused on writeup content.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import logging
from .base import BaseCrawler

logger = logging.getLogger(__name__)


class CTFtimeCrawler(BaseCrawler):
    """Crawl CTF writeups from CTFtime.org"""
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.base_url = "https://ctftime.org"
        self.api_url = f"{self.base_url}/api/v1/events/"
        self.writeups_url = f"{self.base_url}/writeups"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def crawl(self, limit: int = 200) -> List[Dict]:
        """
        Crawl CTFtime writeups.
        
        Strategy:
        1. Get writeup listing pages (pagination)
        2. Extract individual writeup URLs
        3. Fetch full writeup content
        4. Validate and filter web-related writeups
        
        Args:
            limit: Maximum number of writeups to collect
            
        Returns:
            List of writeup dictionaries
        """
        logger.info(f"🎯 CTFtime: Starting crawl (target: {limit} writeups)")
        
        writeups = []
        page = 1
        consecutive_empty = 0
        max_empty_pages = 3
        
        while len(writeups) < limit and consecutive_empty < max_empty_pages:
            logger.info(f"   📄 Crawling page {page}...")
            
            # Get writeup listing page
            page_writeups = self._crawl_page(page, limit - len(writeups))
            
            if not page_writeups:
                consecutive_empty += 1
                logger.warning(f"   ⚠️ Empty page {page} ({consecutive_empty}/{max_empty_pages})")
            else:
                consecutive_empty = 0
                writeups.extend(page_writeups)
                logger.info(f"   ✅ Page {page}: +{len(page_writeups)} writeups (total: {len(writeups)}/{limit})")
            
            page += 1
            time.sleep(self.rate_limit)
        
        logger.info(f"   🎉 CTFtime crawl complete: {len(writeups)} writeups")
        return writeups
    
    def _crawl_page(self, page: int, remaining: int) -> List[Dict]:
        """Crawl a single page of writeup listings."""
        try:
            # CTFtime writeup listing with pagination
            url = f"{self.writeups_url}?page={page}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find writeup links
            # CTFtime structure: <a href="/writeup/12345">Writeup Title</a>
            writeup_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/writeup/' in href and href.count('/') == 2:  # /writeup/12345
                    writeup_links.append(href)
            
            # Remove duplicates
            writeup_links = list(set(writeup_links))[:remaining]
            
            logger.debug(f"      Found {len(writeup_links)} writeup links on page {page}")
            
            # Fetch individual writeups
            writeups = []
            for link in writeup_links:
                writeup = self._fetch_writeup(link)
                
                if writeup:
                    # Validate
                    valid, reason = self.validator.validate(writeup)
                    
                    if valid and not self.dedup.is_duplicate(writeup['content']):
                        writeups.append(writeup)
                        self._log_progress(len(writeups), remaining, writeup['title'])
                    else:
                        logger.debug(f"      ⏭️ Skipped: {reason}")
                
                time.sleep(self.rate_limit)
            
            return writeups
            
        except Exception as e:
            logger.error(f"   ❌ Error crawling page {page}: {e}")
            return []
    
    def _fetch_writeup(self, writeup_path: str) -> Optional[Dict]:
        """
        Fetch individual writeup content.
        
        Args:
            writeup_path: Path like /writeup/12345
            
        Returns:
            Writeup dict or None
        """
        try:
            url = f"{self.base_url}{writeup_path}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h2')
            title = title_elem.text.strip() if title_elem else 'Untitled'
            
            # Extract content
            # CTFtime writeups are in <div class="well">
            content_div = soup.find('div', class_='well')
            
            if not content_div:
                logger.debug(f"      No content found for {writeup_path}")
                return None
            
            # Get text content
            content = content_div.get_text(separator='\n').strip()
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'source': 'ctftime',
                'crawled_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.debug(f"      Failed to fetch {writeup_path}: {e}")
            return None
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract additional metadata from writeup page."""
        metadata = {}
        
        try:
            # CTF name
            ctf_link = soup.find('a', href=lambda x: x and '/event/' in x)
            if ctf_link:
                metadata['ctf_name'] = ctf_link.text.strip()
            
            # Challenge name
            challenge_elem = soup.find('span', class_='label')
            if challenge_elem:
                metadata['challenge'] = challenge_elem.text.strip()
            
            # Tags/Categories
            tags = soup.find_all('span', class_='label label-info')
            if tags:
                metadata['tags'] = [tag.text.strip() for tag in tags]
            
        except Exception as e:
            logger.debug(f"      Error extracting metadata: {e}")
        
        return metadata


def main():
    """Test CTFtime crawler."""
    logging.basicConfig(level=logging.INFO)
    
    crawler = CTFtimeCrawler(rate_limit=2.0)
    writeups = crawler.crawl(limit=10)
    
    print(f"\n{'='*60}")
    print(f"Crawled {len(writeups)} writeups")
    print(f"{'='*60}")
    
    for i, w in enumerate(writeups[:3]):
        print(f"\n{i+1}. {w['title']}")
        print(f"   URL: {w['url']}")
        print(f"   Length: {len(w['content'])} chars")
        if w.get('metadata'):
            print(f"   Metadata: {w['metadata']}")


if __name__ == "__main__":
    main()
