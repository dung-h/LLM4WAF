"""
Medium & RSS crawler - Extract security writeups from Medium and other RSS feeds.
Focuses on recent bug bounty reports and CTF writeups.
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from .base import BaseCrawler

logger = logging.getLogger(__name__)


class MediumRSSCrawler(BaseCrawler):
    """Crawl security writeups from Medium RSS feeds and other blogs."""
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        
        # Medium RSS feeds by tag
        self.medium_feeds = {
            'bug-bounty': 'https://medium.com/feed/tag/bug-bounty',
            'web-security': 'https://medium.com/feed/tag/web-security',
            'xss': 'https://medium.com/feed/tag/xss',
            'penetration-testing': 'https://medium.com/feed/tag/penetration-testing',
            'ctf-writeup': 'https://medium.com/feed/tag/ctf-writeup',
            'cybersecurity': 'https://medium.com/feed/tag/cybersecurity'
        }
        
        # Additional RSS feeds
        self.other_feeds = {
            'portswigger': 'https://portswigger.net/research/rss',
            'pentesterlab': 'https://blog.pentesterlab.com/feed',
            'owasp': 'https://owasp.org/blog/feed.xml'
        }
    
    def crawl(self, limit: int = 100) -> List[Dict]:
        """
        Crawl writeups from RSS feeds.
        
        Strategy:
        1. Fetch Medium feeds by tag
        2. Fetch other security blog feeds
        3. Filter for web security keywords
        4. Extract full article content
        
        Args:
            limit: Maximum number of writeups
            
        Returns:
            List of writeup dictionaries
        """
        logger.info(f"🎯 Medium/RSS: Starting crawl (target: {limit} writeups)")
        
        all_writeups = []
        
        # Crawl Medium feeds
        logger.info("   📰 Crawling Medium feeds...")
        for tag, feed_url in self.medium_feeds.items():
            if len(all_writeups) >= limit:
                break
            
            logger.info(f"      Tag: {tag}")
            writeups = self._crawl_feed(feed_url, tag, limit - len(all_writeups))
            all_writeups.extend(writeups)
            logger.info(f"      ✅ {tag}: +{len(writeups)} writeups (total: {len(all_writeups)})")
        
        # Crawl other feeds
        logger.info("   📰 Crawling other security blogs...")
        for source, feed_url in self.other_feeds.items():
            if len(all_writeups) >= limit:
                break
            
            logger.info(f"      Source: {source}")
            writeups = self._crawl_feed(feed_url, source, limit - len(all_writeups))
            all_writeups.extend(writeups)
            logger.info(f"      ✅ {source}: +{len(writeups)} writeups (total: {len(all_writeups)})")
        
        logger.info(f"   🎉 RSS crawl complete: {len(all_writeups)} writeups")
        return all_writeups[:limit]
    
    def _crawl_feed(self, feed_url: str, source: str, remaining: int) -> List[Dict]:
        """Crawl a single RSS feed."""
        writeups = []
        
        try:
            # Parse RSS feed
            logger.debug(f"         Fetching {feed_url}...")
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                logger.warning(f"         No entries found in {feed_url}")
                return []
            
            logger.debug(f"         Found {len(feed.entries)} entries")
            
            # Process entries
            for entry in feed.entries[:remaining * 2]:  # Get more to filter
                if len(writeups) >= remaining:
                    break
                
                writeup = self._process_entry(entry, source, feed_url)
                
                if writeup:
                    # Validate
                    valid, reason = self.validator.validate(writeup)
                    
                    if valid and not self.dedup.is_duplicate(writeup['content']):
                        writeups.append(writeup)
                        logger.debug(f"         ✅ {writeup['title'][:40]}... ({len(writeup['content'])} chars)")
                    else:
                        logger.debug(f"         ⏭️ Skipped: {reason}")
            
        except Exception as e:
            logger.error(f"         ❌ Error crawling {feed_url}: {e}")
        
        return writeups
    
    def _process_entry(self, entry: Dict, source: str, feed_url: str) -> Optional[Dict]:
        """Process a single RSS feed entry."""
        try:
            # Extract basic info
            title = entry.get('title', 'Untitled').strip()
            link = entry.get('link', '')
            published = entry.get('published', '')
            
            # Extract content
            content_raw = entry.get('content', [{}])[0].get('value', '') or \
                         entry.get('summary', '') or \
                         entry.get('description', '')
            
            if not content_raw:
                return None
            
            # Clean HTML
            soup = BeautifulSoup(content_raw, 'html.parser')
            
            # Remove scripts, styles
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer']):
                tag.decompose()
            
            # Get text
            content = soup.get_text(separator='\n', strip=True)
            
            # Filter by date (last 2 years)
            if published:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date < datetime.now() - timedelta(days=730):
                        logger.debug(f"         Old article: {title[:40]}...")
                        return None
                except:
                    pass
            
            return {
                'url': link,
                'title': title,
                'content': content,
                'source': f'rss_{source}',
                'crawled_at': datetime.now().isoformat(),
                'metadata': {
                    'feed_url': feed_url,
                    'published': published,
                    'author': entry.get('author', 'Unknown')
                }
            }
            
        except Exception as e:
            logger.debug(f"         Error processing entry: {e}")
            return None


class DevToCrawler(BaseCrawler):
    """Crawl writeups from Dev.to using their API."""
    
    def __init__(self, rate_limit: float = 2.0):
        super().__init__(rate_limit)
        self.api_url = "https://dev.to/api/articles"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
    
    def crawl(self, limit: int = 50) -> List[Dict]:
        """
        Crawl Dev.to articles using their public API.
        
        API: https://developers.forem.com/api
        No authentication needed for public articles.
        """
        logger.info(f"🎯 Dev.to: Starting crawl (target: {limit} writeups)")
        
        writeups = []
        page = 1
        per_page = 30
        
        tags = ['cybersecurity', 'bugbounty', 'ctf', 'websecurity', 'infosec']
        
        for tag in tags:
            if len(writeups) >= limit:
                break
            
            logger.info(f"   🏷️ Tag: {tag}")
            
            try:
                params = {
                    'tag': tag,
                    'per_page': per_page,
                    'page': page
                }
                
                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                
                articles = response.json()
                
                for article in articles:
                    if len(writeups) >= limit:
                        break
                    
                    writeup = self._process_article(article)
                    
                    if writeup:
                        valid, reason = self.validator.validate(writeup)
                        
                        if valid and not self.dedup.is_duplicate(writeup['content']):
                            writeups.append(writeup)
                            logger.debug(f"      ✅ {writeup['title'][:40]}...")
                
                logger.info(f"      Tag {tag}: +{len([w for w in writeups if tag in w.get('metadata', {}).get('tags', [])])} writeups")
                
            except Exception as e:
                logger.error(f"      ❌ Error fetching Dev.to {tag}: {e}")
        
        logger.info(f"   🎉 Dev.to complete: {len(writeups)} writeups")
        return writeups
    
    def _process_article(self, article: Dict) -> Optional[Dict]:
        """Process a Dev.to article."""
        try:
            # Get full article content
            article_url = article.get('url', '')
            
            # Fetch full HTML
            response = requests.get(article_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article body
            article_body = soup.find('div', class_='crayons-article__body')
            
            if not article_body:
                return None
            
            content = article_body.get_text(separator='\n', strip=True)
            
            return {
                'url': article_url,
                'title': article.get('title', 'Untitled'),
                'content': content,
                'source': 'devto',
                'crawled_at': datetime.now().isoformat(),
                'metadata': {
                    'tags': article.get('tag_list', []),
                    'published': article.get('published_at', ''),
                    'author': article.get('user', {}).get('username', 'Unknown')
                }
            }
            
        except Exception as e:
            logger.debug(f"      Error processing article: {e}")
            return None


def main():
    """Test RSS crawlers."""
    logging.basicConfig(level=logging.INFO)
    
    # Test Medium
    print("\n" + "="*60)
    print("Testing Medium RSS Crawler")
    print("="*60)
    medium_crawler = MediumRSSCrawler(rate_limit=3.0)
    medium_writeups = medium_crawler.crawl(limit=10)
    print(f"\nCrawled {len(medium_writeups)} writeups from Medium/RSS")
    
    # Test Dev.to
    print("\n" + "="*60)
    print("Testing Dev.to Crawler")
    print("="*60)
    devto_crawler = DevToCrawler(rate_limit=2.0)
    devto_writeups = devto_crawler.crawl(limit=10)
    print(f"\nCrawled {len(devto_writeups)} writeups from Dev.to")
    
    # Show samples
    print("\n" + "="*60)
    print("Sample Writeups")
    print("="*60)
    
    all_writeups = medium_writeups + devto_writeups
    for i, w in enumerate(all_writeups[:5]):
        print(f"\n{i+1}. [{w['source']}] {w['title']}")
        print(f"   URL: {w['url']}")
        print(f"   Length: {len(w['content'])} chars")


if __name__ == "__main__":
    main()
