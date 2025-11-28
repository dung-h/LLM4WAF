"""
Security Researcher Blogs Crawler
Crawl personal blogs from top security researchers (2020-2022 focus)
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

from .base import BaseCrawler, WriteupValidator


class SecurityResearchersCrawler(BaseCrawler):
    """Crawl security researcher blogs with focus on 2020-2022 XSS/SQLi content"""
    
    # Top security researchers known for web exploitation
    RESEARCHERS = {
        # XSS Specialists
        'brutelogic': {
            'name': 'Brute Logic (Rodolfo Assis)',
            'blog': 'https://brutelogic.com.br/blog/',
            'rss': 'https://brutelogic.com.br/blog/feed/',
            'focus': 'xss',
            'quality': 5  # 1-5 rating
        },
        'terjanq': {
            'name': 'terjanq',
            'blog': 'https://blog.terjanq.me/',
            'rss': 'https://blog.terjanq.me/feed.xml',
            'focus': 'xss',
            'quality': 5
        },
        'garethheyes': {
            'name': 'Gareth Heyes',
            'blog': 'https://www.garethheyes.co.uk/blog/',
            'rss': None,  # No RSS, need HTML parsing
            'focus': 'xss',
            'quality': 5
        },
        
        # SQLi Specialists
        'sqlmap': {
            'name': 'SQLMap Development',
            'blog': 'http://www.sqlmap.org/',
            'rss': None,
            'focus': 'sqli',
            'quality': 4
        },
        
        # General Web Security (with XSS/SQLi content)
        'portswigger_james': {
            'name': 'James Kettle (PortSwigger)',
            'blog': 'https://portswigger.net/research',
            'rss': 'https://portswigger.net/research/rss',
            'focus': 'web',
            'quality': 5
        },
        'orange_tsai': {
            'name': 'Orange Tsai',
            'blog': 'https://blog.orange.tw/',
            'rss': 'https://blog.orange.tw/feeds/posts/default',
            'focus': 'web',
            'quality': 5
        },
        'filedescriptor': {
            'name': 'filedescriptor',
            'blog': 'https://blog.innerht.ml/',
            'rss': 'https://blog.innerht.ml/feeds/posts/default',
            'focus': 'xss',
            'quality': 5
        },
        'liveoverflow': {
            'name': 'LiveOverflow',
            'blog': 'https://liveoverflow.com/blog/',
            'rss': 'https://liveoverflow.com/blog/index.xml',
            'focus': 'web',
            'quality': 4
        },
        'albinowax': {
            'name': 'albinowax (James Kettle)',
            'blog': 'https://skeletonscribe.net/',
            'rss': 'https://skeletonscribe.net/feed.xml',
            'focus': 'web',
            'quality': 5
        },
        'meder': {
            'name': 'Meder Kydyraliev',
            'blog': 'https://www.meder.ws/',
            'rss': None,
            'focus': 'web',
            'quality': 4
        },
        
        # Bug Bounty Hunters (active 2020-2022)
        'nahamsec': {
            'name': 'Ben Sadeghipour',
            'blog': 'https://nahamsec.com/',
            'rss': 'https://nahamsec.com/feed/',
            'focus': 'web',
            'quality': 4
        },
        'zseano': {
            'name': 'Sean',
            'blog': 'https://blog.zseano.com/',
            'rss': 'https://blog.zseano.com/feed',
            'focus': 'web',
            'quality': 4
        },
        'stok': {
            'name': 'Fredrik Alexandersson',
            'blog': 'https://www.stokfredrik.com/',
            'rss': 'https://www.stokfredrik.com/feed',
            'focus': 'web',
            'quality': 4
        },
        'jhaddix': {
            'name': 'Jason Haddix',
            'blog': 'https://jhaddix.com/',
            'rss': 'https://jhaddix.com/feed/',
            'focus': 'web',
            'quality': 4
        },
        'ron_chan': {
            'name': 'Ron Chan',
            'blog': 'https://ngailong.wordpress.com/',
            'rss': 'https://ngailong.wordpress.com/feed/',
            'focus': 'xss',
            'quality': 4
        },
        'edward_z': {
            'name': 'EdOverflow',
            'blog': 'https://edoverflow.com/',
            'rss': None,
            'focus': 'web',
            'quality': 4
        },
        
        # CTF Players with writeups
        'liveoverflow_ctf': {
            'name': 'LiveOverflow CTF',
            'blog': 'https://liveoverflow.com/tag/ctf/',
            'rss': None,
            'focus': 'web',
            'quality': 4
        },
        'hxp': {
            'name': 'hxp Security',
            'blog': 'https://hxp.io/blog/',
            'rss': 'https://hxp.io/feed.xml',
            'focus': 'web',
            'quality': 5
        },
        
        # Platform Blogs
        'hackerone_hacktivity': {
            'name': 'HackerOne Hacktivity',
            'blog': 'https://hackerone.com/hacktivity',
            'rss': None,
            'focus': 'web',
            'quality': 4
        },
        'bugcrowd': {
            'name': 'Bugcrowd Blog',
            'blog': 'https://www.bugcrowd.com/blog/',
            'rss': 'https://www.bugcrowd.com/feed/',
            'focus': 'web',
            'quality': 3
        },
        
        # Security Companies
        'acunetix': {
            'name': 'Acunetix Blog',
            'blog': 'https://www.acunetix.com/blog/',
            'rss': 'https://www.acunetix.com/blog/feed/',
            'focus': 'web',
            'quality': 3
        },
        'netsparker': {
            'name': 'Netsparker Blog',
            'blog': 'https://www.netsparker.com/blog/',
            'rss': 'https://www.netsparker.com/blog/feed/',
            'focus': 'web',
            'quality': 3
        },
    }
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
        
        # Focus on 2020-2022
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2022, 12, 31)
    
    def crawl(self, limit: int = 50) -> List[Dict]:
        """Crawl security researcher blogs"""
        print(f"\n🔬 Crawling {len(self.RESEARCHERS)} security researcher blogs...")
        print(f"📅 Focus: {self.start_date.year}-{self.end_date.year}")
        
        all_writeups = []
        
        # Sort by quality rating
        sorted_researchers = sorted(
            self.RESEARCHERS.items(),
            key=lambda x: x[1]['quality'],
            reverse=True
        )
        
        for researcher_id, info in sorted_researchers:
            if len(all_writeups) >= limit:
                break
            
            print(f"\n👤 {info['name']} ({info['focus'].upper()}) [Quality: {info['quality']}/5]")
            
            try:
                if info['rss']:
                    writeups = self._crawl_rss(researcher_id, info)
                else:
                    writeups = self._crawl_html(researcher_id, info)
                
                print(f"   Found: {len(writeups)} posts in date range")
                all_writeups.extend(writeups)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
            
            time.sleep(self.rate_limit)
        
        print(f"\n✅ Total collected: {len(all_writeups)} posts")
        return all_writeups[:limit]
    
    def _crawl_rss(self, researcher_id: str, info: Dict) -> List[Dict]:
        """Crawl blog via RSS feed"""
        writeups = []
        
        feed = feedparser.parse(info['rss'])
        
        for entry in feed.entries:
            # Parse date
            if hasattr(entry, 'published_parsed'):
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed'):
                pub_date = datetime(*entry.updated_parsed[:6])
            else:
                continue
            
            # Filter by date range
            if not (self.start_date <= pub_date <= self.end_date):
                continue
            
            # Get content
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            # Create writeup
            writeup = {
                'title': entry.get('title', 'No Title'),
                'url': entry.get('link', ''),
                'date': pub_date.strftime('%Y-%m-%d'),
                'content': self._clean_html(content),
                'author': info['name'],
                'source': f"researcher_{researcher_id}",
                'tags': [info['focus'], 'blog', str(pub_date.year)]
            }
            
            # Validate
            is_valid, reason = self.validator.validate(writeup)
            if is_valid:
                writeups.append(writeup)
                print(f"   ✓ {writeup['title'][:50]}... ({pub_date.year})")
            else:
                print(f"   ⊘ {writeup['title'][:50]}... - {reason}")
        
        return writeups
    
    def _crawl_html(self, researcher_id: str, info: Dict) -> List[Dict]:
        """Crawl blog via HTML parsing (when no RSS)"""
        writeups = []
        
        try:
            response = requests.get(info['blog'], timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Generic blog post detection
            articles = soup.find_all(['article', 'div'], class_=lambda x: x and any(
                term in str(x).lower() for term in ['post', 'entry', 'article', 'blog']
            ))
            
            if not articles:
                # Fallback: find all links
                articles = soup.find_all('a', href=True)
            
            for article in articles[:20]:  # Limit to avoid too many requests
                try:
                    # Extract title and URL
                    if article.name == 'a':
                        title = article.get_text(strip=True)
                        url = article['href']
                    else:
                        title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                        if not title_elem:
                            continue
                        title = title_elem.get_text(strip=True)
                        
                        link_elem = article.find('a', href=True)
                        if not link_elem:
                            continue
                        url = link_elem['href']
                    
                    # Make absolute URL
                    if url.startswith('/'):
                        from urllib.parse import urljoin
                        url = urljoin(info['blog'], url)
                    
                    # Try to extract date
                    date_elem = article.find(['time', 'span'], class_=lambda x: x and 'date' in str(x).lower())
                    if date_elem:
                        date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                        try:
                            # Try multiple date formats
                            for fmt in ['%Y-%m-%d', '%B %d, %Y', '%d %B %Y', '%Y/%m/%d']:
                                try:
                                    pub_date = datetime.strptime(date_str, fmt)
                                    break
                                except:
                                    continue
                            else:
                                pub_date = datetime.now()  # Fallback
                        except:
                            pub_date = datetime.now()
                    else:
                        pub_date = datetime.now()
                    
                    # Filter by date range
                    if not (self.start_date <= pub_date <= self.end_date):
                        continue
                    
                    # Fetch full article
                    time.sleep(self.rate_limit)
                    article_response = requests.get(url, timeout=10)
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract content
                    content_elem = article_soup.find(['article', 'div'], class_=lambda x: x and any(
                        term in str(x).lower() for term in ['content', 'post', 'entry']
                    ))
                    
                    content = content_elem.get_text(separator='\n', strip=True) if content_elem else ""
                    
                    writeup = {
                        'title': title,
                        'url': url,
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'content': content,
                        'author': info['name'],
                        'source': f"researcher_{researcher_id}",
                        'tags': [info['focus'], 'blog', str(pub_date.year)]
                    }
                    
                    # Validate
                    is_valid, reason = self.validator.validate(writeup)
                    if is_valid:
                        writeups.append(writeup)
                        print(f"   ✓ {title[:50]}... ({pub_date.year})")
                    else:
                        print(f"   ⊘ {title[:50]}... - {reason}")
                
                except Exception as e:
                    print(f"   ⚠ Article error: {e}")
                    continue
        
        except Exception as e:
            print(f"   ❌ Blog fetch error: {e}")
        
        return writeups


class HistoricalCTFCrawler(BaseCrawler):
    """Crawl historical CTF writeups from 2020-2022"""
    
    # Known CTF writeup repositories
    CTF_REPOS = [
        {
            'name': 'CTFtime 2020 Archive',
            'base_url': 'https://ctftime.org/event/list/',
            'year': 2020,
            'quality': 4
        },
        {
            'name': 'CTFtime 2021 Archive',
            'base_url': 'https://ctftime.org/event/list/',
            'year': 2021,
            'quality': 4
        },
        {
            'name': 'CTFtime 2022 Archive',
            'base_url': 'https://ctftime.org/event/list/',
            'year': 2022,
            'quality': 4
        }
    ]
    
    # Top CTF teams with public writeups
    CTF_TEAMS = {
        'perfect_blue': 'https://perfect.blue/writeups.html',
        'organizers': 'https://organizers.github.io/',
        'ctf_kosenctf': 'https://kosenctf.com/writeups/',
        'dragonsector': 'https://blog.dragonsector.pl/',
        'sekai': 'https://sekaictf.com/writeups/',
    }
    
    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit)
    
    def crawl(self, limit: int = 50) -> List[Dict]:
        """Crawl historical CTF writeups from 2020-2022"""
        print(f"\n🏆 Crawling historical CTF writeups (2020-2022)...")
        
        all_writeups = []
        
        # Method 1: CTFtime archive with date filtering
        for year in [2020, 2021, 2022]:
            if len(all_writeups) >= limit:
                break
            
            print(f"\n📅 Year {year}")
            
            # Use existing CTFtime crawler with year parameter
            from .ctftime import CTFtimeCrawler
            ctf_crawler = CTFtimeCrawler(rate_limit=self.rate_limit)
            
            # Crawl with year filter
            year_writeups = self._crawl_year(ctf_crawler, year, limit // 3)
            all_writeups.extend(year_writeups)
            
            print(f"   Collected: {len(year_writeups)} writeups")
        
        print(f"\n✅ Total historical writeups: {len(all_writeups)}")
        return all_writeups[:limit]
    
    def _crawl_year(self, crawler, year: int, limit: int) -> List[Dict]:
        """Crawl CTFtime writeups for specific year"""
        writeups = []
        
        # Construct year-specific URL
        url = f"https://ctftime.org/writeups?year={year}"
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find writeup links (similar to CTFtime crawler)
            writeup_links = soup.find_all('a', href=lambda x: x and '/writeup/' in str(x))
            
            for link in writeup_links[:limit]:
                try:
                    writeup_url = f"https://ctftime.org{link['href']}"
                    title = link.get_text(strip=True)
                    
                    # Fetch writeup content
                    time.sleep(self.rate_limit)
                    writeup_response = requests.get(writeup_url, timeout=10)
                    writeup_soup = BeautifulSoup(writeup_response.content, 'html.parser')
                    
                    content_div = writeup_soup.find('div', id='writeup')
                    if not content_div:
                        continue
                    
                    content = content_div.get_text(separator='\n', strip=True)
                    
                    writeup = {
                        'title': title,
                        'url': writeup_url,
                        'date': f"{year}-01-01",  # Approximate
                        'content': content,
                        'source': f'ctftime_{year}',
                        'tags': ['ctf', 'web', str(year)]
                    }
                    
                    is_valid, reason = self.validator.validate(writeup)
                    if is_valid:
                        writeups.append(writeup)
                        print(f"   ✓ {title[:50]}...")
                
                except Exception as e:
                    print(f"   ⚠ Error: {e}")
                    continue
        
        except Exception as e:
            print(f"   ❌ Year {year} crawl error: {e}")
        
        return writeups


# Export crawlers
__all__ = ['SecurityResearchersCrawler', 'HistoricalCTFCrawler']
