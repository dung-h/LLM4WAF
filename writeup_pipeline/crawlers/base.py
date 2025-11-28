"""Base crawler class for all writeup sources."""

from abc import ABC, abstractmethod
from typing import List, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)


class WriteupValidator:
    """Validate writeup content for XSS and SQLi ONLY."""
    
    # XSS-specific keywords
    XSS_KEYWORDS = [
        'xss', 'cross-site scripting', 'cross site scripting',
        '<script>', 'javascript:', 'onerror=', 'onload=', 'onclick=',
        '<svg', '<img', '<iframe', 'alert(', 'prompt(', 'confirm(',
        'document.cookie', 'eval(', 'innerhtml', 'outerhtml',
        'dom-based', 'reflected xss', 'stored xss', 'dom xss',
        'self-xss', 'mutation xss'
    ]
    
    # SQLi-specific keywords
    SQLI_KEYWORDS = [
        'sqli', 'sql injection', 'sql-injection',
        'union select', "' or 1=1", "or '1'='1", '-- -', '#',
        'information_schema', 'sleep(', 'benchmark(',
        'extractvalue', 'updatexml', 'concat(',
        'blind sql', 'boolean-based', 'time-based',
        'error-based sql', 'union-based', 'stacked queries'
    ]
    
    # SKIP ONLY non-injection challenges (keep SSTI, XXE - they have payloads)
    SKIP_KEYWORDS = [
        'shellcode', 'rop', 'pwn', 'buffer overflow', 'heap',
        'reverse engineering', 'binary', 'forensics', 'crypto',
        'brainfuck', 'esoteric', 'steganography', 'stego',
        'osint', 'open source intelligence', 'geoint',
        'ai challenge', 'machine learning', 'ml model',
        'coding challenge', 'algorithm', 'programming puzzle',
        'networking', 'packet', 'pcap'
    ]
    
    def validate(self, writeup: Dict) -> tuple[bool, str]:
        """Validate if writeup contains XSS or SQLi."""
        content = writeup.get('content', '').lower()
        title = writeup.get('title', '').lower()
        
        # Check length
        if len(content) < 1000:
            return False, "Too short (< 1000 chars)"
        
        if len(content) > 500000:
            return False, "Too long (> 500KB)"
        
        # Check for XSS
        has_xss = any(kw in content for kw in self.XSS_KEYWORDS)
        
        # Check for SQLi
        has_sqli = any(kw in content for kw in self.SQLI_KEYWORDS)
        
        # Must have XSS OR SQLi
        if not (has_xss or has_sqli):
            return False, "No XSS/SQLi keywords"
        
        # IMPROVED: Only skip if has SKIP keywords but NO XSS/SQLi
        # If it has both XSS/SQLi AND skip keywords, prioritize XSS/SQLi
        has_skip = any(kw in content or kw in title for kw in self.SKIP_KEYWORDS)
        
        if has_skip and not (has_xss or has_sqli):
            # Only non-web content
            skip_found = [kw for kw in self.SKIP_KEYWORDS if kw in content or kw in title]
            return False, f"Not XSS/SQLi ({skip_found[0]})"
        
        # If has XSS/SQLi, accept even if also has other attack types

            return False, "No XSS/SQLi keywords"
        
        # Check for payloads
        has_code = '```' in writeup.get('content', '')
        has_payloads = '<' in content or "'" in content or '"' in content
        
        if not (has_code or has_payloads):
            return False, "No code/payloads"
        
        attack_types = []
        if has_xss:
            attack_types.append('XSS')
        if has_sqli:
            attack_types.append('SQLi')
        
        return True, f"Valid ({', '.join(attack_types)})"


class ContentDeduplicator:
    """Track and remove duplicate content."""
    
    def __init__(self):
        self.seen_hashes = set()
    
    def get_hash(self, content: str) -> str:
        """Get content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate(self, content: str) -> bool:
        """Check if content is duplicate."""
        content_hash = self.get_hash(content)
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False


class BaseCrawler(ABC):
    """Abstract base class for all crawlers."""
    
    def __init__(self, rate_limit: float = 2.0):
        self.rate_limit = rate_limit
        self.validator = WriteupValidator()
        self.dedup = ContentDeduplicator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def crawl(self, limit: int) -> List[Dict]:
        """
        Crawl writeups from source.
        
        Args:
            limit: Maximum number of writeups to crawl
            
        Returns:
            List of writeup dictionaries
        """
        pass
    
    def _log_progress(self, current: int, total: int, title: str):
        """Log crawling progress."""
        self.logger.info(f"   ✅ [{current}/{total}] {title[:50]}...")
