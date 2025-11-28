"""Writeup Payload Extraction Pipeline - Crawlers Module"""

from .base import BaseCrawler, WriteupValidator, ContentDeduplicator
from .ctftime import CTFtimeCrawler
from .medium import MediumRSSCrawler, DevToCrawler
from .github import GitHubWriteupCrawler, GitHubRepoWriteupCrawler
from .security_blogs import (
    HackerOneCrawler,
    PortSwiggerCrawler,
    PentesterLabCrawler,
    BugBountyBlogsCrawler
)
from .security_researchers import (
    SecurityResearchersCrawler,
    HistoricalCTFCrawler
)
from .specialized import (
    PortSwiggerResearchCrawler,
    OrangeTsaiGitHubCrawler,
    WaybackMachineCrawler,
    CTFtimeHistoricalCrawler,
)

__all__ = [
    'BaseCrawler', 
    'WriteupValidator', 
    'ContentDeduplicator',
    'CTFtimeCrawler',
    'MediumRSSCrawler',
    'DevToCrawler',
    'GitHubWriteupCrawler',
    'GitHubRepoWriteupCrawler',
    'HackerOneCrawler',
    'PortSwiggerCrawler',
    'PentesterLabCrawler',
    'BugBountyBlogsCrawler',
    'SecurityResearchersCrawler',
    'HistoricalCTFCrawler',
    # Specialized crawlers (site-specific strategies)
    'PortSwiggerResearchCrawler',
    'OrangeTsaiGitHubCrawler',
    'WaybackMachineCrawler',
    'CTFtimeHistoricalCrawler',
]
