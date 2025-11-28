"""
Writeup Payload Extraction Pipeline

Customized pipeline inspired by CyberLLMInstruct for extracting
WAF bypass payloads from CTF writeups.
"""

__version__ = "1.0.0"
__author__ = "LLM4WAF Team"

from . import crawlers
from . import extractors
from . import validators

__all__ = ['crawlers', 'extractors', 'validators']
