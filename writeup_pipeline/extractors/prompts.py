"""Prompt templates for LLM-based payload extraction."""

PROMPT_V1_BASIC = """You are a security researcher analyzing CTF writeups.

TASK: Extract all payloads that successfully bypassed WAF or security filters.

WRITEUP:
{writeup}

OUTPUT FORMAT (JSON):
[{{"payload": "...", "attack_type": "...", "bypass_reason": "..."}}]
"""


PROMPT_V2_STRICT = """You are a security researcher analyzing CTF writeups and bug bounty reports.

TASK: Extract ONLY payloads that successfully bypassed WAF/security filters.

STRICT EXTRACTION RULES:
1. ONLY extract if explicitly marked as: "PASSED", "SUCCESS", "WORKED", "BYPASSED", "✓"
2. ONLY HTML/JavaScript payloads (XSS) or SQL injection payloads
3. NO URLs, NO code snippets, NO CTF flags, NO incomplete examples
4. Preserve EXACT payload format with all special characters
5. Must explain the specific bypass technique used

VALID PAYLOAD EXAMPLES:
✅ "<svg/onload=alert(1)>" - XSS payload
✅ "' OR '1'='1" - SQL injection
✅ "<img src=x onerror=alert(document.cookie)>" - XSS with event handler
✅ "admin'--" - SQL comment injection

INVALID EXAMPLES (DO NOT EXTRACT):
❌ "https://example.com/xss?payload=..." - URL
❌ "function exploit() {{ ... }}" - Code snippet
❌ "flag{{CTF_FLAG_HERE}}" - CTF flag
❌ "import requests; ..." - Python code
❌ "curl -X POST ..." - Shell command

WRITEUP CONTENT:
{writeup}

OUTPUT FORMAT (JSON array):
[
  {{
    "payload": "exact payload string with all special chars",
    "attack_type": "xss|sqli",
    "bypass_technique": "specific technique used (e.g., null byte, case variation, encoding)",
    "waf_bypassed": "name of WAF if mentioned (e.g., ModSecurity, Cloudflare)",
    "context": "brief description of where/how it was used"
  }}
]

IMPORTANT: 
- Return empty array [] if no valid payloads found
- Each payload must be standalone executable code
- Do not include explanatory text, only valid JSON

Extract all successful bypass payloads now:
"""


PROMPT_V3_FEWSHOT = """You are an expert security researcher specializing in WAF bypass techniques.

TASK: Extract payloads that successfully bypassed security filters from this CTF writeup.

EXTRACTION EXAMPLES:

Example 1 - XSS Bypass:
Writeup: "After testing various payloads, <svg/onload=alert(1)> WORKED and bypassed ModSecurity."
Output: [{{"payload": "<svg/onload=alert(1)>", "attack_type": "xss", "bypass_technique": "self-closing SVG tag", "waf_bypassed": "ModSecurity", "context": "Reflected XSS in search parameter"}}]

Example 2 - SQL Injection:
Writeup: "The payload ' OR '1'='1 PASSED the filter and returned all records."
Output: [{{"payload": "' OR '1'='1", "attack_type": "sqli", "bypass_technique": "tautology", "waf_bypassed": "unknown", "context": "Login bypass"}}]

Example 3 - No Valid Payloads:
Writeup: "We tried several approaches but nothing worked. The challenge was eventually solved using a different technique."
Output: []

NOW EXTRACT FROM THIS WRITEUP:
{writeup}

OUTPUT (JSON array only):
"""


def get_prompt(version: str = "v2_strict") -> str:
    """
    Get prompt template by version.
    
    Args:
        version: Prompt version ('v1_basic', 'v2_strict', 'v3_fewshot')
        
    Returns:
        Prompt template string
    """
    prompts = {
        'v1_basic': PROMPT_V1_BASIC,
        'v2_strict': PROMPT_V2_STRICT,
        'v3_fewshot': PROMPT_V3_FEWSHOT
    }
    
    return prompts.get(version, PROMPT_V2_STRICT)
