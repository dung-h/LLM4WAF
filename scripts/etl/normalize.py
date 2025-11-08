from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import unquote, unquote_plus


def multi_decode(s: str, rounds: int = 2) -> str:
    out = s
    for _ in range(rounds):
        try:
            out = unquote_plus(out)
        except Exception:
            pass
        try:
            out = html.unescape(out)
        except Exception:
            pass
    return out


def normalize_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def canonicalize(s: str) -> str:
    return normalize_ws(multi_decode(s, rounds=2).lower())

