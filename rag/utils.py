from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import quote, unquote_plus
from difflib import SequenceMatcher
from typing import Iterable


def canonicalize(s: str) -> str:
    try:
        s = unquote_plus(s)
    except Exception:
        pass
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s.strip()).lower()
    return s


def sim_ratio(a: str, b: str) -> float:
    return SequenceMatcher(a=canonicalize(a), b=canonicalize(b)).ratio()


def url_encode_partial(s: str, chars: Iterable[str] = ("'", '"', " ", "(", ")")) -> str:
    out = []
    for ch in s:
        if ch in chars:
            out.append(quote(ch))
        else:
            out.append(ch)
    return "".join(out)


def toggle_case_keywords(s: str, kws=("union", "select", "sleep", "benchmark", "or", "and")) -> str:
    def _flip(m):
        w = m.group(0)
        return "".join(ch.lower() if ch.isupper() else ch.upper() for ch in w)
    for kw in kws:
        s = re.sub(rf"(?i)\b{kw}\b", _flip, s)
    return s


def insert_comment_in_keywords(s: str, kws=("union", "select", "sleep", "benchmark")) -> str:
    def _ins(m):
        w = m.group(0)
        return "/**/".join(list(w))
    for kw in kws:
        s = re.sub(rf"(?i)\b{kw}\b", _ins, s)
    return s


def whitespace_inflate(s: str) -> str:
    return re.sub(r"\s", lambda m: m.group(0) * 3, s)


def add_trailing_comment(s: str) -> str:
    if "--" in s or "#" in s:
        return s
    return s + " -- 1"


def simple_mutations(s: str) -> list[str]:
    return [
        url_encode_partial(s),
        toggle_case_keywords(s),
        insert_comment_in_keywords(s),
        whitespace_inflate(s),
        add_trailing_comment(s),
    ]

