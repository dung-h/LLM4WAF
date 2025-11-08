from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "data" / "raw" / "seeds"


def git_clone_or_update(url: str, dest: Path) -> None:
    if dest.exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def collect_lines_txt(root: Path) -> List[str]:
    lines: List[str] = []
    for p in root.rglob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    lines.append(s)
        except Exception:
            continue
    return lines


def main() -> None:
    SEEDS.mkdir(parents=True, exist_ok=True)

    xss_repo = SEEDS / "xss-payload-list"
    sqli_repo = SEEDS / "sql-injection-payload-list"

    git_clone_or_update("https://github.com/payloadbox/xss-payload-list.git", xss_repo)
    git_clone_or_update("https://github.com/payloadbox/sql-injection-payload-list.git", sqli_repo)

    xss_lines = collect_lines_txt(xss_repo)
    sqli_lines = collect_lines_txt(sqli_repo)

    # Write unified seed files expected by our indexer
    out_xss = ROOT / "data" / "raw" / "seed_xss.csv"
    out_sqli = ROOT / "data" / "raw" / "seed_sqli.csv"
    if xss_lines:
        pd.DataFrame({"payload": xss_lines}).drop_duplicates().to_csv(out_xss, index=False)
    if sqli_lines:
        pd.DataFrame({"payload": sqli_lines}).drop_duplicates().to_csv(out_sqli, index=False)
    print(f"Wrote seeds: XSS={len(xss_lines)} SQLi={len(sqli_lines)}")


if __name__ == "__main__":
    main()

