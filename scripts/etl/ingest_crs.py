from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "crs"
PROC = ROOT / "data" / "processed"


SEC_RULE_RE = re.compile(r"^\s*SecRule\b(.*)$")
ID_RE = re.compile(r"id\s*:\s*([0-9]+)")
PHASE_RE = re.compile(r"phase\s*:\s*([0-9]+)")
MSG_RE = re.compile(r"msg\s*:\s*'([^']*)'")
T_RE = re.compile(r"\bt:([a-zA-Z0-9]+)")


def git_clone_or_update(url: str, dest: Path) -> None:
    if dest.exists():
        # Try to pull latest
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def parse_rules(rule_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for conf in sorted((rule_dir).glob("*.conf")):
        with open(conf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not SEC_RULE_RE.search(line):
                    continue
                text = line.strip()
                rid = None
                msg = None
                phase = None
                transforms: List[str] = []
                # Attempt to parse id, msg, phase
                m = ID_RE.search(text)
                if m:
                    rid = m.group(1)
                m = MSG_RE.search(text)
                if m:
                    msg = m.group(1)
                m = PHASE_RE.search(text)
                if m:
                    phase = m.group(1)
                transforms = T_RE.findall(text)

                rows.append({
                    "id": rid,
                    "msg": msg,
                    "phase": phase,
                    "regex": text,
                    "transforms": ",".join(transforms) if transforms else None,
                    "file": conf.name,
                })
    return pd.DataFrame(rows)


def main() -> None:
    git_clone_or_update("https://github.com/coreruleset/coreruleset.git", RAW)
    rules_dir = RAW / "rules"
    if not rules_dir.exists():
        raise SystemExit("CRS rules directory not found after clone.")
    df = parse_rules(rules_dir)
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "crs_rules.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df)} rules")


if __name__ == "__main__":
    main()

