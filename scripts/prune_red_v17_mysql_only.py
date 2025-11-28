import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def looks_mysql_sqli(payload: str) -> bool:
    """
    Heuristic: giữ SQLi phù hợp với MySQL, loại bỏ payload rõ ràng thuộc DB khác.
    """
    p = payload.lower()

    banned_patterns = [
        "pg_sleep",      # PostgreSQL
        "pg_database",
        "sysibm.",       # DB2
        "db2 ",
        "ora_",          # Oracle
        "utl_http",
        "dbms_",         # Oracle packages
        "waitfor",       # MSSQL
        "nvarchar",      # MSSQL-ish
        "sp_",           # MSSQL stored proc prefix
        "bigquery",
    ]
    if any(b in p for b in banned_patterns):
        return False

    # Phải có dấu hiệu là câu SQL, không phải mô tả
    if not any(k in p for k in ("select", "insert", "update", "delete", "sleep(", "benchmark(", "union", " or ", " and ")):
        return False

    # Loại các dòng thuần văn bản/markdown
    if p.startswith("the payload should"):
        return False
    if p.startswith("scenario:") or p.startswith("###"):
        return False

    return True


def looks_xss(payload: str) -> bool:
    """
    Heuristic: giữ XSS thực sự, bỏ các dòng mô tả.
    """
    p = payload.strip()
    pl = p.lower()

    if not p:
        return False

    # Loại mô tả / markdown rõ ràng
    if pl.startswith("the payload should"):
        return False
    if pl.startswith("scenario:") or pl.startswith("###"):
        return False
    if "reflected xss" in pl and "<" not in p and ">" not in p:
        return False

    # Yêu cầu có ít nhất một dấu hiệu XSS
    indicators = [
        "<", ">", "javascript:", "onerror", "onload", "onclick",
        "onmouseover", "onfocus", "src=", "href=", "alert(",
        "document.", "window.", "eval(", "img ", "svg",
    ]
    if not any(ind in pl for ind in indicators) and not any(ch in p for ch in ("<", ">")):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune RED v17 MySQL-only augment set to keep only real SQLi/XSS payloads."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/red_v17_mysql_only.jsonl",
        help="Input RED v17 MySQL-only JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/red_v17_mysql_only_pruned.jsonl",
        help="Output JSONL with pruned augment records.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    kept_sqli = 0
    kept_xss = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in iter_jsonl(in_path):
            total += 1
            atk = (rec.get("attack_type") or "").upper()
            payload = str(rec.get("payload", "")).strip()
            if not payload:
                continue

            if atk == "SQLI":
                if not looks_mysql_sqli(payload):
                    continue
                kept_sqli += 1
            elif atk == "XSS":
                if not looks_xss(payload):
                    continue
                kept_xss += 1
            else:
                continue

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Scanned {total} records from {in_path}")
    print(f"Kept {kept_sqli} SQLi and {kept_xss} XSS augment records.")
    print(f"Wrote {kept} records to {out_path}")


if __name__ == "__main__":
    main()

