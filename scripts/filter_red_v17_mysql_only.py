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

    # Loại bỏ Postgres / DB2 / MSSQL / Oracle đặc thù
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

    # Đảm bảo trông giống SQLi chứ không phải text thuần
    if not any(k in p for k in ("select", "insert", "update", "delete", "sleep(", "benchmark(", "union", " or ", " and ")):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter RED v17 records to keep only MySQL-friendly SQLi and all XSS."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/red_v17_from_seeds_deepseek_merged.jsonl",
        help="Input RED v17 JSONL file (from DeepSeek seeds).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/red_v17_mysql_only.jsonl",
        help="Output JSONL with filtered records.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: List[Dict[str, Any]] = []
    total = 0
    total_sqli = 0
    total_xss = 0

    for rec in iter_jsonl(in_path):
        total += 1
        attack_type = (rec.get("attack_type") or "").upper()
        payload = str(rec.get("payload", "")).strip()

        if attack_type == "SQLI":
            if not payload:
                continue
            if not looks_mysql_sqli(payload):
                continue
            kept.append(rec)
            total_sqli += 1
        elif attack_type == "XSS":
            kept.append(rec)
            total_xss += 1
        else:
            # Bỏ qua loại khác nếu có
            continue

    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Scanned {total} records from {in_path}")
    print(f"Kept {total_sqli} SQLi (MySQL-friendly) and {total_xss} XSS records.")
    print(f"Wrote {len(kept)} records to {out_path}")


if __name__ == "__main__":
    main()

