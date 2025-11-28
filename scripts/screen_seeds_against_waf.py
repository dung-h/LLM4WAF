import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import httpx


DVWA_BASE_URL = "http://localhost:8000"
DVWA_LOGIN_URL = f"{DVWA_BASE_URL}/login.php"
DVWA_SQLI_URL = f"{DVWA_BASE_URL}/vulnerabilities/sqli/"
DVWA_XSS_URL = f"{DVWA_BASE_URL}/vulnerabilities/xss_r/"
USERNAME = "admin"
PASSWORD = "password"


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


def iter_csv_payloads(path: Path, attack_type: str) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = (row.get("payload") or "").strip()
            if not payload:
                continue
            yield {"payload": payload, "attack_type": attack_type}


def dvwa_login() -> httpx.Cookies:
    cookies = httpx.Cookies()
    with httpx.Client(follow_redirects=True, cookies=cookies, timeout=15.0) as client:
        r = client.get(DVWA_LOGIN_URL)
        r.raise_for_status()
        m = re.search(r"user_token' value='([a-f0-9]{32})'", r.text, re.I)
        if not m:
            raise RuntimeError("Failed to find user_token on DVWA login page.")
        token = m.group(1)
        data = {"username": USERNAME, "password": PASSWORD, "user_token": token, "Login": "Login"}
        r = client.post(DVWA_LOGIN_URL, data=data)
        r.raise_for_status()
        if "login.php" in str(r.url):
            raise RuntimeError("DVWA login failed.")
    return cookies


SQL_ERROR_PATTERNS = [
    r"You have an error in your SQL syntax",
    r"Warning: mysql_fetch_array\(\)",
    r"supplied argument is not a valid MySQL result resource",
    r"SQLSTATE\{",
    r"Unclosed quotation mark",
    r"ODBC SQL Server Driver",
    r"Microsoft OLE DB Provider for SQL Server",
    r"Incorrect syntax near",
    r"ORA-\d{5}",
    r"PostgreSQL error",
    r"pg_query\(\) ",
    r"SQLite error",
    r"syntax error at or near",
    r"mysql_num_rows\(\) ",
    r"mysql_fetch_assoc\(\) ",
    r"mysql_result\(\) ",
    r"DB_ERROR",
    r"SQL Error",
    r"Fatal error: Uncaught PDOException",
    r"\{SQLSTATE",
]

DVWA_SUCCESS_MARKERS = [
    "user id exists in the database",
    "first name",
    "surname",
]


def test_sqli(payload: str, client: httpx.Client) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = client.get(DVWA_SQLI_URL, params={"id": payload}, timeout=20.0)
    elapsed = time.perf_counter() - t0

    if r.status_code == 403:
        return {"result": "blocked", "status_code": r.status_code, "elapsed": elapsed}

    text = r.text.lower()
    for pattern in SQL_ERROR_PATTERNS:
        if re.search(pattern.lower(), text):
            return {
                "result": "sql_error_bypass",
                "status_code": r.status_code,
                "elapsed": elapsed,
            }

    if elapsed > 3.0:
        return {"result": "time_delay", "status_code": r.status_code, "elapsed": elapsed}

    for marker in DVWA_SUCCESS_MARKERS:
        if marker in text:
            return {"result": "data_leak", "status_code": r.status_code, "elapsed": elapsed}

    return {"result": "passed", "status_code": r.status_code, "elapsed": elapsed}


def test_xss(payload: str, client: httpx.Client) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = client.get(DVWA_XSS_URL, params={"name": payload}, timeout=20.0)
    elapsed = time.perf_counter() - t0

    if r.status_code == 403:
        return {"result": "blocked", "status_code": r.status_code, "elapsed": elapsed}

    text = r.text
    # Heuristic: xem payload (hoặc HTML-escaped) có xuất hiện trong response không
    if payload in text:
        return {"result": "reflected", "status_code": r.status_code, "elapsed": elapsed}

    return {"result": "passed", "status_code": r.status_code, "elapsed": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen seed payloads against DVWA+WAF and label them with WAF result."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (JSONL with payload/attack_type or CSV with 'payload' column).",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="SQLi",
        help="Attack type for CSV input (SQLi or XSS). Ignored for JSONL if attack_type present.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/seeds_screened_against_waf.jsonl",
        help="Output JSONL file with payload + attack_type + result.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load seeds
    seeds: Iterable[Dict[str, Any]]
    if in_path.suffix.lower() == ".jsonl":
        seeds = iter_jsonl(in_path)
    elif in_path.suffix.lower() == ".csv":
        seeds = iter_csv_payloads(in_path, args.attack_type)
    else:
        raise SystemExit("Unsupported input format. Use .jsonl or .csv")

    print("[Screen] Logging into DVWA...")
    cookies = dvwa_login()
    print("[Screen] Login successful.")

    total = 0
    with httpx.Client(follow_redirects=True, cookies=cookies, timeout=20.0) as client, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for rec in seeds:
            payload = str(rec.get("payload", "")).strip()
            if not payload:
                continue
            attack_type = (rec.get("attack_type") or args.attack_type).upper()

            if attack_type == "SQLI":
                res = test_sqli(payload, client)
            elif attack_type == "XSS":
                res = test_xss(payload, client)
            else:
                continue

            out_rec = {
                "payload": payload,
                "attack_type": attack_type,
                "result": res["result"],
                "status_code": res["status_code"],
                "elapsed": res["elapsed"],
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            total += 1
            if total % 50 == 0:
                print(f"[Screen] Tested {total} seeds...")

    print(f"[Screen] Done. Tested {total} seeds. Results written to {out_path}")


if __name__ == "__main__":
    main()

