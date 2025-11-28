import argparse
import json
from collections import Counter
from pathlib import Path


SQLI_POSITIVE_RESULTS = {"sql_error_bypass", "time_delay", "data_leak"}
XSS_POSITIVE_RESULTS = {"reflected"}


def is_sqli_payload(payload: str) -> bool:
    p = payload.lower()
    keywords = [
        "select ",
        " union ",
        " and ",
        " or ",
        " sleep(",
        " benchmark(",
        "updatexml(",
        "extractvalue(",
        "waitfor delay",
        "information_schema",
    ]
    return any(k in p for k in keywords)


def is_xss_payload(payload: str) -> bool:
    p = payload.lower()
    if "<" in p or ">" in p:
        return True
    keywords = [
        "script",
        "onerror",
        "onload",
        "<svg",
        "<img",
        "<iframe",
        "javascript:",
    ]
    return any(k in p for k in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select high-quality SQLi/XSS seeds from a screened JSONL file "
            "(e.g. seeds_extra_screened.jsonl) produced by screen_seeds_against_waf.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input screened JSONL file (with result/status_code/elapsed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for selected good seeds",
    )
    args = parser.parse_args()

    total = 0
    kept = 0
    result_counter: Counter[str] = Counter()
    kept_result_counter: Counter[str] = Counter()
    seen_payloads: set[str] = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            payload = rec.get("payload", "")
            if not payload:
                continue

            result = str(rec.get("result", "")).lower()
            result_counter[result] += 1

            attack_type_raw = rec.get("attack_type", "")
            attack_type = str(attack_type_raw).lower()

            keep = False

            if attack_type in {"sqli", "sql", "mysql"}:
                # Strong positives from WAF testing
                if result in SQLI_POSITIVE_RESULTS:
                    keep = True
                # Also keep some "passed" if they look like genuine MySQL SQLi
                elif result == "passed" and is_sqli_payload(payload):
                    keep = True
            elif attack_type in {"xss"}:
                if result in XSS_POSITIVE_RESULTS:
                    keep = True
                elif result == "passed" and is_xss_payload(payload):
                    keep = True

            if not keep:
                continue

            # Deduplicate by payload string
            if payload in seen_payloads:
                continue
            seen_payloads.add(payload)

            kept += 1
            kept_result_counter[result] += 1

            out_rec = {
                "payload": payload,
                "attack_type": attack_type,
                "result": result,
                "status_code": rec.get("status_code"),
                "elapsed": rec.get("elapsed"),
            }
            source = rec.get("source")
            if source is not None:
                out_rec["source"] = source

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"Scanned {total} records from {args.input}")
    print(f"Kept {kept} records into {args.output}")
    print("Result distribution (all):", dict(result_counter))
    print("Result distribution (kept):", dict(kept_result_counter))


if __name__ == "__main__":
    main()

