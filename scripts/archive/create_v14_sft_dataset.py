import argparse
import json
import os
import re
from typing import Dict, Any, Iterable


def detect_db_type(context: str) -> str:
    ctx = context.lower()
    if "mysql" in ctx:
        return "MySQL"
    if "postgres" in ctx or "postgresql" in ctx:
        return "PostgreSQL"
    if "mssql" in ctx or "sql server" in ctx:
        return "MSSQL"
    if "oracle" in ctx:
        return "Oracle"
    return "Unknown"


def detect_waf_name(context: str) -> str:
    """
    Very rough heuristic based on context text.
    In real runs we may later replace this by wafw00f output.
    """
    ctx = context.lower()
    if "modsecurity" in ctx or "crs" in ctx:
        return "ModSecurity CRS"
    if "cloudflare" in ctx:
        return "Cloudflare"
    if "generic waf" in ctx:
        return "Generic WAF"
    return "Unknown"


def detect_attack_subtype(payload: str) -> str:
    p = payload.lower()
    # Time-based
    if "sleep(" in p or "benchmark(" in p:
        return "time_based"
    # Union-based
    if " union " in p:
        return "union_based"
    # Error-based (rough)
    if "extractvalue(" in p or "updatexml(" in p or "@@version" in p:
        return "error_based"
    # Tautology / boolean-based
    tautology_patterns = [
        " or 1=1",
        "' or '1'='1",
        "\" or \"1\"=\"1",
        " or 1 = 1",
    ]
    if any(t in p for t in tautology_patterns):
        return "tautology"
    # Stacked queries
    if ";--" in p or "; --" in p or ";\n" in p:
        return "stacked"
    return "unknown"


def build_prompt(instruction: str, context: str, constraints: str) -> str:
    parts = []
    if instruction:
        parts.append(instruction)
    if context:
        parts.append(context)
    if constraints:
        parts.append(constraints)
    user_block = "\n".join(parts)
    return f"<|user|>\n{user_block}\n<|assistant|>\n"


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def convert_record_v13_to_v14(rec: Dict[str, Any]) -> Dict[str, Any]:
    instruction = rec.get("instruction", "")
    context = rec.get("context", "")
    constraints = rec.get("constraints", "")
    payload = rec.get("payload", "")
    reasoning = rec.get("reasoning", "")
    attack_type = rec.get("attack_type", "SQLi")

    db_type = detect_db_type(context)
    waf_name = detect_waf_name(context)
    attack_subtype = detect_attack_subtype(payload)

    prompt = build_prompt(instruction, context, constraints)

    # Ensure we end with exactly one payload + <|endoftext|>
    # Strip existing end tokens if present
    clean_payload = str(payload).strip()
    clean_payload = re.sub(r"(<\|endoftext\|>)+$", "", clean_payload).strip()
    chosen = f"{prompt}{clean_payload}<|endoftext|>"

    return {
        "instruction": instruction,
        "context": context,
        "constraints": constraints,
        "payload": clean_payload,
        "reasoning": reasoning,
        "attack_type": attack_type,
        "attack_subtype": attack_subtype,
        "db_type": db_type,
        "waf_name": waf_name,
        "prompt": prompt,
        "chosen": chosen,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert v13 SFT JSONL to v14 schema.")
    ap.add_argument(
        "--input",
        type=str,
        default="data/processed/v13_sft_data.jsonl",
        help="Path to v13 SFT JSONL file.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/processed/v14_sft_data.jsonl",
        help="Output path for v14 JSONL file.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = 0
    subtype_counts: Dict[str, int] = {}

    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(args.input):
            v14 = convert_record_v13_to_v14(rec)
            total += 1
            st = v14.get("attack_subtype", "unknown")
            subtype_counts[st] = subtype_counts.get(st, 0) + 1
            out_f.write(json.dumps(v14, ensure_ascii=False) + "\n")

    print(f"Converted {total} records to {args.output}")
    print("attack_subtype distribution:")
    for k in sorted(subtype_counts.keys()):
        print(f"  {k}: {subtype_counts[k]}")


if __name__ == "__main__":
    main()

