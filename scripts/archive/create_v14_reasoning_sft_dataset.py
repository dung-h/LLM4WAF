import argparse
import json
from typing import Dict, Any, Iterable


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


def build_reasoning_prompt(rec: Dict[str, Any]) -> str:
    instruction = rec.get("instruction", "")
    context = rec.get("context", "")
    constraints = rec.get("constraints", "")
    payload = rec.get("payload", "")
    attack_subtype = rec.get("attack_subtype", "unknown")
    db_type = rec.get("db_type", "Unknown")
    waf_name = rec.get("waf_name", "Unknown")

    user_lines = [
        "You are a security analyst. Given the following SQL injection payload and context,",
        "explain in 3-5 concise bullet points what the payload does, why it can be effective",
        "against the target (and WAF if relevant), and any notable evasion techniques.",
        "",
        f"Original generation instruction: {instruction}",
        f"Context: {context}",
        f"Constraints: {constraints}",
        f"Database type: {db_type}",
        f"WAF name: {waf_name}",
        f"Attack subtype: {attack_subtype}",
        f"Payload: {payload}",
    ]
    user_block = "\n".join(user_lines)
    return f"<|user|>\n{user_block}\n<|assistant|>\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create reasoning-only SFT dataset from v14 reasoning subset."
    )
    ap.add_argument(
        "--input",
        default="data/processed/v14_sft_reasoning_subset.jsonl",
        help="Reasoning-enriched subset JSONL.",
    )
    ap.add_argument(
        "--output",
        default="data/processed/v14_reasoning_sft_data.jsonl",
        help="Output JSONL with `chosen` field for reasoning SFT.",
    )
    args = ap.parse_args()

    count_in = 0
    count_out = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(args.input):
            count_in += 1
            reasoning = str(rec.get("reasoning", "")).strip()
            if not reasoning:
                continue
            prompt = build_reasoning_prompt(rec)
            chosen = f"{prompt}{reasoning}<|endoftext|>"
            out_rec = dict(rec)
            out_rec["prompt_reasoning"] = prompt
            out_rec["chosen"] = chosen
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            count_out += 1

    print(
        f"Processed {count_in} records, wrote {count_out} reasoning SFT examples to {args.output}"
    )


if __name__ == "__main__":
    main()

