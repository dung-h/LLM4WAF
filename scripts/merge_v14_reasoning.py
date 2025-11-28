import argparse
import json
from typing import Dict, Any, Iterable, Tuple


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


def build_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    # Use (instruction, payload) as a stable key for merge
    return (
        rec.get("instruction", "").strip(),
        rec.get("payload", "").strip(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge DeepSeek reasoning subset back into v14 dataset.")
    ap.add_argument(
        "--base",
        default="data/processed/v14_sft_data.jsonl",
        help="Path to full v14 SFT dataset.",
    )
    ap.add_argument(
        "--subset",
        default="data/processed/v14_sft_reasoning_subset.jsonl",
        help="Path to enriched reasoning subset.",
    )
    ap.add_argument(
        "--output",
        default="data/processed/v14_sft_data_enriched.jsonl",
        help="Output path for merged dataset.",
    )
    args = ap.parse_args()

    # Load enriched reasoning subset into a lookup table
    reasoning_map: Dict[Tuple[str, str], str] = {}
    for rec in iter_jsonl(args.subset):
        key = build_key(rec)
        reasoning = rec.get("reasoning", "")
        if reasoning:
            reasoning_map[key] = reasoning

    updated_count = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(args.base):
            total += 1
            key = build_key(rec)
            if key in reasoning_map:
                rec["reasoning"] = reasoning_map[key]
                updated_count += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Merged reasoning for {updated_count}/{total} records into {args.output}")


if __name__ == "__main__":
    main()

