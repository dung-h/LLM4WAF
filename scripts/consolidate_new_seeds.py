import json
from pathlib import Path
import argparse

def main() -> None:
    """
    Consolidates multiple JSONL seed files into a single file with unique payloads.
    """
    parser = argparse.ArgumentParser(description="Consolidate multiple JSONL seed files.")
    parser.add_argument(
        "--input-files",
        nargs='+',
        default=[
            "data/processed/seeds_from_payloadbox_sqli.jsonl",
            "data/processed/seeds_from_payloadbox_xss.jsonl",
            "data/processed/seeds_from_humblelad_xss.jsonl",
        ],
        help="List of input JSONL seed files to consolidate.",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/seeds_newly_extracted_consolidated.jsonl",
        help="Output consolidated JSONL file.",
    )
    args = parser.parse_args()

    seen_payloads = set()
    total_records_written = 0
    output_path = Path(args.output_file)

    print("Starting consolidation...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for input_str in args.input_files:
            input_path = Path(input_str)
            if not input_path.exists():
                print(f"  [WARN] Input file not found, skipping: {input_path}")
                continue
            
            print(f"  - Processing {input_path}...")
            with input_path.open("r", encoding="utf-8", errors="ignore") as in_f:
                for line in in_f:
                    try:
                        record = json.loads(line)
                        payload = record.get("payload")

                        if not payload or payload in seen_payloads:
                            continue

                        seen_payloads.add(payload)
                        out_f.write(line) # Write the original line to preserve all fields
                        total_records_written += 1
                    except (json.JSONDecodeError, AttributeError):
                        continue # Skip malformed lines

    print(f"\nConsolidation complete.")
    print(f"Wrote {total_records_written} unique records to {output_path}")

if __name__ == "__main__":
    main()
