import argparse
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.index import build_payload_index, build_crs_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", action="store_true", help="Build payload TF-IDF index")
    parser.add_argument("--crs", type=str, default="", help="Path to CRS rules parquet to index")
    args = parser.parse_args()

    if args.payload:
        p = build_payload_index()
        print(f"Payload index: {p}")

    if args.crs:
        crs_parquet = Path(args.crs)
        if not crs_parquet.exists():
            raise SystemExit(f"CRS parquet not found: {crs_parquet}")
        p = build_crs_index(crs_parquet)
        print(f"CRS index: {p}")


if __name__ == "__main__":
    main()
