import argparse
import sys
from typing import Dict
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rag.retrievers import TFIDFRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to payload TF-IDF joblib")
    parser.add_argument("--context", default="DB=MySQL; sink=query; method=GET; CRS=PL2", help="Target context string")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    retr = TFIDFRetriever.load(args.index)
    res = retr.query(args.context, top_k=args.top_k)
    for r in res:
        meta = r.get("meta", {})
        print(f"score={r['score']:.3f}\tsource={meta.get('source','')}\tcat={meta.get('category','')}\tid={r['id']}")


if __name__ == "__main__":
    main()
