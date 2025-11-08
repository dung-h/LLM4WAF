from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def _default_preprocess(x: str) -> str:
    return x if isinstance(x, str) else str(x)


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any]


class TFIDFRetriever:
    def __init__(self, analyzer: str = "char_wb", ngram_range: Tuple[int, int] = (3, 5)):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=True,
            min_df=1,
        )
        self.doc_ids: List[str] = []
        self.doc_meta: List[Dict[str, Any]] = []
        self.mat = None

    def build(self, docs: List[Doc]):
        texts = [_default_preprocess(d.text) for d in docs]
        self.doc_ids = [d.id for d in docs]
        self.doc_meta = [d.meta for d in docs]
        self.mat = self.vectorizer.fit_transform(texts)
        return self

    def query(self, query_text: str, top_k: int = 10, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if self.mat is None:
            raise RuntimeError("Retriever not built or loaded.")
        q = self.vectorizer.transform([_default_preprocess(query_text)])
        sims = cosine_similarity(q, self.mat).ravel()

        indices = np.argsort(-sims)
        results: List[Dict[str, Any]] = []
        for idx in indices:
            item = {
                "id": self.doc_ids[int(idx)],
                "score": float(sims[int(idx)]),
                "meta": self.doc_meta[int(idx)],
            }
            if filters:
                ok = True
                for k, v in filters.items():
                    if item["meta"].get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str):
        joblib.dump({
            "vectorizer": self.vectorizer,
            "doc_ids": self.doc_ids,
            "doc_meta": self.doc_meta,
            "mat": self.mat,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TFIDFRetriever":
        obj = joblib.load(path)
        inst = cls()
        inst.vectorizer = obj["vectorizer"]
        inst.doc_ids = obj["doc_ids"]
        inst.doc_meta = obj["doc_meta"]
        inst.mat = obj["mat"]
        return inst

