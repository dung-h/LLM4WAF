from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Iterable, Tuple, Dict, Any


def text_hash(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return h[:10]


def preview(text: str, n: int = 120) -> str:
    t = text.replace("\n", " ")
    return t[:n]


@contextmanager
def measure_time() -> float:
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        globals()["_last_elapsed_ms"] = (end - start) * 1000.0


def last_elapsed_ms(default: float | None = None) -> float | None:
    return globals().get("_last_elapsed_ms", default)


def grad_global_norm(parameters: Iterable) -> float:
    import math
    import torch

    total = 0.0
    for p in parameters:
        if getattr(p, "grad", None) is not None:
            g = p.grad
            total += float(torch.norm(g, 2).item() ** 2)
    return float(math.sqrt(total)) if total > 0 else 0.0


def cuda_mem_stats() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda_alloc": int(torch.cuda.memory_allocated()),
                "cuda_alloc_max": int(torch.cuda.max_memory_allocated()),
            }
    except Exception:
        pass
    return {"cuda_alloc": 0, "cuda_alloc_max": 0}

