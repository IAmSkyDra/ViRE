# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np

from .tokenization import vi_segment


def _summarize(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0, "p95": 0, "p99": 0}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _lengths_char(texts: List[str]) -> np.ndarray:
    return np.array([len(t or "") for t in texts], dtype=np.int32)


def _lengths_token(texts: List[str]) -> np.ndarray:
    return np.array([len(vi_segment(t or "")) for t in texts], dtype=np.int32)


def compute_dataset_stats(questions: List[str], contexts: List[str]) -> Dict[str, object]:
    """
    Thống kê dựa trên TẬP THỰC SỰ đem đi eval (sau sampling/dedup/lower nếu có).
    - Ký tự & token cho question / context
    - Đếm số context duy nhất (raw)
    """
    q_len_char = _lengths_char(questions)
    q_len_tok  = _lengths_token(questions)
    c_len_char = _lengths_char(contexts)
    c_len_tok  = _lengths_token(contexts)

    num_rows = len(questions)
    num_docs = len(contexts)
    num_unique_raw = len(set(contexts))

    return {
        "counts": {
            "num_rows": int(num_rows),
            "num_docs": int(num_docs),
            "num_unique_contexts": int(num_unique_raw),
            "dup_contexts": int(num_docs - num_unique_raw),
        },
        "question_len_char": _summarize(q_len_char),
        "question_len_token": _summarize(q_len_tok),
        "context_len_char": _summarize(c_len_char),
        "context_len_token": _summarize(c_len_tok),
    }
