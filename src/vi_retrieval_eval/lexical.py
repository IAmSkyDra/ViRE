# src/vi_retrieval_eval/lexical.py
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .tokenization import vi_segment
from .progress import iter_progress

HAVE_BM25 = False
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    HAVE_BM25 = True
except Exception:
    BM25Okapi = object  # type: ignore


# ---------- TF-IDF ----------
def build_tfidf(docs: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vect = TfidfVectorizer(analyzer=vi_segment)
    X_docs = vect.fit_transform(docs)
    return vect, X_docs


def tfidf_scores(vect: TfidfVectorizer, X_docs, queries: List[str]) -> np.ndarray:
    X_q = vect.transform(queries)
    sims = cosine_similarity(X_q, X_docs)
    return sims.astype(np.float32)


# ---------- BM25 Okapi ----------
def build_bm25(
    docs: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> "BM25Okapi":
    """
    Tạo BM25Okapi với tham số Okapi:
      - k1: (thường 1.2 ~ 2.0)
      - b:  (0 ~ 1)
    """
    if not HAVE_BM25:
        raise RuntimeError("BM25 skipped: please `pip install rank_bm25`.")
    tokenized_docs = [vi_segment(d) for d in docs]
    return BM25Okapi(tokenized_docs, k1=k1, b=b)


def bm25_scores(
    bm25: "BM25Okapi",
    queries: List[str],
    show_progress: bool = False,
) -> np.ndarray:
    """
    Tính điểm BM25 cho từng query với progress bar (nếu bật).
    """
    rows: List[np.ndarray] = []
    it = iter_progress(queries, enable=show_progress, tqdm_desc="BM25 scoring", total=len(queries))
    for q in it:
        rows.append(np.array(bm25.get_scores(vi_segment(q)), dtype=np.float32))
    return np.vstack(rows)
