from typing import List, Dict, Optional, Any
import os
import csv
import numpy as np

from .io_utils import save_json
from .lexical import build_tfidf, tfidf_scores, build_bm25, bm25_scores
from .fusion import minmax_rowwise, ranks_from_scores, rrf_fuse_ranks
from .qrels import rank_of_first_gold
from .dense_index import DenseFAISS
from .metrics import evaluate_all

# Kích hoạt registry các backend (side-effect import)
from .embeddings import *  # noqa: F401
from .embeddings.base import get_embedder

# optional logging
try:
    from .logging_utils import setup_logger
except Exception:
    def setup_logger(level: str = "info"):
        class _Dummy:
            def info(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        return _Dummy()


def safe_model_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_").replace(" ", "_")


# ----------------------- NEW: helpers for error CSV -----------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_csv(path: str, rows: List[Dict[str, Any]], field_order: List[str]):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            rr = {}
            for k, v in r.items():
                if isinstance(v, (list, tuple, dict)):
                    import json as _json
                    rr[k] = _json.dumps(v, ensure_ascii=False)
                else:
                    rr[k] = v
            w.writerow(rr)


def _topk_indices_rowwise(score_mat: np.ndarray, k: int) -> np.ndarray:
    """
    Trả về (Q, k) là doc_id top-k theo score giảm dần cho mỗi query.
    Dùng argpartition rồi sort trong nhóm k để tối ưu.
    """
    Q, N = score_mat.shape
    k = min(max(k, 1), N)
    part = np.argpartition(-score_mat, kth=k-1, axis=1)[:, :k]  # (Q, k)
    row_idx = np.arange(Q)[:, None]
    ord_in_k = np.argsort(-score_mat[row_idx, part], axis=1)
    return part[row_idx, ord_in_k]  # (Q, k)


# -------------------------------------------------------------------------


def run(
    method: str,
    fusion: str,
    questions: List[str],
    contexts: List[str],
    gold_lists: List[List[int]],
    out_dir: str,
    *,
    dense_backend: str,
    embed_model: str,
    sbert_model: str,
    gemini_model: str,
    batch_size: int,
    index_metric: str,
    alpha: float,
    rrf_k: int,
    force: bool,
    ks: List[int],
    show_progress: bool = False,
    log_level: str = "info",
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    qids: Optional[List[str]] = None,     # NEW: optional qids for error CSV
    error_k: Optional[int] = None,        # NEW: reference K for "fail@K" (default: max(ks))
) -> Dict[str, float]:

    logger = setup_logger(log_level)
    os.makedirs(out_dir, exist_ok=True)
    ranks_path = os.path.join(out_dir, "ranks.json")
    metrics_path = os.path.join(out_dir, "metrics.json")

    logger.info(f"Method={method} | Fusion={fusion} | OutDir={out_dir}")
    logger.debug(f"Questions={len(questions)} | Contexts={len(contexts)} | Ks={ks}")

    scores = None  # (Q, N)

    if method == "tfidf":
        logger.info("Building TF-IDF...")
        vect, X_docs = build_tfidf(contexts)
        logger.info("Scoring TF-IDF...")
        scores = tfidf_scores(vect, X_docs, questions)

    elif method == "bm25":
        logger.info(f"Building BM25 (k1={bm25_k1}, b={bm25_b})...")
        bm25 = build_bm25(contexts, k1=bm25_k1, b=bm25_b)
        logger.info("Scoring BM25...")
        scores = bm25_scores(bm25, questions, show_progress=show_progress)

    else:
        logger.info(f"Preparing dense backend: {dense_backend}")
        if dense_backend == "openai":
            embedder = get_embedder("openai", model=embed_model, batch_size=batch_size, show_progress=show_progress)
            model_name = embed_model
        elif dense_backend == "gemini":
            embedder = get_embedder("gemini", model=gemini_model or "text-embedding-004", batch_size=batch_size, show_progress=show_progress)
            model_name = gemini_model or "text-embedding-004"
        elif dense_backend == "sbert":
            embedder = get_embedder("sbert", model_name=sbert_model, batch_size=batch_size, show_progress=show_progress)
            model_name = sbert_model
        else:
            raise ValueError(f"Unknown dense_backend: {dense_backend}")

        # Cache path corrected to follow backend + model
        dataset_name = os.path.normpath(out_dir).split(os.sep)[-2]
        model_tag = f"{dense_backend}_{safe_model_name(model_name)}"
        cache_path = os.path.join("cache", dataset_name, model_tag)
        logger.info(f"Using FAISS DB path: {cache_path}")
        dense = DenseFAISS(base_dir=cache_path, index_metric=index_metric)

        dense.build_or_load_docs(
            contexts,
            embed_fn=embedder.embed,
            force=force,
            show_progress=show_progress,
            batch_note=model_tag,
        )

        logger.info("Embedding queries...")
        q_embs = dense.embed_queries(
            questions,
            embed_fn=embedder.embed,
            force=force,
            show_progress=show_progress,
        )

        logger.info("Dense scoring (q @ d.T)...")
        d_scores = dense.dense_scores_for_queries(q_embs, show_progress=show_progress)

        if method == "dense":
            scores = d_scores

        elif method == "dense+tfidf":
            logger.info("Building TF-IDF for hybrid...")
            vect, X_docs = build_tfidf(contexts)
            logger.info("Scoring TF-IDF for hybrid...")
            t_scores = tfidf_scores(vect, X_docs, questions)

            if fusion == "alpha":
                logger.info(f"Fusing dense+tfidf with alpha={alpha} (min-max row-wise)...")
                scores = alpha * minmax_rowwise(d_scores) + (1 - alpha) * minmax_rowwise(t_scores)
            elif fusion == "rrf":
                logger.info(f"Fusing dense+tfidf with RRF (k={rrf_k})...")
                Q, N = len(questions), len(contexts)
                ranks_dense = np.zeros((Q, N), dtype=np.int32)
                ranks_tfidf = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_dense[i] = ranks_from_scores(d_scores[i])
                    ranks_tfidf[i] = ranks_from_scores(t_scores[i])
                scores = rrf_fuse_ranks([ranks_dense, ranks_tfidf], k=rrf_k)
            else:
                raise ValueError("For dense+tfidf you must set --fusion alpha|rrf")

        elif method == "dense+bm25":
            logger.info(f"Building BM25 for hybrid (k1={bm25_k1}, b={bm25_b})...")
            bm25 = build_bm25(contexts, k1=bm25_k1, b=bm25_b)
            logger.info("Scoring BM25 for hybrid...")
            b_scores = bm25_scores(bm25, questions, show_progress=show_progress)

            if fusion == "alpha":
                logger.info(f"Fusing dense+bm25 with alpha={alpha} (min-max row-wise)...")
                scores = alpha * minmax_rowwise(d_scores) + (1 - alpha) * minmax_rowwise(b_scores)
            elif fusion == "rrf":
                logger.info(f"Fusing dense+bm25 with RRF (k={rrf_k})...")
                Q, N = len(questions), len(contexts)
                ranks_dense = np.zeros((Q, N), dtype=np.int32)
                ranks_bm25 = np.zeros((Q, N), dtype=np.int32)
                for i in range(Q):
                    ranks_dense[i] = ranks_from_scores(d_scores[i])
                    ranks_bm25[i] = ranks_from_scores(b_scores[i])
                scores = rrf_fuse_ranks([ranks_dense, ranks_bm25], k=rrf_k)
            else:
                raise ValueError("For dense+bm25 you must set --fusion alpha|rrf")
        else:
            raise ValueError(f"Unknown method: {method}")

    assert scores is not None, "Internal error: scores not computed"

    logger.info("Saving ranks.json (first relevant ranks)...")
    ranks_first = rank_of_first_gold(scores, gold_lists)
    save_json(ranks_first, ranks_path)

    logger.info("Evaluating metrics...")
    metrics = evaluate_all(scores, gold_lists, ks=ks, show_progress=show_progress)
    save_json(metrics, metrics_path)

    # ---------------- NEW: save per-method fail@K CSV ----------------
    k_ref = error_k if (error_k is not None) else (max(ks) if ks else 10)
    errors_dir = os.path.join(out_dir, "errors")
    _ensure_dir(errors_dir)

    # Compute top-k ids & scores for K_ref
    topk_ids = _topk_indices_rowwise(scores, k_ref)             # (Q, k_ref)
    row_idx = np.arange(scores.shape[0])[:, None]
    topk_scores = scores[row_idx, topk_ids]                     # (Q, k_ref)

    rows = []
    Q = len(questions)
    for qi in range(Q):
        gold_id_set = set(gold_lists[qi]) if gold_lists else set()
        ret_ids = topk_ids[qi].tolist()
        # FAIL @K if none of gold ids appear in top-K
        if any(g in ret_ids for g in gold_id_set):
            continue

        gold_ids = list(gold_id_set)
        gold_texts = [contexts[g] for g in gold_ids if 0 <= g < len(contexts)]
        ret_texts = [contexts[d] for d in ret_ids]
        ret_scores = topk_scores[qi].tolist()

        rows.append({
            "k_ref": k_ref,
            "qid": (qids[qi] if (qids is not None and qi < len(qids)) else str(qi)),
            "question": questions[qi],
            "gold_doc_ids": gold_ids,
            "gold_texts": gold_texts,
            "retrieved_doc_ids": ret_ids,
            "retrieved_texts": ret_texts,
            "retrieved_scores": ret_scores,
        })

    fail_csv = os.path.join(errors_dir, f"fail@{k_ref}.csv")
    _save_csv(
        fail_csv,
        rows,
        field_order=[
            "k_ref", "qid", "question",
            "gold_doc_ids", "gold_texts",
            "retrieved_doc_ids", "retrieved_texts", "retrieved_scores",
        ],
    )
    logger.info(f"Saved per-method fail cases → {fail_csv}")
    # -----------------------------------------------------------------

    logger.info(f"Done. Artifacts written to: {out_dir}")
    return metrics
