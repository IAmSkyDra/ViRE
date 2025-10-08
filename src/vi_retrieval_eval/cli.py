# src/vi_retrieval_eval/cli.py
# -*- coding: utf-8 -*-
import os
import argparse
import json
import pandas as pd

# side-effect import to register embedders
from . import embeddings  # noqa: F401

from .io_utils import load_dataset, save_json
from .qrels import load_qrels, build_gold_from_identity, _read_qrels_safely
from .runner import run
from .embeddings.base import _debug_registry  # for --list-backends
from .dedup import dedup_by_content, remap_gold
from .sampling import sample_with_flags
from .textnorm import normalize_for_dedup
from .stats import compute_dataset_stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate lexical/dense/hybrid retrieval for Vietnamese QA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # IO
    p.add_argument("--csv", required=True, help="Input dataset file (CSV/JSONL) with columns or keys: question, context")
    p.add_argument("--qrels", default=None, help="Optional qrels file (CSV/TSV/JSONL) with columns: qid/doc_id/rel")
    p.add_argument("--qid-col", default="qid", help="Qrels column for query id")
    p.add_argument("--docid-col", default="doc_id", help="Qrels column for doc id")
    p.add_argument("--rel-col", default="rel", help="Qrels column for relevance (>0)")
    p.add_argument("--csv-qid-col", default=None, help="Column name in main CSV/JSONL for qid (optional)")
    p.add_argument("--csv-docid-col", default=None, help="Column name in main CSV/JSONL for doc_id (optional)")
    p.add_argument("--output-dir", default="outputs", help="Root folder to save results")

    # Method & fusion
    p.add_argument(
        "--method",
        required=True,
        choices=["tfidf", "bm25", "dense", "dense+tfidf", "dense+bm25"],
        help="Retrieval method",
    )
    p.add_argument("--fusion", choices=["none", "alpha", "rrf"], default="none",
                   help="Fusion for hybrid methods (dense+tfidf / dense+bm25]")
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha for score fusion (0..1)")
    p.add_argument("--rrf-k", type=int, default=60, help="RRF constant k (>=1)")

    # Sampling
    p.add_argument("--max-samples", type=int, default=None, help="Random subset size, e.g. 1000")
    # IMPORTANT: use 10%% in help to avoid argparse formatting issues
    p.add_argument("--sample-frac", type=float, default=None, help="Random subset fraction, e.g. 0.1 (10%%)")
    p.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--prefer-unique", action="store_true",
                   help="Prefer samples with unique contexts when sampling (normalized)")
    p.add_argument("--unique-col", default="context",
                   help="Column name used as uniqueness key for --prefer-unique")

    # Normalization knobs for unique-key & corpus dedup
    p.add_argument("--dedup-lower", action="store_true",
                   help="Lowercase in normalization key for unique/dedup")
    p.add_argument("--dedup-remove-emoji", action="store_true",
                   help="Strip emoji-like symbols in normalization key for unique/dedup")

    # 🔹 NEW: Normalize toàn bộ (trước snapshot, dedup, eval)
    p.add_argument("--normalize-all", action="store_true",
                   help="Normalize ALL questions/contexts with NFKC, remove invisibles/controls, strip emoji, collapse spaces, lowercase BEFORE snapshot/dedup/eval")

    # BM25 hyperparams
    p.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 Okapi k1")
    p.add_argument("--bm25-b", type=float, default=0.75, help="BM25 Okapi b")

    # Dense backends
    p.add_argument("--dense-backend", default="openai",
                   choices=["openai", "gemini", "sbert"], help="Dense embedding backend")
    p.add_argument("--embed-model", default="text-embedding-3-large", help="OpenAI embedding model")
    p.add_argument("--gemini-model", default="text-embedding-004", help="Gemini embedding model")
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-Transformers model")
    p.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    p.add_argument("--index-metric", choices=["ip", "l2"], default="ip",
                   help="FAISS metric (recommend 'ip' with normalized vectors)")

    # QoL
    p.add_argument("--force", action="store_true", help="Force rebuild embeddings and index")
    p.add_argument("--lower", action="store_true", help="Lowercase text before processing (kept for backward-compat; ignored if --normalize-all)")
    p.add_argument("--ks", default="1,3,5,10,20,50,100", help="Comma-separated k values")
    p.add_argument("--show-size", action="store_true", help="Print dataset head + sizes")

    # Progress & logging
    p.add_argument("--progress", action="store_true", help="Show progress bars")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Logger level (if runner uses logging_utils)")
    p.add_argument("--log-file", default=None, help="Optional log file path")

    # Utilities
    p.add_argument("--list-backends", action="store_true",
                   help="List available dense embedding backends and exit")

    # Corpus dedup (before indexing)
    p.add_argument("--dedup", action="store_true",
                   help="Deduplicate identical contexts in the corpus before indexing (queries remain unchanged)")

    # 🔎 Error analysis controls
    p.add_argument("--error-k", type=int, default=None,
                   help="Reference K used to mark a query as FAIL (default: max(ks)).")
    p.add_argument("--save-intersection", action="store_true",
                   help="After run, save the intersection of fail@K across ALL methods for this dataset.")
    p.add_argument("--max-errors", type=int, default=30,
                   help="When printing previews, max rows to show.")

    return p.parse_args()


def _infer_dataset_name(csv_path: str, qrels_path: str or None) -> str:
    """
    Ưu tiên: nếu có qrels -> lấy tên thư mục cha chung của CSV & QRELS.
    Nếu không, suy luận từ tên file CSV; nếu tên file 'generic' -> dùng thư mục cha của CSV.
    """
    generic = {
        "data", "dataset", "pairs", "beir_pairs", "corpus", "queries",
        "train", "test", "valid", "dev", "eval", "sample"
    }

    csv_abs = os.path.abspath(csv_path)
    csv_dir = os.path.dirname(csv_abs)
    csv_base = os.path.basename(csv_abs)
    stem, _ = os.path.splitext(csv_base)
    parent = os.path.basename(csv_dir)

    # Nếu có qrels: lấy thư mục cha chung
    if qrels_path:
        qrels_abs = os.path.abspath(qrels_path)
        qrels_dir = os.path.dirname(qrels_abs)
        try:
            common = os.path.commonpath([csv_dir, qrels_dir])
        except ValueError:
            common = ""
        common_base = os.path.basename(common) if common else ""
        if common_base and common_base.lower() not in {"", "/", ".", "data"}:
            return common_base

    # Nếu không có qrels, hoặc common không dùng được:
    if stem.lower() in generic and parent:
        return parent
    return stem or "dataset"


# ---------- Helpers for intersection across methods ----------

def _join_key(qid: str or None, question: str) -> str:
    """Key join giữa các method: ưu tiên qid; fallback hash(question)."""
    if qid is not None and str(qid).strip() != "":
        return f"QID::{qid}"
    return "QHASH::" + str(abs(hash(question)))


def _read_fail_csv(path: str) -> pd.DataFrame:
    """Đọc 1 fail CSV từ runner; thêm _join_key để gộp giao."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # Chuẩn hoá cột chính
    for col in ["qid", "question", "k_ref"]:
        if col not in df.columns:
            df[col] = None

    df["_join_key"] = [
        _join_key(None if pd.isna(q) else str(q), "" if pd.isna(qq) else str(qq))
        for q, qq in zip(df["qid"], df["question"])
    ]
    return df


def main():
    args = parse_args()

    # Optional: list backends
    if getattr(args, "list_backends", False):
        print("Available dense backends:", ", ".join(sorted(_debug_registry().keys())))
        return

    # Load full dataset
    df_full = load_dataset(args.csv)
    orig_n = len(df_full)

    # 🔹 Normalize ALL (nếu bật) — làm TRƯỚC snapshot / sampling / dedup / eval
    if args.normalize_all:
        def _norm(s):
            return normalize_for_dedup(s, do_lower=True, remove_emoji=True)
        if "question" in df_full.columns:
            df_full["question"] = df_full["question"].astype(str).apply(_norm)
        if "context" in df_full.columns:
            df_full["context"] = df_full["context"].astype(str).apply(_norm)

    # Unified sampling (supports prefer-unique with normalized key)
    df = sample_with_flags(
        df_full,
        max_samples=args.max_samples,
        sample_frac=args.sample_frac,
        seed=args.sample_seed,
        prefer_unique=args.prefer_unique,
        unique_col=args.unique_col,
        norm_lower=args.dedup_lower,
        norm_remove_emoji=args.dedup_remove_emoji,
    )

    if args.show_size:
        print(df.head(3))
        print(f"[INFO] Loaded {orig_n} rows. Using {len(df)} rows after sampling.")
        key_series = df[args.unique_col].astype(str).apply(
            lambda s: normalize_for_dedup(s, do_lower=args.dedup_lower, remove_emoji=args.dedup_remove_emoji)
        )
        print(f"[INFO] Unique {args.unique_col} after sampling (normalized): {key_series.nunique()}")

    # === Persist the sampled set (1–1, BEFORE additional lower & BEFORE corpus dedup) ===
    tag_parts = []
    if args.max_samples:
        tag_parts.append(f"s{args.max_samples}")
    elif args.sample_frac:
        tag_parts.append(f"sf{args.sample_frac:g}")
    if args.prefer_unique:
        tag_parts.append("uniq")
    if args.dedup:
        tag_parts.append("dedup")  # snapshot is still pre-dedup; tag for traceability
    if args.normalize_all:
        tag_parts.append("norm")
    tag_parts.append(f"seed{args.sample_seed}")
    sampling_tag = "-".join(tag_parts) if tag_parts else "full"

    # Suy luận tên dataset (ưu tiên thư mục chung khi có qrels)
    dataset_name = _infer_dataset_name(args.csv, args.qrels)

    stats_dir = os.path.join(args.output_dir, dataset_name, "statistics", sampling_tag)
    os.makedirs(stats_dir, exist_ok=True)

    # Lưu snapshot CSV (dataset) — KHÔNG lưu JSONL theo yêu cầu
    df_snapshot = df[["question", "context"]].copy()
    # nếu có qid/doc_id thì lưu kèm để rerun y hệt
    for c in ("qid", "doc_id"):
        if c in df.columns:
            df_snapshot[c] = df[c]

    csv_out = os.path.join(stats_dir, "sampled.csv")
    df_snapshot.to_csv(csv_out, index=False)
    if args.show_size:
        print(f"[SNAPSHOT] Sampled dataset saved to {csv_out}")

    # Thống kê dataset đã sample (pre-dedup)
    stats = compute_dataset_stats(
        questions=df_snapshot["question"].astype(str).tolist(),
        contexts=df_snapshot["context"].astype(str).tolist(),
    )
    save_json(stats, os.path.join(stats_dir, "stats.json"))
    if args.show_size:
        print(f"[SNAPSHOT] Dataset statistics saved to {stats_dir}/stats.json")

    # Nếu có qrels: lọc qrels theo subset df và lưu snapshot qrels.filtered.csv
    qrels_filtered_csv = None
    qrels_filtered_count = None
    total_positive_in_qrels = None

    if args.qrels:
        qc = args.qid_col.lower()
        dc = args.docid_col.lower()
        rc = args.rel_col.lower()

        # đọc qrels an toàn (hỗ trợ csv/tsv/jsonl/json + đồng bộ tên cột)
        qrels_raw = _read_qrels_safely(
            qrels_path=args.qrels,
            qid_col=qc,
            docid_col=dc,
            rel_col=rc,
        )

        # tập id giữ lại dựa trên df đã sample
        qid_col_csv = (args.csv_qid_col.lower() if args.csv_qid_col else None)
        docid_col_csv = (args.csv_docid_col.lower() if args.csv_docid_col else None)

        if qid_col_csv and qid_col_csv in df.columns:
            keep_qids = set(df[qid_col_csv].astype(str).str.strip().tolist())
        else:
            keep_qids = set(str(i) for i in range(len(df)))  # identity fallback

        if docid_col_csv and docid_col_csv in df.columns:
            keep_docids = set(df[docid_col_csv].astype(str).str.strip().tolist())
        else:
            keep_docids = set(str(i) for i in range(len(df)))  # identity fallback

        # lọc qrels theo subset + rel > 0
        total_positive_in_qrels = int((qrels_raw[rc] > 0).sum())
        qrels_f = qrels_raw[
            (qrels_raw[rc] > 0)
            & (qrels_raw[qc].astype(str).str.strip().isin(keep_qids))
            & (qrels_raw[dc].astype(str).str.strip().isin(keep_docids))
        ].copy()
        qrels_filtered_count = int(qrels_f.shape[0])

        # lưu CSV
        qrels_filtered_csv = os.path.join(stats_dir, "qrels.filtered.csv")
        qrels_f[[qc, dc, rc]].to_csv(qrels_filtered_csv, index=False)

        if args.show_size:
            print(f"[SNAPSHOT] Qrels filtered saved: {qrels_filtered_count}/{total_positive_in_qrels} → {qrels_filtered_csv}")

    # Meta: trỏ đầy đủ snapshot để rerun
    meta = {
        # nguồn
        "source_dataset": os.path.abspath(args.csv),
        "source_qrels": os.path.abspath(args.qrels) if args.qrels else None,

        # snapshot path
        "sampled_csv": csv_out,
        "qrels_filtered_csv": qrels_filtered_csv,

        # đếm
        "num_rows_loaded": int(orig_n),
        "num_rows_sampled": int(len(df)),
        "num_unique_contexts_after_sampling_norm": int(
            df[args.unique_col].astype(str).apply(
                lambda s: normalize_for_dedup(s, do_lower=args.dedup_lower, remove_emoji=args.dedup_remove_emoji)
            ).nunique()
        ),
        "qrels_filtered_count": qrels_filtered_count,
        "qrels_total_positive": total_positive_in_qrels,

        # cấu hình
        "qid_col": args.qid_col,
        "docid_col": args.docid_col,
        "rel_col": args.rel_col,
        "csv_qid_col": args.csv_qid_col,
        "csv_docid_col": args.csv_docid_col,
        "method": args.method,
        "fusion": args.fusion,
        "alpha": args.alpha,
        "rrf_k": args.rrf_k,
        "dense_backend": args.dense_backend,
        "embed_model": args.embed_model if args.dense_backend == "openai" else None,
        "gemini_model": args.gemini_model if args.dense_backend == "gemini" else None,
        "sbert_model": args.sbert_model if args.dense_backend == "sbert" else None,
        "max_samples": args.max_samples,
        "sample_frac": args.sample_frac,
        "sample_seed": args.sample_seed,
        "prefer_unique": args.prefer_unique,
        "unique_col": args.unique_col,
        "dedup": args.dedup,
        "dedup_lower": args.dedup_lower,
        "dedup_remove_emoji": args.dedup_remove_emoji,
        "normalize_all": args.normalize_all,  # 🔹 NEW
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "index_metric": args.index_metric,
        "lower": args.lower,
        "ks": args.ks,
        "output_dir": args.output_dir,
        "dataset_name": dataset_name,
        "sampling_tag": sampling_tag,
        "stats_dir": stats_dir,
    }
    save_json(meta, os.path.join(stats_dir, "meta.json"))
    if args.show_size:
        print(f"[SNAPSHOT] Meta saved to {os.path.join(stats_dir, 'meta.json')}")

    # ==== Build questions/contexts for EVAL ====
    questions = df["question"].astype(str).tolist()
    contexts = df["context"].astype(str).tolist()

    # lowercase (only for eval pipeline)
    if args.lower and not args.normalize_all:
        questions = [q.lower() for q in questions]
        contexts = [c.lower() for c in contexts]

    # build gold (on sampled df)
    if args.qrels:
        gold_lists = load_qrels(
            qrels_path=args.qrels,
            qid_col=args.qid_col.lower(),
            docid_col=args.docid_col.lower(),
            rel_col=args.rel_col.lower(),
            df=df,  # important: sampled df
            csv_qid_col=(args.csv_qid_col.lower() if args.csv_qid_col else None),
            csv_docid_col=(args.csv_docid_col.lower() if args.csv_docid_col else None),
        )
    else:
        gold_lists = build_gold_from_identity(len(df))

    # Optional corpus dedup (for indexing/eval only)
    dedup_suffix = ""
    if args.dedup:
        old_n = len(contexts)
        contexts, mapping = dedup_by_content(contexts)
        gold_lists = remap_gold(gold_lists, mapping)
        new_n = len(contexts)
        if old_n == new_n:
            print(f"[INFO] Dedup enabled: no duplicates found (kept {new_n})")
        else:
            print(f"[INFO] Dedup enabled: removed {old_n - new_n} duplicates (kept {new_n})")
        dedup_suffix = "-dedup"

    # ks & label / out_dir
    ks = [int(x) for x in args.ks.split(",") if x.strip().isdigit()]

    label = args.method.lower()
    if label in ("dense+tfidf", "dense+bm25"):
        if args.fusion == "alpha":
            label += f"-alpha{args.alpha:.2f}"
        elif args.fusion == "rrf":
            label += f"-rrf{args.rrf_k}"

    if args.max_samples:
        label += f"-s{args.max_samples}"
    elif args.sample_frac:
        label += f"-sf{args.sample_frac:g}"
    if args.prefer_unique:
        label += "-uniq"
    if args.normalize_all:
        label += "-norm"
    label += dedup_suffix

    if label.startswith("dense"):
        label += f"-{args.dense_backend}"
        if args.dense_backend == "openai":
            label += f"-{args.embed_model}"
        elif args.dense_backend == "gemini":
            label += f"-{args.gemini_model}"
        elif args.dense_backend == "sbert":
            model_tail = args.sbert_model.split("/")[-1]
            label += f"-{model_tail}"

    out_dir = os.path.join(args.output_dir, dataset_name, label)

    # Prepare qids for runner (optional)
    qids = None
    if args.csv_qid_col and args.csv_qid_col in df.columns:
        qids = df[args.csv_qid_col].astype(str).tolist()

    # Run evaluation
    metrics = run(
        method=args.method.lower(),
        fusion=args.fusion.lower(),
        questions=questions,
        contexts=contexts,
        gold_lists=gold_lists,
        out_dir=out_dir,
        dense_backend=args.dense_backend,
        embed_model=args.embed_model,
        sbert_model=args.sbert_model,
        gemini_model=args.gemini_model,
        batch_size=args.batch_size,
        index_metric=args.index_metric,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
        force=args.force,
        ks=ks,
        # QoL flags
        show_progress=args.progress,
        log_level=args.log_level,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        # 🔹 error CSV controls
        qids=qids,
        error_k=(args.error_k if args.error_k is not None else (max(ks) if ks else 10)),
    )

    print(f"\nMethod: {args.method.upper()}   Dataset: {dataset_name}")
    print(f"Dense backend: {args.dense_backend}")
    print(f"Output dir: {out_dir}")
    from pprint import pprint as _pp
    _pp(metrics)

    # ---------- Save intersection of fail@K across ALL methods (optional) ----------
    if args.save_intersection:
        k_ref = args.error_k if args.error_k is not None else (max(ks) if ks else 10)
        dataset_root = os.path.join(args.output_dir, dataset_name)

        # Tìm tất cả method dirs có errors/fail@K.csv
        method_csvs = []
        if os.path.isdir(dataset_root):
            for name in os.listdir(dataset_root):
                mdir = os.path.join(dataset_root, name)
                if not os.path.isdir(mdir):
                    continue
                fpath = os.path.join(mdir, "errors", f"fail@{k_ref}.csv")
                if os.path.exists(fpath):
                    method_csvs.append((name, fpath))

        if not method_csvs:
            print(f"[INTERSECTION] No fail@{k_ref}.csv found under {dataset_root}")
        else:
            frames = []
            for mname, fpath in method_csvs:
                dfm = _read_fail_csv(fpath)
                if not dfm.empty:
                    dfm["__method__"] = mname
                    frames.append(dfm)

            if not frames:
                print(f"[INTERSECTION] No rows present in fail CSVs for k={k_ref}")
            else:
                all_fail = pd.concat(frames, ignore_index=True)

                # Đếm số method fail cho mỗi query
                mat = (
                    all_fail.assign(val=1)
                    .pivot_table(index="_join_key", columns="__method__", values="val", fill_value=0, aggfunc="max")
                    .reset_index()
                )
                method_cols = [c for c in mat.columns if c != "_join_key"]

                # Lấy thông tin đại diện (qid/question/…)
                rep_cols = ["qid", "question", "k_ref",
                            "gold_doc_ids", "gold_texts",
                            "retrieved_doc_ids", "retrieved_texts", "retrieved_scores"]
                rep = (all_fail
                       .sort_values(["_join_key", "__method__"])
                       .drop_duplicates(subset=["_join_key"])[["_join_key"] + rep_cols])

                mat = rep.merge(mat, on="_join_key", how="left")
                mat["__fail_methods__"] = mat[method_cols].sum(axis=1)

                # Giao: fail ở TẤT CẢ method
                inter = mat[mat["__fail_methods__"] == len(method_cols)].copy()

                summary_dir = os.path.join(dataset_root, "errors_summary")
                os.makedirs(summary_dir, exist_ok=True)
                mat_out = os.path.join(summary_dir, f"fail@{k_ref}__MATRIX.csv")
                inter_out = os.path.join(summary_dir, f"fail@{k_ref}__ALL_METHODS.csv")
                mat.to_csv(mat_out, index=False)
                inter.to_csv(inter_out, index=False)

                print(f"[INTERSECTION] Saved per-method matrix → {mat_out}")
                print(f"[INTERSECTION] Saved ALL-METHODS intersection → {inter_out}")
                with pd.option_context("display.max_colwidth", 120, "display.width", 180):
                    cols_preview = ["qid", "question", "__fail_methods__"] + method_cols
                    print("\n[INTERSECTION] Preview:")
                    print(inter[cols_preview].head(args.max_errors).to_string(index=False))


if __name__ == "__main__":
    main()
