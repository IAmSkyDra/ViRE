# -*- coding: utf-8 -*-
from typing import List, Optional, DefaultDict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import os, json, gzip

def build_gold_from_identity(num_q: int) -> List[List[int]]:
    return [[i] for i in range(num_q)]

def _lower_cols_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower().strip().lstrip("\ufeff") for c in df.columns]
    return df

def _read_jsonl(path: str) -> pd.DataFrame:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return pd.DataFrame(rows)

def _read_qrels_safely(qrels_path: str, qid_col: str, docid_col: str, rel_col: str) -> pd.DataFrame:
    ext = os.path.splitext(qrels_path.lower())[1]
    # 1) đọc thành DataFrame
    if ext == ".tsv":
        qrels = pd.read_csv(qrels_path, sep="\t")
    elif ext in (".jsonl", ".json"):
        if ext == ".jsonl":
            qrels = _read_jsonl(qrels_path)
        else:
            data = json.load(open(qrels_path, encoding="utf-8"))
            qrels = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        # CSV hoặc không rõ
        try:
            qrels = pd.read_csv(qrels_path)
        except Exception:
            qrels = pd.read_csv(qrels_path, sep=None, engine="python")
    _lower_cols_inplace(qrels)

    # 2) chuẩn hóa tên cột
    synonyms = {
        qid_col: {"query-id", "qid", "query_id"},
        docid_col: {"corpus-id", "doc-id", "doc_id", "document-id", "did"},
        rel_col: {"score", "rel", "relevance"},
    }
    def _ensure_col(target_name):
        if target_name in qrels.columns:
            return
        for syn in synonyms.get(target_name, set()):
            if syn in qrels.columns:
                qrels.rename(columns={syn: target_name}, inplace=True)
                return
    _ensure_col(qid_col)
    _ensure_col(docid_col)
    _ensure_col(rel_col)

    missing = [c for c in (qid_col, docid_col, rel_col) if c not in qrels.columns]
    if missing:
        raise ValueError(f"Qrels must contain column(s) {missing}. Found: {list(qrels.columns)}")

    # 3) ép rel về int
    qrels[rel_col] = pd.to_numeric(qrels[rel_col], errors="coerce").fillna(0).astype(int)
    # strip id
    qrels[qid_col] = qrels[qid_col].astype(str).str.strip()
    qrels[docid_col] = qrels[docid_col].astype(str).str.strip()
    return qrels

def load_qrels(qrels_path: str, qid_col: str, docid_col: str, rel_col: str,
               df: pd.DataFrame, csv_qid_col: Optional[str],
               csv_docid_col: Optional[str]) -> List[List[int]]:
    qrels = _read_qrels_safely(qrels_path, qid_col, docid_col, rel_col)

    # mapping từ df đã sample
    def _build_map(col_name: Optional[str], label: str) -> Tuple[dict, Optional[str]]:
        if col_name and col_name in df.columns:
            series = df[col_name].astype(str).str.strip().tolist()
            return {v: i for i, v in enumerate(series)}, col_name
        elif col_name and col_name not in df.columns:
            print(f"[WARN] csv_{label}_col='{col_name}' not found in CSV. Falling back to identity indices.")
        return {str(i): i for i in range(len(df))}, None

    qid_to_idx, used_qcol = _build_map(csv_qid_col, "qid")
    docid_to_idx, used_dcol = _build_map(csv_docid_col, "docid")

    gold_map: DefaultDict[int, List[int]] = defaultdict(list)
    missed_q, missed_d = 0, 0

    # chỉ lấy rel > 0
    pos = qrels[qrels[rel_col] > 0]
    total_pos = int(pos.shape[0])

    for _, r in pos.iterrows():
        q_key = str(r[qid_col]).strip()
        d_key = str(r[docid_col]).strip()
        q_idx = qid_to_idx.get(q_key)
        d_idx = docid_to_idx.get(d_key)
        if q_idx is None:
            missed_q += 1
            continue
        if d_idx is None:
            missed_d += 1
            continue
        gold_map[q_idx].append(d_idx)

    mapped_pairs = sum(len(v) for v in gold_map.values())
    if missed_q or missed_d or mapped_pairs < total_pos:
        print(f"[WARN] Unmapped qrels: {missed_q} queries, {missed_d} docs. "
              f"Mapped positives {mapped_pairs}/{total_pos} "
              f"(csv_qid_col={used_qcol}, csv_docid_col={used_dcol}).")

    out: List[List[int]] = []
    for qi in range(len(df)):
        if qi in gold_map and gold_map[qi]:
            out.append(sorted(set(gold_map[qi])))
        else:
            # fallback identity để không rỗng
            out.append([qi])
    return out

def rank_of_first_gold(scores_matrix: np.ndarray, gold_lists: List[List[int]]) -> List[float]:
    """
    Trả về hạng (1-based) của positive tốt nhất cho mỗi query; inf nếu không có.
    Dùng inverse-permutation để tra cứu O(1)/gold.
    """
    Q, N = scores_matrix.shape
    ranks: List[float] = []
    for i in range(Q):
        order = np.argsort(-scores_matrix[i], kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(1, N + 1)  # inv[j] = hạng của doc j (1-based)
        best = float("inf")
        for g in gold_lists[i]:
            if 0 <= g < N:
                r = float(inv[g])
                if r < best:
                    best = r
        ranks.append(best)
    return ranks
