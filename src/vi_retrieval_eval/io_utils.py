# -*- coding: utf-8 -*-
import os, json, gzip
from typing import Iterable
import pandas as pd
import tiktoken  # pip install tiktoken

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)

def _iter_jsonl(path: str) -> Iterable[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _coerce_qc(obj: dict):
    q = None
    for k in ("question", "query", "q"):
        if k in obj and obj[k] is not None:
            q = obj[k]
            break
    c = None
    if ("title" in obj and obj.get("title")) or ("text" in obj and obj.get("text")):
        title = (obj.get("title") or "").strip()
        text  = (obj.get("text")  or "").strip()
        if title or text:
            c = (f"{title}. {text}".strip(". ").strip()) if title else text
    if c is None:
        for k in ("context", "passage", "document", "doc", "ctx", "text"):
            if k in obj and obj[k] is not None:
                c = obj[k]
                break
    return q, c

def _lower_cols(cols):
    return [c.lower().strip().lstrip("\ufeff") for c in cols]

def _truncate_to_token_limit(text: str, max_tokens: int, encoding) -> str:
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def load_dataset(path: str) -> pd.DataFrame:
    """
    Trả về DataFrame với 2 cột: 'question', 'context', đã truncate context theo token limit.
    """
    if path.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path)
        df.columns = _lower_cols(df.columns)
        required = {"question", "context"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")
        df = df.dropna(subset=["question", "context"]).reset_index(drop=True)

    elif path.endswith((".jsonl", ".jsonl.gz")):
        rows = []
        for obj in _iter_jsonl(path):
            if not isinstance(obj, dict):
                continue
            q, c = _coerce_qc(obj)
            if q is not None and c is not None:
                rows.append({"question": str(q), "context": str(c)})
        if not rows:
            raise ValueError("JSONL did not yield any (question, context) pairs.")
        df = pd.DataFrame(rows).dropna(subset=["question", "context"]).reset_index(drop=True)

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                q, c = _coerce_qc(obj)
                if q is not None and c is not None:
                    rows.append({"question": str(q), "context": str(c)})
            if not rows:
                raise ValueError("JSON list did not yield any (question, context) pairs.")
            df = pd.DataFrame(rows).dropna(subset=["question", "context"]).reset_index(drop=True)
        else:
            raise ValueError("Top-level JSON must be a list of objects.")
    else:
        raise ValueError(f"Unsupported file type for {path}")

    # ✅ Truncate context theo token limit
    max_token_limit = 8192
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")  # hoặc "text-embedding-3-small"
    df["context"] = df["context"].apply(lambda x: _truncate_to_token_limit(x, max_token_limit, encoding))

    return df
