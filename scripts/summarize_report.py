#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a single Markdown report for multiple datasets (VIRE grouped layout).
Bold = best, <u>underline</u> = second-best.
"""

import argparse, json, math, os, re
from typing import Dict, List, Tuple, Optional

METHOD_ORDER_COMMON = ["tfidf", "bm25"]
METHOD_ORDER_DENSE_VARIANTS = [
    "dense",
    "dense+tfidf-alpha",
    "dense+bm25-alpha",
    "dense+tfidf-rrf",
    "dense+bm25-rrf",
]
KIND_PRETTY_LABEL = {
    "tfidf": "tfidf",
    "bm25": "bm25",
    "dense": "dense",
    "dense+tfidf-alpha": "dense + tfidf (alpha)",
    "dense+bm25-alpha": "dense + bm25 (alpha)",
    "dense+tfidf-rrf": "dense + tfidf (rrf)",
    "dense+bm25-rrf": "dense + bm25 (rrf)",
}
BACKEND_MARKERS = ["openai","gemini","sbert"]

# ===== Metric key canonicalization =====
_re_at = re.compile(r"^([a-z\-]+)@(\d+)$", re.IGNORECASE)
def canonicalize_metric_key(name: str) -> str:
    s, low = name.strip(), name.strip().lower()
    m = _re_at.match(low)
    if m:
        head, k = m.group(1), m.group(2)
        if head in ("p","precision"): return f"Precision@{k}"
        if head in ("r","recall"):    return f"Recall@{k}"
        if head in ("mrr",):          return f"MRR@{k}"
        if head in ("ndcg",):         return f"nDCG@{k}"
        if head in ("map",):          return f"MAP@{k}"
        if head in ("hitrate","hit","hit-rate"): return f"HitRate@{k}"
    if low in ("r-precision","rprecision"): return "R-Precision"
    return s

def is_number(x)->bool:
    try: return x is not None and not isinstance(x,bool) and math.isfinite(float(x))
    except Exception: return False

def fmt_val(v: Optional[float], ndigits:int, as_pct:bool)->str:
    if v is None or (isinstance(v,float) and not math.isfinite(v)): return "–"
    return f"{(v*100 if as_pct else v):.{ndigits}f}{'%' if as_pct else ''}"

def find_method_dirs(root:str, dataset:str, include:Optional[str])->List[str]:
    base = os.path.join(root, dataset)
    if not os.path.isdir(base): return []
    out=[]
    for name in sorted(os.listdir(base)):
        p=os.path.join(base,name)
        if os.path.isdir(p) and (not include or re.search(include,name)):
            out.append(p)
    return out

def load_metrics_json(method_dir:str)->Optional[Dict[str,float]]:
    path=os.path.join(method_dir,"metrics.json")
    if not os.path.isfile(path): return None
    try:
        with open(path,"r",encoding="utf-8") as f: data=json.load(f)
        return {k:float(v) for k,v in data.items() if is_number(v)}
    except Exception:
        return None

# ===== Parse method name from folder (matches your screenshot) =====
def _extract_backend(original_name:str)->str:
    low=original_name.lower()
    m=re.search(r"-(openai|sbert|gpt|gemini)[-_]", low)
    if m:
        start = m.start(1)
        return original_name[start:]
    pos=None
    for mk in BACKEND_MARKERS:
        i=low.find(mk)
        if i!=-1 and (pos is None or i<pos): pos=i
    return (original_name[pos:] if pos is not None else "unknown")

def parse_method_name(folder_name:str)->Dict[str,str]:
    name_orig=folder_name
    name=folder_name.lower()
    # lexical
    if name.startswith("tfidf"): return {"kind":"tfidf","backend":"","label":KIND_PRETTY_LABEL["tfidf"]}
    if name.startswith("bm25"):  return {"kind":"bm25","backend":"","label":KIND_PRETTY_LABEL["bm25"]}
    # dense hybrids
    kind=None
    if "dense+tfidf" in name:
        kind = "dense+tfidf-rrf" if re.search(r"\brrf\d+\b", name) else "dense+tfidf-alpha"
    elif "dense+bm25" in name:
        kind = "dense+bm25-rrf"  if re.search(r"\brrf\d+\b", name) else "dense+bm25-alpha"
    elif name.startswith("dense") or "dense" in name:
        kind = "dense"
    else:
        kind = "dense"
    backend=_extract_backend(name_orig)
    return {"kind":kind,"backend":backend,"label":KIND_PRETTY_LABEL.get(kind,kind)}

def compute_column_ranks(rows:List[Dict[str,float]], metrics_keys:List[str])->Dict[str,Tuple[Optional[float],Optional[float]]]:
    out={}
    for m in metrics_keys:
        vals=[r[m] for r in rows if m in r and is_number(r[m])]
        uniq=sorted(set(vals), reverse=True)
        out[m]=(uniq[0], uniq[1]) if len(uniq)>1 else ((uniq[0], None) if uniq else (None,None))
    return out

def build_markdown_table_grouped(entries:List[Dict], metrics_keys:List[str], metrics_labels:List[str],
                                 highlight:Dict[str,Tuple[Optional[float],Optional[float]]],
                                 ndigits:int, as_pct:bool)->str:
    header=["Method"]+metrics_labels
    lines=[" | ".join(header), " | ".join(["---"]*len(header))]
    by_kind:Dict[str,List[Dict]]={}
    for e in entries: by_kind.setdefault(e["kind"],[]).append(e)
    def _append_row(title:str,rowvals:Dict[str,float]):
        cells=[title]
        for m in metrics_keys:
            v=rowvals.get(m,None); base=fmt_val(v,ndigits,as_pct)
            max_v, second_v = highlight.get(m,(None,None))
            cell=base
            if is_number(v) and max_v is not None:
                if abs(v-max_v)<1e-12: cell=f"**{base}**"
                elif second_v is not None and abs(v-second_v)<1e-12: cell=f"<u>{base}</u>"
            cells.append(cell)
        lines.append(" | ".join(cells))
    # tfidf, bm25
    for k in METHOD_ORDER_COMMON:
        for e in sorted(by_kind.get(k,[]), key=lambda x:x["folder"]): _append_row(e["label"], e["row"])
    # dense groups
    dense_entries=[e for e in entries if e["backend"]]
    backends=sorted(sorted(set(e["backend"] for e in dense_entries)), key=lambda s:s.lower())
    for be in backends:
        lines.append(" | ".join([f"**Dense model: {be}**"]+[""]*len(metrics_keys)))
        sub=[e for e in dense_entries if e["backend"]==be]
        by_kind_be:Dict[str,List[Dict]]={}
        for e in sub: by_kind_be.setdefault(e["kind"],[]).append(e)
        for kind in METHOD_ORDER_DENSE_VARIANTS:
            if kind not in by_kind_be: continue
            for e in sorted(by_kind_be[kind], key=lambda x:x["folder"]):
                _append_row("  "+e["label"], e["row"])
    return "\n".join(lines)

# ===== Report builder =====
def build_section_for_dataset(outputs_root:str, dataset:str, metrics_keys:List[str], metrics_labels:List[str],
                              include_regex:Optional[str], ndigits:int, percent:bool)->Optional[str]:
    dirs=find_method_dirs(outputs_root,dataset,include_regex)
    if not dirs: return None
    entries=[]
    for d in dirs:
        folder=os.path.basename(d)
        data=load_metrics_json(d)
        if data is None: continue
        parsed=parse_method_name(folder)
        row={}
        for m in metrics_keys:
            v=data.get(m,None)
            if is_number(v): row[m]=float(v)
        entries.append({"folder":folder,"kind":parsed["kind"],"backend":parsed["backend"],"label":parsed["label"],"row":row})
    if not entries: return None
    highlight=compute_column_ranks([e["row"] for e in entries], metrics_keys)
    table=build_markdown_table_grouped(entries, metrics_keys, metrics_labels, highlight, ndigits, percent)
    return f"## {dataset}\n\n{table}\n"

def parse_args():
    ap=argparse.ArgumentParser(description="Create a single Markdown report for multiple datasets (VIRE grouped layout).",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    ap.add_argument("--metrics", required=True, help="Comma-separated metric names (case-insensitive; aliases OK)")
    ap.add_argument("--include", default=None, help="Regex to filter method folders")
    ap.add_argument("--ndigits", type=int, default=4)
    ap.add_argument("--percent", action="store_true")
    ap.add_argument("--save", required=True)
    return ap.parse_args()

def main():
    args=parse_args()
    datasets=[d.strip() for d in args.datasets.split(",") if d.strip()]
    metrics_in=[m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics_keys=[]; metrics_labels=[]; seen=set()
    for m in metrics_in:
        canon=canonicalize_metric_key(m)
        if canon not in seen:
            metrics_keys.append(canon)
            metrics_labels.append(m)
            seen.add(canon)
    parts=[]
    parts.append("# VIRE Retrieval Report\n")
    parts.append(f"Metrics: `{args.metrics}`{' (shown as %)' if args.percent else ''}\n")
    parts.append("## Datasets\n")
    for d in datasets: parts.append(f"- [{d}](#{d.lower()})")
    parts.append("")
    for d in datasets:
        sec=build_section_for_dataset(args.outputs_root, d, metrics_keys, metrics_labels, args.include, args.ndigits, args.percent)
        parts.append(sec if sec is not None else f"## {d}\n\n*(No results found in `{args.outputs_root}/{d}`)*\n")
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save,"w",encoding="utf-8") as f: f.write("\n".join(parts).rstrip()+"\n")
    print(f"Saved report to: {args.save}")

if __name__ == "__main__":
    main()
