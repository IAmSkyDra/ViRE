# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict
from .textnorm import normalize_for_dedup

def dedup_by_content(
    contexts: List[str],
    *,
    do_lower: bool = False,
    remove_emoji: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Deduplicate corpus by normalized context string.
    Trả về:
      - unique_contexts: danh sách context đã dedup
      - doc_map: len = len(contexts gốc),
                 doc_map[i] = index của context unique tương ứng contexts[i]
    """
    seen: Dict[str, int] = {}
    unique: List[str] = []
    doc_map: List[int] = []

    for c in contexts:
        key = normalize_for_dedup(c, do_lower=do_lower, remove_emoji=remove_emoji)
        idx = seen.get(key)
        if idx is None:
            idx = len(unique)
            seen[key] = idx
            unique.append(c)  # giữ nguyên bản gốc (không cần bản đã normalize)
        doc_map.append(idx)
    return unique, doc_map


def remap_gold(gold_lists: List[List[int]], doc_map: List[int]) -> List[List[int]]:
    """
    Map gold doc indices (original space) sang space đã dedup dùng doc_map.
    Đồng thời dedup các chỉ số gold (set) cho mỗi query.
    """
    out: List[List[int]] = []
    for gold in gold_lists:
        mapped = sorted({doc_map[g] for g in gold})
        out.append(mapped)
    return out
