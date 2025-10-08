# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
from typing import Optional
from .textnorm import normalize_for_dedup

def _add_norm_key(df: pd.DataFrame, *, col: str, lower: bool, rm_emoji: bool) -> pd.DataFrame:
    out = df.copy()
    out["_norm_key"] = out[col].astype(str).apply(
        lambda s: normalize_for_dedup(s, do_lower=lower, remove_emoji=rm_emoji)
    )
    return out

def prefer_unique_context_sampling(
    df: pd.DataFrame,
    n: int,
    *,
    unique_col: str = "context",
    seed: int = 42,
    norm_lower: bool = False,
    norm_remove_emoji: bool = False,
) -> pd.DataFrame:
    """
    Cố gắng lấy đúng n mẫu:
      1) Ưu tiên chọn n mẫu có context unique theo KHÓA CHUẨN HOÁ (_norm_key).
      2) Nếu unique không đủ -> bù thêm ngẫu nhiên từ phần còn lại để ĐỦ n (phần bù có thể trùng).
    Nội dung trả về giữ nguyên cột gốc; _norm_key chỉ dùng để chọn rồi bị drop.
    """
    if n <= 0:
        return df.iloc[0:0].copy()

    # Tạo khoá chuẩn hoá dùng cho so trùng
    df_keyed = _add_norm_key(df, col=unique_col, lower=norm_lower, rm_emoji=norm_remove_emoji)

    # Shuffle để random
    shuffled = df_keyed.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Lấy 1 dòng/ khoá chuẩn hoá theo thứ tự sau shuffle
    deduped_unique = shuffled.drop_duplicates(subset=["_norm_key"], keep="first")

    if len(deduped_unique) >= n:
        res = deduped_unique.head(n).drop(columns=["_norm_key"]).reset_index(drop=True)
        return res

    # Không đủ unique => bù thêm từ phần còn lại để đủ n
    need = n - len(deduped_unique)
    remaining = shuffled.drop(deduped_unique.index)
    filler = remaining.head(min(need, len(remaining)))
    result = pd.concat([deduped_unique, filler], ignore_index=True)
    result = result.head(min(n, len(result))).drop(columns=["_norm_key"]).reset_index(drop=True)
    return result


def sample_with_flags(
    df: pd.DataFrame,
    *,
    max_samples: Optional[int],
    sample_frac: Optional[float],
    seed: int,
    prefer_unique: bool,
    unique_col: str = "context",
    norm_lower: bool = False,
    norm_remove_emoji: bool = False,
) -> pd.DataFrame:
    """
    Sampling thống nhất:
    - Tính n mục tiêu từ sample_frac / max_samples.
    - Nếu prefer_unique=True -> ưu tiên unique theo khoá CHUẨN HOÁ, nếu thiếu bù thêm, đảm bảo tối đa n.
    - Nếu prefer_unique=False -> sample ngẫu nhiên chuẩn.
    - Nếu n >= len(df) -> trả lại df; nếu prefer_unique=True vẫn unique theo khóa rồi bù để giữ size.
    """
    # xác định target n
    n_target: Optional[int] = None
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n_target = max(1, round(sample_frac * len(df)))
    if max_samples is not None and max_samples > 0:
        n_target = max_samples if n_target is None else min(max_samples, n_target)

    if not n_target or n_target >= len(df):
        if prefer_unique:
            return prefer_unique_context_sampling(
                df, len(df),
                unique_col=unique_col, seed=seed,
                norm_lower=norm_lower, norm_remove_emoji=norm_remove_emoji,
            )
        return df

    if prefer_unique:
        return prefer_unique_context_sampling(
            df, n_target,
            unique_col=unique_col, seed=seed,
            norm_lower=norm_lower, norm_remove_emoji=norm_remove_emoji,
        )
    return df.sample(n=n_target, random_state=seed).reset_index(drop=True)
