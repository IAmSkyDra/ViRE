# src/vi_retrieval_eval/progress.py
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

def iter_progress(
    iterable: Iterable[T],
    enable: bool,
    tqdm_desc: Optional[str] = None,
    total: Optional[int] = None,
) -> Iterable[T]:
    """
    Wrap iterable with tqdm if enable=True, else return original iterable.
    - Không yêu cầu cài tqdm (không có thì tự động bỏ qua).
    """
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm  # lazy import
        return tqdm(iterable, desc=tqdm_desc, total=total)
    except Exception:
        return iterable  # no tqdm installed → just no bar
