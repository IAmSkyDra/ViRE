import re
from typing import List

HAVE_UTS = False
try:
    from underthesea import word_tokenize as uts_word_tokenize
    HAVE_UTS = True
except Exception:
    HAVE_UTS = False

def regex_tokens(s: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ_]+", s or "")

def vi_segment(s: str) -> List[str]:
    if HAVE_UTS:
        return uts_word_tokenize(s or "", format="text").split()
    return regex_tokens(s or "")
