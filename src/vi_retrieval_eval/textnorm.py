# -*- coding: utf-8 -*-
import re
import unicodedata

# Zero-width / invisible chars
_ZW_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\ufeff",  # BOM
]

# Collapse all whitespace into a single space
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# A broad emoji / pictograph range (covers most common emoji)
_EMOJI_RE = re.compile(
    "["                       # start char class
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # enclosed characters
    "]+",
    flags=re.UNICODE
)

# Control characters (ASCII 0..31, 127) — loại bỏ cho sạch
_CTRL_RE = re.compile(r"[\x00-\x1F\x7F]")

def normalize_for_dedup(s: str, *, do_lower: bool = False, remove_emoji: bool = False) -> str:
    """
    Chuẩn hoá chuỗi để so trùng/dedup:
      - Unicode normalize NFKC
      - remove control chars (ASCII control)
      - remove zero-width/invisible
      - optional: remove emoji/icon nếu remove_emoji=True
      - strip() hai đầu
      - collapse whitespaces (một khoảng trắng)
      - optional: lower-case nếu do_lower=True
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    # Remove control chars
    s = _CTRL_RE.sub("", s)
    # Remove zero-width/invisible chars
    for ch in _ZW_CHARS:
        s = s.replace(ch, "")
    # Optional: remove emoji/icon
    if remove_emoji:
        s = _EMOJI_RE.sub("", s)
    # Trim + collapse spaces
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    # Optional lower
    if do_lower:
        s = s.lower()
    return s
