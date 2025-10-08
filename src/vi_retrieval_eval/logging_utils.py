# src/vi_retrieval_eval/logging_utils.py
import logging
import sys
from typing import Optional

LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

def setup_logger(level: str = "info", log_file: Optional[str] = None) -> logging.Logger:
    """
    Tạo logger cho vi-retrieval-eval.
    
    Args:
        level: mức log ("debug", "info", "warning", "error").
        log_file: nếu set, log thêm vào file UTF-8.
    """
    logger = logging.getLogger("vi-retrieval-eval")
    logger.setLevel(LEVEL_MAP.get(level.lower(), logging.INFO))
    logger.propagate = False  # tránh log đúp ra root logger

    # clear handler cũ
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
