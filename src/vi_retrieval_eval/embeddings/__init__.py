# src/vi_retrieval_eval/embeddings/__init__.py

# Import các backend để kích hoạt decorator @register(...)
# Thứ tự không quan trọng, chỉ cần import để side-effect đăng ký.
from . import openai_embed  # noqa: F401
from . import gemini_embed  # noqa: F401
from . import sbert_embed   # noqa: F401
