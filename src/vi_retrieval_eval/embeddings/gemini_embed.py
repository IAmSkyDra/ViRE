# src/vi_retrieval_eval/embeddings/gemini_embed.py
from typing import List, Optional
import os
import time
import logging
import numpy as np

from .base import register
from ..progress import iter_progress


@register("gemini")
class GeminiEmbedder:
    """
    Dùng Google Gemini embeddings.

    Yêu cầu:
      - Env: GOOGLE_API_KEY
      - pip install google-generativeai
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        batch_size: int = 128,
        show_progress: bool = False,
        normalize: bool = True,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
    ):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError(
                "Please `pip install google-generativeai` to use Gemini embeddings"
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY in environment")

        genai.configure(api_key=api_key)

        self.model_name = model
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.normalize = normalize
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.genai = genai
        self._logger = logging.getLogger("vi-retrieval-eval")

    def _embed_one(self, text: str) -> np.ndarray:
        """
        Gọi 1 lần API embed_content với retry/backoff.
        Trả về vector np.float32.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = self.genai.embed_content(model=self.model_name, content=text)
                v = np.asarray(r["embedding"], dtype=np.float32)
                return v
            except Exception as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt)
                    self._logger.debug(
                        f"[Gemini] retry {attempt+1}/{self.max_retries-1} in {delay:.1f}s due to: {e}"
                    )
                    time.sleep(delay)
                else:
                    break
        # hết retry
        raise last_err if last_err is not None else RuntimeError("Unknown Gemini error")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embedding cho danh sách texts, có progress bar:
          - Progress batch ngoài (chia theo batch_size)
          - Progress từng call trong batch (do API chưa hỗ trợ batch ổn định)
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out: List[np.ndarray] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        outer = iter_progress(
            range(0, len(texts), self.batch_size),
            enable=self.show_progress,
            tqdm_desc=f"Gemini {self.model_name}",
            total=total_batches,
        )

        for i in outer:
            batch = texts[i : i + self.batch_size]
            vecs: List[np.ndarray] = []

            inner = iter_progress(
                batch,
                enable=self.show_progress,
                tqdm_desc="Gemini calls",
                total=len(batch),
            )

            for t in inner:
                v = self._embed_one(t)
                vecs.append(v)

            out.extend(vecs)

        arr = np.stack(out, axis=0).astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = (arr / norms).astype(np.float32)

        return arr

    @property
    def dim(self) -> int:
        """Tiện ích: chiều embedding (gọi 1 lần nhẹ)."""
        v = self._embed_one("dimension probe")
        return int(v.shape[0])
