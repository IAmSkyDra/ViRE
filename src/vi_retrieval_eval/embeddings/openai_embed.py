# src/vi_retrieval_eval/embeddings/openai_embed.py
from typing import List, Optional
import os
import time
import logging
import numpy as np

from .base import register
from ..progress import iter_progress


@register("openai")
class OpenAIEmbedder:
    """
    Dùng OpenAI embeddings (SDK >= 1.0).
    Yêu cầu:
      - Env: OPENAI_API_KEY
      - pip install openai>=1.0
    """

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 128,
        show_progress: bool = False,
        normalize: bool = True,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install openai>=1.0` to use OpenAI embeddings") from e

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY in environment")

        # Khởi tạo client; hỗ trợ org/project nếu có
        self.client = OpenAI(organization=organization, project=project)  # kwargs None sẽ bị bỏ qua
        self.model = model
        self.batch_size = int(batch_size)
        self.show_progress = bool(show_progress)
        self.normalize = bool(normalize)
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)

        self._logger = logging.getLogger("vi-retrieval-eval")

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Gọi embeddings.create cho 1 batch, có retry/backoff.
        Trả về np.ndarray (B, D) float32 (chưa normalize).
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                vecs = [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]
                return np.stack(vecs, axis=0).astype(np.float32)
            except Exception as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    self._logger.debug(
                        f"[OpenAI] retry {attempt+1}/{self.max_retries-1} in {delay:.1f}s due to: {e}"
                    )
                    time.sleep(delay)
                else:
                    break
        raise last_err if last_err is not None else RuntimeError("Unknown OpenAI error")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Trả về embeddings cho toàn bộ `texts`.
        - Có progress bar theo batch nếu show_progress=True (cần `tqdm`).
        - Normalize L2 (mặc định) để dot = cosine (hợp với FAISS IP).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: List[np.ndarray] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        it = iter_progress(
            range(0, len(texts), self.batch_size),
            enable=self.show_progress,
            tqdm_desc=f"OpenAI {self.model}",
            total=total_batches,
        )
        for i in it:
            batch = texts[i : i + self.batch_size]
            arr = self._embed_batch(batch)  # (B, D)
            out_chunks.append(arr)

        arr_all = np.concatenate(out_chunks, axis=0).astype(np.float32)

        if self.normalize and arr_all.size > 0:
            norms = np.linalg.norm(arr_all, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr_all = (arr_all / norms).astype(np.float32)

        return arr_all

    @property
    def dim(self) -> int:
        """Chiều embedding (gọi 1 batch nhỏ để đo)."""
        tmp = self._embed_batch(["dimension probe"])
        return int(tmp.shape[1])
