# src/vi_retrieval_eval/embeddings/sbert_embed.py
from typing import List, Optional
import logging
import numpy as np

from .base import register
from ..progress import iter_progress


@register("sbert")
class SBERTEmbedder:
    """
    Sentence-Transformers (local).

    Yêu cầu:
      - pip install sentence-transformers
      - GPU (tuỳ chọn): tự dùng 'cuda' nếu có, hoặc bạn có thể chỉ định device.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
        device: Optional[str] = None,
        show_progress: bool = False,
        normalize: bool = True,
        min_batch_size: int = 8,  # chống OOM: không giảm dưới mức này
    ):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError("Please `pip install sentence-transformers` to use SBERT") from e

        # auto-select device nếu không chỉ định
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.show_progress = bool(show_progress)
        self.normalize = bool(normalize)
        self.min_batch_size = int(min_batch_size)

        self._logger = logging.getLogger("vi-retrieval-eval")

        # load model
        self._logger.debug(f"[SBERT] loading model `{model_name}` on `{device}` ...")
        self._st = SentenceTransformer(model_name, device=device, trust_remote_code=True)


    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Encode một batch (đã chia) → np.ndarray (B, D), có normalize nếu bật.
        """
        # show_progress_bar=False để mình tự quản lý tqdm
        arr = self._st.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalize thủ công để thống nhất với các backend
            show_progress_bar=False,
        ).astype(np.float32)

        if self.normalize and arr.size > 0:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            arr = (arr / norms).astype(np.float32)

        return arr

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Trả về embeddings cho toàn bộ `texts` với progress bar theo batch và chống OOM.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out_chunks: List[np.ndarray] = []
        bs = int(self.batch_size)
        total_batches = (len(texts) + bs - 1) // bs

        it = iter_progress(
            range(0, len(texts), bs),
            enable=self.show_progress,
            tqdm_desc=f"SBERT {self.model_name}",
            total=total_batches,
        )

        # Loop qua các batch; nếu OOM, giảm batch_size rồi retry batch hiện tại
        for start in it:
            while True:
                try:
                    batch = texts[start : start + bs]
                    arr = self._encode_batch(batch, batch_size=bs)
                    out_chunks.append(arr)
                    break  # xong batch này
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda" in msg) and bs > self.min_batch_size:
                        new_bs = max(self.min_batch_size, bs // 2)
                        self._logger.warning(
                            f"[SBERT] OOM detected with batch_size={bs}. Retrying with batch_size={new_bs}."
                        )
                        bs = new_bs
                        # cập nhật tổng số batch ước lượng cho tqdm (không cần quá chính xác)
                        continue
                    raise  # lỗi khác → ném ra ngoài

        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @property
    def dim(self) -> int:
        """Chiều embedding (encode 1 câu để đo)."""
        arr = self._encode_batch(["dimension probe"], batch_size=1)
        return int(arr.shape[1])
