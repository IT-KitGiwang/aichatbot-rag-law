# src/ingestion/embedder.py
"""
Embedding Generator — 4.1.3 trong kế hoạch.
Chuyển đổi: list[LegalChunk] → list[vector]

Model: intfloat/multilingual-e5-large
  - Hỗ trợ tiếng Việt tốt (multilingual)
  - Dimension: 1024
  - Yêu cầu prefix: "passage: " cho văn bản, "query: " cho câu hỏi
"""

from __future__ import annotations

from src.ingestion.legal_chunker import LegalChunk


# ===========================================================================
# BƯỚC 3.1 — Cấu hình & Load Model (Singleton)
# ===========================================================================

# Cấu hình mặc định — đồng bộ với config.yaml
_DEFAULT_CONFIG = {
    "model":           "intfloat/multilingual-e5-large",
    "dimension":       1024,
    "normalize":       True,    # L2 normalize — QUAN TRỌNG cho cosine similarity
    "batch_size":      32,
    "max_length":      512,
    "query_prefix":    "query: ",    # Prefix cho câu hỏi của user
    "passage_prefix":  "passage: ",  # Prefix cho đoạn văn bản luật
}


class EmbeddingGenerator:
    """
    Tạo vector embedding cho LegalChunk và câu hỏi của user.

    Singleton pattern: model chỉ được load MỘT LẦN duy nhất,
    mọi instance dùng chung (tránh load lại ~1.2GB mỗi lần gọi).

    Lazy loading: model chỉ load khi thực sự cần embed,
    không load ngay khi khởi tạo object.

    Cách dùng:
        embedder = EmbeddingGenerator()

        # Embed văn bản (dùng "passage: " prefix)
        vectors = embedder.embed_chunks(chunks)   # list[list[float]]

        # Embed câu hỏi (dùng "query: " prefix)
        q_vec   = embedder.embed_query("Điều kiện ly hôn là gì?")  # list[float]
    """

    # Class variable: dùng chung cho tất cả instance
    # None = chưa load, SentenceTransformer = đã load
    _model = None

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: dict cấu hình tùy chỉnh.
                    Nếu None → dùng _DEFAULT_CONFIG.
                    Chỉ cần truyền các key muốn override.
        """
        cfg = {**_DEFAULT_CONFIG, **(config or {})}

        self.model_name      = cfg["model"]
        self.dimension       = cfg["dimension"]
        self.normalize       = cfg["normalize"]
        self.batch_size      = cfg["batch_size"]
        self.max_length      = cfg["max_length"]
        self.query_prefix    = cfg["query_prefix"]
        self.passage_prefix  = cfg["passage_prefix"]

    # ------------------------------------------------------------------
    # Lazy load model — chỉ chạy lần đầu tiên gọi _get_model()
    # ------------------------------------------------------------------

    def _get_model(self):
        """
        Trả về model SentenceTransformer.
        Nếu chưa load → load lần đầu và lưu vào class variable.
        Nếu đã load rồi → trả về ngay, không load lại.

        Dùng class variable EmbeddingGenerator._model (không phải self._model)
        để đảm bảo tất cả instance dùng chung 1 model.
        """
        if EmbeddingGenerator._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Thiếu thư viện. Chạy: pip install sentence-transformers"
                )

            device = self._get_device()
            print(f"[EmbeddingGenerator] Đang load model '{self.model_name}' trên {device}...")

            EmbeddingGenerator._model = SentenceTransformer(
                self.model_name,
                device=device,
            )
            print(f"[EmbeddingGenerator] Model sẵn sàng. Dimension: {self.dimension}")

        return EmbeddingGenerator._model

    @staticmethod
    def _get_device() -> str:
        """
        Tự động chọn thiết bị tính toán:
          - "cuda"  nếu có GPU (NVIDIA)
          - "mps"   nếu có Apple Silicon GPU
          - "cpu"   fallback

        GPU nhanh hơn CPU ~10-50x cho embedding.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ==================================================================
    # BƯỚC 3.2 — Encode batch & Public API
    # ==================================================================

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode danh sách text thành vectors, chia thành batch nhỏ.

        Chia batch để tránh tràn VRAM/RAM khi có nhiều chunks.
        Ví dụ: 300 texts, batch_size=32 → 10 lần encode.

        Args:
            texts: Danh sách text đã có prefix ("passage: ..." hoặc "query: ...")

        Returns:
            list[list[float]] — mỗi phần tử là vector dim=1024,
            thứ tự tương ứng 1-1 với input texts.
        """
        model = self._get_model()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # model.encode() trả về numpy.ndarray shape (batch, 1024)
            vectors = model.encode(
                batch,
                normalize_embeddings=self.normalize,  # L2 normalize
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Chuyển numpy array → list[list[float]] để ChromaDB/JSON dùng được
            all_vectors.extend(vectors.tolist())

        return all_vectors

    def embed_chunks(self, chunks: list[LegalChunk]) -> list[list[float]]:
        """
        Tạo embedding cho danh sách LegalChunk (văn bản luật).

        Chỉ embed child chunks (is_parent=False) vì:
          - Child dùng để TÌM KIẾM → cần vector
          - Parent dùng để TRẢ VỀ context → chỉ cần text, không cần vector

        Prefix "passage: " được thêm vào mỗi chunk text.
        Đây là yêu cầu bắt buộc của multilingual-e5-large.

        Args:
            chunks: list[LegalChunk] — gồm cả parent lẫn child

        Returns:
            list[list[float]] — vector cho TỪNG chunk (kể cả parent).
            Parent chunk trả về vector zero [0.0, 0.0, ...] — không dùng để search.

        Ví dụ:
            chunks = [parent_chunk, child1, child2]
            vectors = embedder.embed_chunks(chunks)
            # vectors[0] = [0.0, ..., 0.0]  ← parent, không search
            # vectors[1] = [0.12, ...]       ← child1, dùng để search
            # vectors[2] = [0.08, ...]       ← child2, dùng để search
        """
        all_vectors: list[list[float]] = []

        for chunk in chunks:
            if chunk.is_parent:
                # Parent không cần embed — trả về vector zero làm placeholder
                all_vectors.append([0.0] * self.dimension)
            else:
                # Child: thêm "passage: " prefix rồi encode
                text_with_prefix = self.passage_prefix + chunk.text
                all_vectors.extend(self._encode_batch([text_with_prefix]))

        return all_vectors

    def embed_chunks_batched(self, chunks: list[LegalChunk]) -> list[list[float]]:
        """
        Phiên bản tối ưu hơn của embed_chunks() — gom tất cả child chunks
        vào một lần gọi _encode_batch() thay vì gọi từng cái một.

        Nhanh hơn đáng kể khi có nhiều chunks vì tận dụng song song hóa GPU.

        Args:
            chunks: list[LegalChunk]

        Returns:
            list[list[float]] — vector cho từng chunk (thứ tự tương ứng 1-1).
        """
        # Tách child chunks và lưu vị trí index của chúng
        child_indices: list[int] = []
        child_texts:   list[str] = []

        for i, chunk in enumerate(chunks):
            if not chunk.is_parent:
                child_indices.append(i)
                child_texts.append(self.passage_prefix + chunk.text)

        # Encode tất cả child trong 1 lần gọi batch
        child_vectors = self._encode_batch(child_texts) if child_texts else []

        # Ghép lại: parent → zero vector, child → vector thật
        result = [[0.0] * self.dimension] * len(chunks)
        for idx, vec in zip(child_indices, child_vectors):
            result[idx] = vec

        return result

    def embed_query(self, query: str) -> list[float]:
        """
        Tạo embedding cho câu hỏi của user.

        Dùng prefix "query: " thay vì "passage: ".
        Prefix khác nhau giúp model tính similarity đúng hướng
        (câu hỏi ↔ đoạn văn bản, không phải văn bản ↔ văn bản).

        Args:
            query: Câu hỏi thuần (không có prefix).
                   VD: "Điều kiện để được ly hôn là gì?"

        Returns:
            list[float] — vector 1024 chiều, đã L2 normalize.

        Ví dụ:
            q_vec = embedder.embed_query("Tài sản chung của vợ chồng gồm những gì?")
            # q_vec: [0.023, -0.041, ...] — 1024 số
        """
        text_with_prefix = self.query_prefix + query.strip()
        vectors = self._encode_batch([text_with_prefix])
        return vectors[0]  # Trả về vector đơn (không phải list of lists)

