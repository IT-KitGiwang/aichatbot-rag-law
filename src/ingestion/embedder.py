# src/ingestion/embedder.py
"""
Embedding Generator — 4.1.3 trong kế hoạch.
Chuyển đổi: list[LegalChunk] → list[vector]

Provider: Voyage AI
Model mặc định: voyage-3-large
    - Embedding API cloud, không phụ thuộc GPU local
    - Dimension mặc định: 1024
"""

from __future__ import annotations

import os
import time

from src.ingestion.legal_chunker import LegalChunk


# ===========================================================================
# BƯỚC 3.1 — Cấu hình & Load Model (Singleton)
# ===========================================================================

# Cấu hình mặc định — đồng bộ với config.yaml
_DEFAULT_CONFIG = {
    "provider":        "voyageai",
    "model":           "voyage-law-2",
    "dimension":       1024,
    "api_key_env":     "VOYAGE_API_KEY",
    "batch_size":      32,
    "max_retries":     3,
    "retry_backoff_s": 1.5,
}


class EmbeddingGenerator:
    """
    Tạo vector embedding cho LegalChunk và câu hỏi của user.

    Client được khởi tạo theo lazy loading, dùng chung giữa các instance
    để tránh tạo lại kết nối nhiều lần.

    Cách dùng:
        embedder = EmbeddingGenerator()

        # Embed văn bản
        vectors = embedder.embed_chunks(chunks)   # list[list[float]]

        # Embed câu hỏi
        q_vec   = embedder.embed_query("Điều kiện ly hôn là gì?")  # list[float]
    """

    # Dùng chung client cho mọi instance
    _client = None

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: dict cấu hình tùy chỉnh.
                    Nếu None → dùng _DEFAULT_CONFIG.
                    Chỉ cần truyền các key muốn override.
        """
        cfg = {**_DEFAULT_CONFIG, **(config or {})}

        self.provider        = cfg["provider"]
        self.model_name      = cfg["model"]
        self.dimension       = cfg["dimension"]
        self.api_key_env     = cfg["api_key_env"]
        self.batch_size      = cfg["batch_size"]
        self.max_retries     = cfg["max_retries"]
        self.retry_backoff_s = cfg["retry_backoff_s"]

        if self.provider.lower() != "voyageai":
            raise ValueError("Hiện tại chỉ hỗ trợ provider='voyageai'.")

    # ------------------------------------------------------------------
    # Lazy load Voyage client — chỉ chạy lần đầu tiên gọi _get_client()
    # ------------------------------------------------------------------

    def _get_client(self):
        """
        Trả về Voyage client.
        Nếu chưa tạo → khởi tạo lần đầu và lưu vào class variable.
        Nếu đã có rồi → trả về ngay.
        """
        if EmbeddingGenerator._client is None:
            try:
                import voyageai
            except ImportError:
                raise ImportError(
                    "Thiếu thư viện. Chạy: pip install voyageai"
                )

            api_key = os.getenv(self.api_key_env, "")
            if not api_key:
                raise ValueError(
                    f"Thiếu API key. Hãy set biến môi trường '{self.api_key_env}'."
                )

            EmbeddingGenerator._client = voyageai.Client(api_key=api_key)
            print(
                f"[EmbeddingGenerator] Voyage client sẵn sàng | "
                f"model='{self.model_name}' | dimension={self.dimension}"
            )

        return EmbeddingGenerator._client

    # ==================================================================
    # BƯỚC 3.2 — Encode batch & Public API
    # ==================================================================

    def _encode_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        """
        Encode danh sách text thành vectors qua Voyage API, chia thành batch nhỏ.

        Chia batch để tránh request quá lớn khi có nhiều chunks.
        Ví dụ: 300 texts, batch_size=32 → 10 lần encode.

        Args:
            texts: Danh sách text đầu vào.
            input_type: "document" hoặc "query".

        Returns:
            list[list[float]] — mỗi phần tử là vector dim=1024,
            thứ tự tương ứng 1-1 với input texts.
        """
        client = self._get_client()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Retry theo exponential backoff để giảm fail do lỗi tạm thời API/network.
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = client.embed(
                        texts=batch,
                        model=self.model_name,
                        input_type=input_type,
                    )
                    vectors = response.embeddings
                    if vectors and len(vectors[0]) != self.dimension:
                        raise ValueError(
                            f"Dimension mismatch: config={self.dimension}, "
                            f"voyage={len(vectors[0])}"
                        )
                    all_vectors.extend(vectors)
                    break
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    sleep_s = self.retry_backoff_s ** (attempt - 1)
                    time.sleep(sleep_s)

        return all_vectors

    def embed_chunks(self, chunks: list[LegalChunk]) -> list[list[float]]:
        """
        Tạo embedding cho danh sách LegalChunk (văn bản luật).

        Chỉ embed child chunks (is_parent=False) vì:
          - Child dùng để TÌM KIẾM → cần vector
          - Parent dùng để TRẢ VỀ context → chỉ cần text, không cần vector

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
                all_vectors.extend(self._encode_batch([chunk.text], input_type="document"))

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
                child_texts.append(chunk.text)

        # Encode tất cả child trong 1 lần gọi batch
        child_vectors = self._encode_batch(child_texts, input_type="document") if child_texts else []

        # Ghép lại: parent → zero vector, child → vector thật
        result = [[0.0] * self.dimension for _ in chunks]
        for idx, vec in zip(child_indices, child_vectors):
            result[idx] = vec

        return result

    def embed_query(self, query: str) -> list[float]:
        """
        Tạo embedding cho câu hỏi của user.

        Args:
            query: Câu hỏi của user.
                   VD: "Điều kiện để được ly hôn là gì?"

        Returns:
            list[float] — vector 1024 chiều.

        Ví dụ:
            q_vec = embedder.embed_query("Tài sản chung của vợ chồng gồm những gì?")
            # q_vec: [0.023, -0.041, ...] — 1024 số
        """
        vectors = self._encode_batch([query.strip()], input_type="query")
        return vectors[0]  # Trả về vector đơn (không phải list of lists)

