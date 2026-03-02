"""
Tạo vector embeddings cho text chunks.
Dùng Google Embedding API — nhẹ, không cần PyTorch/GPU.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import google.generativeai as genai

from src.ingestion.legal_chunker import LegalChunk


class EmbeddingGenerator:
    """
    Tạo embeddings bằng Google Embedding API.
    Hỗ trợ task_type riêng cho document vs query → tăng chất lượng retrieval.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-embedding-001",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "Thiếu GOOGLE_API_KEY. Set trong .env hoặc truyền qua api_key."
            )
        genai.configure(api_key=key)
        self.dimension = 768  # gemini-embedding-001 default
        print(f"[Embedder] Google API model: {model_name} — dimension={self.dimension}")

    # ── embed chunks ────────────────────────

    def embed_chunks(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 100,
        passage_prefix: str = "",
    ) -> list[dict]:
        """
        Tạo embeddings cho danh sách chunks.

        Returns:
            list[dict] — mỗi dict chứa:
              chunk_id, text, embedding (list[float]), metadata
        """
        texts = [passage_prefix + c.text for c in chunks]
        print(f"[Embedder] Encoding {len(texts)} chunks via Google API ...")

        # Google API hỗ trợ batch embed (tối đa ~100 texts/request)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document",
            )
            all_embeddings.extend(result["embedding"])

            done = min(i + batch_size, len(texts))
            print(f"  [{done}/{len(texts)}]")

            # Rate limiting: free tier = 1,500 req/ngày
            if i + batch_size < len(texts):
                time.sleep(0.5)

        results = []
        for chunk, emb in zip(chunks, all_embeddings):
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": emb,
                "metadata": chunk.metadata,
            })

        print(f"[Embedder] Done — {len(results)} embeddings")
        return results

    # ── embed query ─────────────────────────

    def embed_query(self, query: str, query_prefix: str = "") -> list[float]:
        """Embed 1 câu hỏi (dùng khi retrieval)."""
        result = genai.embed_content(
            model=self.model_name,
            content=query_prefix + query,
            task_type="retrieval_query",
        )
        return result["embedding"]
