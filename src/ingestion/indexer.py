"""
Lưu trữ và truy xuất chunks trong ChromaDB.
"""

from __future__ import annotations
import chromadb


class VectorIndexer:
    """
    Quản lý ChromaDB collection cho legal document chunks.
    Hỗ trợ: upsert, search, metadata filter, stats.
    """

    def __init__(
        self,
        persist_directory: str = "./vectorstore",
        collection_name: str = "legal_documents",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[Indexer] ChromaDB ready — "
            f"collection='{collection_name}', "
            f"existing={self.collection.count()} chunks"
        )

    # ── index ───────────────────────────────

    def index_chunks(self, embedded_chunks: list[dict]) -> int:
        """
        Upsert embedded chunks vào ChromaDB.

        Args:
            embedded_chunks: list[{chunk_id, text, embedding, metadata}]

        Returns:
            Số chunks đã lưu.
        """
        if not embedded_chunks:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk in embedded_chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["text"])

            # ChromaDB metadata chỉ chấp nhận str | int | float | bool
            meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        # Upsert theo batch 100
        batch = 100
        total = 0
        for i in range(0, len(ids), batch):
            j = min(i + batch, len(ids))
            self.collection.upsert(
                ids=ids[i:j],
                embeddings=embeddings[i:j],
                documents=documents[i:j],
                metadatas=metadatas[i:j],
            )
            total += j - i

        print(
            f"[Indexer] Upserted {total} chunks. "
            f"Total in collection: {self.collection.count()}"
        )
        return total

    # ── search ──────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Tìm kiếm nearest-neighbor trong ChromaDB.

        Args:
            query_embedding: vector câu hỏi
            top_k: số kết quả
            where: metadata filter (VD: {"chapter": "Chương I"})

        Returns:
            list[dict] — mỗi dict chứa chunk_id, text, metadata, score
        """
        params: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            params["where"] = where

        results = self.collection.query(**params)

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        return formatted

    # ── utils ───────────────────────────────

    def get_stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "total_chunks": self.collection.count(),
            "persist_dir": self.persist_directory,
        }

    def delete_collection(self):
        """Xóa toàn bộ collection (rebuild)."""
        self.client.delete_collection(self.collection_name)
        print(f"[Indexer] Deleted collection: {self.collection_name}")
