# src/ingestion/indexer.py
"""
Legal Indexer — 4.1.4 trong kế hoạch.
Lưu LegalChunk + vector embedding vào ChromaDB.

Hai collection:
    - legal_documents         : child chunks + vectors để search
    - legal_documents_parents : parent chunks text-only để lấy thêm context
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.ingestion.legal_chunker import LegalChunk


# ===========================================================================
# BƯỚC 4.1 — Khởi tạo ChromaDB & Hai Collection
# ===========================================================================

# Tên collection mặc định (đồng bộ với config.yaml)
_DEFAULT_COLLECTION    = "legal_documents"
_DEFAULT_PERSIST_DIR   = "./vectorstore"
_DEFAULT_EMBEDDING_DIM = 1024


class LegalIndexer:
    """
    Quản lý ChromaDB — lưu và truy vấn LegalChunk.

    Hai collection:
      1. legal_documents
         - Chứa: child chunks (is_parent=False)
            - Có vector embedding để similarity search
            - Không chứa parent chunks để tránh làm nhiễu kết quả

      2. legal_documents_parents
         - Chứa: parent chunks (is_parent=True)
            - Không cần vector thật, chỉ giữ text gốc của Điều
            - Lấy theo chunk_id khi cần mở rộng ngữ cảnh trả lời

    Cách dùng:
        indexer = LegalIndexer()

        # Lưu chunks vào DB
        indexer.index_chunks(chunks, vectors)

        # Tìm kiếm
        results = indexer.query(query_vector, n_results=10)

        # Lấy parent để mở rộng context
        parent = indexer.get_parent("uuid-parent-id")
    """

    # Hậu tố để tạo tên collection phụ cho parents
    _PARENT_SUFFIX = "_parents"

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ):
        """
        Khởi tạo ChromaDB client và hai collection.

        Args:
            persist_dir:     Thư mục lưu ChromaDB xuống disk.
                             Tạo mới nếu chưa tồn tại.
            collection_name: Tên collection chính (child chunks).
                             Collection parents = collection_name + "_parents".
            embedding_dim:   Số chiều vector embedding (phải khớp với model).
        """
        self.embedding_dim = embedding_dim

        # Tạo thư mục persist nếu chưa tồn tại.
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # PersistentClient lưu DB xuống disk.
        # anonymized_telemetry=False: không gửi telemetry ẩn danh về ChromaDB.
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Collection chính: child chunks + vectors để search.
        # hnsw:space=cosine: dùng cosine distance cho embedding đã normalize.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Collection phụ: parent chunks text-only để lấy context theo ID.
        parent_collection_name = collection_name + self._PARENT_SUFFIX
        self._parent_collection = self._client.get_or_create_collection(
            name=parent_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(
            f"[LegalIndexer] ChromaDB tại '{persist_dir}' | "
            f"child count={self._collection.count()} | "
            f"parent count={self._parent_collection.count()}"
        )

    # ==================================================================
    # BƯỚC 4.2 — Ghi dữ liệu vào ChromaDB
    # ==================================================================

    def index_chunks(
        self,
        chunks: list[LegalChunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Phân loại chunks và upsert vào đúng collection.

        chunks và vectors phải cùng thứ tự, tương ứng 1-1:
            chunks[i]  ←→  vectors[i]

        Logic phân loại:
            - chunk.is_parent=True  → _parent_collection (text-only, zero vector)
            - chunk.is_parent=False → _collection (child, vector thật)

        Dùng upsert() thay vì add() để an toàn khi re-index:
        nếu chunk_id đã tồn tại → cập nhật thay vì báo lỗi.

        Args:
            chunks:  list[LegalChunk] — gồm cả parent lẫn child.
            vectors: list[list[float]] — vector tương ứng từng chunk.
                     Parent chunk có vector = [0.0, 0.0, ...] (placeholder).
        """
        if not chunks:
            print("[LegalIndexer] Không có chunk nào để index.")
            return

        if len(chunks) != len(vectors):
            raise ValueError(
                "Số lượng chunks và vectors không khớp: "
                f"chunks={len(chunks)}, vectors={len(vectors)}"
            )

        # Tách riêng child và parent để upsert vào hai collection khác nhau.
        child_ids, child_texts, child_vectors, child_metas = [], [], [], []
        parent_ids, parent_texts, parent_metas = [], [], []

        for chunk, vec in zip(chunks, vectors):
            if chunk.is_parent:
                parent_ids.append(chunk.chunk_id)
                parent_texts.append(chunk.text)
                parent_metas.append(chunk.to_chroma_metadata())
            else:
                child_ids.append(chunk.chunk_id)
                child_texts.append(chunk.text)
                child_vectors.append(vec)           # vector thật của child chunk
                child_metas.append(chunk.to_chroma_metadata())

            # Upsert child chunks vào collection chính.
        if child_ids:
            self._upsert_batch(
                collection=self._collection,
                ids=child_ids,
                documents=child_texts,
                embeddings=child_vectors,
                metadatas=child_metas,
            )
            print(f"[LegalIndexer] Đã index {len(child_ids)} child chunks")

        # Upsert parent chunks vào collection phụ.
        if parent_ids:
            self._upsert_batch(
                collection=self._parent_collection,
                ids=parent_ids,
                documents=parent_texts,
                metadatas=parent_metas,
            )
            print(f"[LegalIndexer] Đã lưu {len(parent_ids)} parent chunks")

    def _upsert_batch(
        self,
        collection,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: Optional[list[list[float]]] = None,
        batch_size: int = 100,
    ) -> None:
        """
        Upsert dữ liệu vào ChromaDB collection, chia thành batch nhỏ.

        Chia batch để tránh timeout hoặc lỗi khi upsert số lượng lớn.
        Ví dụ: 500 chunks, batch_size=100 → 5 lần upsert.

        Args:
            collection: ChromaDB collection object.
            ids:        list chunk_id (string UUID).
            documents:  list text nội dung chunk.
            embeddings: list vector float.
            metadatas:  list dict metadata (phẳng, không có None).
            batch_size: Số chunk tối đa mỗi lần upsert.
        """
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end] if embeddings is not None else None,
            )
            print(f"[LegalIndexer]   upsert {end}/{total} chunks")

    def count(self) -> dict:
        """
        Trả về số lượng chunk hiện có trong cả hai collection.

        Returns:
            {"children": int, "parents": int}

        Hữu ích để kiểm tra sau khi index xong:
            indexer.count()
            # → {"children": 245, "parents": 38}
        """
        return {
            "children": self._collection.count(),
            "parents":  self._parent_collection.count(),
        }

    # ==================================================================
    # BƯỚC 4.3 — Truy vấn và xóa dữ liệu
    # ==================================================================

    def query(
        self,
        query_vector: list[float],
        n_results: int = 15,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Tìm kiếm top-N child chunks gần nhất với query_vector.

        Chỉ tìm trong _collection (child chunks) — không tìm trong parents.

        Args:
            query_vector: Vector embedding của câu hỏi người dùng.
                          Độ dài phải khớp embedding_dim (1024).
            n_results:    Số kết quả muốn trả về. Mặc định 15 để
                          retriever có nhiều ứng viên trước khi rerank.
            where:        Filter metadata tùy chọn.
                          Ví dụ: {"law_name": "Luật Đất đai 2024"}
                          hoặc: {"article": "Điều 167."}
                          None = không lọc, tìm toàn bộ collection.

        Returns:
            list[dict], mỗi phần tử gồm:
                - "id":       chunk_id (string UUID)
                - "text":     nội dung chunk
                - "metadata": dict metadata phẳng
                - "score":    float ∈ [0, 1], càng cao càng giống query
                              score = 1 - cosine_distance
        """
        kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        # Chroma trả về list-of-lists vì hỗ trợ multi-query.
        # Ở đây chỉ gửi 1 query nên lấy phần tử [0].
        ids        = raw["ids"][0]
        documents  = raw["documents"][0]
        metadatas  = raw["metadatas"][0]
        distances  = raw["distances"][0]

        results = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            results.append({
                "id":       chunk_id,
                "text":     text,
                "metadata": meta,
                "score":    1.0 - dist,   # cosine distance → similarity score
            })

        return results

    def query_with_parent_context(
        self,
        query_vector: list[float],
        n_results: int = 15,
        where: Optional[dict] = None,
        min_score: float = 0.55,
    ) -> list[dict]:
        """
        Truy vấn child chunks rồi kéo parent full text cho từng kết quả.

        Đây là luồng dùng cho LLM:
          1. tìm child chunk gần nhất với query
          2. lấy parent full text theo parent_chunk_id
          3. trả về cả child + parent để ghép context

        Returns:
            list[dict] với các khóa:
                - id
                - text
                - metadata
                - score
                - parent: dict | None
        """
        child_results = self.query(query_vector=query_vector, n_results=n_results, where=where)
        parent_cache: dict[str, Optional[dict]] = {}
        enriched: list[dict] = []

        for child in child_results:
            parent_id = child.get("metadata", {}).get("parent_chunk_id", "")
            parent = None
            if parent_id and float(child.get("score", 0.0)) >= min_score:
                if parent_id not in parent_cache:
                    parent_cache[parent_id] = self.get_parent(parent_id)
                parent = parent_cache[parent_id]

            enriched.append({
                **child,
                "parent": parent,
            })

        return enriched

    def get_parent(self, parent_chunk_id: str) -> Optional[dict]:
        """
        Lấy toàn bộ parent chunk (tức toàn bộ Điều) theo ID.

        Dùng sau query() khi cần mở rộng ngữ cảnh cho một child chunk:
            child = results[0]
            if child["metadata"].get("parent_chunk_id"):
                parent = indexer.get_parent(child["metadata"]["parent_chunk_id"])

        Args:
            parent_chunk_id: chunk_id của parent (lấy từ metadata child chunk).

        Returns:
            dict gồm "id", "text", "metadata" nếu tìm thấy.
            None nếu ID không tồn tại trong collection.
        """
        raw = self._parent_collection.get(
            ids=[parent_chunk_id],
            include=["documents", "metadatas"],
        )

        # Nếu không tìm thấy, ChromaDB trả về list rỗng
        if not raw["ids"]:
            return None

        return {
            "id":       raw["ids"][0],
            "text":     raw["documents"][0],
            "metadata": raw["metadatas"][0],
        }

    def delete_by_law(self, law_name: str) -> int:
        """
        Xóa toàn bộ chunks (child + parent) thuộc về một bộ luật.

        Hữu ích khi cần re-index một văn bản luật đã thay đổi.
        ChromaDB không hỗ trợ delete theo filter trực tiếp, nên phải lấy IDs
        trước rồi mới gọi delete(ids=[...]).

        Args:
            law_name: Tên bộ luật đúng như lúc index.
                      Ví dụ: "Luật Đất đai 2024"

        Returns:
            Tổng số chunks đã xóa (child + parent).
        """
        where_filter = {"law_name": law_name}
        total_deleted = 0

        # Xóa child chunks trước.
        child_raw = self._collection.get(
            where=where_filter,
            include=[],      # Chỉ cần IDs, không cần documents/metadatas.
        )
        if child_raw["ids"]:
            self._collection.delete(ids=child_raw["ids"])
            total_deleted += len(child_raw["ids"])
            print(f"[LegalIndexer] Xóa {len(child_raw['ids'])} child chunks của '{law_name}'")

        # Xóa parent chunks sau.
        parent_raw = self._parent_collection.get(
            where=where_filter,
            include=[],
        )
        if parent_raw["ids"]:
            self._parent_collection.delete(ids=parent_raw["ids"])
            total_deleted += len(parent_raw["ids"])
            print(f"[LegalIndexer] Xóa {len(parent_raw['ids'])} parent chunks của '{law_name}'")

        if total_deleted == 0:
            print(f"[LegalIndexer] Không tìm thấy chunk nào của '{law_name}'")

        return total_deleted
