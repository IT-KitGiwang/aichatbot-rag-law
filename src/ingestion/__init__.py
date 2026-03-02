# src/ingestion/__init__.py
"""Module nạp dữ liệu luật: PDF → Chunks → Embeddings → ChromaDB."""

# Chỉ import các module nhẹ (không cần torch)
from src.ingestion.pdf_processor import LegalPDFProcessor, LegalArticle
from src.ingestion.legal_chunker import LegalChunker, LegalChunk

# EmbeddingGenerator và VectorIndexer import nặng (torch, chromadb)
# → chỉ import khi thực sự cần dùng (lazy import)

__all__ = [
    "LegalPDFProcessor",
    "LegalArticle",
    "LegalChunker",
    "LegalChunk",
]
