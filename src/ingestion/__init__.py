# src/ingestion/__init__.py
"""
Package ingestion — Pipeline nạp văn bản luật vào ChromaDB.

Luồng xử lý:
    PDF → LegalPDFProcessor → LegalChunker → EmbeddingGenerator → LegalIndexer

Dùng nhanh qua IngestionPipeline:
    from src.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.run("data/raw_pdfs/luat_dat_dai.pdf", law_name="Luật Đất đai 2024")
"""

from src.ingestion.pdf_processor import (
    LegalPDFProcessor,
    LegalStructure,
    RawBlock,
)

from src.ingestion.legal_chunker import (
    LegalChunker,
    LegalChunk,
)

from src.ingestion.embedder import (
    EmbeddingGenerator,
)

from src.ingestion.indexer import (
    LegalIndexer,
)

from src.ingestion.run_ingestion import (
    IngestionPipeline,
)

__all__ = [
    # pdf_processor
    "LegalPDFProcessor",
    "LegalStructure",
    "RawBlock",
    # legal_chunker
    "LegalChunker",
    "LegalChunk",
    # embedder
    "EmbeddingGenerator",
    # indexer
    "LegalIndexer",
    # run_ingestion
    "IngestionPipeline",
]
