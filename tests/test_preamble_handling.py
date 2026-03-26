from __future__ import annotations

from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalStructure, RawBlock


def test_preamble_is_not_indexed_but_is_available_in_metadata() -> None:
    blocks = [
        RawBlock(
            text="Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;",
            page=1,
            block_type="body",
            structure=LegalStructure(),
            law_name="Luật Quản lý thuế",
        ),
        RawBlock(
            text="Quốc hội ban hành Luật Quản lý thuế.",
            page=1,
            block_type="body",
            structure=LegalStructure(),
            law_name="Luật Quản lý thuế",
        ),
        RawBlock(
            text="Điều 15. Quy định chung",
            page=2,
            block_type="article",
            structure=LegalStructure(article="Điều 15. Quy định chung"),
            law_name="Luật Quản lý thuế",
        ),
        RawBlock(
            text="1. Nội dung khoản 1.",
            page=2,
            block_type="clause",
            structure=LegalStructure(article="Điều 15. Quy định chung"),
            law_name="Luật Quản lý thuế",
        ),
    ]

    chunker = LegalChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=80)
    chunks = chunker.chunk(blocks)

    assert chunks, "Expected chunks to be produced"

    # Preamble must not be embedded/indexed as chunk text.
    joined_text = "\n".join(chunk.text for chunk in chunks)
    assert "Căn cứ Hiến pháp" not in joined_text
    assert "Quốc hội ban hành" not in joined_text

    # But it must be available for display via metadata.
    for chunk in chunks:
        meta = chunk.to_chroma_metadata()
        assert "law_preamble" in meta
        assert "Căn cứ Hiến pháp" in meta["law_preamble"]
        assert "Quốc hội ban hành" in meta["law_preamble"]
