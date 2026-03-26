from __future__ import annotations

import re

from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalStructure, RawBlock


def test_clause_point_structure_is_not_broken_when_splitting() -> None:
    structure = LegalStructure(chapter="Chương I", article="Điều 1.")

    article_block = RawBlock(
        text="Điều 1.",
        page=1,
        block_type="article",
        structure=structure,
    )

    long_point_payload = " ".join(["nội_dung"] * 120)
    clause_text = (
        "1. Người nào đó thực hiện nghĩa vụ sau đây:\n"
        f"a) {long_point_payload}\n"
        f"b) {long_point_payload}\n"
        f"c) {long_point_payload}"
    )

    child_blocks = [
        RawBlock(
            text=clause_text,
            page=1,
            block_type="clause",
            structure=structure,
        )
    ]

    # Force splitting with a small chunk size.
    chunker = LegalChunker(
        chunk_size=120,
        chunk_overlap=10,
        min_chunk_size=1,
        on_small_chunk="merge_next",
        include_parent_context=False,
    )

    chunks = chunker._process_article(article_block, child_blocks)

    child_chunks = [chunk for chunk in chunks if (not chunk.is_parent and chunk.chunk_type == "clause")]
    assert child_chunks, "Expected child chunks to be produced for the long clause."

    # No child chunk should start with a bare point label like "b)".
    bad_start = re.compile(r"^[a-zđ]\)\b", re.IGNORECASE)
    for chunk in child_chunks:
        stripped = chunk.text.lstrip()
        assert not bad_start.match(stripped), f"Chunk starts with point label unexpectedly: {stripped[:40]!r}"

        # If this is a split segment, it must still carry the owning clause header "1.".
        assert "1." in stripped[:60], f"Missing clause header in chunk prefix: {stripped[:80]!r}"
