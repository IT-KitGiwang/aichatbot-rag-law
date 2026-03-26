from __future__ import annotations

from pathlib import Path

import yaml

from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalPDFProcessor


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    chunking = cfg.get("chunking") or {}

    pdf = Path("data/raw_pdfs/LUAT-QUAN-LY-THUE-2019.pdf")
    blocks = LegalPDFProcessor().process(pdf_path=str(pdf))

    chunker = LegalChunker(
        chunk_size=int(chunking.get("chunk_size", 512)),
        chunk_overlap=int(chunking.get("chunk_overlap", 50)),
        min_chunk_size=int(chunking.get("min_chunk_size", 80)),
        on_small_chunk=str(chunking.get("on_small_chunk", "merge_next")),
        include_parent_context=bool(chunking.get("include_parent_context", True)),
    )

    chunks = chunker.chunk(blocks)
    child_chunks = [c for c in chunks if not c.is_parent]
    child_chunks.sort(key=lambda c: int(c.token_count))

    print("total_chunks=", len(chunks), "child_chunks=", len(child_chunks))
    print("smallest child chunks:")
    for c in child_chunks[:25]:
        text = (c.text or "").strip()
        first_line = text.splitlines()[0] if text.splitlines() else ""
        print(
            "{token:4d}  type={typ:8s}  page={page:4d}  first_line={line}".format(
                token=int(c.token_count),
                typ=str(c.chunk_type),
                page=int(c.source_page),
                line=repr(first_line[:120]),
            )
        )


if __name__ == "__main__":
    main()
