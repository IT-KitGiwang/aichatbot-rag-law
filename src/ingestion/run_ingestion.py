# src/ingestion/run_ingestion.py
# Script chạy toàn bộ pipeline ingestion: PDF → Chunks → Embeddings → ChromaDB
# Usage: python -m src.ingestion.run_ingestion --pdf data/raw_pdfs/luat_hon_nhan.pdf

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.ingestion.pdf_processor import LegalPDFProcessor
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.indexer import LegalIndexer


# ===========================================================================
# BƯỚC 4.4 — IngestionPipeline: Kết nối 4 module
# ===========================================================================

class IngestionPipeline:
    """
    Orchestrator chạy toàn bộ pipeline ingestion theo thứ tự:

        PDF → RawBlock → LegalChunk → vector → ChromaDB

    Cách dùng đơn giản:
        pipeline = IngestionPipeline()
        pipeline.run("data/raw_pdfs/luat_dat_dai.pdf", law_name="Luật Đất đai 2024")

    Cách dùng cả thư mục:
        pipeline.run_directory("data/raw_pdfs/")
    """

    def __init__(
        self,
        persist_dir: str = "./vectorstore",
        collection_name: str = "legal_documents",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ):
        """
        Khởi tạo 4 module pipeline.

        Args:
            persist_dir:     Thư mục lưu ChromaDB.
            collection_name: Tên collection ChromaDB.
            chunk_size:      Token tối đa mỗi child chunk.
            chunk_overlap:   Token overlap khi chia child chunks.
            min_chunk_size:  Token tối thiểu để giữ lại một chunk.
        """
        print("[Pipeline] Khởi tạo pipeline...")

        self.processor = LegalPDFProcessor()

        self.chunker = LegalChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

        # EmbeddingGenerator dùng Singleton — model chỉ load 1 lần
        self.embedder = EmbeddingGenerator()

        self.indexer = LegalIndexer(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        print("[Pipeline] Sẵn sàng.\n")

    def run(
        self,
        pdf_path: str,
        law_name: str = "",
        law_number: str = "",
        effective_date: str = "",
        overwrite: bool = False,
    ) -> dict:
        """
        Chạy pipeline cho 1 file PDF.

        Args:
            pdf_path:       Đường dẫn đến file PDF.
            law_name:       Tên bộ luật (ví dụ: "Luật Đất đai 2024").
                            Nếu để trống, tự suy ra từ tên file.
            law_number:     Số hiệu văn bản (ví dụ: "45/2024/QH15").
            effective_date: Ngày có hiệu lực (ví dụ: "2025-01-01").
            overwrite:      True = xóa dữ liệu cũ theo law_name trước khi index.

        Returns:
            dict thống kê: {"file", "blocks", "chunks", "children", "parents", "elapsed_s"}
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {pdf_path}")

        print(f"{'='*60}")
        print(f"[Pipeline] Xử lý: {pdf_path.name}")
        t0 = time.time()

        # --- Bước 1: Extract PDF → RawBlock ---
        print("[Pipeline] Bước 1/4: Đọc và phân tích PDF...")
        blocks = self.processor.process(
            pdf_path=str(pdf_path),
            law_name=law_name,
            law_number=law_number,
            effective_date=effective_date,
        )
        print(f"[Pipeline]   → {len(blocks)} blocks")

        if not blocks:
            print("[Pipeline] Không có nội dung. Bỏ qua file này.")
            return {"file": str(pdf_path), "blocks": 0, "chunks": 0,
                    "children": 0, "parents": 0, "elapsed_s": 0}

        # --- Bước 2: Chunk → LegalChunk ---
        print("[Pipeline] Bước 2/4: Chia chunks...")
        chunks = self.chunker.chunk(blocks)
        n_children = sum(1 for c in chunks if not c.is_parent)
        n_parents  = sum(1 for c in chunks if c.is_parent)
        print(f"[Pipeline]   → {len(chunks)} chunks ({n_children} child, {n_parents} parent)")

        # --- Bước 3: Tạo embeddings ---
        print("[Pipeline] Bước 3/4: Tạo embeddings (có thể mất vài phút)...")
        vectors = self.embedder.embed_chunks_batched(chunks)
        print(f"[Pipeline]   → {len(vectors)} vectors (dim={len(vectors[0])})")

        # --- Bước 4: Ghi vào ChromaDB ---
        # Nếu overwrite=True, xóa dữ liệu cũ trước
        actual_law_name = law_name or self.processor.extract_metadata_from_filename(str(pdf_path))
        if overwrite and actual_law_name:
            print(f"[Pipeline] Xóa dữ liệu cũ của '{actual_law_name}'...")
            deleted = self.indexer.delete_by_law(actual_law_name)
            print(f"[Pipeline]   → Đã xóa {deleted} chunks cũ")

        print("[Pipeline] Bước 4/4: Ghi vào ChromaDB...")
        self.indexer.index_chunks(chunks, vectors)

        elapsed = round(time.time() - t0, 1)
        counts = self.indexer.count()
        print(f"[Pipeline] Xong! {elapsed}s | DB hiện có: {counts}")
        print(f"{'='*60}\n")

        return {
            "file":      str(pdf_path),
            "blocks":    len(blocks),
            "chunks":    len(chunks),
            "children":  n_children,
            "parents":   n_parents,
            "elapsed_s": elapsed,
        }

    def run_directory(
        self,
        pdf_dir: str,
        overwrite: bool = False,
    ) -> list[dict]:
        """
        Quét thư mục và chạy pipeline cho tất cả file .pdf tìm thấy.

        Metadata (law_name, law_number, effective_date) sẽ được suy ra
        tự động từ tên file cho từng PDF.

        Args:
            pdf_dir:   Đường dẫn thư mục chứa PDF.
            overwrite: Truyền vào run() cho từng file.

        Returns:
            list[dict] thống kê của từng file đã xử lý.
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.is_dir():
            raise NotADirectoryError(f"Không phải thư mục: {pdf_dir}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[Pipeline] Không tìm thấy file .pdf trong '{pdf_dir}'")
            return []

        print(f"[Pipeline] Tìm thấy {len(pdf_files)} file PDF trong '{pdf_dir}'")
        all_stats = []

        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[Pipeline] [{i}/{len(pdf_files)}] {pdf_path.name}")
            try:
                stats = self.run(str(pdf_path), overwrite=overwrite)
                all_stats.append(stats)
            except Exception as exc:
                print(f"[Pipeline] LỖI khi xử lý '{pdf_path.name}': {exc}")
                all_stats.append({"file": str(pdf_path), "error": str(exc)})

        # Tổng kết
        total_chunks = sum(s.get("chunks", 0) for s in all_stats)
        total_time   = sum(s.get("elapsed_s", 0) for s in all_stats)
        print(f"\n[Pipeline] HOÀN TẤT: {len(pdf_files)} files | "
              f"{total_chunks} chunks | {total_time}s")

        return all_stats


# ===========================================================================
# CLI — Chạy từ terminal
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingestion pipeline: PDF → ChromaDB",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Nguồn dữ liệu (chọn 1 trong 2)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pdf",
        metavar="PATH",
        help="Đường dẫn đến 1 file PDF cần index.",
    )
    source.add_argument(
        "--pdf-dir",
        metavar="DIR",
        help="Thư mục chứa nhiều file PDF cần index.",
    )

    # Metadata bổ sung (chỉ dùng với --pdf)
    parser.add_argument("--law-name",       default="", help="Tên bộ luật.")
    parser.add_argument("--law-number",     default="", help="Số hiệu văn bản.")
    parser.add_argument("--effective-date", default="", help="Ngày có hiệu lực (YYYY-MM-DD).")

    # Tùy chọn
    parser.add_argument(
        "--clear",
        metavar="LAW_NAME",
        default=None,
        help="Xóa bộ luật này khỏi DB trước khi index (re-index).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa dữ liệu cũ của cùng law_name trước khi index.",
    )
    parser.add_argument("--persist-dir",     default="./vectorstore", help="Thư mục ChromaDB.")
    parser.add_argument("--collection-name", default="legal_documents", help="Tên collection.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    # --clear: xóa 1 bộ luật thủ công
    if args.clear:
        print(f"[CLI] Xóa '{args.clear}' khỏi ChromaDB...")
        deleted = pipeline.indexer.delete_by_law(args.clear)
        print(f"[CLI] Đã xóa {deleted} chunks.")

    if args.pdf:
        pipeline.run(
            pdf_path=args.pdf,
            law_name=args.law_name,
            law_number=args.law_number,
            effective_date=args.effective_date,
            overwrite=args.overwrite,
        )
    elif args.pdf_dir:
        pipeline.run_directory(
            pdf_dir=args.pdf_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
