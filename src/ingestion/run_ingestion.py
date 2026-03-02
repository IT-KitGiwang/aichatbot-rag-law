"""
CLI chạy toàn bộ pipeline nạp dữ liệu luật:
  PDF → Text → Articles → Chunks → Embeddings → ChromaDB

Usage:
  python -m src.ingestion.run_ingestion                     (mặc định)
  python -m src.ingestion.run_ingestion --dry-run            (chỉ parse, không embed)
  python -m src.ingestion.run_ingestion --preview 10         (xem 10 chunks đầu)
"""

import argparse
import json
import sys
from pathlib import Path

# Thêm root vào path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env (GOOGLE_API_KEY, etc.)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.ingestion.pdf_processor import LegalPDFProcessor
from src.ingestion.legal_chunker import LegalChunker


# ──────────────────────────────────────────────
# Cấu hình các luật cần nạp
# ──────────────────────────────────────────────

LAWS_CONFIG = [
    {
        "pdf_file": "LUAT-QUAN-LY-THUE-2019.pdf",
        "law_name": "Luật Quản lý thuế",
        "law_number": "38/2019/QH14",
        "effective_date": "2020-07-01",
        "issuing_body": "Quốc hội",
        "preamble_context": (
            "LUẬT QUẢN LÝ THUẾ. "
            "Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam; "
            "Quốc hội ban hành Luật Quản lý thuế."
        ),
    },
    # Thêm luật khác ở đây
    # {
    #     "pdf_file": "luat_hon_nhan_gia_dinh_2014.pdf",
    #     "law_name": "Luật Hôn nhân và Gia đình",
    #     "law_number": "52/2014/QH13",
    #     "effective_date": "2015-01-01",
    #     ...
    # },
]


def run_ingestion(
    dry_run: bool = False,
    preview: int = 0,
    rebuild: bool = False,
):
    """Chạy pipeline nạp dữ liệu."""
    pdf_dir = ROOT / "data" / "raw_pdfs"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # ── PHASE 1: PDF Processing + Chunking ──
    for law_cfg in LAWS_CONFIG:
        pdf_path = pdf_dir / law_cfg["pdf_file"]
        if not pdf_path.exists():
            print(f"[SKIP] Không tìm thấy: {pdf_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Đang xử lý: {law_cfg['law_name']}")
        print(f"  File: {pdf_path.name}")
        print(f"{'='*60}")

        # 1a. Trích xuất & parse
        processor = LegalPDFProcessor(
            law_name=law_cfg["law_name"],
            law_number=law_cfg["law_number"],
            effective_date=law_cfg["effective_date"],
            issuing_body=law_cfg["issuing_body"],
            preamble_context=law_cfg.get("preamble_context", ""),
        )
        articles = processor.process(str(pdf_path))

        # 1b. Chunking
        chunker = LegalChunker(min_child_length=80)
        chunks = chunker.chunk_articles(articles)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[ERROR] Không có chunks nào được tạo!")
        return

    # ── Lưu preview ──
    preview_data = []
    for c in all_chunks[:max(preview, 20)]:
        preview_data.append({
            "chunk_id": c.chunk_id,
            "chunk_type": c.chunk_type,
            "text_length": len(c.text),
            "text_preview": c.text[:300] + "..." if len(c.text) > 300 else c.text,
            "metadata": c.metadata,
        })

    preview_path = processed_dir / "chunks_preview.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Preview] Đã lưu {len(preview_data)} chunks → {preview_path}")

    if preview > 0:
        print(f"\n{'='*60}")
        print(f"  PREVIEW {preview} chunks đầu tiên:")
        print(f"{'='*60}")
        for item in preview_data[:preview]:
            print(f"\n--- {item['chunk_id']} ({item['chunk_type']}) ---")
            print(f"    {item['metadata'].get('hierarchy_path', '')}")
            print(f"    Length: {item['text_length']} chars")
            print(f"    {item['text_preview'][:200]}")

    if dry_run:
        n_parent = sum(1 for c in all_chunks if c.chunk_type == "parent")
        n_search = sum(1 for c in all_chunks if c.chunk_type != "parent")
        print(f"\n[DRY RUN] Dừng ở đây — không embedding, không lưu DB.")
        print(f"  Tổng: {len(all_chunks)} chunks từ {len(LAWS_CONFIG)} luật")
        print(f"  Parent (context): {n_parent} | Search (child+grandchild): {n_search}")
        return

    # ── Tách: search chunks (embed) vs parent chunks (chỉ lưu text) ──
    search_chunks = [c for c in all_chunks if c.chunk_type != "parent"]
    parent_chunks = [c for c in all_chunks if c.chunk_type == "parent"]

    print(f"\n  Search chunks (child+grandchild): {len(search_chunks)} → sẽ embedding")
    print(f"  Parent chunks (full Điều):        {len(parent_chunks)} → lưu riêng, không embed")

    # ── PHASE 2: Embedding (chỉ search chunks) ──
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Embedding ({len(search_chunks)} search chunks)")
    print(f"{'='*60}")

    from src.ingestion.embedder import EmbeddingGenerator
    embedder = EmbeddingGenerator()
    embedded = embedder.embed_chunks(search_chunks)

    # ── Lưu parent chunks ra file (dùng khi trả context) ──
    parent_data = []
    for c in parent_chunks:
        parent_data.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
        })
    parent_path = processed_dir / "parent_chunks.json"
    with open(parent_path, "w", encoding="utf-8") as f:
        json.dump(parent_data, f, ensure_ascii=False, indent=2)
    print(f"  [Parent] Đã lưu {len(parent_data)} parent chunks → {parent_path}")

    # ── PHASE 3: Index vào ChromaDB ──
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Lưu vào ChromaDB")
    print(f"{'='*60}")

    from src.ingestion.indexer import VectorIndexer
    indexer = VectorIndexer(
        persist_directory=str(ROOT / "vectorstore"),
        collection_name="legal_documents",
    )

    if rebuild:
        print("[Rebuild] Xóa collection cũ...")
        indexer.delete_collection()
        indexer = VectorIndexer(
            persist_directory=str(ROOT / "vectorstore"),
            collection_name="legal_documents",
        )

    indexer.index_chunks(embedded)

    # ── Summary ──
    stats = indexer.get_stats()
    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN TẤT")
    print(f"  Tổng chunks trong DB: {stats['total_chunks']}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nạp PDF luật vào ChromaDB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ parse & chunk, không embedding/index")
    parser.add_argument("--preview", type=int, default=0,
                        help="Hiển thị N chunks đầu tiên")
    parser.add_argument("--rebuild", action="store_true",
                        help="Xóa collection cũ và nạp lại từ đầu")

    args = parser.parse_args()
    run_ingestion(
        dry_run=args.dry_run,
        preview=args.preview,
        rebuild=args.rebuild,
    )
