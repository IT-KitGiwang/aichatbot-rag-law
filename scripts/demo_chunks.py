#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo script hiển thị chunk ra terminal, xuất JSON,
và thống kê chunk ngắn/dài theo token.

Usage:
    python scripts/demo_chunks.py --pdf <path_to_pdf>
    python scripts/demo_chunks.py --from-vectorstore
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_chunks")


def _chunk_obj_to_dict(chunk: Any) -> dict:
    """Chuyển object chunk sang dict chuẩn để xử lý và export JSON."""
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "is_parent": chunk.is_parent,
        "parent_chunk_id": chunk.parent_chunk_id,
        "source_page": chunk.source_page,
        "token_count": chunk.token_count,
        "law_name": chunk.law_name,
        "law_number": chunk.law_number,
        "effective_date": chunk.effective_date,
        "part": chunk.part,
        "chapter": chunk.chapter,
        "chapter_title": chunk.chapter_title,
        "section": chunk.section,
        "article": chunk.article,
        "article_title": chunk.article_title,
        "hierarchy_path": chunk.hierarchy_path,
        "text": chunk.text,
    }


def demo_from_pdf(
    pdf_path: str,
    law_name: str = "Luật Mẫu",
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    min_chunk_size: int = 100,
) -> List[dict]:
    """
    Chạy pipeline ingestion trên PDF và trả về 10 chunk đầu tiên.
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        law_name: Tên luật để gán cho chunks
        
    Returns:
        List of first 10 LegalChunk
    """
    # Lazy import để tránh phụ thuộc fitz khi không dùng chế độ --pdf
    from src.ingestion.pdf_processor import LegalPDFProcessor
    from src.ingestion.legal_chunker import LegalChunker

    logger.info(f"📄 Loading PDF: {pdf_path}")
    
    # Bước 1: PDFProcessor
    pdf_processor = LegalPDFProcessor()
    raw_blocks = pdf_processor.process(
        pdf_path=pdf_path,
        law_name=law_name,
    )
    logger.info(f"✓ Extracted {len(raw_blocks)} raw blocks")
    
    # Bước 2: LegalChunker
    chunker = LegalChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    chunks = chunker.chunk(raw_blocks)
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    return [_chunk_obj_to_dict(c) for c in chunks]


def demo_from_vectorstore(
    persist_dir: str = "./vectorstore",
    collection_name: str = "legal_documents",
) -> List[dict]:
    """
    Lấy 10 chunk đầu tiên từ ChromaDB vectorstore.
    
    Returns:
        List of chunk dict (nếu có dữ liệu trong vectorstore)
    """
    try:
        chromadb = importlib.import_module("chromadb")
        
        logger.info("📦 Connecting to ChromaDB vectorstore...")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=collection_name)
        
        # Lấy tất cả chunks (tối đa 1000)
        results = collection.get(limit=1000)
        logger.info(f"✓ Retrieved {len(results['ids'])} chunks from vectorstore")
        
        # Chuyển đổi thành dict chunks
        chunks = []
        for doc_id, metadata, document in zip(
            results['ids'],
            results['metadatas'],
            results['documents']
        ):
            chunks.append({
                "chunk_id": doc_id,
                "text": document,
                "chunk_type": metadata.get('chunk_type', ''),
                "source_page": int(metadata.get('source_page', 0)),
                "law_name": metadata.get('law_name', ''),
                "law_number": metadata.get('law_number', ''),
                "effective_date": metadata.get('effective_date', ''),
                "part": metadata.get('part', ''),
                "chapter": metadata.get('chapter', ''),
                "chapter_title": metadata.get('chapter_title', ''),
                "section": metadata.get('section', ''),
                "article": metadata.get('article', ''),
                "article_title": metadata.get('article_title', ''),
                "hierarchy_path": metadata.get('hierarchy_path', ''),
                "is_parent": metadata.get('is_parent', False),
                "parent_chunk_id": metadata.get('parent_chunk_id', None),
                "token_count": int(metadata.get('token_count', 0)),
            })
        
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Failed to load from vectorstore: {e}")
        return []


def calc_stats(chunks: List[dict], short_max: int, long_min: int) -> dict:
    """Tính thống kê tổng quát và phân loại chunk ngắn/dài."""
    if not chunks:
        return {
            "total": 0,
            "parent": 0,
            "child": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "short_count": 0,
            "medium_count": 0,
            "long_count": 0,
            "short_max": short_max,
            "long_min": long_min,
        }

    token_counts = [int(c.get("token_count", 0)) for c in chunks]
    short_count = sum(1 for c in chunks if int(c.get("token_count", 0)) <= short_max)
    long_count = sum(1 for c in chunks if int(c.get("token_count", 0)) >= long_min)
    medium_count = len(chunks) - short_count - long_count

    return {
        "total": len(chunks),
        "parent": sum(1 for c in chunks if bool(c.get("is_parent", False))),
        "child": sum(1 for c in chunks if not bool(c.get("is_parent", False))),
        "total_tokens": sum(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 2),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "short_count": short_count,
        "medium_count": medium_count,
        "long_count": long_count,
        "short_max": short_max,
        "long_min": long_min,
    }


def write_json_output(output_json: str, all_chunks: List[dict], stats: dict) -> None:
    """Ghi kết quả ra file JSON."""
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": stats,
        "chunks": all_chunks,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nJSON output: {out_path}")


def print_chunk_details(chunk: dict, index: int) -> None:
    """
    In chi tiết của một chunk theo định dạng dễ đọc.
    
    Args:
        chunk: LegalChunk object
        index: Chỉ số của chunk (bắt đầu từ 1)
    """
    print(f"\n{'='*80}")
    print(f"CHUNK #{index}")
    print(f"{'='*80}")
    
    # Thông tin cơ bản
    print(f"\n📋 ID: {chunk.get('chunk_id', '')}")
    print(f"📝 Type: {chunk.get('chunk_type', '')} | Parent: {chunk.get('is_parent', False)}")
    print(f"📄 Page: {chunk.get('source_page', 0)} | Tokens: {chunk.get('token_count', 0)}")
    
    # Thông tin luật
    if chunk.get("law_name"):
        print(f"\n⚖️  Luật: {chunk.get('law_name')}")
    if chunk.get("law_number"):
        print(f"📌 Số hiệu: {chunk.get('law_number')}")
    if chunk.get("effective_date"):
        print(f"📅 Ngày có hiệu lực: {chunk.get('effective_date')}")
    
    # Vị trí trong cấu trúc
    if chunk.get("hierarchy_path"):
        print(f"\n🏛️  Cấu trúc: {chunk.get('hierarchy_path')}")
    if chunk.get("article"):
        print(f"📍 Điều: {chunk.get('article')}")
    if chunk.get("article_title"):
        print(f"   Tiêu đề: {chunk.get('article_title')}")
    
    # Nội dung chunk
    print(f"\n📄 CONTENT:")
    print("-" * 80)
    # In 500 ký tự đầu tiên, nếu dài hơn thì thêm ...
    text = str(chunk.get("text", ""))
    text_preview = text[:500]
    if len(text) > 500:
        text_preview += "\n... [content truncated]"
    print(text_preview)
    print("-" * 80)
    
    # Parent chunk reference nếu có
    if chunk.get("parent_chunk_id"):
        print(f"\n🔗 Parent Chunk ID: {chunk.get('parent_chunk_id')}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo script hiển thị 10 chunk đầu tiên từ ingestion pipeline"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf",
        type=str,
        help="Đường dẫn đến file PDF để xử lý"
    )
    group.add_argument(
        "--from-vectorstore",
        action="store_true",
        help="Lấy chunks từ ChromaDB vectorstore thay vì xử lý PDF"
    )
    
    parser.add_argument(
        "--law-name",
        type=str,
        default="Luật Mẫu",
        help="Tên luật để gán cho chunks (mặc định: 'Luật Mẫu')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Số chunk tối đa hiển thị (mặc định: 10)"
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="data/evaluation/demo_chunks.json",
        help="Đường dẫn file JSON output"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Token tối đa cho mỗi chunk"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Số token overlap khi chia chunk"
    )

    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Ngưỡng token tối thiểu để giữ child chunk"
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./vectorstore",
        help="Thư mục lưu ChromaDB persist directory"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="legal_documents",
        help="Tên collection trong ChromaDB"
    )

    parser.add_argument(
        "--short-max",
        type=int,
        default=120,
        help="Chunk <= ngưỡng này được tính là chunk ngắn"
    )

    parser.add_argument(
        "--long-min",
        type=int,
        default=300,
        help="Chunk >= ngưỡng này được tính là chunk dài"
    )
    
    args = parser.parse_args()

    if args.limit <= 0:
        logger.error("❌ --limit phải > 0")
        sys.exit(1)
    if args.chunk_size <= 0:
        logger.error("❌ --chunk-size phải > 0")
        sys.exit(1)
    if args.chunk_overlap < 0:
        logger.error("❌ --chunk-overlap phải >= 0")
        sys.exit(1)
    if args.min_chunk_size < 0:
        logger.error("❌ --min-chunk-size phải >= 0")
        sys.exit(1)
    if args.short_max >= args.long_min:
        logger.error("❌ Cần đảm bảo short_max < long_min")
        sys.exit(1)
    
    # Lấy chunks
    print("\n🚀 Starting Demo...")
    
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            logger.error(f"❌ PDF file not found: {pdf_path}")
            sys.exit(1)
        chunks = demo_from_pdf(
            str(pdf_path),
            args.law_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_size=args.min_chunk_size,
        )
    else:
        chunks = demo_from_vectorstore(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
        )
    
    if not chunks:
        logger.error("❌ No chunks found!")
        sys.exit(1)

    shown_chunks = chunks[:args.limit]
    full_stats = calc_stats(chunks, short_max=args.short_max, long_min=args.long_min)
    shown_stats = calc_stats(shown_chunks, short_max=args.short_max, long_min=args.long_min)
    
    # In kết quả
    print(f"\n\n✅ Successfully loaded {len(chunks)} chunks (showing {len(shown_chunks)})")
    print(f"\n{'='*80}")
    print(f"DEMO: CHUNKS ĐẦU TIÊN")
    print(f"{'='*80}")
    
    for i, chunk in enumerate(shown_chunks, 1):
        print_chunk_details(chunk, i)
    
    # Tóm tắt
    print(f"\n\n{'='*80}")
    print("📊 TONG KET")
    print(f"{'='*80}")
    print("Toan bo dataset:")
    print(f"  - Tong chunks: {full_stats['total']}")
    print(f"  - Parent chunks: {full_stats['parent']}")
    print(f"  - Child chunks:  {full_stats['child']}")
    print(f"  - Tong token: {full_stats['total_tokens']}")
    print(f"  - Token TB/chunk: {full_stats['avg_tokens']}")
    print(f"  - Min/Max token: {full_stats['min_tokens']} / {full_stats['max_tokens']}")
    print("  - Phan loai do dai:")
    print(f"    + Chunk ngan (<= {full_stats['short_max']}): {full_stats['short_count']}")
    print(f"    + Chunk vua  : {full_stats['medium_count']}")
    print(f"    + Chunk dai  (>= {full_stats['long_min']}): {full_stats['long_count']}")

    print("Du lieu dang hien thi:")
    print(f"  - Tong chunks hien thi: {shown_stats['total']}")
    print(f"  - Parent chunks: {shown_stats['parent']}")
    print(f"  - Child chunks:  {shown_stats['child']}")
    print(f"  - Tong token: {shown_stats['total_tokens']}")
    print(f"  - Token TB/chunk: {shown_stats['avg_tokens']}")
    print(f"  - Min/Max token: {shown_stats['min_tokens']} / {shown_stats['max_tokens']}")

    write_json_output(args.output_json, chunks, full_stats)
    print(f"\n✨ Demo hoàn tất!\n")


if __name__ == "__main__":
    main()
