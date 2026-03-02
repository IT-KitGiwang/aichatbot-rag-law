"""
Xem toàn bộ nội dung các đoạn text SẼ ĐƯỢC EMBEDDING vào ChromaDB.

Hiển thị:
  - Tất cả child + grandchild chunks (không có parent)
  - Nội dung đầy đủ (không cắt)
  - Thống kê kích thước

Lưu kết quả: data/processed/search_chunks_full.json
"""

import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_processor import LegalPDFProcessor
from src.ingestion.legal_chunker import LegalChunker

PDF_PATH = ROOT / "data" / "raw_pdfs" / "LUAT-QUAN-LY-THUE-2019.pdf"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Parse + Chunk ──
processor = LegalPDFProcessor(
    law_name="Luật Quản lý thuế",
    law_number="38/2019/QH14",
    effective_date="2020-07-01",
    issuing_body="Quốc hội",
    preamble_context="LUẬT QUẢN LÝ THUẾ",
)
articles = processor.process(str(PDF_PATH))
chunker = LegalChunker(min_child_length=80, max_clause_length=800)
all_chunks = chunker.chunk_articles(articles)

# ── 2. Tách search chunks (child + grandchild) ──
search_chunks = [c for c in all_chunks if c.chunk_type != "parent"]
parent_chunks = [c for c in all_chunks if c.chunk_type == "parent"]

print(f"\n{'='*70}")
print(f"  NỘI DUNG SẼ EMBEDDING VÀO CHROMADB")
print(f"  Search chunks: {len(search_chunks)} (child + grandchild)")
print(f"  Parent chunks: {len(parent_chunks)} (chỉ lưu file, không embed)")
print(f"{'='*70}")

# ── 3. Hiển thị từng chunk ──
current_article = ""
for i, c in enumerate(search_chunks):
    art = c.metadata.get("article", "")
    # Header khi chuyển Điều mới
    if art != current_article:
        current_article = art
        ch = c.metadata.get("chapter", "")
        sec = c.metadata.get("section", "")
        art_title = c.metadata.get("article_title", "")
        print(f"\n{'─'*70}")
        print(f"📂 {ch} {'> ' + sec + ' ' if sec else ''}> {art}. {art_title}")
        print(f"{'─'*70}")

    clause = c.metadata.get("clause", "")
    point = c.metadata.get("point", "")
    label = f"{clause} {point}".strip() if clause else c.chunk_type

    print(f"\n  ┌─ [{c.chunk_id}] {c.chunk_type.upper()} — {label} ({len(c.text)} chars)")
    print(f"  │")
    # Hiển thị nội dung (bỏ header [...] vì đã hiện ở trên)
    text_lines = c.text.split("\n")
    for line in text_lines:
        print(f"  │  {line}")
    print(f"  └─")

# ── 4. Thống kê ──
lengths = [len(c.text) for c in search_chunks]
too_long = [c for c in search_chunks if len(c.text) > 2000]

print(f"\n{'='*70}")
print(f"  THỐNG KÊ SEARCH CHUNKS")
print(f"{'='*70}")
print(f"  Tổng chunks:        {len(search_chunks)}")
print(f"  Nhỏ nhất:           {min(lengths)} chars")
print(f"  Lớn nhất:           {max(lengths)} chars")
print(f"  Trung bình:         {sum(lengths) // len(lengths)} chars")
print(f"  Chunks > 2000 chars: {len(too_long)} {'⚠️ (có thể vượt token limit)' if too_long else '✅'}")

if too_long:
    print(f"\n  ⚠️  Chunks quá dài:")
    for c in too_long:
        print(f"     - {c.chunk_id}: {len(c.text)} chars | {c.metadata.get('hierarchy_path','')}")

# ── 5. Lưu file JSON ──
output = []
for c in search_chunks:
    output.append({
        "chunk_id": c.chunk_id,
        "chunk_type": c.chunk_type,
        "parent_chunk_id": c.parent_chunk_id,
        "hierarchy_path": c.metadata.get("hierarchy_path", ""),
        "text_length": len(c.text),
        "text": c.text,
    })

out_path = OUTPUT_DIR / "search_chunks_full.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n  📁 Đã lưu → {out_path}")
