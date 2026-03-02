"""Quick test script for ingestion dry-run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.pdf_processor import LegalPDFProcessor
from src.ingestion.legal_chunker import LegalChunker
import json

PDF_PATH = os.path.join("data", "raw_pdfs", "LUAT-QUAN-LY-THUE-2019.pdf")

# 1. Process PDF
processor = LegalPDFProcessor(
    law_name="Luật Quản lý thuế",
    law_number="38/2019/QH14",
    effective_date="2020-07-01",
    issuing_body="Quốc hội",
    preamble_context="LUẬT QUẢN LÝ THUẾ. Căn cứ Hiến pháp nước CHXHCN Việt Nam; Quốc hội ban hành Luật Quản lý thuế.",
)
articles = processor.process(PDF_PATH)

print(f"\n=== ARTICLES: {len(articles)} ===")
for a in articles[:3]:
    print(f"  Điều {a.article_number}: {a.article_title[:60]}... | {a.chapter} | page {a.page_number}")

# 2. Chunk
chunker = LegalChunker(min_child_length=80, max_clause_length=800)
chunks = chunker.chunk_articles(articles)

# 3. Stats
types = {}
for c in chunks:
    types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
print(f"\n=== CHUNK STATS ===")
for t, n in types.items():
    print(f"  {t}: {n}")

# 4. Preview
print(f"\n=== PREVIEW (first 5 chunks) ===")
for c in chunks[:5]:
    print(f"\n--- {c.chunk_id} [{c.chunk_type}] ---")
    print(f"  Path: {c.metadata.get('hierarchy_path','')}")
    print(f"  Text: {c.text[:200]}...")

# 5. Save preview
preview = []
for c in chunks[:30]:
    preview.append({
        "chunk_id": c.chunk_id,
        "chunk_type": c.chunk_type,
        "text_length": len(c.text),
        "hierarchy_path": c.metadata.get("hierarchy_path", ""),
        "text_preview": c.text[:400],
    })

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/chunks_preview.json", "w", encoding="utf-8") as f:
    json.dump(preview, f, ensure_ascii=False, indent=2)
print(f"\nSaved preview to data/processed/chunks_preview.json")
