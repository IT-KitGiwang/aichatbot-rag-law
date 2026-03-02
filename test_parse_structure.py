"""
Test kiểm tra PDF đã tách luật đúng cấu trúc chưa.

Kết quả lưu vào: data/processed/articles_full.json
Hiển thị: Tổng quan Chương → Mục → Điều + nội dung đầy đủ
"""

import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_processor import LegalPDFProcessor

PDF_PATH = ROOT / "data" / "raw_pdfs" / "LUAT-QUAN-LY-THUE-2019.pdf"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Parse PDF ──
processor = LegalPDFProcessor(
    law_name="Luật Quản lý thuế",
    law_number="38/2019/QH14",
    effective_date="2020-07-01",
    issuing_body="Quốc hội",
    preamble_context="LUẬT QUẢN LÝ THUẾ",
)
articles = processor.process(str(PDF_PATH))

# ── 2. Hiển thị cấu trúc tổng quan ──
print(f"\n{'='*70}")
print(f"  LUẬT QUẢN LÝ THUẾ 2019 — CẤU TRÚC TÁCH ĐIỀU")
print(f"  Tổng: {len(articles)} Điều luật")
print(f"{'='*70}\n")

# Nhóm theo Chương
chapters = {}
for a in articles:
    ch = a.chapter or "(Không có chương)"
    if ch not in chapters:
        chapters[ch] = {
            "title": a.chapter_title,
            "sections": {},
            "articles": [],
        }
    sec = a.section or ""
    if sec:
        if sec not in chapters[ch]["sections"]:
            chapters[ch]["sections"][sec] = {
                "title": a.section_title,
                "articles": [],
            }
        chapters[ch]["sections"][sec]["articles"].append(a)
    else:
        chapters[ch]["articles"].append(a)

for ch_name, ch_data in chapters.items():
    print(f"\n📂 {ch_name}: {ch_data['title']}")
    print(f"   {'─'*60}")

    # Điều không thuộc Mục nào
    for a in ch_data["articles"]:
        text_len = len(a.text)
        print(f"   📄 Điều {a.article_number}. {a.article_title[:50]:<50} | {text_len:>5} chars | trang {a.page_number}")

    # Điều thuộc Mục
    for sec_name, sec_data in ch_data["sections"].items():
        print(f"   📁 {sec_name}: {sec_data['title']}")
        for a in sec_data["articles"]:
            text_len = len(a.text)
            print(f"      📄 Điều {a.article_number}. {a.article_title[:47]:<47} | {text_len:>5} chars | trang {a.page_number}")

# ── 3. Thống kê ──
total_chars = sum(len(a.text) for a in articles)
avg_chars = total_chars // len(articles) if articles else 0
min_art = min(articles, key=lambda a: len(a.text))
max_art = max(articles, key=lambda a: len(a.text))

print(f"\n{'='*70}")
print(f"  THỐNG KÊ")
print(f"{'='*70}")
print(f"  Tổng Điều:          {len(articles)}")
print(f"  Tổng Chương:        {len(chapters)}")
print(f"  Tổng ký tự:         {total_chars:,}")
print(f"  Trung bình/Điều:    {avg_chars:,} chars")
print(f"  Điều ngắn nhất:     Điều {min_art.article_number} ({len(min_art.text)} chars)")
print(f"  Điều dài nhất:      Điều {max_art.article_number} ({len(max_art.text)} chars)")

# ── 4. Kiểm tra lỗi ──
print(f"\n{'='*70}")
print(f"  KIỂM TRA LỖI")
print(f"{'='*70}")

errors = []

# Check Điều liên tục
art_nums = [a.article_number for a in articles]
for i in range(1, len(art_nums)):
    if art_nums[i] != art_nums[i-1] + 1:
        errors.append(f"  ⚠️  Thiếu Điều: sau Điều {art_nums[i-1]} → nhảy sang Điều {art_nums[i]}")

# Check Điều trống
for a in articles:
    if len(a.text.strip()) < 30:
        errors.append(f"  ⚠️  Điều {a.article_number} quá ngắn ({len(a.text)} chars): {a.text[:50]}")

# Check thiếu chương
for a in articles:
    if not a.chapter:
        errors.append(f"  ⚠️  Điều {a.article_number} không có Chương")

if errors:
    for e in errors:
        print(e)
else:
    print("  ✅ Không phát hiện lỗi — cấu trúc Điều liên tục, đầy đủ Chương!")

# ── 5. Lưu file JSON đầy đủ ──
output = []
for a in articles:
    output.append({
        "article_number": a.article_number,
        "article_title": a.article_title,
        "chapter": a.chapter,
        "chapter_title": a.chapter_title,
        "section": a.section,
        "section_title": a.section_title,
        "page_number": a.page_number,
        "text_length": len(a.text),
        "text": a.text,  # NỘI DUNG ĐẦY ĐỦ
        "hierarchy_path": a.metadata.get("hierarchy_path", ""),
    })

out_path = OUTPUT_DIR / "articles_full.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n  📁 Đã lưu nội dung đầy đủ → {out_path}")
print(f"     (Mở file này để xem toàn bộ nội dung từng Điều trước khi chunking)")
