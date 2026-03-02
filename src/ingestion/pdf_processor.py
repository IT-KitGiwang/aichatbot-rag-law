"""
Xử lý PDF luật Việt Nam — trích xuất text có nhận biết cấu trúc pháp luật.

Cấu trúc luật VN (không có Phần):
    Chương I, II, ... (số La Mã)
        Mục 1, 2, ...  (số tự nhiên, tùy chọn)
            Điều 1, 2, ... (số tự nhiên)
                1. Khoản (số tự nhiên + dấu chấm)
                    a) Điểm (chữ cái + dấu ngoặc đơn)

Output: list[LegalArticle] — mỗi phần tử = 1 Điều luật + metadata đầy đủ.
"""

import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class LegalArticle:
    """Một Điều luật với metadata đầy đủ."""
    text: str                           # Nội dung nguyên văn
    article_number: int = 0             # Số Điều
    article_title: str = ""             # Tiêu đề Điều
    chapter: str = ""                   # VD: "Chương I"
    chapter_title: str = ""             # VD: "NHỮNG QUY ĐỊNH CHUNG"
    section: str = ""                   # VD: "Mục 1" (nếu có)
    section_title: str = ""
    page_number: int = 0
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Regex patterns cho cấu trúc luật VN
# ──────────────────────────────────────────────

# Chương I, II, III, ... (La Mã), theo sau bởi tiêu đề IN HOA
RE_CHAPTER = re.compile(
    r'^Chương\s+([IVXLCDM]+)\s*$',
    re.MULTILINE
)

# Mục 1, 2, 3, ... theo sau bởi tiêu đề
RE_SECTION = re.compile(
    r'^Mục\s+(\d+)\s*$',
    re.MULTILINE
)

# Điều 1. Tiêu đề điều luật
RE_ARTICLE = re.compile(
    r'^Điều\s+(\d+)\.\s*(.*)',
    re.MULTILINE
)

# Noise patterns cần loại bỏ
NOISE_PATTERNS = [
    # Header trang download
    re.compile(r'(?i)downloaded?\s+from.*$', re.MULTILINE),
    re.compile(r'(?i)thuvienphapluat\.vn.*$', re.MULTILINE),
    re.compile(r'(?i)source:?\s*http.*$', re.MULTILINE),
    # Dòng chỉ chứa gạch ngang
    re.compile(r'^[\-–—=]{3,}\s*$', re.MULTILINE),
]


# ──────────────────────────────────────────────
# PDF Processor
# ──────────────────────────────────────────────

class LegalPDFProcessor:
    """
    Xử lý PDF luật Việt Nam:
      1. Trích xuất text từng trang (PyMuPDF)
      2. Loại bỏ noise: header, footer, số trang, watermark
      3. Chuẩn hóa Unicode tiếng Việt (NFC)
      4. Nhận dạng cấu trúc Chương > Mục > Điều
      5. Trả về danh sách LegalArticle
    """

    def __init__(
        self,
        law_name: str,
        law_number: str,
        effective_date: str = "",
        issuing_body: str = "Quốc hội",
        preamble_context: str = "",
    ):
        self.law_name = law_name
        self.law_number = law_number
        self.effective_date = effective_date
        self.issuing_body = issuing_body
        self.preamble_context = preamble_context

    # ── public ──────────────────────────────

    def process(self, pdf_path: str) -> list[LegalArticle]:
        """Pipeline hoàn chỉnh: PDF → list[LegalArticle]."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {pdf_path}")

        # 1. Trích xuất text thô
        raw_pages = self._extract_pages(pdf_path)

        # 2. Làm sạch & chuẩn hóa
        clean_pages = [
            (pg, self._clean_text(text)) for pg, text in raw_pages
        ]

        # 3. Ghép toàn bộ thành 1 chuỗi + page map
        full_text, page_map = self._merge_pages(clean_pages)

        # 4. Nhận dạng cấu trúc Chương/Mục
        chapters = self._parse_chapters(full_text)
        sections = self._parse_sections(full_text)

        # 5. Tách từng Điều luật + gắn metadata
        articles = self._parse_articles(full_text, page_map, chapters, sections)

        print(f"[PDF Processor] {pdf_path.name}: {len(raw_pages)} trang, {len(articles)} Điều luật")
        return articles

    # ── private: extract ────────────────────

    def _extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Trích xuất text từ PDF, trả về [(page_num, text)]."""
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if text.strip():
                pages.append((i + 1, text))
        doc.close()
        return pages

    # ── private: clean ──────────────────────

    def _clean_text(self, text: str) -> str:
        """Loại bỏ noise, chuẩn hóa Unicode."""
        # NFC normalize (hợp nhất dấu + chữ)
        text = unicodedata.normalize("NFC", text)

        # Loại bỏ noise patterns
        for pattern in NOISE_PATTERNS:
            text = pattern.sub("", text)

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()

            # Bỏ dòng chỉ là số trang
            if stripped.isdigit():
                continue

            # Bỏ dòng trống liên tiếp
            if not stripped and cleaned and not cleaned[-1].strip():
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # ── private: merge ──────────────────────

    def _merge_pages(
        self, pages: list[tuple[int, str]]
    ) -> tuple[str, list[tuple[int, int]]]:
        """
        Ghép tất cả trang thành 1 chuỗi.
        Trả về (full_text, page_map) — page_map = [(char_offset, page_num)].
        """
        parts: list[str] = []
        page_map: list[tuple[int, int]] = []
        offset = 0
        for pg, text in pages:
            page_map.append((offset, pg))
            parts.append(text)
            offset += len(text) + 1  # +1 cho \n nối trang
        return "\n".join(parts), page_map

    def _page_at(self, page_map: list[tuple[int, int]], pos: int) -> int:
        """Tra cứu số trang chứa vị trí ký tự `pos`."""
        result = 1
        for char_offset, pg in page_map:
            if pos >= char_offset:
                result = pg
            else:
                break
        return result

    # ── private: parse structure ─────────────

    def _parse_chapters(self, text: str) -> list[dict]:
        """Tìm tất cả Chương + tiêu đề (dòng IN HOA liền sau)."""
        chapters: list[dict] = []
        for m in RE_CHAPTER.finditer(text):
            chapter_num = m.group(1)
            # Tiêu đề thường ở dòng ngay sau, IN HOA
            after = text[m.end():m.end() + 200]
            title_line = ""
            for line in after.split("\n"):
                line_s = line.strip()
                if line_s:
                    title_line = line_s
                    break
            chapters.append({
                "number": chapter_num,
                "label": f"Chương {chapter_num}",
                "title": title_line,
                "start": m.start(),
            })
        return chapters

    def _parse_sections(self, text: str) -> list[dict]:
        """Tìm tất cả Mục + tiêu đề."""
        sections: list[dict] = []
        for m in RE_SECTION.finditer(text):
            sec_num = m.group(1)
            after = text[m.end():m.end() + 200]
            title_line = ""
            for line in after.split("\n"):
                line_s = line.strip()
                if line_s:
                    title_line = line_s
                    break
            sections.append({
                "number": sec_num,
                "label": f"Mục {sec_num}",
                "title": title_line,
                "start": m.start(),
            })
        return sections

    def _find_parent(
        self, items: list[dict], pos: int
    ) -> tuple[str, str]:
        """Tìm Chương/Mục chứa vị trí `pos` (nhìn ngược lại)."""
        label, title = "", ""
        for item in items:
            if item["start"] <= pos:
                label = item["label"]
                title = item["title"]
            else:
                break
        return label, title

    # ── private: parse articles ──────────────

    def _parse_articles(
        self,
        text: str,
        page_map: list[tuple[int, int]],
        chapters: list[dict],
        sections: list[dict],
    ) -> list[LegalArticle]:
        """Tách từng Điều luật, gắn metadata đầy đủ."""
        matches = list(RE_ARTICLE.finditer(text))
        if not matches:
            print("[WARN] Không tìm thấy Điều nào trong văn bản!")
            return []

        articles: list[LegalArticle] = []

        for i, m in enumerate(matches):
            art_num = int(m.group(1))
            art_title = m.group(2).strip()

            # Phạm vi text: từ Điều này → Điều kế tiếp
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            art_text = text[start:end].strip()

            # Tìm Chương & Mục cha
            ch_label, ch_title = self._find_parent(chapters, start)
            sec_label, sec_title = self._find_parent(sections, start)

            # Hierarchy path
            path_parts = [self.law_name]
            if ch_label:
                path_parts.append(ch_label)
            if sec_label:
                path_parts.append(sec_label)
            path_parts.append(f"Điều {art_num}")
            hierarchy = " > ".join(path_parts)

            page = self._page_at(page_map, start)

            article = LegalArticle(
                text=art_text,
                article_number=art_num,
                article_title=art_title,
                chapter=ch_label,
                chapter_title=ch_title,
                section=sec_label,
                section_title=sec_title,
                page_number=page,
                metadata={
                    "law_name": self.law_name,
                    "law_number": self.law_number,
                    "effective_date": self.effective_date,
                    "issuing_body": self.issuing_body,
                    "preamble_context": self.preamble_context,
                    "chapter": ch_label,
                    "chapter_title": ch_title,
                    "section": sec_label,
                    "section_title": sec_title,
                    "article": f"Điều {art_num}",
                    "article_title": art_title,
                    "article_number": art_num,
                    "hierarchy_path": hierarchy,
                    "source_page": page,
                },
            )
            articles.append(article)

        return articles
