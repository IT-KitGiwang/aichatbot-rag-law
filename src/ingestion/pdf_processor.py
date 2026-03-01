"""
Legal-Aware PDF Processor.
Trích xuất text từ PDF luật Việt Nam với nhận biết cấu trúc:
Phần → Chương → Mục → Điều → Khoản → Điểm
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import fitz 


# ===========================================================================
# BƯỚC 1.2 — Regex Patterns
# ===========================================================================

# --- Nhóm 1: Nhận dạng cấu trúc văn bản luật ---
#
# Thứ tự trong dict quan trọng: Python sẽ thử từ trên xuống dưới.
# "part" và "chapter" phải đứng trước "clause" vì "Chương III" nếu
# không khớp trước có thể bị nhầm thành body.
#
LEGAL_PATTERNS: dict[str, re.Pattern] = {
    "part": re.compile(
        r"^PHẦN\s+THỨ\s+(.+)\b", 
        re.IGNORECASE,
    ),
    "chapter": re.compile(
        r"^Chương\s+([IVXLCDM]+|\d+)\b",  
        re.IGNORECASE,
    ),
    "section": re.compile(
        r"^Mục\s+\d+\b",            
        re.UNICODE,
    ),
    "article": re.compile(
        r"^Điều\s+\d+\b[\.:]",     
        re.UNICODE,
    ),
    "clause": re.compile(
        r"^\d+\.\s+\S.*",           
        re.IGNORECASE,
    ),
    "point": re.compile(
        r"^[a-zđ]\)\s*\S",       
        re.IGNORECASE,
    ),
}

# --- Nhóm 2: Lọc noise (header / footer / số trang) ---
#
# Các dòng khớp bất kỳ pattern nào dưới đây sẽ bị bỏ qua hoàn toàn
# trước khi đưa vào nhận dạng cấu trúc.
#
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),  
    re.compile(r"^\s*Trang\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*QUỐC\s*HỘI\s*$", re.IGNORECASE),
    re.compile(r"^\s*CỘNG\s*HÒA\s*XÃ\s*HỘI\s*$", re.IGNORECASE),
    re.compile(r"^\s*Độc\s*lập\s*[-–]\s*Tự\s*do\s*$", re.IGNORECASE),
]


def detect_line_type(line: str) -> str: # trả về chuỗi str "part" | "chapter" | "section" | "article" | "clause" | "point" | "body"
    """
    Nhận dạng loại của một dòng văn bản dựa trên LEGAL_PATTERNS.

    Trả về một trong các giá trị:
        "part" | "chapter" | "section" | "article" | "clause" | "point" | "body"

    Ví dụ:
        detect_line_type("Điều 15. Quyền...")  → "article"
        detect_line_type("1. Vợ chồng...")     → "clause"
        detect_line_type("bình đẳng với nhau") → "body"
    """
    for block_type, pattern in LEGAL_PATTERNS.items():
        if pattern.match(line):
            return block_type
    return "body"


def is_noise(line: str) -> bool:
    """
    Kiểm tra xem một dòng có phải là noise cần bỏ qua không.

    Ví dụ:
        is_noise("- 12 -")        → True
        is_noise("Điều 15.")      → False
    """
    return any(p.search(line) for p in _NOISE_PATTERNS) # nếu khớp bất kỳ pattern nào → là noise (true), ngược lại → không phải noise (false)


# ===========================================================================
# BƯỚC 1.1 — Data Structures
# ===========================================================================

@dataclass
class LegalStructure:
    """
    Lưu trạng thái vị trí hiện tại trong cấu trúc phân cấp văn bản luật.

    Hoạt động như một "con trỏ GPS": khi duyệt qua từng dòng PDF,
    object này liên tục được cập nhật để phản ánh ta đang đọc đến
    Chương nào, Mục nào, Điều nào.

    Ví dụ sau khi đọc qua "Chương III" → "Mục 1" → "Điều 15":
        LegalStructure(
            chapter = "Chương III",
            section = "Mục 1",
            article = "Điều 15. Quyền và nghĩa vụ về nhân thân"
        )
    """
    part: Optional[str] = None            # PHẦN THỨ NHẤT / HAI / ...
    chapter: Optional[str] = None         # Chương I / II / III / ...
    chapter_title: Optional[str] = None   # Tiêu đề chương (dòng sau Chương)
    section: Optional[str] = None         # Mục 1 / 2 / 3 / ...
    section_title: Optional[str] = None   # Tiêu đề mục
    article: Optional[str] = None         # Điều 1. / Điều 15. / ...
    article_title: Optional[str] = None   # Tiêu đề điều (phần sau dấu chấm)

    def to_hierarchy_path(self) -> str:
        """
        Tạo chuỗi mô tả vị trí phân cấp, dùng làm metadata.
        Ví dụ: "Chương III > Mục 1 > Điều 15"
        """
        parts = []
        if self.chapter:
            parts.append(self.chapter)
        if self.section:
            parts.append(self.section)
        if self.article:
            # Chỉ lấy phần "Điều 15", không lấy tiêu đề dài
            parts.append(self.article.split(".")[0].strip())
        return " > ".join(parts)

    def clone(self) -> "LegalStructure":
        """
        Tạo bản sao độc lập của object hiện tại.

        QUAN TRỌNG: Cần thiết khi gắn structure vào RawBlock.
        Nếu dùng trực tiếp (không clone), tất cả RawBlock sẽ cùng
        trỏ về một object — khi structure thay đổi, mọi block bị
        ảnh hưởng ngược lại.
        """
        return LegalStructure(
            part=self.part,
            chapter=self.chapter,
            chapter_title=self.chapter_title,
            section=self.section,
            section_title=self.section_title,
            article=self.article,
            article_title=self.article_title,
        )


@dataclass
class RawBlock:
    """
    Đơn vị văn bản thô sau khi trích xuất từ PDF.

    Là output của LegalPDFProcessor và input của LegalChunker.
    Mỗi block là một đoạn text liên tục thuộc cùng một loại cấu trúc
    (ví dụ: một Khoản, một Điều, một tiêu đề Chương).

    Ví dụ:
        RawBlock(
            text      = "1. Vợ, chồng bình đẳng với nhau, có quyền...",
            page      = 12,
            block_type = "clause",
            structure  = LegalStructure(chapter="Chương III", article="Điều 15.")
        )
    """
    text: str 
    page: int
    block_type: str           # "part" | "chapter" | "section" | "article"
                              # | "clause" | "point" | "body"
    structure: LegalStructure = field(default_factory=LegalStructure) # Đảm bảo luôn có một structure, dù rỗng riêng biệt

# ===========================================================================
# BƯỚC 1.3 — Trích xuất dòng từ PDF & Lọc noise
# ===========================================================================

class LegalPDFProcessor:
    """Xử lý PDF luật Việt Nam — nhận biết cấu trúc phân cấp."""

    # ------------------------------------------------------------------
    # Bước 1.3a: Trích xuất dòng thô từ PDF
    # ------------------------------------------------------------------

    def _extract_lines(self, pdf_path: Path) -> list[tuple[str, int]]:
        """
        Dùng PyMuPDF đọc PDF, trả về danh sách (dòng_text, số_trang).

        Mỗi trang được đọc bằng page.get_text("text") → chuỗi lớn,
        sau đó splitlines() để tách thành từng dòng riêng.
        Số trang được gắn ngay tại đây để không bị mất thông tin nguồn.

        Ví dụ output:
            [
                ("LUẬT",                       1),
                ("HÔN NHÂN VÀ GIA ĐÌNH",       1),
                ("- 1 -",                       1),
                ("Chương I",                    2),
                ("Điều 1. Phạm vi điều chỉnh", 2),
            ]
        """
        lines: list[tuple[str, int]] = []

        doc = fitz.open(str(pdf_path))
        try:
            for page_num, page in enumerate(doc, start=1): # enumerate có nghĩa là vừa lấy index (số trang) vừa lấy nội dung trang
                # "text" mode: trả về plain text, giữ nguyên layout dòng
                raw_text = page.get_text("text") # xún dòng đúng chỗ mà PDF hiển thị
                for line in raw_text.splitlines():
                    lines.append((line, page_num))
        finally:
            doc.close()  # đảm bảo đóng file dù có lỗi

        return lines 

    # ------------------------------------------------------------------
    # Bước 1.3b: Lọc bỏ noise
    # ------------------------------------------------------------------

    def _filter_noise(
        self, raw_lines: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        """
        Loại bỏ dòng trống và dòng là header/footer/số trang.

        Duyệt qua từng (dòng, trang), giữ lại nếu:
          1. Dòng không rỗng sau strip()
          2. Không khớp bất kỳ pattern nào trong _NOISE_PATTERNS

        Ví dụ:
            Input:  [("- 1 -", 1), ("Chương I", 2), ("", 2)]
            Output: [("Chương I", 2)]
        """
        clean: list[tuple[str, int]] = []

        for line, page in raw_lines:
            stripped = line.strip()

            # Bỏ qua dòng trống
            if not stripped:
                continue

            # Bỏ qua dòng khớp noise pattern
            if is_noise(stripped):
                continue

            clean.append((stripped, page))

        return clean

    # ------------------------------------------------------------------
    # Bước 1.4: Nhận dạng cấu trúc & gom thành RawBlock
    # ------------------------------------------------------------------

    def _identify_structure(
        self, clean_lines: list[tuple[str, int]]
    ) -> list[RawBlock]:
        """
        Duyệt từng dòng sạch, nhận dạng loại, gom thành RawBlock.

        Thuật toán "duyệt + flush":
          - Duy trì buffer (current_lines) tích lũy dòng của block hiện tại.
          - Khi gặp dòng bắt đầu block mới → flush buffer thành RawBlock,
            cập nhật GPS (current_struct), bắt đầu buffer mới.
          - Body chỉ append vào buffer, không flush.

        Quy tắc cập nhật struct khi flush:
          - Chương mới  → reset section, article
          - Mục mới     → reset article (giữ chapter)
          - Điều mới    → chỉ cập nhật article (giữ chapter, section)
          - Khoản/Điểm  → không thay đổi struct (vẫn thuộc Điều hiện tại)
        """
        blocks: list[RawBlock] = []
        current_struct = LegalStructure()
        current_lines: list[str] = []
        current_type: str = "body"
        current_page: int = 1

        def flush():
            """Xuất buffer thành RawBlock và reset buffer."""
            nonlocal current_lines, current_type, current_page
            text = " ".join(current_lines).strip()
            if text:
                blocks.append(RawBlock(
                    text=text,
                    page=current_page,
                    block_type=current_type,
                    structure=current_struct.clone(),  # ← clone để độc lập
                ))
            current_lines = []

        for line, page in clean_lines:
            line_type = detect_line_type(line)

            if line_type == "part":
                flush()
                current_struct.part = line
                current_struct.chapter = None
                current_struct.chapter_title = None
                current_struct.section = None
                current_struct.section_title = None
                current_struct.article = None
                current_struct.article_title = None
                current_type = "part"
                current_page = page
                current_lines = [line]

            elif line_type == "chapter":
                flush()
                current_struct.chapter = line
                current_struct.chapter_title = None
                current_struct.section = None
                current_struct.section_title = None
                current_struct.article = None
                current_struct.article_title = None
                current_type = "chapter"
                current_page = page
                current_lines = [line]

            elif line_type == "section":
                flush()
                current_struct.section = line
                current_struct.section_title = None
                current_struct.article = None
                current_struct.article_title = None
                current_type = "section"
                current_page = page
                current_lines = [line]

            elif line_type == "article":
                flush()
                current_struct.article = line
                current_struct.article_title = None
                current_type = "article"
                current_page = page
                current_lines = [line]

            elif line_type in ("clause", "point"):
                # Flush block khoản/điểm trước đó (nếu có)
                # nhưng KHÔNG thay đổi struct — vẫn thuộc Điều hiện tại
                flush()
                current_type = line_type
                current_page = page
                current_lines = [line]

            else:
                # "body" — chỉ append, không flush
                # Dòng body đầu tiên sau chapter/section thường là tiêu đề
                if current_type == "chapter" and not current_struct.chapter_title:
                    current_struct.chapter_title = line
                elif current_type == "section" and not current_struct.section_title:
                    current_struct.section_title = line
                current_lines.append(line)

        flush()  # flush block cuối cùng
        return blocks

    # ------------------------------------------------------------------
    # Bước 1.5: Public API — gom 3 bước thành một lời gọi duy nhất
    # ------------------------------------------------------------------

    def process(
        self,
        pdf_path: str | Path,
        law_name: str = "",
        law_number: str = "",
        effective_date: str = "",
    ) -> list[RawBlock]:
        """
        Xử lý một file PDF luật và trả về danh sách RawBlock.

        Đây là method duy nhất bên ngoài cần gọi.
        Nội bộ thực hiện 3 bước theo thứ tự:
            1. _extract_lines()      — đọc PDF → (dòng, trang)
            2. _filter_noise()       — lọc header/footer/số trang
            3. _identify_structure() — nhận dạng & gom thành RawBlock

        Sau đó gắn thông tin văn bản luật (law_name, law_number,
        effective_date) vào thuộc tính _law_info của mỗi block —
        thông tin này sẽ được LegalChunker dùng để tạo metadata.

        Args:
            pdf_path:       Đường dẫn file PDF.
            law_name:       Tên đầy đủ (VD: "Luật Hôn nhân và Gia đình 2014").
                            Nếu bỏ trống → tự suy luận từ tên file.
            law_number:     Số hiệu văn bản (VD: "52/2014/QH13").
            effective_date: Ngày có hiệu lực dạng YYYY-MM-DD.

        Returns:
            list[RawBlock] — mỗi block mang đầy đủ cấu trúc + thông tin luật.

        Ví dụ:
            processor = LegalPDFProcessor()
            blocks = processor.process(
                "data/raw_pdfs/luat_hon_nhan.pdf",
                law_name="Luật Hôn nhân và Gia đình 2014",
                law_number="52/2014/QH13",
                effective_date="2015-01-01",
            )
            print(len(blocks))          # số RawBlock
            print(blocks[0].block_type) # "chapter" / "article" / ...
            print(blocks[0].structure.chapter)  # "Chương I"
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

        # Tự suy luận law_name từ tên file nếu không được cung cấp
        if not law_name:
            law_name = self.extract_metadata_from_filename(pdf_path)["law_name"]

        # Bước 1 → 2 → 3
        raw_lines   = self._extract_lines(pdf_path)
        clean_lines = self._filter_noise(raw_lines)
        blocks      = self._identify_structure(clean_lines)

        # Gắn thông tin văn bản luật vào từng block
        # (dùng dict _law_info để không làm bẩn dataclass RawBlock)
        for block in blocks:
            block._law_info = {
                "law_name":       law_name,
                "law_number":     law_number,
                "effective_date": effective_date,
            }

        return blocks

    def extract_metadata_from_filename(self, pdf_path: Path) -> dict:
        """
        Suy luận law_name từ tên file khi không được cung cấp.

        Ví dụ:
            "luat_hon_nhan_2014.pdf" → {"law_name": "Luat Hon Nhan 2014"}
        """
        name = pdf_path.stem.replace("_", " ").title()
        return {"law_name": name, "law_number": "", "effective_date": ""}

