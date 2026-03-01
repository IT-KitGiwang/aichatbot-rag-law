# src/ingestion/legal_chunker.py
"""
Legal-Aware Chunker — 4.1.2 trong kế hoạch.
Chuyển đổi: list[RawBlock] → list[LegalChunk]

Chiến lược: Parent-Child chunking
  - Parent = toàn bộ Điều luật  (trả về khi trả lời, đủ ngữ cảnh)
  - Child  = từng Khoản / Điểm  (dùng khi tìm kiếm, chính xác hơn)
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.pdf_processor import LegalStructure, RawBlock


# ===========================================================================
# BƯỚC 2.1 — LegalChunk dataclass
# ===========================================================================

@dataclass
class LegalChunk:
    """
    Đơn vị văn bản đã xử lý — sẵn sàng để embed và index vào ChromaDB.

    Là output của LegalChunker và input của EmbeddingGenerator + LegalIndexer.

    Hai loại chunk:
      - is_parent=True  : chunk toàn bộ Điều, dùng để TRẢ VỀ ngữ cảnh
      - is_parent=False : chunk từng Khoản/Điểm, dùng để TÌM KIẾM

    Ví dụ parent chunk:
        LegalChunk(
            chunk_id   = "uuid-abc",
            text       = "Chương III | Điều 15\n1. Vợ chồng bình đẳng...\n2. Vợ chồng có quyền...",
            chunk_type = "article",
            is_parent  = True,
            law_name   = "Luật Hôn nhân và Gia đình 2014",
            article    = "Điều 15.",
            ...
        )

    Ví dụ child chunk (con của parent trên):
        LegalChunk(
            chunk_id        = "uuid-xyz",
            text            = "Chương III | Điều 15\n1. Vợ chồng bình đẳng...",
            chunk_type      = "clause",
            is_parent       = False,
            parent_chunk_id = "uuid-abc",   ← trỏ về parent
            ...
        )
    """

    # --- Định danh ---
    chunk_id: str                    # UUID duy nhất, tạo bằng uuid.uuid4()
    text: str                        # Nội dung chunk (có context prefix)
    chunk_type: str                  # "article" | "clause" | "point" | "section" | "body"
    source_page: int                 # Trang trong PDF gốc

    # --- Thông tin văn bản luật ---
    law_name: str = ""
    law_number: str = ""
    effective_date: str = ""

    # --- Vị trí trong cấu trúc luật ---
    part: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    article: str = ""
    article_title: str = ""
    hierarchy_path: str = ""         # "Chương III > Mục 1 > Điều 15"

    # --- Parent-Child ---
    is_parent: bool = False          # True = parent chunk (toàn Điều)
    parent_chunk_id: Optional[str] = None  # None nếu là parent hoặc chunk đơn

    # --- Thống kê ---
    token_count: int = 0             # Ước tính số token (để quyết định cắt)

    def to_chroma_metadata(self) -> dict:
        """
        Chuyển đổi sang dict phẳng để lưu vào ChromaDB.

        Quy tắc ChromaDB:
          - Chỉ chấp nhận: str, int, float, bool
          - Không được có None → dùng "" thay thế
          - Không được có nested (lồng) dict/list

        Returns:
            dict phẳng, an toàn để truyền vào ChromaDB metadatas=[...]
        """
        return {
            "chunk_type":       self.chunk_type,
            "source_page":      self.source_page,
            "law_name":         self.law_name,
            "law_number":       self.law_number,
            "effective_date":   self.effective_date,
            "part":             self.part,
            "chapter":          self.chapter,
            "chapter_title":    self.chapter_title,
            "section":          self.section,
            "article":          self.article,
            "article_title":    self.article_title,
            "hierarchy_path":   self.hierarchy_path,
            "is_parent":        self.is_parent,
            "parent_chunk_id":  self.parent_chunk_id or "",  # None → ""
            "token_count":      self.token_count,
        }


# ===========================================================================
# BƯỚC 2.2 — Hàm tiện ích ở module level
# ===========================================================================

def _approx_token_count(text: str) -> int:
    """
    Ước tính số token bằng cách đếm từ.

    Không dùng tokenizer thực sự (để tránh dependency nặng và chậm).
    Kinh nghiệm: tiếng Việt trung bình ~1.3 từ/token.
    Sai số ±15% chấp nhận được — chỉ dùng để so sánh với ngưỡng chunk_size.

    Ví dụ:
        _approx_token_count("Vợ chồng bình đẳng với nhau")
        → 5 từ → 6 token (gần đúng)
    """
    words = text.split()
    # Hệ số 1.3: tiếng Việt có nhiều từ ghép ngắn → token count cao hơn word count
    return int(len(words) * 1.3)


def _build_context_prefix(struct: LegalStructure) -> str:
    """
    Tạo chuỗi ngữ cảnh prepend vào đầu mỗi chunk trước khi embed.

    Mục đích: giúp embedding model hiểu đoạn text này thuộc
    Chương nào, Mục nào, Điều nào — thay vì chỉ thấy nội dung đơn độc.

    Ví dụ output:
        struct = LegalStructure(
            chapter = "Chương III",
            section = "Mục 1",
            article = "Điều 15. Quyền và nghĩa vụ..."
        )
        → "Chương III | Mục 1 | Điều 15."

    Chỉ lấy phần số của Điều (bỏ tiêu đề dài) để prefix gọn,
    tránh prefix quá dài chiếm nhiều token của chunk_size.
    """
    parts = []

    if struct.chapter:
        parts.append(struct.chapter)

    if struct.section:
        parts.append(struct.section)

    if struct.article:
        # "Điều 15. Quyền và nghĩa vụ..." → chỉ lấy "Điều 15."
        article_short = struct.article.split()[0:2]  # ["Điều", "15."]
        parts.append(" ".join(article_short))

    return " | ".join(parts)  # "Chương III | Mục 1 | Điều 15."


# ===========================================================================
# BƯỚC 2.3 — LegalChunker class & _process_article()
# ===========================================================================

from dataclasses import dataclass

@dataclass
class LegalChunker:
    """Chia văn bản luật thành các LegalChunk tối ưu cho RAG."""

    chunk_size: int = 512
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    include_parent_context: bool = True

    # ------------------------------------------------------------------
    # Bước 2.3: Xử lý một Điều luật → tạo chunk(s)
    # ------------------------------------------------------------------
    
    def _process_article(
        self,
        article_block: RawBlock,
        child_blocks: list[RawBlock],
    ) -> list[LegalChunk]:
        """
        Nhận một Điều luật + danh sách Khoản/Điểm, tạo ra chunk(s).

        Quy trình:
          1. Gom toàn bộ text → parent_text
          2. Tính token_count
          3. Nếu ngắn   → trả về 1 chunk đơn
             Nếu dài    → tạo 1 parent + N children (1 child/khoản)

        Args:
            article_block: RawBlock loại "article" (dòng "Điều X.")
            child_blocks:  Các RawBlock loại "clause"/"point"/"body"
                           nằm sau Điều đó trong PDF.

        Returns:
            list[LegalChunk] — có thể là [1 chunk đơn]
                               hoặc [parent, child1, child2, ...]
        """
        struct = article_block.structure
        law_info = getattr(article_block, "_law_info", {})

        # Tạo context prefix: "Chương III | Mục 1 | Điều 15."
        prefix = _build_context_prefix(struct) if self.include_parent_context else ""

        # ── Bước 1: Gom toàn bộ text của Điều ──────────────────────────
        all_texts = [article_block.text] + [b.text for b in child_blocks]
        full_content = "\n".join(all_texts)

        # Text của parent = prefix + toàn bộ nội dung Điều
        parent_text = f"{prefix}\n{full_content}".strip() if prefix else full_content

        # ── Bước 2: Tính token count ─────────────────────────────────────
        parent_tokens = _approx_token_count(parent_text)

        # Base metadata dùng chung cho mọi chunk của Điều này
        base_meta = self._build_base_meta(struct, law_info, article_block.page)

        # ── Bước 3a: Điều ngắn → 1 chunk đơn ───────────────────────────
        if parent_tokens <= self.chunk_size:
            # Không cần chia — 1 chunk là đủ, không có parent/child
            if parent_tokens < self.min_chunk_size:
                return []  # Quá ngắn, bỏ qua

            return [LegalChunk(
                chunk_id=str(uuid.uuid4()),
                text=parent_text,
                chunk_type=article_block.block_type,
                is_parent=False,        # Chunk đơn, không phải parent
                parent_chunk_id=None,
                token_count=parent_tokens,
                **base_meta,
            )]

        # ── Bước 3b: Điều dài → parent + N children ─────────────────────
        parent_id = str(uuid.uuid4())

        # Parent chunk = toàn bộ Điều (dùng để trả về context đầy đủ)
        parent_chunk = LegalChunk(
            chunk_id=parent_id,
            text=parent_text,
            chunk_type=article_block.block_type,
            is_parent=True,
            parent_chunk_id=None,
            token_count=parent_tokens,
            **base_meta,
        )

        # Children = từng Khoản/Điểm riêng lẻ (dùng để tìm kiếm)
        child_chunks: list[LegalChunk] = []
        for child_block in child_blocks:
            child_content = child_block.text.strip()
            if not child_content:
                continue

            # Child text = prefix + nội dung khoản
            child_text = f"{prefix}\n{child_content}".strip() if prefix else child_content
            child_tokens = _approx_token_count(child_text)

            if child_tokens < self.min_chunk_size:
                continue  # Khoản quá ngắn, bỏ qua

            child_chunks.append(LegalChunk(
                chunk_id=str(uuid.uuid4()),
                text=child_text,
                chunk_type=child_block.block_type,
                is_parent=False,
                parent_chunk_id=parent_id,  # ← trỏ về parent
                token_count=child_tokens,
                **base_meta,
            ))

        # Nếu không tạo được child nào (vì quá ngắn), trả về 1 chunk đơn thay thế
        if not child_chunks:
            return [LegalChunk(
                chunk_id=parent_id,
                text=parent_text,
                chunk_type=article_block.block_type,
                is_parent=False,
                parent_chunk_id=None,
                token_count=parent_tokens,
                **base_meta,
            )]

        return [parent_chunk] + child_chunks

    # ------------------------------------------------------------------
    # Helper: tạo base metadata dùng chung cho mọi chunk của 1 Điều
    # ------------------------------------------------------------------

    def _build_base_meta(
        self,
        struct,
        law_info: dict,
        page: int,
    ) -> dict:
        """
        Trả về dict các keyword argument dùng chung khi tạo LegalChunk.

        Gom ở đây để tránh lặp code 2 lần (cho parent và children).
        Tất cả chunk trong cùng một Điều đều có chung các giá trị này.
        """
        return {
            "source_page":    page,
            "law_name":       law_info.get("law_name", ""),
            "law_number":     law_info.get("law_number", ""),
            "effective_date": law_info.get("effective_date", ""),
            "part":           struct.part or "",
            "chapter":        struct.chapter or "",
            "chapter_title":  struct.chapter_title or "",
            "section":        struct.section or "",
            "article":        struct.article or "",
            "article_title":  struct.article_title or "",
            "hierarchy_path": struct.to_hierarchy_path(),
        }

    # ==================================================================
    # BƯỚC 2.4 — Nhóm blocks theo Điều & Public API chunk()
    # ==================================================================

    def _group_by_article(
        self,
        blocks: list[RawBlock],
    ) -> list[tuple[RawBlock, list[RawBlock]]]:
        """
        Phân nhóm list[RawBlock] thành các nhóm (article, [children]).

        Mỗi tuple gồm:
          - Phần tử 0: RawBlock đại diện cho Điều (hoặc Chương/Mục nếu standalone)
          - Phần tử 1: Danh sách các RawBlock con (Khoản, Điểm, body)

        Ví dụ input (từ _identify_structure()):
            [
                RawBlock("Chương III", type="chapter"),
                RawBlock("QUAN HỆ VỢ CHỒNG", type="body"),
                RawBlock("Điều 15.", type="article"),
                RawBlock("1. Vợ chồng bình đẳng...", type="clause"),
                RawBlock("2. Vợ chồng có quyền...", type="clause"),
                RawBlock("Điều 16.", type="article"),
                RawBlock("1. Đại diện cho nhau...", type="clause"),
            ]

        Ví dụ output:
            [
                (RawBlock("Chương III"), [RawBlock("QUAN HỆ VỢ CHỒNG")]),
                (RawBlock("Điều 15."),  [RawBlock("1. Vợ chồng..."), RawBlock("2. ...")]),
                (RawBlock("Điều 16."),  [RawBlock("1. Đại diện...")]),
            ]
        """
        groups: list[tuple[RawBlock, list[RawBlock]]] = []
        current_header: Optional[RawBlock] = None
        current_children: list[RawBlock] = []

        def flush_group():
            """Đẩy nhóm hiện tại vào danh sách nếu có header."""
            nonlocal current_header, current_children
            if current_header is not None:
                groups.append((current_header, current_children))
            current_header = None
            current_children = []

        for block in blocks:
            if block.block_type == "article":
                # Flush nhóm Điều trước đó, bắt đầu nhóm mới
                flush_group()
                current_header = block
                current_children = []

            elif block.block_type in ("chapter", "section", "part"):
                # Flush nhóm trước, tạo nhóm standalone cho Chương/Mục
                # (để tiêu đề Chương cũng được index)
                flush_group()
                current_header = block
                current_children = []

            else:
                # clause / point / body → thêm vào children của nhóm hiện tại
                if current_header is None:
                    # Block trước khi gặp Điều đầu tiên → tạo nhóm body đầu trang
                    current_header = block
                    current_children = []
                else:
                    current_children.append(block)

        flush_group()  # flush nhóm cuối cùng
        return groups

    def chunk(self, blocks: list[RawBlock]) -> list[LegalChunk]:
        """
        Public API — chuyển đổi toàn bộ list[RawBlock] → list[LegalChunk].

        Đây là method duy nhất bên ngoài cần gọi trên LegalChunker.
        Thực hiện 2 bước nội bộ:
            1. _group_by_article() → nhóm blocks theo Điều
            2. _process_article()  → tạo chunk(s) cho từng Điều

        Args:
            blocks: list[RawBlock] từ LegalPDFProcessor.process()

        Returns:
            list[LegalChunk] — gồm cả parent và child chunks,
            sẵn sàng để EmbeddingGenerator embed và LegalIndexer lưu vào ChromaDB.

        Ví dụ:
            processor = LegalPDFProcessor()
            blocks    = processor.process("luat_hon_nhan.pdf", law_name="...")

            chunker = LegalChunker()
            chunks  = chunker.chunk(blocks)

            print(len(chunks))                      # tổng số chunk
            print(chunks[0].chunk_type)             # "article" / "clause" / ...
            print(chunks[0].hierarchy_path)         # "Chương III > Điều 15"
            print(chunks[0].is_parent)              # True / False
        """
        groups = self._group_by_article(blocks)

        all_chunks: list[LegalChunk] = []
        for article_block, child_blocks in groups:
            chunks = self._process_article(article_block, child_blocks)
            all_chunks.extend(chunks)

        return all_chunks
