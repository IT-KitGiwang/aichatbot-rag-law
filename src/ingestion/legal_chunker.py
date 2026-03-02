"""
Chia nhỏ văn bản luật theo cấu trúc pháp lý — 3-Level Hierarchical Chunking.

Strategy 3 cấp:
  • PARENT  chunk = toàn bộ 1 Điều luật    → dùng để trả context đầy đủ
  • CHILD   chunk = từng Khoản trong Điều   → search chính xác theo khoản
  • GRANDCHILD chunk = từng Điểm trong Khoản → search khi khoản quá dài (a,b,c,...)
  • Mỗi chunk kèm context header: [Luật X > Chương Y > Điều Z]
  • Không bao giờ cắt giữa 1 Điều / 1 Khoản / 1 Điểm
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.pdf_processor import LegalArticle


# ──────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────

@dataclass
class LegalChunk:
    """Một chunk văn bản luật, sẵn sàng để embedding."""
    chunk_id: str
    text: str                               # Nội dung đã có context header
    chunk_type: str                         # "parent" | "child" | "grandchild"
    parent_chunk_id: Optional[str] = None   # None nếu là parent
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Regex tách Khoản và Điểm
# ──────────────────────────────────────────────

# Khoản: dòng bắt đầu bằng "1. ", "2. ", ...
RE_CLAUSE = re.compile(r'(?=^(\d+)\.\s+)', re.MULTILINE)

# Điểm: dòng bắt đầu bằng "a) ", "b) ", ..., "đ) ", ...
RE_POINT = re.compile(r'(?=^([a-zđ])\)\s+)', re.MULTILINE)


# ──────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────

class LegalChunker:
    """
    Chia list[LegalArticle] → list[LegalChunk] theo chiến lược
    3-Level Hierarchical Chunking cho pháp luật VN.

    Cấp 1: Điều (parent)     — luôn tạo
    Cấp 2: Khoản (child)     — tạo khi Điều có ≥ 2 khoản
    Cấp 3: Điểm (grandchild) — tạo khi Khoản có ≥ 2 điểm VÀ vượt max_clause_length
    """

    def __init__(
        self,
        min_child_length: int = 80,
        max_clause_length: int = 800,
        include_context_header: bool = True,
    ):
        self.min_child_length = min_child_length
        self.max_clause_length = max_clause_length  # Nếu khoản dài hơn → tách điểm
        self.include_context_header = include_context_header

    # ── public ──────────────────────────────

    def chunk_articles(self, articles: list[LegalArticle]) -> list[LegalChunk]:
        """Chia toàn bộ danh sách Điều thành chunks."""
        all_chunks: list[LegalChunk] = []

        for article in articles:
            chunks = self._chunk_one_article(article)
            all_chunks.extend(chunks)

        n_parent = sum(1 for c in all_chunks if c.chunk_type == "parent")
        n_child = sum(1 for c in all_chunks if c.chunk_type == "child")
        n_grand = sum(1 for c in all_chunks if c.chunk_type == "grandchild")
        print(
            f"[Chunker] Tổng: {len(all_chunks)} chunks "
            f"({n_parent} parent, {n_child} child, {n_grand} grandchild) "
            f"từ {len(articles)} Điều"
        )
        return all_chunks

    # ── private ─────────────────────────────

    def _make_header(self, article: LegalArticle) -> str:
        """Tạo context header: [Luật X > Chương Y > Mục Z > Điều N]"""
        if not self.include_context_header:
            return ""
        path = article.metadata.get("hierarchy_path", f"Điều {article.article_number}")
        return f"[{path}]\n\n"

    def _chunk_one_article(self, article: LegalArticle) -> list[LegalChunk]:
        """Tạo parent + child + grandchild chunks cho 1 Điều."""
        chunks: list[LegalChunk] = []
        uid = uuid.uuid4().hex[:8]
        parent_id = f"art_{article.article_number}_{uid}"
        header = self._make_header(article)

        # ── CẤP 1 — PARENT: toàn bộ Điều ──
        parent = LegalChunk(
            chunk_id=parent_id,
            text=header + article.text,
            chunk_type="parent",
            metadata={
                **article.metadata,
                "chunk_type": "parent",
                "chunk_id": parent_id,
            },
        )
        chunks.append(parent)

        # ── CẤP 2 — CHILD: từng Khoản ──
        clause_spans = self._split_by_regex(article.text, RE_CLAUSE)

        # Chỉ tạo children nếu Điều có ≥ 2 khoản
        if len(clause_spans) >= 2:
            for clause_num, clause_text in clause_spans:
                clause_text_stripped = clause_text.strip()
                if len(clause_text_stripped) < self.min_child_length:
                    continue

                child_id = f"cl_{article.article_number}_{clause_num}_{uid}"
                clause_hierarchy = (
                    article.metadata.get("hierarchy_path", "")
                    + f" > Khoản {clause_num}"
                )

                child = LegalChunk(
                    chunk_id=child_id,
                    text=header + clause_text_stripped,
                    chunk_type="child",
                    parent_chunk_id=parent_id,
                    metadata={
                        **article.metadata,
                        "chunk_type": "child",
                        "chunk_id": child_id,
                        "parent_chunk_id": parent_id,
                        "clause": f"Khoản {clause_num}",
                        "hierarchy_path": clause_hierarchy,
                    },
                )
                chunks.append(child)

                # ── CẤP 3 — GRANDCHILD: từng Điểm (nếu khoản quá dài) ──
                if len(clause_text_stripped) > self.max_clause_length:
                    point_spans = self._split_by_regex(clause_text_stripped, RE_POINT)
                    if len(point_spans) >= 2:
                        for point_letter, point_text in point_spans:
                            point_text_stripped = point_text.strip()
                            if len(point_text_stripped) < self.min_child_length:
                                continue

                            grand_id = (
                                f"pt_{article.article_number}"
                                f"_{clause_num}_{point_letter}_{uid}"
                            )
                            point_hierarchy = (
                                clause_hierarchy + f" > Điểm {point_letter})"
                            )

                            grand = LegalChunk(
                                chunk_id=grand_id,
                                text=header + point_text_stripped,
                                chunk_type="grandchild",
                                parent_chunk_id=child_id,
                                metadata={
                                    **article.metadata,
                                    "chunk_type": "grandchild",
                                    "chunk_id": grand_id,
                                    "parent_chunk_id": child_id,
                                    "clause": f"Khoản {clause_num}",
                                    "point": f"Điểm {point_letter})",
                                    "hierarchy_path": point_hierarchy,
                                },
                            )
                            chunks.append(grand)

        return chunks

    @staticmethod
    def _split_by_regex(
        text: str, pattern: re.Pattern
    ) -> list[tuple[str, str]]:
        """
        Tách text theo regex pattern (Khoản hoặc Điểm).
        Trả về [(number_or_letter, segment_text), ...].
        """
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        results: list[tuple[str, str]] = []
        for i, m in enumerate(matches):
            label = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            results.append((label, text[start:end]))

        return results
