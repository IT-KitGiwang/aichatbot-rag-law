# src/ingestion/legal_chunker.py
"""
Legal-Aware Chunker — 4.1.2 trong kế hoạch.
Chuyển đổi: list[RawBlock] → list[LegalChunk]

Chiến lược: Parent-Child chunking
    - Parent = toàn bộ Điều luật để LLM có đủ ngữ cảnh khi trả lời
    - Child  = từng Khoản / Điểm để tìm kiếm chính xác và gọn hơn
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
    Đơn vị văn bản sau chunking, sẵn sàng để embed và lưu vào ChromaDB.

    Đây là output của LegalChunker và input cho EmbeddingGenerator + LegalIndexer.

    Hai loại chunk:
            - is_parent=True  : chunk toàn bộ Điều, chỉ dùng để mở rộng ngữ cảnh
            - is_parent=False : chunk Khoản/Điểm, dùng cho semantic search

        Các field chính:
            - chunk_size / chunk_overlap: quyết định độ dài và mức chồng lấn khi cắt
            - min_chunk_size: ngưỡng tối thiểu để cố gắng gộp các mảnh quá ngắn
            - on_small_chunk: chiến lược xử lý chunk nhỏ (hiện dùng merge_next)
            - include_parent_context: có prepend Chương/Mục/Điều vào text hay không

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
    chunk_id: str                    # UUID duy nhất cho từng chunk
    text: str                        # Nội dung chunk, có thể kèm context prefix
    chunk_type: str                  # "article" | "clause" | "point" | "section" | "body"
    source_page: int                 # Trang gốc trong PDF

    # --- Thông tin văn bản luật ---
    law_name: str = ""
    law_number: str = ""
    effective_date: str = ""
    law_preamble: str = ""          # Phần căn cứ/mở đầu: không embed, chỉ hiển thị

    # --- Vị trí trong cấu trúc luật ---
    part: str = ""
    chapter: str = ""
    chapter_title: str = ""
    section: str = ""
    article: str = ""
    article_title: str = ""
    hierarchy_path: str = ""         # Đường dẫn cấu trúc: "Chương III > Mục 1 > Điều 15"

    # --- Parent-Child ---
    is_parent: bool = False          # True nếu là parent chunk (toàn Điều)
    parent_chunk_id: Optional[str] = None  # ID parent; None nếu chunk độc lập

    # --- Thống kê ---
    token_count: int = 0             # Số token ước lượng để quyết định cắt/gộp

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
            "law_preamble":     self.law_preamble,
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
    Ước tính số token bằng cách đếm từ và nhân hệ số gần đúng.

    Không dùng tokenizer thực sự (để tránh dependency nặng và chậm).
    Với tiếng Việt, 1 token thường tương ứng xấp xỉ 1.3 từ.
    Hàm này chỉ dùng để so sánh tương đối với chunk_size/min_chunk_size.

    Ví dụ:
        _approx_token_count("Vợ chồng bình đẳng với nhau")
        → 5 từ → 6 token (gần đúng)
    """
    words = text.split()
    # Hệ số 1.3: tiếng Việt có nhiều từ ghép ngắn → token count cao hơn word count
    return int(len(words) * 1.3)


def _token_budget_to_word_budget(token_budget: int, safety_margin_tokens: int = 0) -> int:
    """
    Quy đổi token budget sang số từ an toàn trước khi cắt text.

    Vì token được ước lượng từ số từ, ta trừ thêm một biên an toàn để
    phần prefix/context không làm đoạn sau khi ghép bị vượt trần.
    """
    safe_tokens = max(1, token_budget - safety_margin_tokens)
    return max(1, int(safe_tokens / 1.35))


def _build_context_prefix(struct: LegalStructure) -> str:
    """
    Tạo prefix ngữ cảnh để embedding hiểu chunk thuộc Chương/Mục/Điều nào.

    Prefix này giúp model không chỉ nhìn thấy nội dung rời rạc,
    mà còn biết văn bản đang nằm trong cấu trúc luật nào.

    Ví dụ output:
        struct = LegalStructure(
            chapter = "Chương III",
            section = "Mục 1",
            article = "Điều 15. Quyền và nghĩa vụ..."
        )
        → "Chương III | Mục 1 | Điều 15."

    Chỉ giữ phần ngắn của Điều để prefix không chiếm quá nhiều token.
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


def _split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Cắt text thành nhiều đoạn chồng lấn theo giới hạn token xấp xỉ.

    Mục tiêu: khi một khoản quá dài, vẫn tách được thành nhiều child chunks
    nhưng giữ được ngữ cảnh liên tục qua overlap.
    """
    words = text.split()
    if not words:
        return []

    if chunk_size <= 0:
        return [text]

    # Quy đổi token budget sang số từ để segment thực tế không bị vượt trần.
    max_words = _token_budget_to_word_budget(chunk_size, safety_margin_tokens=8)

    # Tránh step <= 0 gây vòng lặp vô hạn.
    overlap_words = _token_budget_to_word_budget(chunk_overlap)
    overlap_words = max(0, min(overlap_words, max_words - 1))
    step = max(1, max_words - overlap_words)

    segments: list[str] = []
    min_segment_words = 20
    for start in range(0, len(words), step):
        end = start + max_words
        segment_words = words[start:end]
        if not segment_words:
            break
        # Tránh sinh đoạn quá ngắn. Nếu mảnh cuối quá ít từ thì gộp ngược
        # vào đoạn trước để không làm xuất hiện chunk rác.
        if len(segment_words) < min_segment_words:
            if segments:
                segments[-1] = (segments[-1] + " " + " ".join(segment_words)).strip()
                break
            segments.append(" ".join(segment_words))
        else:
            segments.append(" ".join(segment_words))
        if end >= len(words):
            break

    return segments


def _truncate_text_to_budget(text: str, token_budget: int, safety_margin_tokens: int = 12) -> str:
    """
    Cắt text xuống mức an toàn theo token budget xấp xỉ.

    Dùng cho parent chunk dài để giữ context gọn hơn, giảm chi phí lưu trữ
    và tránh kéo large_rate lên vì các parent quá dài.
    """
    content = text.strip()
    if not content:
        return content

    words = content.split()
    max_words = _token_budget_to_word_budget(token_budget, safety_margin_tokens=safety_margin_tokens)
    if len(words) <= max_words:
        return content

    trimmed = " ".join(words[:max_words]).strip()
    return f"{trimmed} ..."


def _merge_prefix_with_content(prefix: str, content: str) -> str:
    """
    Ghép prefix + content nhưng tránh lặp dòng đầu giống hệt prefix.

    Trường hợp thường gặp: chunk standalone của "Chương I" có prefix="Chương I"
    và content cũng bắt đầu bằng "Chương I".
    """
    if not prefix:
        return content.strip()

    content_stripped = content.strip()
    if not content_stripped:
        return prefix.strip()

    first_line = content_stripped.splitlines()[0].strip()
    normalized_first = " ".join(first_line.split()).lower()
    normalized_prefix = " ".join(prefix.strip().split()).lower()
    if normalized_first == normalized_prefix:
        return content_stripped

    return f"{prefix}\n{content_stripped}".strip()


def _split_structured_legal_block(
    content: str,
    token_budget: int,
    chunk_overlap: int,
) -> list[str]:
    """Split a structured legal block without breaking clause/point semantics.

    Goal: avoid producing segments that start mid-structure (e.g. starting with
    "b)" while losing the owning clause "1.").

    Strategy:
      - Detect a leading clause header line like "1. ...".
      - Detect point lines like "a) ...", "b) ..." and keep each point as an atomic block.
      - Pack points into segments within token_budget, always prefixing with the clause header.
      - If a single point is still too large, fall back to word splitting but keep the label.
    """

    content = content.strip()
    if not content:
        return []

    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    if not lines:
        return []

    def is_clause_header(line: str) -> bool:
        return bool(line.strip() and line.strip()[0].isdigit() and "." in line.split(maxsplit=1)[0])

    def is_point_line(line: str) -> bool:
        s = line.strip().lower()
        if len(s) < 2:
            return False
        # Matches "a) ...", "đ) ..." at line start.
        return bool(s[0] in "abcdefghijklmnopqrstuvwxyzđ" and s[1] == ")")

    header_line = lines[0].strip() if is_clause_header(lines[0]) else ""
    remaining = lines[1:] if header_line else lines

    # Build point blocks from remaining lines.
    blocks: list[str] = []
    current: list[str] = []
    saw_point = False
    for line in remaining:
        if is_point_line(line):
            saw_point = True
            if current:
                blocks.append("\n".join(current).strip())
            current = [line.strip()]
        else:
            if not current:
                current = [line.strip()]
            else:
                current.append(line.strip())
    if current:
        blocks.append("\n".join(current).strip())

    if not blocks:
        return []

    # If we didn't see clear point structure and we have no header, fallback to word splitting.
    if not saw_point and not header_line:
        return _split_text_with_overlap(content, token_budget, chunk_overlap)

    def approx_ok(text: str) -> bool:
        return _approx_token_count(text) <= token_budget

    segments: list[str] = []
    current_lines: list[str] = [header_line] if header_line else []

    for block in blocks:
        candidate_lines = [*current_lines, block] if current_lines else [block]
        candidate_text = "\n".join(candidate_lines).strip()

        if not current_lines:
            current_lines = ([header_line] if header_line else []) + [block]
            continue

        if approx_ok(candidate_text):
            current_lines = candidate_lines
            continue

        # Flush current segment (must contain at least header or some content).
        if "\n".join(current_lines).strip():
            segments.append("\n".join(current_lines).strip())

        # Start new segment with header + this block.
        current_lines = ([header_line] if header_line else []) + [block]

        # If even header+block is too large, split the block while preserving its label if possible.
        if not approx_ok("\n".join(current_lines).strip()):
            label = ""
            block_stripped = block.strip()
            first_word = block_stripped.split(maxsplit=1)[0] if block_stripped else ""
            if first_word.endswith(")") and len(first_word) == 2:
                label = first_word
                remainder = block_stripped[len(first_word):].strip()
            else:
                remainder = block_stripped

            sub_segments = _split_text_with_overlap(remainder, token_budget, chunk_overlap)
            for sub in sub_segments:
                sub = sub.strip()
                if not sub:
                    continue
                rebuilt = " ".join([p for p in [label, sub] if p]).strip()
                if header_line:
                    rebuilt = "\n".join([header_line, rebuilt]).strip()
                segments.append(rebuilt)
            current_lines = [header_line] if header_line else []

    tail = "\n".join(current_lines).strip()
    if tail:
        segments.append(tail)

    # Last resort: ensure we always return something.
    return segments or _split_text_with_overlap(content, token_budget, chunk_overlap)


# ===========================================================================
# BƯỚC 2.3 — LegalChunker class & _process_article()
# ===========================================================================

from dataclasses import dataclass

@dataclass
class LegalChunker:
    """Chia văn bản luật thành LegalChunk tối ưu cho RAG luật."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 80
    on_small_chunk: str = "merge_next"
    include_parent_context: bool = True

    # chunk_size: trần token ước lượng cho mỗi child chunk.
    # chunk_overlap: phần chồng lấn khi phải cắt thành nhiều đoạn.
    # min_chunk_size: ngưỡng dưới để ưu tiên gộp chunk nhỏ thay vì giữ riêng.
    # on_small_chunk: chính sách xử lý chunk nhỏ; hiện dùng "merge_next".
    # include_parent_context: prepend Chương/Mục/Điều vào text khi tạo chunk.

    # ------------------------------------------------------------------
    # Bước 2.3: Xử lý một Điều luật → tạo chunk(s)
    # ------------------------------------------------------------------
    
    def _process_article(
        self,
        article_block: RawBlock,
        child_blocks: list[RawBlock],
        law_preamble: str = "",
    ) -> list[LegalChunk]:
        """
        Nhận một Điều luật và danh sách Khoản/Điểm, rồi tạo chunk(s).

        Quy trình:
          1. Gom toàn bộ text → parent_text
          2. Tính token_count
          3. Nếu ngắn   → trả về 1 chunk đơn
             Nếu dài    → tạo 1 parent + N children (1 child/khoản)

        Args:
            article_block: RawBlock loại "article" (dòng "Điều X.")
            child_blocks: RawBlock loại "clause"/"point"/"body" nằm sau Điều đó.

        Returns:
            list[LegalChunk] — có thể là [1 chunk đơn]
                               hoặc [parent, child1, child2, ...]
        """
        struct = article_block.structure
        law_info = {
            "law_name": article_block.law_name,
            "law_number": article_block.law_number,
            "effective_date": article_block.effective_date,
            "law_preamble": law_preamble,
        }

        # Prefix ngữ cảnh để giữ mối liên hệ với Chương/Mục/Điều khi embed.
        prefix = _build_context_prefix(struct) if self.include_parent_context else ""

        # Bước 1: ghép toàn bộ nội dung của Điều để tính parent chunk.
        all_texts = [article_block.text] + [b.text for b in child_blocks]
        full_content = "\n".join(all_texts)

        # Parent text = prefix + toàn bộ nội dung Điều.
        parent_text = _merge_prefix_with_content(prefix, full_content)

        # Bước 2: ước lượng token để biết Điều này có cần tách parent/child hay không.
        parent_tokens = _approx_token_count(parent_text)
        parent_is_long = parent_tokens > self.chunk_size

        # Metadata dùng chung cho mọi chunk của cùng một Điều.
        base_meta = self._build_base_meta(struct, law_info, article_block.page)

        # Bước 3a: Điều ngắn → giữ nguyên thành 1 chunk đơn, không tách parent/child.
        if not parent_is_long:
            # Không drop chunk nhỏ trong luật, vì có thể làm mất nghĩa pháp lý.
            return [LegalChunk(
                chunk_id=str(uuid.uuid4()),
                text=parent_text,
                chunk_type=article_block.block_type,
                is_parent=False,
                parent_chunk_id=None,
                token_count=parent_tokens,
                **base_meta,
            )]

        # Bước 3b: Điều dài → tạo parent chunk + các child chunk.
        parent_id = str(uuid.uuid4())

        # Parent chunk giữ toàn bộ Điều để dùng khi cần mở rộng ngữ cảnh.
        parent_chunk = LegalChunk(
            chunk_id=parent_id,
            text=parent_text,
            chunk_type=article_block.block_type,
            is_parent=True,
            parent_chunk_id=None,
            token_count=parent_tokens,
            **base_meta,
        )

        # Child chunks là Khoản/Điểm để phục vụ semantic search.
        child_chunks: list[LegalChunk] = []

        # Chọn budget cho child: vừa đủ dư cho prefix, vừa tránh cắt vụn đoạn nội dung.
        prefix_tokens = _approx_token_count(prefix) if prefix else 0
        child_budget = max(1, max(self.chunk_size - prefix_tokens - 8, int(self.chunk_size * 0.85)))

        def append_child_chunk(text_content: str, chunk_type: str) -> None:
            content = text_content.strip()
            if not content:
                return

            min_child_tokens = 80
            produced_base_index = len(child_chunks)

            def strip_prefix(text: str) -> str:
                if not prefix:
                    return text.strip()
                lines = text.splitlines()
                if lines:
                    first = " ".join(lines[0].split()).lower()
                    pref = " ".join(prefix.split()).lower()
                    if first == pref:
                        return "\n".join(lines[1:]).strip()
                return text.strip()

            def maybe_merge_small_with_previous(new_text: str, new_tokens: int) -> tuple[str, int]:
                """Nếu chunk mới quá nhỏ, thử gộp với chunk trước đó trong cùng lượt xử lý."""
                if new_tokens >= min_child_tokens:
                    return new_text, new_tokens
                if len(child_chunks) <= produced_base_index:
                    return new_text, new_tokens

                prev = child_chunks.pop()
                prev_content = strip_prefix(prev.text)
                new_content = strip_prefix(new_text)
                merged_content = "\n".join([prev_content, new_content]).strip()
                merged_text = _merge_prefix_with_content(prefix, merged_content)
                merged_tokens = _approx_token_count(merged_text)

                if merged_tokens <= self.chunk_size and prev.chunk_type == chunk_type:
                    return merged_text, merged_tokens

                child_chunks.append(prev)
                return new_text, new_tokens

            # Với clause/point: ưu tiên giữ nguyên cấu trúc pháp lý.
            # - Vừa budget thì giữ nguyên.
            # - Quá dài thì split theo cấu trúc, không để rơi mất header "1.".
            if chunk_type in ("clause", "point"):
                whole_text = _merge_prefix_with_content(prefix, content)
                whole_tokens = _approx_token_count(whole_text)
                if whole_tokens <= self.chunk_size:
                    whole_text, whole_tokens = maybe_merge_small_with_previous(whole_text, whole_tokens)
                    child_chunks.append(LegalChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=whole_text,
                        chunk_type=chunk_type,
                        is_parent=False,
                        parent_chunk_id=parent_id,
                        token_count=whole_tokens,
                        **base_meta,
                    ))
                    return

                segments = _split_structured_legal_block(
                    content=content,
                    token_budget=child_budget,
                    chunk_overlap=self.chunk_overlap,
                )
            else:
                # Body/other: cắt theo token ước lượng.
                segments = _split_text_with_overlap(content, child_budget, self.chunk_overlap)

            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                child_text = _merge_prefix_with_content(prefix, segment)
                child_tokens = _approx_token_count(child_text)

                # Nếu vẫn vượt trần do prefix hoặc sai số ước lượng, cắt tiếp.
                if child_tokens > self.chunk_size and len(segment.split()) > 1:
                    tighter_budget = max(1, child_budget // 2)
                    # Với clause/point, giữ lại header đầu dòng kiểu "1.".
                    header_line = ""
                    seg_lines = [ln for ln in segment.splitlines() if ln.strip()]
                    if chunk_type in ("clause", "point") and seg_lines:
                        first = seg_lines[0].strip()
                        if first and first[0].isdigit() and "." in first.split(maxsplit=1)[0]:
                            header_line = first
                            remainder = "\n".join(seg_lines[1:]).strip()
                        else:
                            remainder = segment
                    else:
                        remainder = segment

                    for tighter_segment in _split_text_with_overlap(remainder, tighter_budget, self.chunk_overlap):
                        tighter_segment = tighter_segment.strip()
                        if not tighter_segment:
                            continue
                        rebuilt = tighter_segment
                        if header_line and not tighter_segment.lstrip().startswith(header_line):
                            rebuilt = "\n".join([header_line, tighter_segment]).strip()
                        tighter_text = _merge_prefix_with_content(prefix, rebuilt)
                        tighter_tokens = _approx_token_count(tighter_text)
                        tighter_text, tighter_tokens = maybe_merge_small_with_previous(tighter_text, tighter_tokens)
                        child_chunks.append(LegalChunk(
                            chunk_id=str(uuid.uuid4()),
                            text=tighter_text,
                            chunk_type=chunk_type,
                            is_parent=False,
                            parent_chunk_id=parent_id,
                            token_count=tighter_tokens,
                            **base_meta,
                        ))
                    continue

                child_text, child_tokens = maybe_merge_small_with_previous(child_text, child_tokens)
                child_chunks.append(LegalChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=child_text,
                    chunk_type=chunk_type,
                    is_parent=False,
                    parent_chunk_id=parent_id,  # ← trỏ về parent
                    token_count=child_tokens,
                    **base_meta,
                ))

        # Buffer cho small chunks: merge_next = gộp child nhỏ với child kế tiếp.
        small_buffer: list[str] = []
        small_buffer_type: Optional[str] = None

        def flush_small_buffer_with_next(next_text: str, next_type: str) -> None:
            """Gộp các đoạn nhỏ đang treo với đoạn kế tiếp lớn hơn."""
            nonlocal small_buffer, small_buffer_type

            # Nếu buffer đang là clause, chỉ nên merge với point/body (thuộc clause đó),
            # không merge với clause khác.
            if small_buffer_type == "clause" and next_type == "clause":
                merged_text = "\n".join(small_buffer).strip()
                merged_type = small_buffer_type
                small_buffer = []
                small_buffer_type = None
                append_child_chunk(merged_text, merged_type)
                append_child_chunk(next_text, next_type)
                return

            merged_text = "\n".join([*small_buffer, next_text]).strip()
            merged_type = small_buffer_type or next_type
            small_buffer = []
            small_buffer_type = None
            append_child_chunk(merged_text, merged_type)

        def flush_small_buffer_as_tail() -> None:
            """Xử lý phần nhỏ còn lại ở cuối để không mất dữ liệu."""
            nonlocal small_buffer, small_buffer_type
            if not small_buffer:
                return

            tail_text = "\n".join(small_buffer).strip()
            tail_type = small_buffer_type or "clause"

            if child_chunks:
                # Thử gộp vào chunk ngay trước đó để giảm chunk vụn ở cuối.
                previous_chunk = child_chunks.pop()
                # previous_chunk.text đã có prefix; nếu ghép thẳng rồi gọi append_child_chunk()
                # sẽ bị add prefix lần nữa → lặp prefix. Vì vậy, ghép theo "content thuần".
                previous_text = previous_chunk.text.strip()
                previous_content = previous_text
                if prefix:
                    first_line = previous_text.splitlines()[0].strip() if previous_text else ""
                    normalized_first = " ".join(first_line.split()).lower()
                    normalized_prefix = " ".join(prefix.strip().split()).lower()
                    if normalized_first == normalized_prefix:
                        previous_content = "\n".join(previous_text.splitlines()[1:]).strip()

                candidate_content = "\n".join([previous_content, tail_text]).strip()
                candidate_text = _merge_prefix_with_content(prefix, candidate_content)
                candidate_tokens = _approx_token_count(candidate_text)

                if candidate_tokens <= self.chunk_size and previous_chunk.chunk_type == tail_type:
                    append_child_chunk(candidate_content, previous_chunk.chunk_type)
                else:
                    # Nếu gộp vượt trần, trả lại chunk trước đó rồi giữ tail riêng.
                    child_chunks.append(previous_chunk)
                    append_child_chunk(tail_text, tail_type)
            else:
                append_child_chunk(tail_text, tail_type)

            small_buffer = []
            small_buffer_type = None

        # Merge theo CLAUSE (giữ cấu trúc pháp lý): gom point (a)/b)/c)) ngay sau clause
        # vào cùng một đơn vị clause trước khi split theo budget.
        logical_units: list[tuple[str, str]] = []
        i = 0
        while i < len(child_blocks):
            block = child_blocks[i]
            text = block.text.strip()
            if not text:
                i += 1
                continue

            if block.block_type == "clause":
                collected = [text]
                j = i + 1
                while j < len(child_blocks) and child_blocks[j].block_type == "point":
                    point_text = child_blocks[j].text.strip()
                    if point_text:
                        collected.append(point_text)
                    j += 1
                logical_units.append(("\n".join(collected).strip(), "clause"))
                i = j
                continue

            if block.block_type == "point":
                # Point rời rạc (không có clause phía trước) — giữ nguyên.
                logical_units.append((text, "point"))
                i += 1
                continue

            logical_units.append((text, block.block_type))
            i += 1

        # Pack clause liên tiếp để tăng density, nhưng giữ <= chunk_size để tránh
        # split trên multi-clause (có thể làm lệch cấu trúc nếu phải split).
        packed_units: list[tuple[str, str]] = []
        clause_pack: list[str] = []
        pack_limit = max(1, self.chunk_size - 12)

        def flush_clause_pack() -> None:
            nonlocal clause_pack
            if not clause_pack:
                return
            packed_units.append(("\n".join(clause_pack).strip(), "clause"))
            clause_pack = []

        for unit_text, unit_type in logical_units:
            if unit_type == "clause":
                candidate_pack = "\n".join([*clause_pack, unit_text]).strip()
                candidate_text = _merge_prefix_with_content(prefix, candidate_pack)
                candidate_tokens = _approx_token_count(candidate_text)

                if clause_pack and candidate_tokens > pack_limit:
                    flush_clause_pack()
                    clause_pack = [unit_text]
                else:
                    clause_pack.append(unit_text)
                continue

            flush_clause_pack()
            packed_units.append((unit_text, unit_type))

        flush_clause_pack()

        for unit_text, unit_type in packed_units:
            # Tránh split child 2 lần: KHÔNG split ở vòng ngoài.
            # Chỉ delegate cho append_child_chunk(), và append sẽ chỉ split khi cần.
            content_with_prefix = _merge_prefix_with_content(prefix, unit_text)
            content_tokens = _approx_token_count(content_with_prefix)

            if content_tokens < self.min_chunk_size:
                # Chunk nhỏ: chưa giữ riêng ngay mà đưa vào buffer để chờ gộp với phần sau.
                if not small_buffer:
                    small_buffer_type = unit_type
                small_buffer.append(unit_text)
                continue

            # Khi gặp đoạn đủ dài, gộp luôn phần nhỏ đang treo với đoạn này.
            if small_buffer:
                flush_small_buffer_with_next(unit_text, unit_type)
            else:
                append_child_chunk(unit_text, unit_type)

        # Xử lý phần còn treo ở cuối để không làm rơi nội dung ngắn.
        flush_small_buffer_as_tail()

        # Nếu chưa sinh được child nào, fallback về cách cắt thô để tránh chỉ còn parent chunk.
        if not child_chunks:
            fallback_segments = _split_text_with_overlap(
                text=full_content,
                chunk_size=child_budget,
                chunk_overlap=self.chunk_overlap,
            )

            for segment in fallback_segments:
                segment = segment.strip()
                if not segment:
                    continue
                child_text = _merge_prefix_with_content(prefix, segment)
                child_tokens = _approx_token_count(child_text)
                child_chunks.append(LegalChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=child_text,
                    chunk_type=article_block.block_type,
                    is_parent=False,
                    parent_chunk_id=parent_id,
                    token_count=child_tokens,
                    **base_meta,
                ))

        if not child_chunks:
            return [parent_chunk]

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
        Trả về các keyword argument dùng chung khi tạo LegalChunk.

        Gom ở đây để tránh lặp logic giữa parent chunk và child chunk.
        Các chunk cùng một Điều sẽ dùng chung bộ metadata này.
        """
        return {
            "source_page":    page,
            "law_name":       law_info.get("law_name", ""),
            "law_number":     law_info.get("law_number", ""),
            "effective_date": law_info.get("effective_date", ""),
            "law_preamble":   law_info.get("law_preamble", ""),
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

        Lưu ý: các block "part/chapter/section" chỉ là header cấu trúc rất ngắn.
        Chúng không cần đi vào luồng index riêng vì thông tin đó đã nằm trong
        metadata/prefix của các chunk Điều/Khoản phía sau.
        Vì vậy, chunk() sẽ lọc chúng ra trước khi grouping.

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
        # 1) Lọc các header cấu trúc ngắn (part/chapter/section) trước khi grouping.
        # Các thông tin này đã được giữ trong metadata/prefix của chunk phía sau.
        non_header_blocks = [
            block for block in blocks if block.block_type not in ("part", "chapter", "section")
        ]

        # 2) Tách preamble/căn cứ ra khỏi luồng embedding/indexing, nhưng vẫn lưu để hiển thị.
        # Preamble ở đây là các block "body" nằm trước Điều đầu tiên.
        preamble_texts: list[str] = []
        preamble_cut_index = 0
        for idx, block in enumerate(non_header_blocks):
            if block.block_type == "article":
                preamble_cut_index = idx
                break
            if block.block_type == "body":
                preamble_texts.append(block.text.strip())
        else:
            preamble_cut_index = len(non_header_blocks)

        law_preamble = "\n".join([t for t in preamble_texts if t]).strip()
        # Giới hạn độ dài để metadata không phình quá lớn; chỉ dùng cho hiển thị.
        if len(law_preamble) > 800:
            law_preamble = law_preamble[:800].rstrip() + " ..."

        # Loại bỏ phần preamble khỏi luồng chunking/indexing.
        filtered_blocks = non_header_blocks[preamble_cut_index:]

        groups = self._group_by_article(filtered_blocks)

        all_chunks: list[LegalChunk] = []
        for article_block, child_blocks in groups:
            chunks = self._process_article(article_block, child_blocks, law_preamble=law_preamble)
            all_chunks.extend(chunks)

        return all_chunks
