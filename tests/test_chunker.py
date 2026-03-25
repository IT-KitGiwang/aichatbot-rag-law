from dataclasses import dataclass

import pytest

from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalPDFProcessor, LegalStructure, RawBlock


def test_identify_structure_keeps_chapter_title_in_text() -> None:
	processor = LegalPDFProcessor()
	clean_lines = [
		("Chương I", 1),
		("NHỮNG QUY ĐỊNH CHUNG", 1),
		("Điều 1. Phạm vi điều chỉnh", 1),
		("1. Luật này quy định...", 1),
	]

	blocks = processor._identify_structure(clean_lines)

	chapter_blocks = [b for b in blocks if b.block_type == "chapter"]
	assert len(chapter_blocks) == 1
	assert chapter_blocks[0].structure.chapter_title == "NHỮNG QUY ĐỊNH CHUNG"
	assert "NHỮNG QUY ĐỊNH CHUNG" in chapter_blocks[0].text


def test_identify_structure_keeps_section_title_in_text() -> None:
	processor = LegalPDFProcessor()
	clean_lines = [
		("Chương I", 1),
		("NHỮNG QUY ĐỊNH CHUNG", 1),
		("Mục 1", 1),
		("NGUYÊN TẮC CƠ BẢN", 1),
		("Điều 1. Phạm vi điều chỉnh", 1),
	]

	blocks = processor._identify_structure(clean_lines)

	section_blocks = [b for b in blocks if b.block_type == "section"]
	assert len(section_blocks) == 1
	assert section_blocks[0].structure.section_title == "NGUYÊN TẮC CƠ BẢN"
	assert "NGUYÊN TẮC CƠ BẢN" in section_blocks[0].text


@dataclass
class _FakeEmbedResponse:
	embeddings: list[list[float]]


class _FakeClientMismatchedCount:
	def embed(self, texts: list[str], model: str, input_type: str) -> _FakeEmbedResponse:
		_ = (model, input_type)
		# Cố tình trả thiếu vector để mô phỏng lỗi provider/network.
		if len(texts) >= 2:
			return _FakeEmbedResponse(embeddings=[[0.1, 0.2, 0.3, 0.4]])
		return _FakeEmbedResponse(embeddings=[[0.1, 0.2, 0.3, 0.4]])


def test_encode_batch_raises_on_embedding_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
	embedder = EmbeddingGenerator(config={"dimension": 4, "batch_size": 4})
	monkeypatch.setattr(embedder, "_get_client", lambda: _FakeClientMismatchedCount())

	with pytest.raises(ValueError, match="Embedding count mismatch"):
		embedder._encode_batch(["text 1", "text 2"], input_type="document")


def test_process_article_keeps_long_chunks_under_budget() -> None:
	chunker = LegalChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=20)
	structure = LegalStructure(chapter="Chương I", article="Điều 1. Phạm vi điều chỉnh")
	article_block = RawBlock(
		text="Điều 1. Phạm vi điều chỉnh",
		page=1,
		block_type="article",
		structure=structure.clone(),
		law_name="Luật mẫu",
	)
	long_clause_text = "1. " + "noi dung " * 260
	child_blocks = [
		RawBlock(
			text=long_clause_text,
			page=1,
			block_type="clause",
			structure=structure.clone(),
			law_name="Luật mẫu",
		),
	]

	chunks = chunker._process_article(article_block, child_blocks)

	assert chunks
	assert any(chunk.is_parent for chunk in chunks)
	assert all(chunk.token_count <= 120 for chunk in chunks if not chunk.is_parent)
	assert any(chunk.token_count > 120 for chunk in chunks if chunk.is_parent)


def test_process_article_fallback_splits_when_no_child_blocks() -> None:
	chunker = LegalChunker(chunk_size=120, chunk_overlap=20, min_chunk_size=20)
	structure = LegalStructure(chapter="Chương I", article="Điều 1. Phạm vi điều chỉnh")
	article_block = RawBlock(
		text="Điều 1. Phạm vi điều chỉnh " + ("noi dung " * 320),
		page=1,
		block_type="article",
		structure=structure.clone(),
		law_name="Luật mẫu",
	)

	chunks = chunker._process_article(article_block, [])

	assert any(chunk.is_parent for chunk in chunks)
	assert all(chunk.token_count <= 120 for chunk in chunks if not chunk.is_parent)
	assert len([chunk for chunk in chunks if not chunk.is_parent]) >= 2
