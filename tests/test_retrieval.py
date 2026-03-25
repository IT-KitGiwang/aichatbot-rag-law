# tests/test_retrieval.py
# Unit tests cho retrieval path: child query + parent context hydration

from __future__ import annotations


from src.ingestion.indexer import LegalIndexer


class _FakeChildCollection:
	def query(self, **kwargs):
		_ = kwargs
		return {
			"ids": [["child-1"]],
			"documents": [["child text"]],
			"metadatas": [[{"parent_chunk_id": "parent-1"}]],
			"distances": [[0.1]],
		}


class _FakeParentCollection:
	def get(self, **kwargs):
		_ = kwargs
		return {
			"ids": ["parent-1"],
			"documents": ["full parent text"],
			"metadatas": [{"is_parent": True}],
		}


def test_query_with_parent_context_attaches_full_parent_text() -> None:
	indexer = LegalIndexer.__new__(LegalIndexer)
	indexer._collection = _FakeChildCollection()
	indexer._parent_collection = _FakeParentCollection()

	results = indexer.query_with_parent_context(query_vector=[0.1, 0.2], n_results=1)

	assert len(results) == 1
	assert results[0]["text"] == "child text"
	assert results[0]["metadata"]["parent_chunk_id"] == "parent-1"
	assert results[0]["parent"]["text"] == "full parent text"


def test_query_with_parent_context_skips_low_score_hydration() -> None:
	class _LowScoreChildCollection:
		def query(self, **kwargs):
			_ = kwargs
			return {
				"ids": [["child-1"]],
				"documents": [["child text"]],
				"metadatas": [[{"parent_chunk_id": "parent-1"}]],
				"distances": [[0.8]],
			}

	indexer = LegalIndexer.__new__(LegalIndexer)
	indexer._collection = _LowScoreChildCollection()
	indexer._parent_collection = _FakeParentCollection()

	results = indexer.query_with_parent_context(query_vector=[0.1, 0.2], n_results=1, min_score=0.55)

	assert len(results) == 1
	assert results[0]["parent"] is None
