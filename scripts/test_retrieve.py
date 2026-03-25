#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retrieval test harness.

Use an existing vectorstore to check whether retrieval returns useful child
chunks and whether full parent context can be hydrated correctly.

Usage:
    python scripts/test_retrieve.py --query "Thủ tục ly hôn là gì?"
    python scripts/test_retrieve.py --query-file data/evaluation/queries.txt

Optional:
    --top-k  Top-N child chunks to retrieve.
    --json   Write the report to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")


from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.indexer import LegalIndexer


def _read_queries(query: str | None, query_file: str | None) -> list[str]:
    if query:
        return [query.strip()]

    if query_file:
        path = Path(query_file)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file query: {path}")

        queries: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                queries.append(cleaned)
        return queries

    return []


def _score_parent_hydration(results: list[dict]) -> dict:
    total = len(results)
    hydrated = sum(1 for item in results if item.get("parent"))
    parent_scores = [float(item["parent"].get("metadata", {}).get("token_count", 0)) for item in results if item.get("parent")]

    return {
        "total": total,
        "hydrated": hydrated,
        "hydration_rate": round((hydrated / total) if total else 0.0, 4),
        "avg_parent_tokens": round((sum(parent_scores) / len(parent_scores)) if parent_scores else 0.0, 2),
    }


def _print_result(query: str, results: list[dict]) -> dict:
    stats = _score_parent_hydration(results)
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)
    print(
        f"Hydration: {stats['hydrated']}/{stats['total']} | "
        f"rate={stats['hydration_rate']} | avg_parent_tokens={stats['avg_parent_tokens']}"
    )

    for i, item in enumerate(results, 1):
        score = float(item.get("score", 0.0))
        meta = item.get("metadata", {})
        parent = item.get("parent")
        parent_tokens = parent.get("metadata", {}).get("token_count", 0) if parent else 0
        parent_len = len(parent.get("text", "")) if parent else 0
        print(
            f"[{i}] score={score:.4f} | chunk_type={meta.get('chunk_type', '')} | "
            f"article={meta.get('article', '')} | parent={bool(parent)} | parent_tokens={parent_tokens} | parent_len={parent_len}"
        )

    return {
        "query": query,
        "results": results,
        "stats": stats,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test retrieval quality with parent context hydration.")
    parser.add_argument("--query", default="", help="One query string to test.")
    parser.add_argument("--query-file", default="", help="File containing one query per line.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of child chunks to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum similarity score to hydrate parent context.")
    parser.add_argument("--persist-dir", default="./vectorstore", help="ChromaDB persist directory.")
    parser.add_argument("--collection-name", default="legal_documents", help="Main child collection name.")
    parser.add_argument("--json", default="", help="Write report to JSON.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    queries = _read_queries(args.query, args.query_file)
    if not queries:
        parser.error("Cần --query hoặc --query-file")

    api_key = os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        parser.error("Thiếu VOYAGE_API_KEY để embed query.")

    embedder = EmbeddingGenerator()
    indexer = LegalIndexer(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_dim=embedder.dimension,
    )

    all_reports: list[dict[str, Any]] = []
    for query in queries:
        query_vector = embedder.embed_query(query)
        results = indexer.query_with_parent_context(
            query_vector=query_vector,
            n_results=args.top_k,
            min_score=args.min_score,
        )
        all_reports.append(_print_result(query, results))

    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({"reports": all_reports}, handle, ensure_ascii=False, indent=2)
        print(f"\nJSON written to: {output_path}")


if __name__ == "__main__":
    main()