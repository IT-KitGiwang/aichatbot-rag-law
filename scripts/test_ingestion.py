#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingestion test harness.

Run a PDF through the ingestion stages and print metrics that help judge
whether chunking and embedding are reasonably optimized.

Usage:
    python scripts/test_ingestion.py --pdf data/raw_pdfs/example.pdf
    python scripts/test_ingestion.py --pdf-dir data/raw_pdfs

Optional:
    --index         Persist chunks to ChromaDB after the test run.
    --output-json   Write the report to a JSON file.
    --compare       Alias for --benchmark (compare multiple chunking configs).
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parent.parent))


from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.indexer import LegalIndexer
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalPDFProcessor


def _chunk_to_dict(chunk: Any) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "is_parent": chunk.is_parent,
        "parent_chunk_id": chunk.parent_chunk_id,
        "source_page": chunk.source_page,
        "token_count": chunk.token_count,
        "law_name": chunk.law_name,
        "law_number": chunk.law_number,
        "effective_date": chunk.effective_date,
        "part": chunk.part,
        "chapter": chunk.chapter,
        "chapter_title": chunk.chapter_title,
        "section": chunk.section,
        "article": chunk.article,
        "article_title": chunk.article_title,
        "hierarchy_path": chunk.hierarchy_path,
        "text": chunk.text,
    }


def _stats_from_chunks(chunks: list[dict], chunk_size: int, min_chunk_size: int) -> dict:
    token_counts = [int(chunk.get("token_count", 0)) for chunk in chunks]
    total = len(chunks)
    parents = sum(1 for chunk in chunks if bool(chunk.get("is_parent", False)))
    children = total - parents
    small_count = sum(1 for count in token_counts if count < min_chunk_size)
    large_count = sum(1 for count in token_counts if count > chunk_size)

    if token_counts:
        avg_tokens = round(statistics.mean(token_counts), 2)
        median_tokens = round(statistics.median(token_counts), 2)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
    else:
        avg_tokens = 0
        median_tokens = 0
        min_tokens = 0
        max_tokens = 0

    utilization = round((avg_tokens / chunk_size) if chunk_size else 0, 4)
    small_rate = round((small_count / total) if total else 0, 4)
    large_rate = round((large_count / total) if total else 0, 4)
    parent_rate = round((parents / total) if total else 0, 4)

    return {
        "total": total,
        "parent": parents,
        "child": children,
        "avg_tokens": avg_tokens,
        "median_tokens": median_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "small_count": small_count,
        "large_count": large_count,
        "small_rate": small_rate,
        "large_rate": large_rate,
        "parent_rate": parent_rate,
        "utilization": utilization,
        "chunk_size": chunk_size,
        "min_chunk_size": min_chunk_size,
    }


def _child_only_stats(chunks: list[dict], chunk_size: int, min_chunk_size: int) -> dict:
    child_chunks = [chunk for chunk in chunks if not bool(chunk.get("is_parent", False))]
    return _stats_from_chunks(child_chunks, chunk_size, min_chunk_size)


def _optimization_notes(stats: dict) -> list[str]:
    notes: list[str] = []

    if stats["total"] == 0:
        return ["No chunks were produced, so the ingestion path needs review."]

    large_rate = stats["large_rate"]
    if large_rate == 0:
        notes.append("Good: no chunk exceeds chunk_size.")
    elif large_rate <= 0.05:
        notes.append("Good: a very small share of chunks exceeds chunk_size.")
    elif large_rate <= 0.15:
        notes.append("Warning: a moderate share of chunks exceeds chunk_size; consider splitting long child blocks more aggressively.")
    else:
        notes.append("Fail: too many chunks exceed chunk_size; this will hurt retrieval quality and increase embedding cost.")

    if 0.45 <= stats["utilization"] <= 0.9:
        notes.append("Good: average token utilization is in a healthy range.")
    else:
        notes.append("Warning: average token utilization is outside the preferred range (0.45-0.90).")

    if stats["small_rate"] <= 0.2:
        notes.append("Good: small-chunk share is under control.")
    else:
        notes.append("Warning: too many small chunks; consider increasing min_chunk_size or merging more aggressively.")

    if stats["parent"] > 0 and stats["child"] > 0:
        notes.append("Good: parent-child split is active, which usually improves retrieval quality.")
    elif stats["parent"] == 0:
        notes.append("Note: no parent chunks were created; the document may be too short or chunking may be too strict.")
    else:
        notes.append("Note: only parent chunks were created; this is unusual and worth checking.")

    if large_rate > 0:
        notes.append(
            "Cost-saving fix: cap long chunks earlier by splitting oversized clauses/articles into smaller child parts, "
            "then keep only one compact parent chunk for retrieval context."
        )

    return notes


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _threshold_status(summary: dict) -> str:
    """
    Đánh giá nhanh dựa trên ngưỡng người dùng đưa ra.

    Chỉ chấm trên child chunks vì đây là phần thực sự được embed.
    """
    utilization = float(summary.get("utilization", 0.0))
    small_rate = float(summary.get("small_rate", 0.0))
    large_rate = float(summary.get("large_rate", 0.0))
    parent_rate = float(summary.get("parent_rate", 0.0))

    score = 0
    score += 1 if 0.55 <= utilization <= 0.85 else 0
    score += 1 if small_rate < 0.10 else 0
    score += 1 if large_rate <= 0.05 else 0
    score += 1 if 0.10 <= parent_rate <= 0.25 else 0

    if score == 4:
        return "PASS"
    if score >= 2:
        return "WARN"
    return "FAIL"


def _run_benchmark(
    pdf_path: Path,
    processor: LegalPDFProcessor,
    configs: list[dict],
    law_name: str,
    law_number: str,
    effective_date: str,
) -> list[dict]:
    """Chạy cùng một PDF qua nhiều cấu hình để so sánh ngưỡng chunk."""
    results: list[dict] = []
    for cfg in configs:
        chunker = LegalChunker(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            min_chunk_size=cfg["min_chunk_size"],
        )

        t0 = time.perf_counter()
        blocks = processor.process(
            pdf_path=str(pdf_path),
            law_name=law_name,
            law_number=law_number,
            effective_date=effective_date,
        )
        process_s = round(time.perf_counter() - t0, 4)

        t1 = time.perf_counter()
        chunks = chunker.chunk(blocks)
        chunk_s = round(time.perf_counter() - t1, 4)

        chunks_dict = [_chunk_to_dict(chunk) for chunk in chunks]
        summary_all = _stats_from_chunks(chunks_dict, chunker.chunk_size, chunker.min_chunk_size)
        summary_child = _child_only_stats(chunks_dict, chunker.chunk_size, chunker.min_chunk_size)

        results.append({
            "config": cfg,
            "process_s": process_s,
            "chunk_s": chunk_s,
            "summary_all": summary_all,
            "summary": summary_child,
            "status": _threshold_status(summary_child),
        })

    return results


def _print_benchmark(results: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)
    for item in results:
        cfg = item["config"]
        summary = item["summary"]
        summary_all = item["summary_all"]
        print(
            f"{item['status']:<4} | "
            f"chunk_size={cfg['chunk_size']:<4} min_chunk_size={cfg['min_chunk_size']:<4} overlap={cfg['chunk_overlap']:<3} | "
            f"child_large_rate={summary.get('large_rate', 0):<6} "
            f"child_util={summary.get('utilization', 0):<6} "
            f"child_small_rate={summary.get('small_rate', 0):<6} "
            f"parent_rate={summary_all.get('parent_rate', 0):<6} | "
            f"chunks={summary_all.get('total', 0):<4} parents={summary_all.get('parent', 0):<3} children={summary_all.get('child', 0):<3}"
        )


def _default_benchmark_configs() -> list[dict]:
    """Các cấu hình bám sát ngưỡng người dùng đưa ra."""
    return [
        {"chunk_size": 512, "min_chunk_size": 150, "chunk_overlap": 50},
        {"chunk_size": 544, "min_chunk_size": 150, "chunk_overlap": 50},
        {"chunk_size": 576, "min_chunk_size": 150, "chunk_overlap": 50},
        {"chunk_size": 640, "min_chunk_size": 150, "chunk_overlap": 50},
    ]


def _run_single_pdf(
    pdf_path: Path,
    processor: LegalPDFProcessor,
    chunker: LegalChunker,
    embedder: EmbeddingGenerator | None,
    indexer: LegalIndexer | None,
    law_name: str,
    law_number: str,
    effective_date: str,
    do_index: bool,
    do_embed: bool,
) -> dict:
    report: dict[str, Any] = {"file": str(pdf_path)}

    started = time.perf_counter()

    t0 = time.perf_counter()
    blocks = processor.process(
        pdf_path=str(pdf_path),
        law_name=law_name,
        law_number=law_number,
        effective_date=effective_date,
    )
    report["process_s"] = round(time.perf_counter() - t0, 4)
    report["blocks"] = len(blocks)

    if not blocks:
        report["chunks"] = 0
        report["children"] = 0
        report["parents"] = 0
        report["chunk_s"] = 0.0
        report["embed_s"] = 0.0
        report["index_s"] = 0.0
        report["total_s"] = round(time.perf_counter() - started, 4)
        report["summary"] = _stats_from_chunks([], chunker.chunk_size, chunker.min_chunk_size)
        report["notes"] = ["No blocks extracted from the PDF."]
        return report

    t1 = time.perf_counter()
    chunks = chunker.chunk(blocks)
    report["chunk_s"] = round(time.perf_counter() - t1, 4)
    report["chunks"] = len(chunks)

    chunks_dict = [_chunk_to_dict(chunk) for chunk in chunks]
    summary_all = _stats_from_chunks(chunks_dict, chunker.chunk_size, chunker.min_chunk_size)
    summary_child = _child_only_stats(chunks_dict, chunker.chunk_size, chunker.min_chunk_size)
    report["summary_all"] = summary_all
    report["summary"] = summary_child
    report["notes"] = _optimization_notes(summary_child)

    if do_embed:
        if embedder is None:
            raise RuntimeError("Embedder is not initialized.")
        t2 = time.perf_counter()
        vectors = embedder.embed_chunks_batched(chunks)
        report["embed_s"] = round(time.perf_counter() - t2, 4)
        report["vectors"] = len(vectors)
    else:
        vectors = []
        report["embed_s"] = 0.0
        report["vectors"] = 0
        report["notes"].append("Embedding skipped: no VOYAGE_API_KEY or --skip-embed was used.")

    report["children"] = summary_all["child"]
    report["parents"] = summary_all["parent"]

    if do_index:
        if indexer is None:
            raise RuntimeError("Indexer is not initialized.")
        if not do_embed:
            raise RuntimeError("Cannot index without embeddings. Run with embedding enabled or omit --index.")
        t3 = time.perf_counter()
        indexer.index_chunks(chunks, vectors)
        report["index_s"] = round(time.perf_counter() - t3, 4)
    else:
        report["index_s"] = 0.0

    report["total_s"] = round(time.perf_counter() - started, 4)
    return report


def _print_report(report: dict) -> None:
    print("\n" + "=" * 88)
    print(f"FILE: {report['file']}")
    print("=" * 88)

    print(f"Blocks: {report.get('blocks', 0)}")
    print(f"Chunks: {report.get('chunks', 0)} | Parents: {report.get('parents', 0)} | Children: {report.get('children', 0)}")

    summary = report.get("summary", {})
    summary_all = report.get("summary_all", summary)
    print(
        "Token stats: "
        f"avg={summary.get('avg_tokens', 0)} | "
        f"median={summary.get('median_tokens', 0)} | "
        f"min={summary.get('min_tokens', 0)} | "
        f"max={summary.get('max_tokens', 0)}"
    )
    print(
        "All chunks: "
        f"parents={summary_all.get('parent', 0)} | "
        f"children={summary_all.get('child', 0)} | "
        f"large_rate_all={summary_all.get('large_rate', 0)}"
    )
    print(
        "Rates: "
        f"utilization={summary.get('utilization', 0)} | "
        f"small_rate={summary.get('small_rate', 0)} | "
        f"large_rate={summary.get('large_rate', 0)} | "
        f"parent_rate={summary.get('parent_rate', 0)}"
    )

    print(
        "Timing: "
        f"process={_format_seconds(report.get('process_s', 0.0))} | "
        f"chunk={_format_seconds(report.get('chunk_s', 0.0))} | "
        f"embed={_format_seconds(report.get('embed_s', 0.0))} | "
        f"index={_format_seconds(report.get('index_s', 0.0))} | "
        f"total={_format_seconds(report.get('total_s', 0.0))}"
    )

    print("\nOptimization notes:")
    for note in report.get("notes", []):
        print(f"- {note}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test and inspect ingestion optimization results.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="Path to one PDF file.")
    source.add_argument("--pdf-dir", help="Path to a directory containing PDF files.")

    parser.add_argument("--law-name", default="", help="Law name metadata.")
    parser.add_argument("--law-number", default="", help="Law number metadata.")
    parser.add_argument("--effective-date", default="", help="Effective date metadata.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Max tokens per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Token overlap for splitting long chunks.")
    parser.add_argument("--min-chunk-size", type=int, default=150, help="Minimum token threshold for keeping a chunk.")
    parser.add_argument("--persist-dir", default="./vectorstore", help="ChromaDB persist directory.")
    parser.add_argument("--collection-name", default="legal_documents", help="Main ChromaDB collection name.")
    parser.add_argument("--index", action="store_true", help="Persist results to ChromaDB after the test run.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding and only test extract/chunk metrics.")
    parser.add_argument("--benchmark", action="store_true", help="Run multiple chunking configs and compare them.")
    parser.add_argument("--compare", action="store_true", help="Alias for --benchmark.")
    parser.add_argument("--output-json", default="", help="Write the final report to a JSON file.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap must be >= 0")
    if args.min_chunk_size < 0:
        parser.error("--min-chunk-size must be >= 0")

    processor = LegalPDFProcessor()

    if args.benchmark or args.compare:
        pdf_path = Path(args.pdf) if args.pdf else None
        if pdf_path is None:
            parser.error("--benchmark/--compare currently requires --pdf, not --pdf-dir.")
        if not pdf_path.exists():
            parser.error(f"PDF not found: {pdf_path}")

        benchmark_results = _run_benchmark(
            pdf_path=pdf_path,
            processor=processor,
            configs=_default_benchmark_configs(),
            law_name=args.law_name,
            law_number=args.law_number,
            effective_date=args.effective_date,
        )
        _print_benchmark(benchmark_results)
        return

    chunker = LegalChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
    )

    do_embed = not args.skip_embed and bool(os.getenv("VOYAGE_API_KEY", ""))
    embedder = EmbeddingGenerator() if do_embed else None
    indexer = None
    if args.index:
        if not do_embed:
            parser.error("--index requires embeddings. Set VOYAGE_API_KEY or remove --index.")
        indexer = LegalIndexer(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_dim=embedder.dimension,
        )

    if not do_embed:
        print("[test_ingestion] Embedding disabled: running extract/chunk test only.")

    reports: list[dict] = []

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            parser.error(f"PDF not found: {pdf_path}")

        report = _run_single_pdf(
            pdf_path=pdf_path,
            processor=processor,
            chunker=chunker,
            embedder=embedder,
            indexer=indexer,
            law_name=args.law_name,
            law_number=args.law_number,
            effective_date=args.effective_date,
            do_index=args.index,
            do_embed=do_embed,
        )
        reports.append(report)
        _print_report(report)

    else:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            parser.error(f"Not a directory: {pdf_dir}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {pdf_dir}")
            return

        print(f"Found {len(pdf_files)} PDF file(s) in {pdf_dir}")
        for pdf_path in pdf_files:
            report = _run_single_pdf(
                pdf_path=pdf_path,
                processor=processor,
                chunker=chunker,
                embedder=embedder,
                indexer=indexer,
                law_name=args.law_name,
                law_number=args.law_number,
                effective_date=args.effective_date,
                do_index=args.index,
                do_embed=do_embed,
            )
            reports.append(report)
            _print_report(report)

        aggregate = {
            "files": len(reports),
            "blocks": sum(int(item.get("blocks", 0)) for item in reports),
            "chunks": sum(int(item.get("chunks", 0)) for item in reports),
            "parents": sum(int(item.get("parents", 0)) for item in reports),
            "children": sum(int(item.get("children", 0)) for item in reports),
            "process_s": round(sum(float(item.get("process_s", 0.0)) for item in reports), 4),
            "chunk_s": round(sum(float(item.get("chunk_s", 0.0)) for item in reports), 4),
            "embed_s": round(sum(float(item.get("embed_s", 0.0)) for item in reports), 4),
            "index_s": round(sum(float(item.get("index_s", 0.0)) for item in reports), 4),
            "total_s": round(sum(float(item.get("total_s", 0.0)) for item in reports), 4),
        }
        print("\n" + "=" * 88)
        print("AGGREGATE SUMMARY")
        print("=" * 88)
        print(
            f"Files={aggregate['files']} | Blocks={aggregate['blocks']} | "
            f"Chunks={aggregate['chunks']} | Parents={aggregate['parents']} | Children={aggregate['children']}"
        )
        print(
            f"Timing total: process={_format_seconds(aggregate['process_s'])} | "
            f"chunk={_format_seconds(aggregate['chunk_s'])} | "
            f"embed={_format_seconds(aggregate['embed_s'])} | "
            f"index={_format_seconds(aggregate['index_s'])} | "
            f"sum={_format_seconds(aggregate['total_s'])}"
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reports": reports,
        }
        if len(reports) > 1:
            payload["aggregate"] = {
                "files": len(reports),
                "blocks": sum(int(item.get("blocks", 0)) for item in reports),
                "chunks": sum(int(item.get("chunks", 0)) for item in reports),
                "parents": sum(int(item.get("parents", 0)) for item in reports),
                "children": sum(int(item.get("children", 0)) for item in reports),
            }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nJSON written to: {output_path}")


if __name__ == "__main__":
    main()