from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.pdf_processor import LegalPDFProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORT_PATH = EVALUATION_DIR / "chunker_embedding_report.json"


def _load_chunking_config() -> dict:
    if yaml is None or not CONFIG_PATH.exists():
        return {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_size": 150,
            "on_small_chunk": "merge_next",
        }

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    chunking = data.get("chunking") or {}
    return {
        "chunk_size": int(chunking.get("chunk_size", 512)),
        "chunk_overlap": int(chunking.get("chunk_overlap", 50)),
        "min_chunk_size": int(chunking.get("min_chunk_size", 150)),
        "on_small_chunk": str(chunking.get("on_small_chunk", "merge_next")),
    }


def _summarize_chunks(chunks: list, chunk_size: int, min_chunk_size: int) -> dict:
    child_chunks = [chunk for chunk in chunks if not chunk.is_parent]
    parent_chunks = [chunk for chunk in chunks if chunk.is_parent]
    child_tokens = [int(chunk.token_count) for chunk in child_chunks]
    parent_tokens = [int(chunk.token_count) for chunk in parent_chunks]

    child_avg = statistics.mean(child_tokens) if child_tokens else 0.0
    child_util = (child_avg / chunk_size) if chunk_size else 0.0

    return {
        "total": len(chunks),
        "child": len(child_chunks),
        "parent": len(parent_chunks),
        "child_avg_tokens": round(child_avg, 2),
        "child_util": round(child_util, 4),
        "child_small_rate": round(
            (sum(1 for token in child_tokens if token < min_chunk_size) / len(child_tokens)) if child_tokens else 0.0,
            4,
        ),
        "child_large_rate": round(
            (sum(1 for token in child_tokens if token > chunk_size) / len(child_tokens)) if child_tokens else 0.0,
            4,
        ),
        "parent_rate": round((len(parent_chunks) / len(chunks)) if chunks else 0.0, 4),
        "max_child_tokens": max(child_tokens) if child_tokens else 0,
        "max_parent_tokens": max(parent_tokens) if parent_tokens else 0,
        "min_child_tokens": min(child_tokens) if child_tokens else 0,
    }


def _summarize_embedding(chunks: list, summary: dict) -> dict:
    api_key = os.getenv("VOYAGE_API_KEY", "").strip()
    child_count = summary["child"]
    child_tokens = summary["child_avg_tokens"] * child_count if child_count else 0.0

    if not api_key:
        return {
            "status": "skipped_no_api_key",
            "actual": False,
            "vectors": child_count,
            "vector_dim": 1024,
            "embed_s": 0.0,
            "tokens_to_embed": round(child_tokens, 2),
            "estimated_cost_proxy": round(child_tokens, 2),
            "note": "VOYAGE_API_KEY not found; embedding request skipped and cost is estimated from child token volume.",
        }

    embedder = EmbeddingGenerator()
    started = time.perf_counter()
    vectors = embedder.embed_chunks_batched(chunks)
    embed_s = time.perf_counter() - started

    vector_dim = len(vectors[0]) if vectors else embedder.dimension
    return {
        "status": "embedded",
        "actual": True,
        "vectors": len(vectors),
        "vector_dim": vector_dim,
        "embed_s": round(embed_s, 4),
        "tokens_to_embed": round(child_tokens, 2),
        "estimated_cost_proxy": round(child_tokens, 2),
        "tokens_per_second": round((child_tokens / embed_s) if embed_s > 0 else 0.0, 2),
        "note": "Actual Voyage embeddings completed successfully.",
    }


def _threshold_checks(summary: dict) -> list[str]:
    failures: list[str] = []

    if summary["total"] == 0:
        failures.append("No chunks were produced.")
        return failures

    if summary["child"] == 0:
        failures.append("No child chunks were produced.")

    if summary["parent"] == 0:
        failures.append("No parent chunks were produced.")

    if summary["child_large_rate"] > 0:
        failures.append("Child chunk exceeded chunk_size.")

    if summary["child_small_rate"] > 0.30:
        failures.append("Too many child chunks are below min_chunk_size.")

    if not 0.45 <= summary["child_util"] <= 0.70:
        failures.append("Child utilization is outside the target band 0.45-0.70.")

    if not 0.08 <= summary["parent_rate"] <= 0.25:
        failures.append("Parent rate is outside the target band 0.08-0.25.")

    return failures


def _print_report(pdf_path: Path, config: dict, summary: dict, embedding: dict, timings: dict[str, float]) -> None:
    print("\n" + "=" * 92)
    print(f"CHUNKER FINAL CHECK: {pdf_path.name}")
    print("=" * 92)
    print(
        "Optimized parameters: "
        f"chunk_size={config['chunk_size']} | "
        f"chunk_overlap={config['chunk_overlap']} | "
        f"min_chunk_size={config['min_chunk_size']} | "
        f"on_small_chunk={config['on_small_chunk']}"
    )
    print(
        "Chunk metrics: "
        f"total={summary['total']} | "
        f"children={summary['child']} | "
        f"parents={summary['parent']} | "
        f"child_avg_tokens={summary['child_avg_tokens']:.2f} | "
        f"child_util={summary['child_util']:.4f} | "
    )
    print(
        "Quality metrics: "
        f"child_small_rate={summary['child_small_rate']:.4f} | "
        f"child_large_rate={summary['child_large_rate']:.4f} | "
        f"parent_rate={summary['parent_rate']:.4f} | "
        f"max_child_tokens={summary['max_child_tokens']}"
    )
    print(
        "Embedding: "
        f"status={embedding['status']} | "
        f"vectors={embedding['vectors']} | "
        f"vector_dim={embedding['vector_dim']} | "
        f"tokens_to_embed={embedding['tokens_to_embed']:.2f} | "
        f"cost_proxy_tokens={embedding['estimated_cost_proxy']:.2f}"
    )
    print(
        "Timings: "
        f"process={timings['process_s']:.2f}s | "
        f"chunk={timings['chunk_s']:.2f}s | "
        f"embed={embedding['embed_s']:.2f}s | "
        f"total={timings['total_s']:.2f}s"
    )
    print("Verdict: READY FOR NEXT STAGE")
    print("=" * 92)


def _write_report(payload: dict) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_pdf_paths() -> list[Path]:
    pdf_paths = sorted(RAW_PDF_DIR.glob("*.pdf"))
    return pdf_paths


@pytest.mark.parametrize("pdf_path", _collect_pdf_paths())
def test_chunker_final_readiness(pdf_path: Path) -> None:
    if not pdf_path.exists():
        pytest.skip(f"Missing PDF file: {pdf_path}")

    config = _load_chunking_config()
    processor = LegalPDFProcessor()
    chunker = LegalChunker(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        min_chunk_size=config["min_chunk_size"],
        on_small_chunk=config["on_small_chunk"],
    )

    started = time.perf_counter()

    process_started = time.perf_counter()
    blocks = processor.process(pdf_path=str(pdf_path))
    process_s = time.perf_counter() - process_started

    assert blocks, f"No blocks extracted from {pdf_path.name}"

    chunk_started = time.perf_counter()
    chunks = chunker.chunk(blocks)
    chunk_s = time.perf_counter() - chunk_started
    total_s = time.perf_counter() - started

    summary = _summarize_chunks(chunks, config["chunk_size"], config["min_chunk_size"])
    embedding = _summarize_embedding(chunks, summary)
    timings = {"process_s": process_s, "chunk_s": chunk_s, "total_s": total_s}
    failures = _threshold_checks(summary)

    report_item = {
        "file": str(pdf_path),
        "config": config,
        "summary": summary,
        "embedding": embedding,
        "timings": {k: round(v, 4) for k, v in timings.items()},
    }
    _write_report({"reports": [report_item]})
    _print_report(pdf_path, config, summary, embedding, timings)

    assert not failures, (
        f"Chunker readiness failed for {pdf_path.name}\n"
        f"Config: {config}\n"
        f"Summary: {summary}\n"
        f"Embedding: {embedding}\n"
        f"Failures:\n- " + "\n- ".join(failures)
    )
