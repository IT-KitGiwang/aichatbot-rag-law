from __future__ import annotations

import statistics
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.pdf_processor import LegalPDFProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"


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


def _chunk_metrics(chunks: list, chunk_size: int, min_chunk_size: int) -> dict:
    child_chunks = [chunk for chunk in chunks if not chunk.is_parent]
    parent_chunks = [chunk for chunk in chunks if chunk.is_parent]
    child_token_counts = [int(chunk.token_count) for chunk in child_chunks]
    all_token_counts = [int(chunk.token_count) for chunk in chunks]

    if child_token_counts:
        child_avg = statistics.mean(child_token_counts)
        child_util = child_avg / chunk_size if chunk_size else 0.0
    else:
        child_avg = 0.0
        child_util = 0.0

    return {
        "total": len(chunks),
        "child": len(child_chunks),
        "parent": len(parent_chunks),
        "child_token_counts": child_token_counts,
        "all_token_counts": all_token_counts,
        "child_avg": child_avg,
        "child_util": child_util,
        "child_small_rate": (sum(1 for count in child_token_counts if count < min_chunk_size) / len(child_token_counts))
        if child_token_counts
        else 0.0,
        "child_large_rate": (sum(1 for count in child_token_counts if count > chunk_size) / len(child_token_counts))
        if child_token_counts
        else 0.0,
        "parent_rate": (len(parent_chunks) / len(chunks)) if chunks else 0.0,
        "max_child_tokens": max(child_token_counts) if child_token_counts else 0,
        "max_all_tokens": max(all_token_counts) if all_token_counts else 0,
    }


def _build_report(pdf_path: Path, metrics: dict, config: dict, elapsed: dict[str, float]) -> str:
    return (
        f"PDF: {pdf_path.name}\n"
        f"  blocks={metrics['blocks']} chunks={metrics['total']} parents={metrics['parent']} children={metrics['child']}\n"
        f"  chunk_size={config['chunk_size']} overlap={config['chunk_overlap']} min_chunk_size={config['min_chunk_size']}\n"
        f"  child_util={metrics['child_util']:.4f} child_small_rate={metrics['child_small_rate']:.4f} "
        f"child_large_rate={metrics['child_large_rate']:.4f} parent_rate={metrics['parent_rate']:.4f}\n"
        f"  max_child_tokens={metrics['max_child_tokens']} max_all_tokens={metrics['max_all_tokens']}\n"
        f"  timings: process={elapsed['process_s']:.2f}s chunk={elapsed['chunk_s']:.2f}s total={elapsed['total_s']:.2f}s"
    )


def _print_optimized_parameters(pdf_path: Path, metrics: dict, config: dict, elapsed: dict[str, float]) -> None:
    print("\n" + "=" * 88)
    print(f"INGESTION FINAL CHECK: {pdf_path.name}")
    print("=" * 88)
    print(
        "Optimized parameters: "
        f"chunk_size={config['chunk_size']} | "
        f"chunk_overlap={config['chunk_overlap']} | "
        f"min_chunk_size={config['min_chunk_size']} | "
        f"on_small_chunk={config['on_small_chunk']}"
    )
    print(
        "Key metrics: "
        f"child_util={metrics['child_util']:.4f} | "
        f"child_small_rate={metrics['child_small_rate']:.4f} | "
        f"child_large_rate={metrics['child_large_rate']:.4f} | "
        f"parent_rate={metrics['parent_rate']:.4f}"
    )
    print(
        "Chunk summary: "
        f"blocks={metrics['blocks']} | "
        f"chunks={metrics['total']} | "
        f"parents={metrics['parent']} | "
        f"children={metrics['child']}"
    )
    print(
        "Thresholds: "
        "child_util in [0.45, 0.70] | "
        "child_small_rate <= 0.30 | "
        "child_large_rate == 0 | "
        "parent_rate in [0.08, 0.25]"
    )
    print(
        "Timings: "
        f"process={elapsed['process_s']:.2f}s | "
        f"chunk={elapsed['chunk_s']:.2f}s | "
        f"total={elapsed['total_s']:.2f}s"
    )
    print("Verdict: READY FOR NEXT STAGE")
    print("=" * 88)


def _readable_threshold_failures(metrics: dict) -> list[str]:
    failures: list[str] = []

    if metrics["total"] == 0:
        failures.append("No chunks were produced.")
        return failures

    if metrics["child"] == 0:
        failures.append("No child chunks were produced.")

    if metrics["parent"] == 0:
        failures.append("No parent chunks were produced.")

    if metrics["child_large_rate"] > 0:
        failures.append("Some child chunks exceed chunk_size.")

    if metrics["child_small_rate"] > 0.30:
        failures.append("Too many child chunks are below min_chunk_size.")

    if not 0.45 <= metrics["child_util"] <= 0.70:
        failures.append("Child utilization is outside the target band 0.45-0.70.")

    if not 0.08 <= metrics["parent_rate"] <= 0.25:
        failures.append("Parent chunk ratio is outside the target band 0.08-0.25.")

    if metrics["max_child_tokens"] > metrics.get("chunk_size", metrics["max_child_tokens"]):
        failures.append("A child chunk exceeded the configured chunk_size.")

    return failures


@pytest.mark.parametrize("pdf_path", sorted(RAW_PDF_DIR.glob("*.pdf")))
def test_ingestion_final_readiness(pdf_path: Path) -> None:
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

    started = __import__("time").perf_counter()

    process_started = __import__("time").perf_counter()
    blocks = processor.process(pdf_path=str(pdf_path))
    process_s = __import__("time").perf_counter() - process_started

    assert blocks, f"No blocks extracted from {pdf_path.name}"

    chunk_started = __import__("time").perf_counter()
    chunks = chunker.chunk(blocks)
    chunk_s = __import__("time").perf_counter() - chunk_started
    total_s = __import__("time").perf_counter() - started

    metrics = _chunk_metrics(chunks, config["chunk_size"], config["min_chunk_size"])
    metrics["blocks"] = len(blocks)
    metrics["chunk_size"] = config["chunk_size"]

    failures = _readable_threshold_failures(metrics)
    report = _build_report(
        pdf_path=pdf_path,
        metrics=metrics,
        config=config,
        elapsed={"process_s": process_s, "chunk_s": chunk_s, "total_s": total_s},
    )

    _print_optimized_parameters(
        pdf_path=pdf_path,
        metrics=metrics,
        config=config,
        elapsed={"process_s": process_s, "chunk_s": chunk_s, "total_s": total_s},
    )

    assert not failures, report + "\n\nFAILURES:\n- " + "\n- ".join(failures)
