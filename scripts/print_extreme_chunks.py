#!/usr/bin/env python
"""
In đầy đủ metadata và nội dung của chunk ngắn nhất + dài nhất từ file JSON.

Usage:
    python scripts/print_extreme_chunks.py --json data/evaluation/demo_chunks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _print_separator(title: str) -> None:
    line = "=" * 120
    print(f"\n{line}")
    print(title)
    print(line)


def _print_full_chunk(title: str, chunk: dict[str, Any]) -> None:
    _print_separator(title)

    print("METADATA (FULL):")
    for key in sorted(chunk.keys()):
        if key == "text":
            continue
        print(f"- {key}: {chunk.get(key)}")

    print("\nTEXT (FULL):")
    print(chunk.get("text", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="In chunk dài nhất và ngắn nhất (full metadata + full text)."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="data/evaluation/demo_chunks.json",
        help="Đường dẫn tới file JSON chứa trường chunks",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("File JSON không có danh sách chunks hợp lệ.")

    chunks_with_tokens = [
        c for c in chunks
        if isinstance(c, dict) and c.get("token_count") is not None
    ]
    if not chunks_with_tokens:
        raise ValueError("Không có chunk nào chứa token_count.")

    max_chunk = max(chunks_with_tokens, key=lambda c: _to_int(c.get("token_count")))
    min_chunk = min(chunks_with_tokens, key=lambda c: _to_int(c.get("token_count")))

    _print_full_chunk("CHUNK DAI NHAT (MAX TOKEN)", max_chunk)
    _print_full_chunk("CHUNK NGAN NHAT (MIN TOKEN)", min_chunk)


if __name__ == "__main__":
    main()
