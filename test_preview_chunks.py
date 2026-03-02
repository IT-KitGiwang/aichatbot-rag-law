"""Preview chunks dài nhất — nội dung đầy đủ, không cắt."""

import json
from pathlib import Path

FILE = Path("data/processed/search_chunks_full.json")
data = json.load(open(FILE, "r", encoding="utf-8"))

# Sắp xếp theo text_length giảm dần
data_sorted = sorted(data, key=lambda d: d["text_length"], reverse=True)

print(f"Tổng: {len(data)} search chunks\n")

for i, chunk in enumerate(data_sorted[:3]):
    print(f"{'='*70}")
    print(f"  CHUNK DAI THU {i+1}")
    print(f"  ID:     {chunk['chunk_id']}")
    print(f"  Type:   {chunk['chunk_type']}")
    print(f"  Parent: {chunk['parent_chunk_id']}")
    print(f"  Path:   {chunk['hierarchy_path']}")
    print(f"  Length: {chunk['text_length']} chars")
    print(f"{'='*70}")
    print()
    print(chunk["text"])
    print()
