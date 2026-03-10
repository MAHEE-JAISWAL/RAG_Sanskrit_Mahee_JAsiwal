"""
debug_chunks.py — Inspect what's in your index and test retrieval.
Run from code/ directory: python debug_chunks.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from retriever import SanskritRetriever

retriever = SanskritRetriever()
retriever.load_index()

print("=" * 60)
print(f"Total chunks: {len(retriever._chunks)}")
print("=" * 60)

# Print all chunks with their section
for i, chunk in enumerate(retriever._chunks):
    section = chunk.get("metadata", {}).get("section", "?")
    preview = chunk["text"][:120].replace("\n", " ")
    print(f"\n[{i}] Section: {section}")
    print(f"     {preview}")

print("\n" + "=" * 60)
print("RETRIEVAL TEST")
print("=" * 60)

test_queries = [
    "Who is Shankhanaad?",
    "What did Shankhanaad do with the sugar?",
    "Ghantakarna bell monkey",
    "शंखनादः",
    "Kalidasa King Bhoj",
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = retriever.retrieve(q, top_k=2)
    for r in results:
        section = r.get("metadata", {}).get("section", "?")
        print(f"  score={r['score']:.3f} | [{section}] {r['text'][:100]}")