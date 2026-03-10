"""
ingest.py — Standalone ingestion script.

Run ONCE before using the RAG pipeline to build the vector index.

  cd <project_root>/code
  python ingest.py

Options:
  --force   Rebuild index even if one already exists.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run(force: bool = False) -> None:
    from document_loader import SanskritDocumentLoader
    from retriever import SanskritRetriever

    retriever = SanskritRetriever()

    if retriever.is_index_built() and not force:
        print("[Ingest] Index already exists. Use --force to rebuild.")
        return

    print("=" * 60)
    print("  Sanskrit RAG — Ingestion Pipeline")
    print("=" * 60)

    t0 = time.time()

    # 1. Load and chunk corpus
    loader = SanskritDocumentLoader()
    chunks = loader.load()

    lengths = [len(c["text"]) for c in chunks]
    print(f"\nChunk statistics:")
    print(f"  Total : {len(chunks)}")
    print(f"  Avg   : {sum(lengths) // max(len(lengths), 1)} chars")
    print(f"  Range : {min(lengths)} – {max(lengths)} chars")

    # 2. Build FAISS/numpy index
    retriever.build_index(chunks)

    # 3. Smoke test
    print("\nSmoke test — 'Kalidasa clever poet':")
    for r in retriever.retrieve("Kalidasa clever poet", top_k=3):
        print(f"  score={r['score']:.3f} | {r['text'][:80]} …")

    print(f"\n✓ Ingestion complete in {time.time() - t0:.1f}s")
    print("  Run:  python rag_pipeline.py --interactive")
    print("  Run:  streamlit run app.py")


# ── IMPORTANT: nothing runs on import — only when called directly ─────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanskrit RAG — Ingestion")
    parser.add_argument("--force", action="store_true", help="Force index rebuild")
    args = parser.parse_args()
    run(force=args.force)