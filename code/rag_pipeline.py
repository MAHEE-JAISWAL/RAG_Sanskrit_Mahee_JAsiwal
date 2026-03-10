"""
rag_pipeline.py — Orchestrates the full Sanskrit RAG pipeline.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config          import TOP_K, LLM_MODEL_NAME
from document_loader import SanskritDocumentLoader
from retriever       import SanskritRetriever
from generator       import SanskritGenerator


class SanskritRAGSystem:

    def __init__(
        self,
        data_dir:  str = None,
        llm_model: str = LLM_MODEL_NAME,   # kept for API compatibility but unused
        top_k:     int = TOP_K,
    ):
        from config import CORPUS_FILE

        corpus_path = (
            str(Path(data_dir) / "sanskrit_corpus.txt")
            if data_dir
            else str(CORPUS_FILE)
        )

        self.top_k     = top_k
        self.loader    = SanskritDocumentLoader(corpus_path=corpus_path)
        self.retriever = SanskritRetriever(top_k=top_k)
        self.generator = SanskritGenerator()   # no model_name arg needed anymore
        self._ready    = False

    def ingest(self, force_rebuild: bool = False) -> None:
        if self.retriever.is_index_built() and not force_rebuild:
            print("[RAG] Existing index found — loading …")
            self.retriever.load_index()
        else:
            print("[RAG] Building index from corpus …")
            chunks = self.loader.load()
            self.retriever.build_index(chunks)

        print("[RAG] Loading LLM generator …")
        self.generator.load()
        self._ready = True
        print("[RAG] System ready ✓")

    def query(self, user_query: str) -> Dict:
        if not self._ready:
            raise RuntimeError("System not ready. Call ingest() first.")

        t0      = time.time()
        results = self.retriever.retrieve(user_query, top_k=self.top_k)
        answer  = self.generator.generate(user_query, results)

        return {
            "query":            user_query,
            "answer":           answer,
            "retrieved_chunks": results,
            "retrieval_scores": [r["score"] for r in results],
            "latency_seconds":  round(time.time() - t0, 3),
        }

    def interactive(self) -> None:
        print("\n" + "=" * 60)
        print("  Sanskrit RAG System — Interactive Mode")
        print("  Type 'quit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                q = input("Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            r   = self.query(q)
            sep = "─" * 60
            print(f"\n{sep}")
            print(f"Answer:\n{r['answer']}")
            print(f"\nLatency: {r['latency_seconds']}s")
            print(f"{sep}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanskrit RAG System")
    parser.add_argument("--ingest",      action="store_true")
    parser.add_argument("--query",       type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--data-dir",    type=str, default=None)
    parser.add_argument("--top-k",       type=int, default=TOP_K)
    args = parser.parse_args()

    rag = SanskritRAGSystem(data_dir=args.data_dir, top_k=args.top_k)
    rag.ingest(force_rebuild=args.ingest)

    if args.query:
        r = rag.query(args.query)
        print(f"\nQuery  : {r['query']}")
        print(f"Answer : {r['answer']}")
        print(f"Latency: {r['latency_seconds']}s")
    else:
        rag.interactive()