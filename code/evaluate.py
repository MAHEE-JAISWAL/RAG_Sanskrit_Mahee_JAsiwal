"""
evaluate.py — Evaluation script for the Sanskrit RAG System.

Measures: retrieval similarity scores, answer latency, memory usage.

  cd <project_root>/code
  python evaluate.py
"""

import json
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import SanskritRAGSystem
from config import DATA_DIR


TEST_QUERIES = [
    # English
    "Who is Shankhanaad and what did he do?",
    "What is the story of King Bhoj and the poet Kalidasa?",
    "What happened with the bell in the forest?",
    "What is the moral of the story about the devotee?",
    # Sanskrit (Devanagari)
    "शंखनादः कः अस्ति?",
    "कालिदासः कः अस्ति?",
    "घण्टाकर्णः किम् अभवत्?",
    # Transliteration
    "Shankhanaada sharkara",
    "Bhoja raja kalidasa",
]


def evaluate() -> dict:
    print("=" * 60)
    print("  Sanskrit RAG — Evaluation")
    print("=" * 60)

    rag = SanskritRAGSystem()
    rag.ingest()

    process   = psutil.Process(os.getpid())
    per_query = []
    latencies = []

    for q in TEST_QUERIES:
        mem_before = process.memory_info().rss / 1024 / 1024
        result     = rag.query(q)
        mem_after  = process.memory_info().rss / 1024 / 1024

        top_score = result["retrieval_scores"][0] if result["retrieval_scores"] else 0.0
        latencies.append(result["latency_seconds"])

        entry = {
            "query":               q,
            "answer_preview":      result["answer"][:200],
            "top_retrieval_score": round(top_score, 4),
            "latency_s":           result["latency_seconds"],
            "memory_delta_mb":     round(mem_after - mem_before, 2),
        }
        per_query.append(entry)

        print(f"\nQ : {q}")
        print(f"  Answer : {result['answer'][:150]} …")
        print(f"  Score  : {top_score:.4f}  |  Latency: {result['latency_seconds']}s")

    avg_lat   = sum(latencies) / len(latencies)
    avg_score = sum(e["top_retrieval_score"] for e in per_query) / len(per_query)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Queries tested        : {len(TEST_QUERIES)}")
    print(f"  Avg latency           : {avg_lat:.3f}s")
    print(f"  Min / Max latency     : {min(latencies):.3f}s / {max(latencies):.3f}s")
    print(f"  Avg top retrieval sim : {avg_score:.4f}")

    output = {
        "summary": {
            "num_queries":              len(TEST_QUERIES),
            "avg_latency_s":            round(avg_lat, 3),
            "min_latency_s":            round(min(latencies), 3),
            "max_latency_s":            round(max(latencies), 3),
            "avg_top_retrieval_score":  round(avg_score, 4),
        },
        "per_query": per_query,
    }

    out_path = DATA_DIR / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")

    return output


if __name__ == "__main__":
    evaluate()