"""
retriever.py — Embed chunks and query via FAISS (CPU-only).

Key fix: English queries are translated to Hindi/Sanskrit equivalents
before embedding so they match Devanagari corpus chunks properly.
"""

import json
import os
import re
import time
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import EMBEDDING_MODEL, FAISS_INDEX_FILE, CHUNKS_FILE, TOP_K


# ── English → Devanagari query translation map ────────────────────────────────
# Translates common English query terms to Hindi/Sanskrit equivalents
# so they match the Devanagari corpus chunks via the multilingual embedder.

QUERY_TRANSLATION = {
    # Characters
    "shankhanaad":   "शंखनाद",
    "shankhanaada":  "शंखनाद",
    "shankanaad":    "शंखनाद",
    "govardhanaadas": "गोवर्धनदास",
    "kalidasa":      "कालीदास",
    "kalidas":       "कालीदास",
    "king bhoj":     "भोजराजा",
    "bhoj":          "भोजराजा",
    "ghantakarna":   "घण्टाकर्ण",
    "ghantkarna":    "घण्टाकर्ण",
    "devotee":       "भक्त",
    "old woman":     "वृद्धा",
    "scholar":       "पण्डित",
    "poet":          "कवि",
    "servant":       "भृत्य",
    "monkey":        "वानर",
    "monkeys":       "वानराः",
    "god":           "देव",

    # Story topics
    "sugar":         "शर्करा",
    "sharkara":      "शर्करा",
    "bell":          "घण्टा",
    "milk":          "दुग्ध",
    "puppy":         "श्वानशावक",
    "dog":           "श्वान",
    "bag":           "सन्चिका",
    "cloth":         "वस्त्र",
    "torn":          "जीर्ण",
    "cold":          "शीत",
    "palanquin":     "पालखी",
    "prayer":        "प्रार्थना",
    "cart":          "शकट",
    "lakh":          "लक्ष",
    "rupees":        "रुप्यक",
    "poem":          "काव्य",
    "court":         "दरबार",
    "forest":        "वन",
    "demon":         "राक्षस",
    "reward":        "सुवर्ण",
    "gold":          "सुवर्ण",
    "moral":         "नीति",
    "heaven":        "स्वर्ग",
    "grammar":       "व्याकरण",
    "disguise":      "रूप",
    "clever":        "चतुर",
    "foolish":       "मूर्ख",
    "effort":        "प्रयत्न",
    "help":          "साहाय्य",
    "afraid":        "भय",
    "city":          "नगर",
    "face":          "मुख",
    "black":         "कृष्ण",
    "rope":          "दोरक",
    "spilled":       "स्त्रवति",
    "suffocated":    "श्वास रुध्द",
    "king":          "राजा",
    "story":         "कथा",
}


def translate_query_to_devanagari(query: str) -> str:
    """
    Replace English words in query with Devanagari equivalents.
    Keeps original structure, just swaps known terms.
    E.g. "Who is Shankhanaad?" → "Who is शंखनाद?"
    """
    result = query
    # Sort by length desc so longer phrases match before substrings
    for eng, dev in sorted(QUERY_TRANSLATION.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(eng), re.IGNORECASE)
        result  = pattern.sub(dev, result)
    return result


class SanskritRetriever:

    def __init__(
        self,
        model_name:  str = EMBEDDING_MODEL,
        index_file:  str = FAISS_INDEX_FILE,
        chunks_file: str = CHUNKS_FILE,
        top_k:       int = TOP_K,
    ):
        self.model_name  = model_name
        self.index_file  = index_file
        self.chunks_file = chunks_file
        self.top_k       = top_k

        self._model      = None
        self._index      = None
        self._embeddings = None
        self._chunks: List[Dict] = []
        self._use_faiss  = False

    def build_index(self, chunks: List[Dict]) -> None:
        self._chunks = chunks
        model  = self._get_model()
        texts  = [c["text"] for c in chunks]

        print(f"[Retriever] Embedding {len(chunks)} chunk(s) …")
        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"[Retriever] Embedding done in {time.time()-t0:.1f}s — shape {embeddings.shape}")
        self._embeddings = embeddings.astype(np.float32)
        self._try_build_faiss()
        self._save()

    def load_index(self) -> None:
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(
                f"Chunks not found at '{self.chunks_file}'. Run ingest.py first."
            )
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

        emb_file = self.index_file + ".npy"
        if os.path.exists(emb_file):
            self._embeddings = np.load(emb_file).astype(np.float32)

        try:
            import faiss
            if os.path.exists(self.index_file):
                self._index     = faiss.read_index(self.index_file)
                self._use_faiss = True
                print(f"[Retriever] FAISS index loaded ({self._index.ntotal} vectors).")
            elif self._embeddings is not None:
                self._try_build_faiss()
        except ImportError:
            self._use_faiss = False
            print(f"[Retriever] Loaded {len(self._chunks)} chunks (numpy mode).")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        if not self._chunks:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        k = top_k or self.top_k

        # Translate English terms to Devanagari before embedding
        translated = translate_query_to_devanagari(query)
        if translated != query:
            print(f"[Retriever] Query translated → '{translated}'")

        model = self._get_model()
        q_emb = model.encode(
            [translated],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(q_emb, k)
            pairs = list(zip(scores[0], indices[0]))
        else:
            sims  = (self._embeddings @ q_emb[0]).flatten()
            top_i = np.argsort(sims)[::-1][:k]
            pairs = [(float(sims[i]), int(i)) for i in top_i]

        results = []
        for score, idx in pairs:
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk          = dict(self._chunks[int(idx)])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def is_index_built(self) -> bool:
        return os.path.exists(self.chunks_file)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Retriever] Loading embedding model '{self.model_name}' …")
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _try_build_faiss(self):
        try:
            import faiss
            dim          = self._embeddings.shape[1]
            self._index  = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings)
            self._use_faiss = True
            print(f"[Retriever] FAISS index built ({self._index.ntotal} vectors).")
        except ImportError:
            self._use_faiss = False
            print("[Retriever] FAISS not available — using numpy cosine similarity.")

    def _save(self):
        Path(self.chunks_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.index_file + ".npy", self._embeddings)
        if self._use_faiss:
            import faiss
            faiss.write_index(self._index, self.index_file)
            print(f"[Retriever] FAISS index saved → {self.index_file}")
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)
        print(f"[Retriever] Chunks saved → {self.chunks_file}")