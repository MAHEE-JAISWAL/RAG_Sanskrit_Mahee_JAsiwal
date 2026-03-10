"""
config.py — Central configuration for Sanskrit RAG System.
"""

from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
CODE_DIR  = Path(__file__).resolve().parent
BASE_DIR  = CODE_DIR.parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = CODE_DIR / "index_store"
MODEL_DIR = CODE_DIR / "models"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Document Settings ─────────────────────────────────────────────────────────
CORPUS_FILE   = DATA_DIR / "sanskrit_corpus.txt"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── FAISS Index ───────────────────────────────────────────────────────────────
FAISS_INDEX_FILE = str(INDEX_DIR / "sanskrit_faiss.index")
CHUNKS_FILE      = str(INDEX_DIR / "chunks.json")
TOP_K            = 4

# ── LLM — TinyLlama GGUF (CPU) ───────────────────────────────────────────────
LLM_MODEL_NAME    = "TinyLlama-1.1B-Chat"
LLM_MODEL_PATH    = str(MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
LLM_MODEL_URL     = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    "/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
LLM_MAX_NEW_TOKENS = 300
LLM_TEMPERATURE    = 0.2   # low = more factual, less hallucination
LLM_CONTEXT_LENGTH = 2048

# ── flan-t5 fallback (unused but kept for config compatibility) ───────────────
LLM_MAX_TOKENS = 200
LLM_NUM_BEAMS  = 4

# ── Prompt (used by flan-t5 fallback only) ────────────────────────────────────
RAG_PROMPT_TEMPLATE = (
    "You are a Sanskrit literature expert. "
    "Answer using ONLY the context below. Be specific.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)