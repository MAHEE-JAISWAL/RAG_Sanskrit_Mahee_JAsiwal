
# Sanskrit RAG System

A fully CPU-based Retrieval-Augmented Generation (RAG) system for querying Sanskrit documents.


## How It Works — Step by Step

### Step 1: Document Loading
We load Sanskrit text from `sanskrit_corpus.txt`. This is the raw knowledge base the system will query against. Plain `.txt` format was chosen for simplicity and compatibility.

### Step 2: Chunking
The loaded text is split into **300-word overlapping chunks** with a **50-word overlap** between consecutive chunks.
- Why 300 words? Large enough to preserve context, small enough for accurate retrieval.
- Why overlap? Prevents losing meaning at chunk boundaries.

### Step 3: Embedding
Each chunk is converted into a vector using:
> **`paraphrase-multilingual-MiniLM-L12-v2`** (~470MB)

- Supports **Devanagari script** natively — critical for Sanskrit text.
- Lightweight enough to run on CPU without GPU.
- Multilingual capability handles transliterated queries too (e.g., `kAlIdAsa bhoja raja`).

### Step 4: Indexing (FAISS)
All chunk embeddings are stored in a **FAISS `IndexFlatIP`** vector index.
- Why FAISS? Sub-millisecond retrieval even on CPU.
- Index is saved to disk after first run — no re-embedding needed on restart.
- `IndexFlatIP` uses **cosine similarity** (inner product on normalized vectors).

### Step 5: Retrieval
On receiving a query, it is embedded using the same model, then the **top-k=3** most similar chunks are retrieved from the FAISS index.
- Why top-3? Enough context without overwhelming the generator with noise.

### Step 6: Generation (LLM)
Retrieved chunks are passed as context to:
> **`google/flan-t5-base`** (~990MB)

- Why Flan-T5? Instruction-tuned, CPU-friendly, and strong at question-answering tasks.
- Runs with `num_beams=2` for a balance between speed and answer quality.
- No GPU required — works on any standard machine.

---

## Models Used

| Component | Model | Size | Why |
|-----------|-------|------|-----|
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | ~470MB | Devanagari support, lightweight, multilingual |
| Generator | `google/flan-t5-base` | ~990MB | CPU-friendly, instruction-tuned, good at QA |
| Index | FAISS `IndexFlatIP` | — | Fast cosine search, disk-cacheable |

---

## Project Structure

```
RAG_Sanskrit/
├── code/
│   ├── rag_pipeline.py   # Core RAG pipeline
│   ├── evaluate.py       # Evaluation script
│   ├── app.py            # Streamlit web UI
│   └── requirements.txt
├── data/
│   ├── sanskrit_corpus.txt
│   └── index/            # Auto-generated after first run
├── report/
│   └── technical_report.pdf
└── README.md
```

---

## Setup & Usage

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run interactive mode
cd code
python rag_pipeline.py --interactive

# 4. Single query
python rag_pipeline.py --query "Who is Shankhanaad?"

# 5. Force re-ingest
python rag_pipeline.py --ingest --interactive

# 6. Web UI
streamlit run code/app.py
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `../data` | Path to data directory |
| `--llm` | `google/flan-t5-base` | HuggingFace model ID |
| `--top-k` | `3` | Retrieved chunks per query |
| `chunk_size` | `300` | Words per chunk |
| `overlap` | `50` | Overlap between chunks |

---

## CPU Optimization Notes

- All models run explicitly on `device="cpu"`
- Embeddings cached to disk after first ingestion
- `num_beams=2` keeps generation fast without sacrificing quality
- FAISS provides near-instant retrieval even without a GPU
```