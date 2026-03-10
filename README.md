# Sanskrit RAG System

A fully CPU-based Retrieval-Augmented Generation (RAG) system for querying Sanskrit documents.

## Project Structure

```
RAG_Sanskrit/
├── code/
│   ├── rag_pipeline.py   # Core RAG pipeline (loader, chunker, retriever, generator)
│   ├── evaluate.py       # Evaluation script
│   ├── app.py            # Streamlit web UI
│   └── requirements.txt  # Python dependencies
├── data/
│   ├── sanskrit_corpus.txt   # Sanskrit source documents
│   └── index/                # Auto-generated vector index (after first run)
├── report/
│   └── technical_report.pdf  # Technical report
└── README.md
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              SanskritRAGSystem                  │
│                                                 │
│  ┌──────────────┐     ┌───────────────────────┐ │
│  │   Retriever  │     │      Generator        │ │
│  │              │     │                       │ │
│  │ Multilingual │     │  google/flan-t5-base  │ │
│  │  MiniLM-L12  │────▶│  (CPU inference)      │ │
│  │  + FAISS     │     │                       │ │
│  └──────────────┘     └───────────────────────┘ │
│         ▲                        │              │
│         │                        ▼              │
│  ┌──────────────┐          Final Answer         │
│  │ Vector Index │                               │
│  │ (FAISS / np) │                               │
│  └──────────────┘                               │
└─────────────────────────────────────────────────┘
         ▲
         │  Ingestion (one-time)
┌────────────────────┐
│  Document Loader   │  → Reads sanskrit_corpus.txt
│  + Chunker         │  → Splits into 300-word overlapping chunks
│  + Embedder        │  → paraphrase-multilingual-MiniLM-L12-v2
└────────────────────┘
```

## Setup

### 1. Clone / unzip the project

```bash
cd RAG_Sanskrit
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Run the pipeline

#### Terminal (interactive mode)
```bash
cd code
python rag_pipeline.py --interactive
```

#### Single query
```bash
python rag_pipeline.py --query "Who is Shankhanaad?"
python rag_pipeline.py --query "शंखनादः कः अस्ति?"
```

#### Force re-ingest documents
```bash
python rag_pipeline.py --ingest --interactive
```

#### Streamlit Web UI
```bash
streamlit run code/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

#### Run Evaluation
```bash
python code/evaluate.py
```

## Configuration

All parameters can be passed via CLI flags or changed in the `SanskritRAGSystem` constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `../data` | Path to data directory |
| `--llm` | `google/flan-t5-base` | HuggingFace model ID |
| `--top-k` | `3` | Number of retrieved chunks |
| `chunk_size` | `300` | Words per chunk |
| `overlap` | `50` | Overlap between chunks |

## Models Used

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | ~470MB | Supports Devanagari |
| Generator | `google/flan-t5-base` | ~990MB | CPU-friendly T5 |
| Index | FAISS `IndexFlatIP` | — | Exact cosine search |

## CPU Optimization Notes

- All models run on `device="cpu"` explicitly
- T5-base uses `num_beams=2` for speed/quality balance
- Embeddings are cached to disk after first ingestion
- FAISS provides sub-millisecond retrieval even on CPU

## Example Queries

- `Who is Shankhanaad and what mistakes did he make?`
- `What is the story of King Bhoj and Kalidasa?`
- `What happened with the bell in the forest?`
- `शंखनादः कः अस्ति?` (Sanskrit)
- `kAlIdAsa bhoja raja` (transliteration)
