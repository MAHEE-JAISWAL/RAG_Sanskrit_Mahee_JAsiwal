"""
app.py — Streamlit web UI for the Sanskrit RAG System.

Run from the code/ directory:
    streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import SanskritRAGSystem
from config import LLM_MODEL_NAME, TOP_K

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sanskrit RAG",
    page_icon="📜",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📜 Sanskrit RAG System")
st.markdown(
    "Ask questions about Sanskrit texts — in **Sanskrit (Devanagari)**, "
    "**transliteration**, or **English**."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=8, value=TOP_K)
    llm_model = st.selectbox(
        "Generator model",
        ["google/flan-t5-base", "google/flan-t5-small"],
        index=0,
    )
    force_rebuild = st.checkbox("Force re-ingest on startup", value=False)
    st.markdown("---")
    st.caption("CPU-only inference · No GPU required")

# ── Load RAG (cached so it only loads once per session) ───────────────────────
@st.cache_resource(show_spinner="⏳ Loading RAG system…")
def load_rag(model: str, k: int, rebuild: bool) -> SanskritRAGSystem:
    rag = SanskritRAGSystem(llm_model=model, top_k=k)
    rag.ingest(force_rebuild=rebuild)
    return rag

rag = load_rag(llm_model, top_k, force_rebuild)

# ── Sample buttons ────────────────────────────────────────────────────────────
SAMPLES = [
    "Who is Shankhanaad?",
    "What did Shankhanaad do with the sugar?",
    "What is the story of King Bhoj and Kalidasa?",
    "What is the moral about the devotee and God?",
    "How did the old woman solve the Ghantakarna problem?",
    "शंखनादः कः अस्ति?",
    "घण्टाकर्णः किम् अभवत्?",
]

st.markdown("**Try a sample query:**")
cols = st.columns(len(SAMPLES))
for col, sample in zip(cols, SAMPLES):
    if col.button(sample, key=f"btn_{sample}"):
        st.session_state.query_text = sample
        st.session_state.result     = None
        st.session_state.error      = None

# ── Query input box ───────────────────────────────────────────────────────────
st.markdown("### 🔍 Ask a Question")
col1, col2 = st.columns([4, 1])
with col1:
    typed_query = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="e.g. Who is Kalidasa? / कालिदासः कः अस्ति?",
        value=st.session_state.query_text,
        key="input_box",
    )
with col2:
    ask_clicked = st.button("Ask →", use_container_width=True, type="primary")

# ── Decide final query ────────────────────────────────────────────────────────
final_query = typed_query.strip() or st.session_state.query_text.strip()

should_run = ask_clicked or (
    st.session_state.query_text
    and st.session_state.result is None
    and st.session_state.error  is None
    and final_query
)

st.markdown("---")

if should_run and final_query:
    with st.spinner("🔎 Retrieving context and generating answer…"):
        try:
            result = rag.query(final_query)
            st.session_state.result     = result
            st.session_state.error      = None
            st.session_state.query_text = final_query
        except Exception as e:
            st.session_state.error  = str(e)
            st.session_state.result = None

elif ask_clicked and not final_query:
    st.warning("Please enter a query first.")

# ── Show results ──────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"❌ Error: {st.session_state.error}")

elif st.session_state.result:
    result = st.session_state.result

    st.markdown("### 💬 Answer")
    st.success(result["answer"])
    st.caption(
        f"⏱ Latency: `{result['latency_seconds']}s` &nbsp;|&nbsp; "
        f"Query: _{result['query']}_"
    )

    st.markdown("### 📚 Retrieved Context Chunks")
    for i, (chunk, score) in enumerate(
        zip(result["retrieved_chunks"], result["retrieval_scores"]), 1
    ):
        section = chunk.get("metadata", {}).get("section", "Unknown")
        with st.expander(f"[{i}]  {section}  —  similarity: {score:.4f}"):
            st.text(chunk["text"])