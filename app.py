"""
app.py
------
Streamlit demo for the LoL Analytics RAG Assistant.
Run with: streamlit run app.py

This is the "show, don't tell" piece of the project -- a recruiter or
interviewer can open this and ask their own questions about the LoL
capstone findings, rather than just reading a README description of
what RAG theoretically does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from retriever import get_retriever
from generator import answer_question

st.set_page_config(page_title="LoL Analytics RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 LoL Analytics — RAG Assistant")
st.markdown(
    "Ask questions about the statistical findings from the "
    "[LoL Match Analytics capstone](https://github.com/harshp9945/lol-match-analytics) "
    "(51,490 ranked matches). Answers are grounded in the project's actual "
    "notebook outputs — the assistant will say *\"I don't know\"* rather than "
    "guess if something isn't in the knowledge base."
)

with st.expander("ℹ️ How this works (RAG architecture)"):
    st.markdown("""
    1. **Knowledge base**: 14 statistical findings extracted from the 12 capstone notebooks, each tagged with its source notebook.
    2. **Retrieval**: your question is matched against the knowledge base using TF-IDF cosine similarity (a free, local, offline retrieval method).
    3. **Threshold guardrail**: if nothing scores above 0.15 similarity, the system refuses to answer rather than guessing.
    4. **Generation**: the retrieved context + your question is passed to an LLM, which is instructed to answer *only* from the provided context.

    **Known limitation**: TF-IDF matches on keyword overlap, not meaning — so phrasing matters more than it would with real embeddings. This demo intentionally uses TF-IDF to keep it free to run; see the README for the embedding-based upgrade path.
    """)

# Initialize retriever once (cached across reruns)
@st.cache_resource
def load_retriever():
    return get_retriever("tfidf")

retriever = load_retriever()

# Sample questions for easy testing
st.markdown("**Try a sample question:**")
sample_cols = st.columns(2)
sample_questions = [
    "What's the win rate for securing Baron?",
    "Is Yasuo overpowered?",
    "How accurate is the win prediction model?",
    "Can teams come back from being behind?",
]

selected_query = None
for i, q in enumerate(sample_questions):
    col = sample_cols[i % 2]
    if col.button(q, use_container_width=True):
        selected_query = q

query = st.text_input("Or ask your own question:", value=selected_query or "")

if query:
    with st.spinner("Retrieving and generating answer..."):
        result = answer_question(query, retriever, backend="mock")

    st.markdown("### Answer")
    st.info(result["answer"])

    st.markdown("### Sources")
    for src in result["sources"]:
        score = src["score"]
        confidence = "🟢 High" if score and score >= 0.3 else "🟡 Low" if score and score >= 0.15 else "⚪ Below threshold"
        st.markdown(f"- `{src['notebook']}` — similarity: {score} ({confidence})")

st.markdown("---")
st.caption(
    "This demo uses a free mock generator (returns retrieved text directly) so it runs with no API key. "
    "For a real LLM, run Ollama locally for free (see README) and change `backend=\"mock\"` to `backend=\"ollama\"` above, "
    "or use `backend=\"anthropic\"` with an API key for best quality."
)
