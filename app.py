"""Start from the repository root with: streamlit run app.py."""
import hashlib

import streamlit as st
from lab.config import ROOT, AzureSettings
from lab.retrieval import EMBEDDING_MODEL, FaissStore, answer_question, get_embeddings, load_pdf_chunks

st.set_page_config(page_title="LangChain PDF Q&A", page_icon="📚")
st.title("Ask the New York brochure")
st.caption("Explore a supplied training brochure with retrieved passages and source citations.")
try:
    AzureSettings.load()
except ValueError as error:
    st.info(str(error))
    st.stop()


@st.cache_resource
def load_store(pdf_path, source_hash, model_id):
    # Hash and model are explicit cache keys, so changed inputs rebuild the index.
    return FaissStore.build(load_pdf_chunks(pdf_path), get_embeddings(), model_id=model_id)


pdf_path = ROOT / "assets/new_york_brochure.pdf"
top_k = st.sidebar.slider("Passages to retrieve", 1, 8, 3)
use_filter = st.sidebar.checkbox("Filter by cosine similarity", value=False)
threshold = st.sidebar.slider("Minimum cosine similarity", -1.0, 1.0, 0.5, 0.05) if use_filter else None
st.sidebar.caption("Similarity is a ranking score; the optional cutoff is not calibrated confidence.")
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved passages"):
                for label in message["sources"]:
                    st.text(label)

if question := st.chat_input("Ask a standalone question about the brochure"):
    if not question.strip():
        st.stop()
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Retrieving passages and generating an answer..."):
            store = load_store(str(pdf_path), hashlib.sha256(pdf_path.read_bytes()).hexdigest(), EMBEDDING_MODEL)
            hits = store.search(question, k=top_k, minimum_score=threshold)
            answer = answer_question(question, hits)
            sources = [f"[Source {i}] {h.document.metadata['source']}, page {h.document.metadata['page']} "
                       f"(cosine {h.score:.3f})\n{h.document.page_content}" for i, h in enumerate(hits, start=1)]
    except Exception:
        st.error("Request failed. Check configuration, deployment access, model downloads, and connectivity.")
    else:
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "sources": sources},
        ])
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Retrieved passages"):
                for source in sources:
                    st.text(source)
