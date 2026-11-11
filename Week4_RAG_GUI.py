import io
import os
import importlib.util
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def load_week3_module():
    """Load the Week 3 pipeline module once and cache it."""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file or environment."
        )

    module_path = os.path.join(os.getcwd(), "week3_rag_pipeline.py")
    spec = importlib.util.spec_from_file_location("week3_rag_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load week3_rag_pipeline.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_uploaded_documents(uploaded_files: List[Any]) -> List[Dict[str, str]]:
    """Read uploaded PDFs and text files into the same document schema."""
    documents: List[Dict[str, str]] = []
    for uploaded in uploaded_files or []:
        name = uploaded.name
        data = uploaded.getvalue()
        if name.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(data))
                pages = [
                    (page.extract_text() or "").strip() for page in reader.pages
                ]
                text = "\n\n".join(p for p in pages if p).strip()
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.warning(f"Could not read PDF '{name}': {exc}")
                continue
        else:
            try:
                text = data.decode("utf-8", errors="ignore").strip()
            except Exception:
                text = str(data, errors="ignore").strip()

        if text:
            documents.append({"source": name, "text": text})
    return documents


def build_rag_state(module: Any, raw_documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run the Week 3 pipeline stages 1-4 to create the searchable index."""
    documents = module.stage1_ingest(raw_documents)
    chunks = module.stage2_chunk(documents)
    chunks, embeddings = module.stage3_embed(chunks)
    index, normalised = module.stage4_index(embeddings)
    return {
        "documents": documents,
        "chunks": chunks,
        "index": index,
        "normalised_embeddings": normalised,
        "ready": True,
    }


st.set_page_config(page_title="RAG GUI", page_icon="🤖", layout="wide")
st.title("RAG GUI")
st.caption("Ingest documents, build the vector index, then ask questions using the RAG pipeline.")

if "rag_state" not in st.session_state:
    st.session_state.rag_state = {"ready": False}

try:
    rag_module = load_week3_module()
except Exception as exc:
    rag_module = None
    st.error(str(exc))
    with st.expander("Setup instructions"):
        st.code(
            """# .env
OPENROUTER_API_KEY=your_key_here

# then restart Streamlit
streamlit run Week4_RAG_GUI.py
"""
        )
    st.stop()

with st.sidebar:
    st.header("Knowledge base")
    st.subheader("Sources")
    load_sample = st.button("Load sample corpus", use_container_width=True)
    load_pdf_folder = st.button("Load PDFs from docs/", use_container_width=True)
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=3)

if load_sample:
    st.session_state.rag_state = build_rag_state(rag_module, rag_module.SAMPLE_DOCUMENTS)
    st.success(f"Sample corpus ingested: {len(st.session_state.rag_state['documents'])} documents")

if load_pdf_folder:
    pdf_docs = rag_module.read_pdf_files("docs")
    if pdf_docs:
        st.session_state.rag_state = build_rag_state(rag_module, pdf_docs)
        st.success(f"PDF files ingested: {len(st.session_state.rag_state['documents'])} documents")
    else:
        st.warning("No PDF files were found in the docs/ folder.")

if uploaded_files:
    docs = read_uploaded_documents(uploaded_files)
    if docs:
        st.session_state.rag_state = build_rag_state(rag_module, docs)
        st.success(f"Uploaded files ingested: {len(st.session_state.rag_state['documents'])} documents")
    else:
        st.warning("No readable text was found in the uploaded files.")

state = st.session_state.rag_state
if state.get("ready"):
    st.metric("Documents", len(state["documents"]))
    st.metric("Chunks", len(state["chunks"]))
    st.caption(f"Index total: {state['index'].ntotal}")
else:
    st.info("Load a sample corpus or upload files to build the knowledge base first.")

st.divider()
st.subheader("Ask a question")
question = st.text_input("Question", placeholder="What is RAG?")

if st.button("Generate answer"):
    if not state.get("ready"):
        st.error("Please ingest a document set first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving, augmenting, and generating..."):
            retrieved = rag_module.stage5_retrieve(
                question,
                state["index"],
                state["normalised_embeddings"],
                state["chunks"],
                top_k=top_k,
            )
            system_prompt, user_prompt = rag_module.stage6_build_prompt(question, retrieved)
            answer = rag_module.stage7_generate(system_prompt, user_prompt)

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved sources"):
            for chunk in retrieved:
                st.markdown(
                    f"**{chunk['source']}** — chunk #{chunk['chunk_index']} | similarity {chunk['retrieval_score']:.4f}"
                )
                st.caption(chunk["text"])

        with st.expander("Prompt debug"):
            st.code(system_prompt)
            st.code(user_prompt)
