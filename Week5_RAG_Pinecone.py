"""
=============================================================================
  Week 5: RAG (Retrieval-Augmented Generation) Pipeline — Pinecone + Streamlit
=============================================================================

Same 7-stage pipeline as week3_rag_pipeline.py, but backed by a managed
Pinecone vector database and wrapped in a Streamlit GUI:

  STAGE 1 — INGESTION       : Load raw documents (uploads / sample / docs folder)
  STAGE 2 — CHUNKING        : Split documents using different strategies
  STAGE 3 — EMBEDDING       : Convert chunks to vectors via OpenRouter
  STAGE 4 — INDEXING        : Create a Pinecone index and upsert vectors
  STAGE 5 — RETRIEVAL       : Query the index and get top-k chunks
  STAGE 6 — AUGMENTATION    : Build a prompt from retrieved chunks
  STAGE 7 — GENERATION      : Call the LLM via OpenRouter and return answer

Models used (via OpenRouter):
  Embedding : openai/text-embedding-3-small  (1536-dim)
  LLM       : openai/gpt-4o-mini

Vector DB  : Pinecone (serverless, cosine metric)

Run the GUI:
  streamlit run Week5_RAG_Pinecone.py

Run the CLI (index setup + ingestion without the GUI):
  python Week5_RAG_Pinecone.py create-index
  python Week5_RAG_Pinecone.py ingest --source sample
  python Week5_RAG_Pinecone.py ingest --source docs --path docs --strategy recursive
  python Week5_RAG_Pinecone.py ask "What is RAG?"
  python Week5_RAG_Pinecone.py stats

Install dependencies:
  pip install openai pinecone streamlit python-dotenv pypdf numpy
=============================================================================
"""

import argparse
import io
import os
import re
import sys
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM   = 1536          # dimensions produced by text-embedding-3-small
LLM_MODEL       = "openai/gpt-4o-mini"

# Retrieval: how many chunks to pull back from Pinecone
TOP_K = 3

# Pinecone defaults (override via .env or GUI / CLI flags)
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-week5")
DEFAULT_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "Default")
DEFAULT_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_REGION     = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_BATCH_SIZE  = 20          # chunks per embeddings request
UPSERT_BATCH_SIZE = 100         # vectors per Pinecone upsert request

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE CORPUS  (replace with file I/O, PDF parsing, web scraping, etc.)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "source": "doc_llm_basics.txt",
        "text": """
        Large Language Models (LLMs) are deep learning models trained on massive text corpora.
        They use the Transformer architecture introduced in the seminal paper
        'Attention Is All You Need' (Vaswani et al., 2017).
        LLMs learn statistical patterns over tokens and can generate coherent, contextually
        relevant text. Prominent examples include GPT-4, Claude, Gemini, and Llama.
        Training involves next-token prediction on billions of tokens and is extremely
        compute-intensive, often requiring thousands of GPUs running for weeks.
        After pre-training, models are fine-tuned with RLHF (Reinforcement Learning from
        Human Feedback) to align outputs with human preferences.
        """,
    },
    {
        "source": "doc_rag_overview.txt",
        "text": """
        Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by
        injecting relevant external knowledge into the prompt at inference time.
        Rather than relying solely on weights learned during training, a RAG system first
        retrieves the most semantically similar document chunks from a vector database,
        then constructs a prompt that combines the user's question with those chunks.
        The LLM is then asked to answer using only the provided context, reducing hallucination
        and keeping responses grounded in real, up-to-date information.
        RAG is widely used in enterprise Q&A chatbots, coding assistants, and search engines.
        """,
    },
    {
        "source": "doc_embeddings.txt",
        "text": """
        Embeddings are dense, fixed-length numeric vectors that encode the semantic meaning
        of text. Similar texts are placed close together in the high-dimensional embedding
        space. OpenAI's text-embedding-3-small model produces 1536-dimensional vectors and
        is a strong baseline for retrieval tasks. To compare two embeddings, cosine similarity
        or dot product is used. In a RAG system, both the document chunks and the user query
        are converted to embeddings, and the nearest neighbours in vector space are returned
        as the most relevant context. This is far more powerful than keyword search because
        it captures meaning, not just surface-level word overlap.
        """,
    },
    {
        "source": "doc_vector_databases.txt",
        "text": """
        Vector databases are purpose-built storage systems for high-dimensional embeddings.
        They support efficient approximate nearest-neighbour (ANN) search, making it possible
        to query millions of vectors in milliseconds. Pinecone is a managed vector database:
        it provides serverless indexes, namespaces for logical partitioning, and metadata
        filtering without any infrastructure to run yourself. FAISS, by contrast, is an
        in-process library that stores vectors in local memory. For production workloads
        with millions of records, a managed store such as Pinecone removes the operational
        burden of sharding, replication, and persistence.
        """,
    },
    {
        "source": "doc_chunking.txt",
        "text": """
        Chunking is the process of splitting long documents into smaller, overlapping pieces
        before embedding. Common strategies include:
        1. Fixed-size chunking: split every N characters/tokens regardless of content.
        2. Sentence-based chunking: split on sentence boundaries to preserve coherence.
        3. Paragraph-based chunking: split on double newlines — natural semantic units.
        4. Recursive character splitting: try paragraph, then sentence, then word boundaries.
        5. Semantic chunking: use embedding similarity to find natural topic boundaries.
        Chunk overlap (e.g., 10–20 % of the chunk size) is added so that context at
        boundaries is not lost. Smaller chunks give more precise retrieval; larger chunks
        give more context per result. Typical chunk sizes range from 256 to 1024 tokens.
        """,
    },
    {
        "source": "doc_prompt_engineering.txt",
        "text": """
        Prompt engineering in RAG involves carefully constructing the final prompt sent to
        the LLM. A well-designed RAG prompt has three parts:
        1. System instruction — tells the model to answer using only the provided context
           and to say 'I don't know' when the answer cannot be found.
        2. Context block — the concatenated retrieved chunks, clearly labelled.
        3. User question — the original query from the user.
        Adding source citations (chunk metadata like filename and chunk index) in the context
        block allows the LLM to attribute its answer, improving trust and debuggability.
        Temperature should be set low (0.0–0.2) for factual Q&A tasks to reduce randomness.
        """,
    },
]

SAMPLE_DOCUMENTS = [doc for doc in SAMPLE_DOCUMENTS if doc["text"].strip()]


# ─────────────────────────────────────────────────────────────────────────────
# CLI HELPERS  (used by the command-line entry point, not by the GUI)
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    """Print a clearly visible stage banner."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: str, detail: str = "") -> None:
    """Print a numbered sub-step with optional detail."""
    print(f"\n  ▶ {step}")
    if detail:
        for line in textwrap.wrap(detail, width=68):
            print(f"    {line}")


def cli_progress(fraction: float, label: str) -> None:
    """Progress sink used by CLI commands."""
    print(f"    [{fraction * 100:5.1f}%] {label}")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

_openai_clients: Dict[str, OpenAI] = {}
_pinecone_clients: Dict[str, Pinecone] = {}


def get_openai_client() -> OpenAI:
    """Return a cached OpenRouter client (embeddings + chat share the same one)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env or the environment."
        )
    if api_key not in _openai_clients:
        _openai_clients[api_key] = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
    return _openai_clients[api_key]


def get_pinecone_client() -> Pinecone:
    """Return a cached Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to .env or the environment."
        )
    if api_key not in _pinecone_clients:
        _pinecone_clients[api_key] = Pinecone(api_key=api_key)
    return _pinecone_clients[api_key]


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_pdf_files(folder_path: str = "docs") -> List[Dict[str, str]]:
    """Read every PDF in a folder and return a list of document dicts.

    Each dict is normalized to the schema used by the rest of the pipeline:
    {"source": filename, "text": extracted_text}.
    """
    if not os.path.isdir(folder_path):
        return []

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF reading requires the 'pypdf' package. Install it with: pip install pypdf"
        ) from exc

    documents: List[Dict[str, str]] = []
    pdf_files = sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith(".pdf")
    )

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        text = "\n\n".join(pages).strip()
        if text:
            documents.append({"source": filename, "text": text})

    return documents


def read_uploaded_documents(uploaded_files: List[Any]) -> List[Dict[str, str]]:
    """Read Streamlit uploads (PDF / TXT / Markdown) into the document schema."""
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
            except Exception as exc:
                st.warning(f"Could not read PDF '{name}': {exc}")
                continue
        else:
            text = data.decode("utf-8", errors="ignore").strip()

        if text:
            documents.append({"source": name, "text": text})
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def stage1_ingest(documents: List[Dict]) -> List[Dict]:
    """
    Load raw documents.  In a real system this step would read from:
      - Local files  (PDF, DOCX, Markdown, plain text)
      - Databases    (SQL, MongoDB)
      - APIs         (Confluence, Notion, SharePoint)
      - Web scrapers

    Each document is normalised to a dict with at minimum:
        { 'source': str, 'text': str }
    """
    loaded = []
    for doc in documents:
        # Collapse line wrapping inside paragraphs, but keep blank-line
        # separators so paragraph-based chunking can still detect boundaries.
        paragraphs = [
            " ".join(paragraph.split())
            for paragraph in re.split(r"\n\s*\n", doc["text"].strip())
            if paragraph.strip()
        ]
        clean_text = "\n\n".join(paragraphs)
        if clean_text:
            loaded.append({"source": doc["source"], "text": clean_text})
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

class ChunkingStrategy:
    """Collection of chunking strategies."""

    @staticmethod
    def fixed_size(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
        """
        Fixed-size character chunking.
        Simple but can split mid-sentence.  Good for uniform retrieval latency.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - overlap  # slide window with overlap
        return [c for c in chunks if c]

    @staticmethod
    def sentence_based(text: str) -> List[str]:
        """
        Split on sentence boundaries ('. ', '! ', '? ').
        Preserves complete sentences → better semantic coherence per chunk.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk_size = 2  # sentences per chunk
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i : i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def paragraph_based(text: str) -> List[str]:
        """
        Split on paragraph boundaries (double newline).
        Natural semantic units; works well for structured documents.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]

    @staticmethod
    def recursive(text: str, max_chars: int = 300) -> List[str]:
        """
        Recursive splitting: try paragraph → sentence → word.
        Guarantees chunks ≤ max_chars while preferring natural boundaries.

        Any part that is itself larger than max_chars is split recursively
        instead of being emitted as-is.
        """
        if len(text) <= max_chars:
            return [text]
        for sep in ["\n\n", ". ", " "]:
            parts = text.split(sep)
            if len(parts) > 1:
                chunks = []
                current = ""
                for part in parts:
                    candidate = (current + sep + part).strip() if current else part
                    if len(candidate) <= max_chars:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        sub = ChunkingStrategy.recursive(part, max_chars)
                        chunks.extend(sub[:-1])
                        current = sub[-1] if sub else ""
                if current:
                    chunks.append(current)
                return [c for c in chunks if c]
        # Hard fallback: no separator found — slice at the character boundary
        return [text[:max_chars], text[max_chars:]]


STRATEGY_FUNCTIONS: Dict[str, Callable[[str], List[str]]] = {
    "paragraph": ChunkingStrategy.paragraph_based,
    "recursive": ChunkingStrategy.recursive,
    "sentence": ChunkingStrategy.sentence_based,
    "fixed": ChunkingStrategy.fixed_size,
}

# GUI labels → strategy keys
GUI_STRATEGY_LABELS: Dict[str, str] = {
    "Paragraph-based (recommended)": "paragraph",
    "Recursive (300-char cap)": "recursive",
    "Sentence-based (2 sentences)": "sentence",
    "Fixed-size (200 chars, 40 overlap)": "fixed",
}


def stage2_chunk(documents: List[Dict], strategy: str = "paragraph") -> List[Dict]:
    """
    Chunk every document with the selected strategy.
    Returns a flat list of chunk dicts:
        { 'source', 'chunk_index', 'text', 'strategy' }
    """
    strategy_fn = STRATEGY_FUNCTIONS.get(strategy)
    if strategy_fn is None:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. "
            f"Choose one of: {', '.join(STRATEGY_FUNCTIONS)}"
        )

    all_chunks: List[Dict] = []
    for doc in documents:
        for idx, chunk_text in enumerate(strategy_fn(doc["text"])):
            all_chunks.append(
                {
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": chunk_text,
                    "strategy": strategy,
                }
            )
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> np.ndarray:
    """
    Call OpenRouter's embeddings endpoint in batches.
    Returns an (N, EMBEDDING_DIM) float32 numpy array.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_vectors = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vectors)

        if progress_cb:
            done = i + len(batch)
            progress_cb(done / len(texts), f"Embedded {done}/{len(texts)} chunks")

        # Polite rate-limit buffer between batches
        if i + batch_size < len(texts):
            time.sleep(0.3)

    return np.array(all_embeddings, dtype=np.float32)


def stage3_embed(
    chunks: List[Dict],
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """Embed every chunk and return it with the (N, 1536) embedding matrix."""
    embeddings = get_embeddings(
        [c["text"] for c in chunks],
        progress_cb=progress_cb,
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(embeddings)} vectors for {len(chunks)} chunks."
        )
    return chunks, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — INDEXING (PINECONE)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_pinecone_index(
    index_name: str,
    cloud: str = DEFAULT_CLOUD,
    region: str = DEFAULT_REGION,
    wait: bool = True,
    timeout_s: int = 120,
) -> Tuple[Any, bool]:
    """
    Create the Pinecone index if it does not exist, then return a handle.

    The index is serverless with cosine similarity and the embedding model's
    dimension, so vectors can be upserted without normalising them here
    (Pinecone normalises for cosine).

    Returns (index_handle, created).
    """
    pc = get_pinecone_client()
    created = False

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        created = True

        if wait:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                try:
                    if getattr(pc.describe_index(index_name).status, "ready", False):
                        break
                except Exception:
                    pass  # index may not be describable for a few seconds
                time.sleep(2)

    description = pc.describe_index(index_name)
    dimension = int(getattr(description, "dimension", EMBEDDING_DIM))
    metric = getattr(description, "metric", "cosine")
    metric_value = str(getattr(metric, "value", metric)).lower()

    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"Index '{index_name}' has dimension {dimension}, but '{EMBEDDING_MODEL}' "
            f"produces {EMBEDDING_DIM}. Use a different index name."
        )
    if metric_value != "cosine":
        raise RuntimeError(
            f"Index '{index_name}' uses metric '{metric_value}'; this pipeline expects 'cosine'."
        )

    return pc.Index(index_name), created


def connect_existing_index(index_name: str) -> Any:
    """Connect to an existing index or raise a helpful error."""
    pc = get_pinecone_client()
    if not pc.has_index(index_name):
        raise RuntimeError(
            f"Index '{index_name}' does not exist. Create it from the Streamlit "
            f"sidebar or run: python {os.path.basename(__file__)} create-index"
        )
    return pc.Index(index_name)


def make_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic vector id → re-ingesting a document overwrites its chunks."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", source).strip("-").lower()
    return f"{slug or 'doc'}::chunk-{chunk_index}"


def stage4_upsert(
    index: Any,
    chunks: List[Dict],
    embeddings: np.ndarray,
    namespace: str = DEFAULT_NAMESPACE,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Upsert chunk vectors into Pinecone in batches.

    Metadata carries the chunk text plus provenance, so retrieval needs no
    secondary lookup: Pinecone returns the content straight back.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    vectors = [
        {
            "id": make_chunk_id(chunk["source"], chunk["chunk_index"]),
            "values": embedding.tolist(),
            "metadata": {
                "source": chunk["source"],
                "chunk_index": int(chunk["chunk_index"]),
                "text": chunk["text"],
                "strategy": chunk["strategy"],
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    written = 0
    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        written += len(batch)

        if progress_cb:
            progress_cb(written / len(vectors), f"Upserted {written}/{len(vectors)} vectors")

    return written


def clear_namespace(index: Any, namespace: str) -> None:
    """Delete every vector in a namespace; a missing namespace is already empty."""
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
        if status == 404 or "not found" in str(exc).lower():
            return
        raise


def _namespace_count(index: Any, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    description = namespaces.get(namespace)
    if description is None:
        return 0
    if isinstance(description, dict):
        return int(description.get("vector_count", 0) or 0)
    return int(getattr(description, "vector_count", 0) or 0)


def wait_for_vectors(
    index: Any,
    namespace: str,
    expected: int,
    timeout_s: int = 60,
) -> bool:
    """Serverless indexes are eventually consistent — poll until visible."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _namespace_count(index, namespace) >= expected:
            return True
        time.sleep(2)
    return False


def index_stats(index: Any) -> Dict[str, Any]:
    """Return total vector count, dimension and per-namespace counts."""
    stats = index.describe_index_stats()
    namespaces = {}
    for name, description in (stats.namespaces or {}).items():
        if isinstance(description, dict):
            namespaces[name] = int(description.get("vector_count", 0) or 0)
        else:
            namespaces[name] = int(getattr(description, "vector_count", 0) or 0)
    return {
        "total_vector_count": int(getattr(stats, "total_vector_count", 0) or 0),
        "dimension": int(getattr(stats, "dimension", EMBEDDING_DIM) or EMBEDDING_DIM),
        "namespaces": namespaces,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION ORCHESTRATION  (Stages 1-4 in one call, shared by GUI and CLI)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_documents(
    documents: List[Dict],
    index_name: str,
    namespace: str = DEFAULT_NAMESPACE,
    strategy: str = "paragraph",
    cloud: str = DEFAULT_CLOUD,
    region: str = DEFAULT_REGION,
    clear_first: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run ingestion end to end: normalise → chunk → embed → index → upsert.

    With clear_first=True the namespace is emptied before writing, so the
    namespace always reflects exactly the documents just ingested.
    """
    if not documents:
        raise ValueError("No documents were provided for ingestion.")

    def report(fraction: float, label: str) -> None:
        if progress_cb:
            progress_cb(min(max(fraction, 0.0), 1.0), label)

    report(0.02, "Normalising documents…")
    ingested = stage1_ingest(documents)
    if not ingested:
        raise ValueError("No readable text was found in the provided documents.")

    report(0.08, f"Chunking {len(ingested)} documents with the '{strategy}' strategy…")
    chunks = stage2_chunk(ingested, strategy=strategy)
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    report(0.10, f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}…")
    chunks, embeddings = stage3_embed(
        chunks,
        progress_cb=lambda f, msg: report(0.10 + 0.55 * f, msg),
    )

    report(0.68, f"Connecting to Pinecone index '{index_name}'…")
    index, created = ensure_pinecone_index(index_name, cloud=cloud, region=region)

    if clear_first:
        report(0.72, f"Clearing namespace '{namespace}'…")
        clear_namespace(index, namespace)
        existing = 0
    else:
        existing = _namespace_count(index, namespace)

    report(0.75, f"Upserting {len(chunks)} vectors into '{namespace}'…")
    written = stage4_upsert(
        index,
        chunks,
        embeddings,
        namespace=namespace,
        progress_cb=lambda f, msg: report(0.75 + 0.20 * f, msg),
    )

    report(0.97, "Waiting for the index to make the vectors queryable…")
    ready = wait_for_vectors(index, namespace, expected=existing + written)
    report(1.0, "Ingestion complete.")

    return {
        "index_name": index_name,
        "namespace": namespace,
        "strategy": strategy,
        "index_created": created,
        "document_count": len(ingested),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "written": written,
        "ready": ready,
        "stats": index_stats(index),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def stage5_retrieve(
    query: str,
    index: Any,
    namespace: str = DEFAULT_NAMESPACE,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    1. Embed the user query with the same model used for indexing.
    2. Query Pinecone for the top-k nearest neighbours.
    3. Return the matching chunk dicts with their similarity scores.
    """
    query_vec = get_embeddings([query])[0]
    response = index.query(
        vector=query_vec.tolist(),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    retrieved = []
    for rank, match in enumerate(response.matches, start=1):
        metadata = match.metadata or {}
        retrieved.append(
            {
                "id": match.id,
                "source": metadata.get("source", "unknown"),
                "chunk_index": int(metadata.get("chunk_index", -1)),
                "text": metadata.get("text", ""),
                "strategy": metadata.get("strategy", "unknown"),
                "retrieval_score": float(match.score),
                "retrieval_rank": rank,
            }
        )
    return retrieved


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — PROMPT AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def stage6_build_prompt(query: str, retrieved_chunks: List[Dict]) -> Tuple[str, str]:
    """
    Assemble the final prompt for the LLM.

    Structure:
        SYSTEM : strict instruction to use only the provided context
        USER   : context block (all retrieved chunks, labelled) + question
    """
    context_lines = []
    for chunk in retrieved_chunks:
        context_lines.append(
            f"[Source: {chunk['source']} | Chunk #{chunk['chunk_index']} "
            f"| Similarity: {chunk['retrieval_score']:.4f}]\n{chunk['text']}"
        )
    context_block = "\n\n---\n\n".join(context_lines)

    system_prompt = (
        "You are a precise and helpful assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "If the answer cannot be found in the context, respond with: "
        "'I don't have enough information to answer this question.' "
        "Cite the source file name when possible."
    )

    user_prompt = (
        f"CONTEXT:\n\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {query}"
    )

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — GENERATION (LLM CALL)
# ─────────────────────────────────────────────────────────────────────────────

def stage7_generate(system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
    """Call the LLM via OpenRouter; return the answer and token usage."""
    response = get_openai_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = (response.choices[0].message.content or "").strip()
    usage_obj = response.usage
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return answer, usage


# ─────────────────────────────────────────────────────────────────────────────
# FINAL ANSWER DISPLAY (CLI)
# ─────────────────────────────────────────────────────────────────────────────

def display_answer(
    query: str,
    answer: str,
    usage: Dict[str, int],
    retrieved_chunks: List[Dict],
) -> None:
    """Pretty-print the full RAG result to the terminal."""
    banner("FINAL ANSWER")

    print(f"\n  Question : {query}\n")
    print("  " + "─" * 68)
    print()
    for line in textwrap.wrap(answer, width=68):
        print(f"  {line}")
    print()
    print("  " + "─" * 68)
    print(f"\n  Prompt tokens    : {usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {usage['completion_tokens']:,}")
    print(f"  Total tokens     : {usage['total_tokens']:,}")

    print("\n  Sources used:")
    if not retrieved_chunks:
        print("    (none)")
    for chunk in retrieved_chunks:
        print(
            f"    • {chunk['source']}  "
            f"(chunk #{chunk['chunk_index']}, "
            f"similarity={chunk['retrieval_score']:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT GUI
# ─────────────────────────────────────────────────────────────────────────────

def _render_key_input(label: str, env_var: str) -> None:
    """Show whether an API key is configured, or accept one for this session."""
    if os.getenv(env_var):
        st.caption(f"{label}: configured")
        return
    entered = st.text_input(
        f"{label} API key",
        type="password",
        placeholder=f"Paste key, or set {env_var} in .env",
    )
    if entered.strip():
        os.environ[env_var] = entered.strip()
        st.rerun()


def _render_stats() -> None:
    stats = st.session_state.get("index_stats")
    if not stats:
        return
    st.caption(
        f"Vectors: {stats['total_vector_count']:,} · "
        f"Dimension: {stats['dimension']}"
    )
    if stats["namespaces"]:
        listing = ", ".join(
            f"{name} ({count:,})" for name, count in stats["namespaces"].items()
        )
        st.caption(f"Namespaces: {listing}")
    else:
        st.caption("Namespaces: none yet")


def render_sidebar() -> Dict[str, Any]:
    """Connection settings, index management and retrieval options."""
    with st.sidebar:
        st.header("Connections")
        _render_key_input("OpenRouter", "OPENROUTER_API_KEY")
        _render_key_input("Pinecone", "PINECONE_API_KEY")

        st.divider()
        st.header("Pinecone index")

        index_name = st.text_input(
            "Index name",
            value=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        )
        namespace = st.text_input(
            "Namespace",
            value=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        )

        clouds = ["aws", "gcp", "azure"]
        cloud_default = os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD)
        cloud = st.selectbox(
            "Cloud",
            clouds,
            index=clouds.index(cloud_default) if cloud_default in clouds else 0,
        )
        region = st.text_input(
            "Region",
            value=os.getenv("PINECONE_REGION", DEFAULT_REGION),
        )

        create_col, stats_col = st.columns(2)
        if create_col.button("Create index", type="primary"):
            if not os.getenv("PINECONE_API_KEY"):
                st.error("Set PINECONE_API_KEY first.")
            else:
                try:
                    with st.spinner("Creating or connecting to the index…"):
                        index, created = ensure_pinecone_index(index_name, cloud, region)
                    st.session_state["index_stats"] = index_stats(index)
                    st.success(
                        f"Index '{index_name}' {'created' if created else 'already exists'}."
                    )
                except Exception as exc:
                    st.error(f"Index setup failed: {exc}")

        if stats_col.button("Refresh stats"):
            try:
                st.session_state["index_stats"] = index_stats(
                    connect_existing_index(index_name)
                )
            except Exception as exc:
                st.error(f"Could not read index stats: {exc}")

        _render_stats()

        st.divider()
        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=TOP_K)

    return {
        "index_name": index_name,
        "namespace": namespace,
        "cloud": cloud,
        "region": region,
        "top_k": top_k,
    }


def collect_source_documents(source: str, uploaded_files: List[Any]) -> List[Dict[str, str]]:
    if source == "Upload files":
        return read_uploaded_documents(uploaded_files)
    if source == "Sample corpus":
        return list(SAMPLE_DOCUMENTS)
    return read_pdf_files("docs")


def _run_ingest(
    documents: List[Dict[str, str]],
    settings: Dict[str, Any],
    strategy: str,
    clear_first: bool,
) -> None:
    with st.status("Ingesting documents…", expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")

        def progress(fraction: float, label: str) -> None:
            bar.progress(fraction, text=label)
            status.write(label)

        try:
            result = ingest_documents(
                documents=documents,
                index_name=settings["index_name"],
                namespace=settings["namespace"],
                strategy=strategy,
                cloud=settings["cloud"],
                region=settings["region"],
                clear_first=clear_first,
                progress_cb=progress,
            )
        except Exception as exc:
            status.update(label="Ingestion failed", state="error")
            st.error(str(exc))
            return

        status.update(label="Ingestion complete", state="complete")

    st.session_state["ingest_result"] = result
    st.session_state["index_stats"] = result["stats"]
    st.session_state["last_answer"] = None


def render_ingestion_result(result: Dict[str, Any]) -> None:
    st.divider()
    st.subheader("Last ingestion")

    doc_col, chunk_col, vector_col = st.columns(3)
    doc_col.metric("Documents", result["document_count"])
    chunk_col.metric("Chunks", result["chunk_count"])
    vector_col.metric("Vectors upserted", result["written"])

    st.caption(
        f"Index '{result['index_name']}' · namespace '{result['namespace']}' · "
        f"strategy '{result['strategy']}' · "
        f"total vectors in index {result['stats']['total_vector_count']:,}"
    )
    if not result["ready"]:
        st.info(
            "Pinecone is still indexing these vectors. Queries may return fewer "
            "results for a few seconds."
        )

    preview = [
        {
            "source": chunk["source"],
            "chunk": chunk["chunk_index"],
            "chars": len(chunk["text"]),
            "preview": chunk["text"][:90].replace("\n", " "),
        }
        for chunk in result["chunks"][:10]
    ]
    st.dataframe(preview, width="stretch")
    if result["chunk_count"] > len(preview):
        st.caption(f"Showing the first {len(preview)} of {result['chunk_count']} chunks.")


def render_ingest_tab(settings: Dict[str, Any]) -> None:
    st.subheader("Ingest documents")
    st.caption(
        "Upload files, use the built-in sample corpus, or read PDFs from the "
        "./docs folder. Ingestion chunks, embeds and upserts into Pinecone."
    )

    source = st.radio(
        "Document source",
        ["Upload files", "Sample corpus", "docs/ folder (PDFs)"],
        horizontal=True,
    )

    uploaded_files: List[Any] = []
    if source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT or Markdown files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
    elif source == "docs/ folder (PDFs)":
        st.caption("Reads every PDF in the ./docs folder next to this script.")

    strategy_label = st.selectbox("Chunking strategy", list(GUI_STRATEGY_LABELS))
    strategy = GUI_STRATEGY_LABELS[strategy_label]

    clear_first = st.checkbox(
        "Clear namespace before ingest",
        value=True,
        help=(
            "Recommended: the namespace will contain exactly the documents you "
            "just ingested. Uncheck to merge with existing vectors."
        ),
    )
    st.caption(
        f"Target: index '{settings['index_name']}' · "
        f"namespace '{settings['namespace']}' · embedding {EMBEDDING_MODEL}"
    )

    if st.button("Ingest into Pinecone", type="primary"):
        if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("PINECONE_API_KEY"):
            st.error("Both OPENROUTER_API_KEY and PINECONE_API_KEY must be configured.")
        else:
            documents = collect_source_documents(source, uploaded_files)
            if not documents:
                st.warning("No readable documents were found for this source.")
            else:
                _run_ingest(documents, settings, strategy, clear_first)

    result = st.session_state.get("ingest_result")
    if result:
        render_ingestion_result(result)


def render_answer(result: Dict[str, Any]) -> None:
    usage = result["usage"]

    st.divider()
    st.subheader("Answer")
    st.caption(f"Question: {result['question']}")
    st.markdown(result["answer"])
    st.caption(
        f"Prompt tokens {usage['prompt_tokens']:,} · "
        f"completion tokens {usage['completion_tokens']:,} · "
        f"total {usage['total_tokens']:,}"
    )

    retrieved = result["retrieved"]
    with st.expander(f"Sources used ({len(retrieved)})"):
        if not retrieved:
            st.warning("No chunks were retrieved for this query.")
        for chunk in retrieved:
            st.markdown(
                f"**{chunk['source']}** — chunk #{chunk['chunk_index']} · "
                f"similarity {chunk['retrieval_score']:.4f} · "
                f"rank {chunk['retrieval_rank']}"
            )
            preview = chunk["text"][:500]
            if len(chunk["text"]) > len(preview):
                preview += "…"
            st.caption(preview)

    with st.expander("Prompt debug"):
        st.code(result["system_prompt"], language=None)
        st.code(result["user_prompt"], language=None)


def _run_query(question: str, settings: Dict[str, Any]) -> None:
    try:
        index = connect_existing_index(settings["index_name"])
        stats = index_stats(index)
    except Exception as exc:
        st.error(f"Could not connect to the Pinecone index: {exc}")
        return

    if stats["namespaces"].get(settings["namespace"], 0) == 0:
        st.error(
            f"Namespace '{settings['namespace']}' has no vectors. "
            "Ingest documents into this namespace first."
        )
        return

    with st.spinner("Embedding the query, searching Pinecone and generating the answer…"):
        try:
            retrieved = stage5_retrieve(
                question,
                index,
                namespace=settings["namespace"],
                top_k=settings["top_k"],
            )
            system_prompt, user_prompt = stage6_build_prompt(question, retrieved)
            answer, usage = stage7_generate(system_prompt, user_prompt)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            return

    st.session_state["last_answer"] = {
        "question": question,
        "answer": answer,
        "usage": usage,
        "retrieved": retrieved,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def render_ask_tab(settings: Dict[str, Any]) -> None:
    st.subheader("Ask questions")
    st.caption(
        f"Index '{settings['index_name']}' · namespace '{settings['namespace']}' · "
        f"top {settings['top_k']} chunks"
    )

    question = st.text_input(
        "Question",
        placeholder="What is RAG and why does it reduce hallucinations?",
    )

    if st.button("Generate answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not os.getenv("OPENROUTER_API_KEY") or not os.getenv("PINECONE_API_KEY"):
            st.error("Both OPENROUTER_API_KEY and PINECONE_API_KEY must be configured.")
        else:
            _run_query(question.strip(), settings)

    result = st.session_state.get("last_answer")
    if result:
        render_answer(result)


def main_gui() -> None:
    st.set_page_config(page_title="RAG Pipeline — Pinecone", layout="wide")
    st.title("RAG Pipeline — Pinecone")
    st.caption(
        "Upload documents, build a serverless Pinecone index, and ask grounded "
        "questions using the RAG pipeline."
    )

    settings = render_sidebar()

    ingest_tab, ask_tab = st.tabs(["Ingest documents", "Ask questions"])
    with ingest_tab:
        render_ingest_tab(settings)
    with ask_tab:
        render_ask_tab(settings)


# ─────────────────────────────────────────────────────────────────────────────
# CLI COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def _print_stats(stats: Dict[str, Any]) -> None:
    print(f"\n    [+] Total vectors : {stats['total_vector_count']:,}")
    print(f"    [+] Dimension     : {stats['dimension']}")
    if stats["namespaces"]:
        for name, count in stats["namespaces"].items():
            print(f"    [+] Namespace     : {name} ({count:,} vectors)")
    else:
        print("    [+] Namespace     : (empty)")


def cli_create_index(args: argparse.Namespace) -> int:
    banner("PINECONE — CREATE INDEX")

    if args.recreate:
        pc = get_pinecone_client()
        if pc.has_index(args.index):
            print(f"    [!] Deleting existing index '{args.index}' …")
            pc.delete_index(args.index)

    index, created = ensure_pinecone_index(args.index, args.cloud, args.region)
    print(
        f"    [+] Index '{args.index}' "
        f"{'created' if created else 'already exists'} "
        f"(dimension={EMBEDDING_DIM}, metric=cosine, cloud={args.cloud}, region={args.region})"
    )
    _print_stats(index_stats(index))
    return 0


def cli_ingest(args: argparse.Namespace) -> int:
    banner("INGEST DOCUMENTS INTO PINECONE")

    documents = list(SAMPLE_DOCUMENTS) if args.source == "sample" else read_pdf_files(args.path)
    if not documents:
        print(f"    [!] No documents found (source='{args.source}', path='{args.path}').")
        return 1

    result = ingest_documents(
        documents=documents,
        index_name=args.index,
        namespace=args.namespace,
        strategy=args.strategy,
        cloud=args.cloud,
        region=args.region,
        clear_first=not args.keep_existing,
        progress_cb=cli_progress,
    )

    print(f"\n    [+] Documents : {result['document_count']}")
    print(f"    [+] Chunks    : {result['chunk_count']}")
    print(f"    [+] Vectors   : {result['written']}")
    _print_stats(result["stats"])
    if not result["ready"]:
        print("    [!] Pinecone has not indexed every vector yet — queries may lag briefly.")
    return 0


def cli_ask(args: argparse.Namespace) -> int:
    index = connect_existing_index(args.index)

    banner("STAGE 5 — RETRIEVAL FROM PINECONE")
    retrieved = stage5_retrieve(
        args.question,
        index,
        namespace=args.namespace,
        top_k=args.top_k,
    )
    if not retrieved:
        print("    [!] No chunks retrieved. Is the namespace populated?")
        return 1

    print(f"\n  {'Rank':<6}{'Score':>8}   {'Source':<30}  Preview")
    print("  " + "-" * 70)
    for chunk in retrieved:
        preview = chunk["text"][:60].replace("\n", " ")
        print(
            f"  {chunk['retrieval_rank']:<6}{chunk['retrieval_score']:>8.4f}   "
            f"{chunk['source']:<30}  \"{preview}…\""
        )

    banner("STAGE 6 — PROMPT AUGMENTATION")
    system_prompt, user_prompt = stage6_build_prompt(args.question, retrieved)
    print_step("System prompt", system_prompt)
    print_step("User prompt preview (first 300 chars)", user_prompt[:300] + "…")

    banner("STAGE 7 — LLM GENERATION (ANSWER)")
    answer, usage = stage7_generate(system_prompt, user_prompt)
    display_answer(args.question, answer, usage, retrieved)
    return 0


def cli_stats(args: argparse.Namespace) -> int:
    banner("PINECONE INDEX STATS")
    index = connect_existing_index(args.index)
    _print_stats(index_stats(index))
    return 0


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Pinecone-backed RAG pipeline (GUI: streamlit run this file).",
    )
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", default=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME))
    common.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))
    common.add_argument("--cloud", default=os.getenv("PINECONE_CLOUD", DEFAULT_CLOUD))
    common.add_argument("--region", default=os.getenv("PINECONE_REGION", DEFAULT_REGION))

    create_parser = subparsers.add_parser(
        "create-index", parents=[common], help="Create the Pinecone index if it is missing."
    )
    create_parser.add_argument(
        "--recreate", action="store_true", help="Delete the existing index first."
    )

    ingest_parser = subparsers.add_parser(
        "ingest", parents=[common], help="Chunk, embed and upsert documents."
    )
    ingest_parser.add_argument("--source", choices=["sample", "docs"], default="sample")
    ingest_parser.add_argument("--path", default="docs", help="Folder to read PDFs from.")
    ingest_parser.add_argument(
        "--strategy", choices=sorted(STRATEGY_FUNCTIONS), default="paragraph"
    )
    ingest_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the namespace before upserting.",
    )

    ask_parser = subparsers.add_parser(
        "ask", parents=[common], help="Run retrieval and generation for one question."
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=TOP_K)

    subparsers.add_parser("stats", parents=[common], help="Print index statistics.")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "create-index":
        return cli_create_index(args)
    if args.command == "ingest":
        return cli_ingest(args)
    if args.command == "ask":
        return cli_ask(args)
    return cli_stats(args)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — Streamlit GUI or CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if st.runtime.exists():
        main_gui()
    else:
        sys.exit(main_cli())
