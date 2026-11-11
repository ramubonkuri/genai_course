"""
=============================================================================
  Week 3: RAG (Retrieval-Augmented Generation) Pipeline — Step by Step
=============================================================================

This script walks through every stage of a RAG application:

  STAGE 1 — INGESTION       : Load raw documents into memory
  STAGE 2 — CHUNKING        : Split documents using different strategies
  STAGE 3 — EMBEDDING       : Convert chunks to vectors via OpenRouter
  STAGE 4 — INDEXING        : Store vectors in a FAISS index
  STAGE 5 — RETRIEVAL       : Query the index and get top-k chunks
  STAGE 6 — AUGMENTATION    : Build a prompt from retrieved chunks
  STAGE 7 — GENERATION      : Call the LLM via OpenRouter and return answer

Models used (via OpenRouter):
  Embedding : openai/text-embedding-3-small  (1536-dim)
  LLM       : openai/gpt-4o-mini

Vector DB  : FAISS (facebook/faiss — local, in-memory)
install dependencies: pip install openai faiss-cpu python-dotenv tiktoken openai numpy
=============================================================================
"""

import os
import re
import sys
import textwrap
import time
from typing import List, Dict, Tuple

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("ERROR: OPENROUTER_API_KEY not found in environment / .env file.")

# OpenRouter base URL — same client handles both embeddings and chat
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM   = 1536          # dimensions produced by text-embedding-3-small
LLM_MODEL       = "openai/gpt-4o-mini"

# Retrieval: how many chunks to pull back from FAISS
TOP_K = 3

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
        to query millions of vectors in milliseconds. FAISS (Facebook AI Similarity Search)
        is a highly optimised open-source library for this task. It supports multiple index
        types: Flat (exact), IVF (inverted file, approximate), HNSW (graph-based), and PQ
        (product quantization for compression). For small corpora a FlatL2 or FlatIP index
        gives exact results with no configuration overhead. For production workloads with
        millions of records, IVF + PQ is the standard choice.
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


SAMPLE_DOCUMENTS = [doc for doc in SAMPLE_DOCUMENTS if doc["text"].strip()]  # remove empty docs



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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


def read_pdf_files(folder_path: str = "docs") -> List[Dict[str, str]]:
    """Read every PDF in a folder and return a list of document dicts.

    Each dict is normalized to the same schema used by the rest of the RAG
    pipeline: {"source": filename, "text": extracted_text}.
    """
    if not os.path.isdir(folder_path):
        print(f"    [!] PDF folder '{folder_path}' not found. No PDFs loaded.")
        return []

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF reading requires the 'pypdf' package. Install it with: "
            "pip install pypdf"
        ) from exc

    documents: List[Dict[str, str]] = []
    pdf_files = sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"    [!] No PDF files found in '{folder_path}'.")
        return documents

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
            print(f"    [+] Loaded PDF '{filename}' — {len(text)} chars")
        else:
            print(f"    [!] PDF '{filename}' contained no readable text.")

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
    banner("STAGE 1 — DOCUMENT INGESTION")
    print_step(
        "What happens here?",
        "Raw content is loaded from its original source and normalised into "
        "a list of document objects. Metadata (filename, URL, creation date) "
        "is preserved alongside the text so it can be surfaced later.",
    )

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
        loaded.append({"source": doc["source"], "text": clean_text})
        print(f"    [+] Loaded '{doc['source']}' — {len(clean_text)} chars")

    print(f"\n  ✔  {len(loaded)} documents ingested.")
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CHUNKING
# ─────────────────────────────────────────────────────────────────────────────


class ChunkingStrategy:
    """Collection of chunking strategies, each demonstrated in turn."""

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
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Group sentences into chunks of ~2-3 sentences
        chunk_size = 2
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
        This is what LangChain's RecursiveCharacterTextSplitter implements.

        The guarantee is enforced by recursing into any part that is itself
        larger than max_chars, rather than emitting it as-is.
        """
        if len(text) <= max_chars:
            return [text]
        # Try separators from coarsest to finest
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
                        # Recurse into oversized parts instead of emitting them
                        # as-is — this is what actually guarantees the max_chars
                        # limit holds for every chunk in the output.
                        sub = ChunkingStrategy.recursive(part, max_chars)
                        chunks.extend(sub[:-1])
                        current = sub[-1] if sub else ""
                if current:
                    chunks.append(current)
                return [c for c in chunks if c]
        # Hard fallback: no separator found — slice at the byte boundary
        return [text[:max_chars], text[max_chars:]]


def stage2_chunk(documents: List[Dict]) -> List[Dict]:
    """
    Demonstrate all four strategies on the first document, then apply
    paragraph-based chunking to the full corpus (the best default).
    Returns a flat list of chunk dicts:
        { 'source', 'chunk_index', 'text', 'strategy' }
    """
    banner("STAGE 2 — CHUNKING STRATEGIES")

    strategies = {
        "Fixed-size (char)": ChunkingStrategy.fixed_size,
        "Sentence-based   ": ChunkingStrategy.sentence_based,
        "Paragraph-based  ": ChunkingStrategy.paragraph_based,
        "Recursive        ": ChunkingStrategy.recursive,
    }

    sample_text = documents[0]["text"]

    print_step(
        "Demonstrating chunking strategies on the first document",
        f"Document: '{documents[0]['source']}' ({len(sample_text)} chars)",
    )
    print(
        "  Preview note: chunks are printed in source order with zero-based "
        "indices; only the first two previews are shown."
    )
    print(
        "  Each preview is capped at 90 characters. Fixed-size chunks overlap "
        "by 40 characters and may split words at character boundaries."
    )
    print(
        "  Paragraph-based splitting looks for blank lines, which ingestion "
        "now preserves."
    )

    for name, fn in strategies.items():
        chunks = fn(sample_text)
        print(f"\n  ── {name.strip()}: {len(chunks)} chunk(s)")
        for i, c in enumerate(chunks[:2]):  # show first 2 chunks as preview
            preview = c[:90].replace("\n", " ")
            if len(c) > 90:
                preview += "… [preview truncated]"
            print(f"     Chunk {i} ({len(c)} chars): \"{preview}\"")
        if len(chunks) > 2:
            print(f"     … {len(chunks) - 2} more chunk(s) not shown")

    # ── Apply paragraph-based chunking to the full corpus ──────────────────
    print_step(
        "Applying PARAGRAPH-BASED chunking to all documents",
        "Paragraph-based chunking preserves natural semantic boundaries and "
        "works well with the structured text in this corpus.",
    )

    all_chunks: List[Dict] = []
    for doc in documents:
        raw_chunks = ChunkingStrategy.paragraph_based(doc["text"])
        for idx, chunk_text in enumerate(raw_chunks):
            all_chunks.append(
                {
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": chunk_text,
                    "strategy": "paragraph",
                }
            )
        print(f"    [+] '{doc['source']}' → {len(raw_chunks)} chunk(s)")

    print(f"\n  ✔  Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts: List[str], batch_size: int = 20) -> np.ndarray:
    """
    Call OpenRouter's embeddings endpoint in batches.
    Returns an (N, EMBEDDING_DIM) float32 numpy array.

    OpenRouter endpoint : POST https://openrouter.ai/api/v1/embeddings
    Compatible with the OpenAI Python SDK — just set base_url.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_vectors = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vectors)
        # Polite rate-limit buffer between batches
        if i + batch_size < len(texts):
            time.sleep(0.3)

    return np.array(all_embeddings, dtype=np.float32)


def stage3_embed(chunks: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
    """
    Embed every chunk using the OpenRouter embeddings API.
    Returns the updated chunk list and the embedding matrix.
    """
    banner("STAGE 3 — EMBEDDING CHUNKS → VECTORS")

    print_step(
        "What is an embedding?",
        f"Each chunk is passed through '{EMBEDDING_MODEL}' which maps it to "
        f"a {EMBEDDING_DIM}-dimensional float vector. Semantically similar "
        "chunks end up close together in this high-dimensional space, enabling "
        "similarity-based retrieval.",
    )
    print_step(
        f"Embedding {len(chunks)} chunks via OpenRouter …",
        "Batching requests to respect rate limits.",
    )

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    # Attach the embedding to each chunk dict for reference
    for i, chunk in enumerate(chunks):
        chunk["embedding_preview"] = embeddings[i, :4].tolist()  # first 4 dims

    print(f"\n  ✔  Embedding matrix shape : {embeddings.shape}")
    print(f"  ✔  dtype                  : {embeddings.dtype}")
    print(f"  ✔  Sample vector (first 4 dims of chunk 0): "
          f"{embeddings[0, :4].tolist()}")
    return chunks, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — INDEXING (FAISS)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from the embedding matrix.

    Index type chosen: IndexFlatIP (Inner Product / cosine similarity).
    We L2-normalise the vectors first so that inner product == cosine similarity.
    For large corpora (>100k docs), swap in IndexIVFFlat or IndexHNSWFlat.
    """
    banner("STAGE 4 — INDEXING VECTORS IN FAISS")

    print_step(
        "Why FAISS?",
        "FAISS (Facebook AI Similarity Search) provides extremely fast "
        "nearest-neighbour search over float32 vectors. It runs entirely "
        "in-memory with no external server required — ideal for prototyping "
        "and small-to-medium datasets.",
    )
    print_step(
        "Index type: IndexFlatIP (exact cosine similarity)",
        "We L2-normalise all vectors before indexing so that the inner "
        "product equals cosine similarity. 'Flat' means no approximation — "
        "every query does a brute-force scan, giving exact results.",
    )

    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalised = embeddings / np.maximum(norms, 1e-10)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(normalised)

    print(f"\n  ✔  Vectors added to index : {index.ntotal}")
    print(f"  ✔  Index dimension        : {index.d}")
    print(f"  ✔  Normalised for cosine similarity")
    return index, normalised


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def stage5_retrieve(
    query: str,
    index: faiss.Index,
    normalised_embeddings: np.ndarray,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    1. Embed the user query with the same model used for indexing.
    2. L2-normalise the query vector.
    3. Search the FAISS index for the top-k nearest neighbours.
    4. Return the matching chunk dicts with their similarity scores.
    """
    banner("STAGE 5 — RETRIEVAL FROM FAISS VECTOR DB")

    print_step(
        "Step 5a — Embed the user query",
        f"Query: \"{query}\"  →  converted to a {EMBEDDING_DIM}-dim vector "
        f"using the same '{EMBEDDING_MODEL}' model.",
    )

    query_vec = get_embeddings([query])              # shape (1, 1536)
    query_norm = query_vec / np.maximum(
        np.linalg.norm(query_vec, axis=1, keepdims=True), 1e-10
    )

    print_step(
        f"Step 5b — Search FAISS index for top-{top_k} chunks",
        "FAISS returns (scores, indices).  Score = cosine similarity in [−1, 1]. "
        "Higher is more relevant.",
    )

    scores, indices = index.search(query_norm, top_k)   # both shape (1, top_k)

    retrieved = []
    print(f"\n  {'Rank':<6}{'Score':>8}   {'Source':<30}  Preview")
    print("  " + "-" * 70)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        chunk = chunks[idx].copy()
        chunk["retrieval_score"] = float(score)
        chunk["retrieval_rank"]  = rank
        retrieved.append(chunk)
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"  {rank:<6}{score:>8.4f}   {chunk['source']:<30}  \"{preview}…\"")

    print(f"\n  ✔  Retrieved {len(retrieved)} chunks.")
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
    banner("STAGE 6 — PROMPT AUGMENTATION")

    print_step(
        "Why augment the prompt?",
        "The LLM has no knowledge of the documents we just indexed — they "
        "may be private, domain-specific, or newer than its training cut-off. "
        "By injecting the retrieved chunks directly into the prompt we give "
        "the model the exact facts it needs to answer accurately.",
    )

    # Build the context block
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

    print_step("System prompt", system_prompt)
    print_step(
        f"User prompt preview (first 300 chars)",
        user_prompt[:300] + "…",
    )
    print(f"\n  ✔  Total context tokens ≈ {len(user_prompt.split()):,} words "
          f"({len(user_prompt):,} chars)")

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — GENERATION (LLM CALL)
# ─────────────────────────────────────────────────────────────────────────────

def stage7_generate(system_prompt: str, user_prompt: str) -> str:
    """
    Call the LLM via OpenRouter's chat completions endpoint.
    Temperature is kept low (0.1) for factual, deterministic answers.
    """
    banner("STAGE 7 — LLM GENERATION (ANSWER)")

    print_step(
        f"Calling '{LLM_MODEL}' via OpenRouter …",
        "The model receives the system instruction + the augmented user "
        "prompt containing all retrieved chunks as context.",
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()

    usage = response.usage
    print(f"\n  ── Token usage ──────────────────────────────────────────────")
    print(f"     Prompt tokens    : {usage.prompt_tokens:,}")
    print(f"     Completion tokens: {usage.completion_tokens:,}")
    print(f"     Total tokens     : {usage.total_tokens:,}")

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# FINAL ANSWER DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_answer(query: str, answer: str, retrieved_chunks: List[Dict]) -> None:
    """Pretty-print the full RAG result to the user."""
    banner("FINAL ANSWER")

    print(f"\n  Question : {query}\n")
    print("  " + "─" * 68)
    print()
    for line in textwrap.wrap(answer, width=68):
        print(f"  {line}")
    print()
    print("  " + "─" * 68)
    print("\n  Sources used:")
    for chunk in retrieved_chunks:
        print(
            f"    • {chunk['source']}  "
            f"(chunk #{chunk['chunk_index']}, "
            f"similarity={chunk['retrieval_score']:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(query: str) -> None:
    """
    Execute all 7 stages of the RAG pipeline for a given query.
    """
    print("\n" + "█" * 72)
    print("  RAG PIPELINE — END TO END WALKTHROUGH")
    print("█" * 72)
    print(f"\n  Query : \"{query}\"")
    print(f"  LLM   : {LLM_MODEL}")
    print(f"  Embed : {EMBEDDING_MODEL}  ({EMBEDDING_DIM}-dim)")
    print(f"  VDB   : FAISS IndexFlatIP")
    print(f"  Top-K : {TOP_K}")

    # Stage 1 — Ingest
    pdf_documents = read_pdf_files("docs")
    documents = stage1_ingest(pdf_documents if pdf_documents else SAMPLE_DOCUMENTS)

    # Stage 2 — Chunk
    chunks = stage2_chunk(documents)

    # Stage 3 — Embed
    chunks, embeddings = stage3_embed(chunks)

    # Stage 4 — Index
    faiss_index, normalised_embeddings = stage4_index(embeddings)

    # Stage 5 — Retrieve
    retrieved_chunks = stage5_retrieve(
        query, faiss_index, normalised_embeddings, chunks
    )

    # Stage 6 — Build augmented prompt
    system_prompt, user_prompt = stage6_build_prompt(query, retrieved_chunks)

    # Stage 7 — Generate answer
    answer = stage7_generate(system_prompt, user_prompt)

    # Display
    display_answer(query, answer, retrieved_chunks)

    print("\n" + "█" * 72)
    print("  Pipeline complete.")
    print("█" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run multiple example queries to showcase the pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "What is RAG and why does it reduce hallucinations?",
        "What are the different chunking strategies and when should I use each?",
        "How does FAISS store and search vectors efficiently?",
    ]

    for i, q in enumerate(queries, start=1):
        print(f"\n\n{'▓' * 72}")
        print(f"  EXAMPLE {i} of {len(queries)}")
        print(f"{'▓' * 72}")
        run_rag_pipeline(q)
        if i < len(queries):
            print("\n  [Pausing 2 s before next query to respect rate limits…]")
            time.sleep(2)
