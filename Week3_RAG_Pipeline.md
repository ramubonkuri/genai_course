# Week 3 — RAG Pipeline: Step by Step

> **Script:** `week3_rag_pipeline.py`  
> **Models (via OpenRouter):** `openai/text-embedding-3-small` (1536-dim) · `openai/gpt-4o-mini`  
> **Vector DB:** FAISS (`IndexFlatIP`, in-memory)

---

## Overview

RAG (Retrieval-Augmented Generation) lets an LLM answer questions using documents it was never trained on. Instead of relying on baked-in weights, the system retrieves the most relevant pieces of your private/domain knowledge at query time and injects them directly into the prompt.

The pipeline has **7 stages**:

```
Documents → Ingest → Chunk → Embed → Index → Retrieve → Augment → Generate → Answer
```

---

## Configuration

```python
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM   = 1536
LLM_MODEL       = "openai/gpt-4o-mini"
TOP_K           = 3        # chunks returned per query
```

The `OpenAI` Python client is pointed at OpenRouter's base URL. The same client instance handles both embeddings and chat completions — no extra SDK needed.

```python
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
```

---

## Stage 1 — Document Ingestion

**Function:** `stage1_ingest(documents)`

Loads raw text and normalises it into a list of document dicts:
```python
{ "source": "doc_rag_overview.txt", "text": "..." }
```

In production, this step reads from:
- Local files — PDF, DOCX, Markdown, plain text
- Databases — SQL, MongoDB
- APIs — Confluence, Notion, SharePoint
- Web scrapers

The implementation collapses internal whitespace inside each paragraph but **preserves blank-line separators** between paragraphs, so the chunker can detect natural boundaries later.

---

## Stage 2 — Chunking Strategies

**Class:** `ChunkingStrategy`  
**Function:** `stage2_chunk(documents)`

Chunking splits long documents into smaller overlapping pieces before embedding. The script demonstrates **four strategies** on the first document, then applies paragraph-based chunking to the full corpus.

### 1. Fixed-Size (Character)

```python
ChunkingStrategy.fixed_size(text, chunk_size=200, overlap=40)
```

- Splits every `chunk_size` characters with an `overlap` sliding window.
- Simple, predictable, uniform retrieval latency.
- **Downside:** can split mid-sentence or mid-word.
- Best for: quick prototyping, uniform-length pipelines.

### 2. Sentence-Based

```python
ChunkingStrategy.sentence_based(text)
```

- Splits on sentence boundaries (`. `, `! `, `? `) using a regex lookbehind.
- Groups 2 sentences per chunk.
- **Preserves complete sentences** → better semantic coherence.
- Best for: factual documents, Q&A corpora.

### 3. Paragraph-Based ✅ *(used for indexing)*

```python
ChunkingStrategy.paragraph_based(text)
```

- Splits on double newlines (`\n\n`).
- Natural semantic units with minimal config.
- **Best default** for structured text.
- Best for: articles, documentation, reports.

### 4. Recursive

```python
ChunkingStrategy.recursive(text, max_chars=300)
```

- Tries separators from coarsest to finest: `\n\n` → `. ` → ` `
- Groups parts greedily until the next addition would exceed `max_chars`.
- **Recurses into any part that is itself larger than `max_chars`**, so the limit holds for every chunk in the output — not just at the top level.
- Falls back to a hard character slice only when no separator exists at all.
- Best for: mixed-format documents where paragraph splits alone are too large.

**How the guarantee works:**

```
text (600 chars)
 └─ split on "\n\n" → [part_A (120 chars), part_B (480 chars)]
                                               │
                                           > max_chars → recurse
                                            └─ split on ". " → [chunk1, chunk2, ...]
```

Without recursion, `part_B` would be emitted as a 480-char chunk even though `max_chars=300`. With recursion, it keeps splitting until every leaf is ≤ `max_chars`.

### Chunk Overlap

All size-bounded strategies add overlap (10–20% of chunk size) so context at boundaries is not lost between adjacent chunks.

| Strategy | Unit | Overlap | Coherence | Config |
|---|---|---|---|---|
| Fixed-size | characters | ✅ yes | ⚠️ low | minimal |
| Sentence | sentences | ❌ no | ✅ high | minimal |
| Paragraph | paragraphs | ❌ no | ✅ high | none |
| Recursive | adaptive | ✅ yes | ✅ high | max_chars |

> Typical production chunk sizes: **256–1024 tokens**.

Each chunk is stored as:
```python
{
  "source": "doc_chunking.txt",
  "chunk_index": 0,
  "text": "...",
  "strategy": "paragraph"
}
```

---

## Stage 3 — Embedding

**Function:** `get_embeddings(texts, batch_size=20)`  
**Function:** `stage3_embed(chunks)`

Every chunk is converted to a **1536-dimensional float32 vector** using `openai/text-embedding-3-small` via OpenRouter's embeddings endpoint:

```
POST https://openrouter.ai/api/v1/embeddings
```

```python
response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=["chunk text 1", "chunk text 2", ...],
)
```

Chunks are sent in batches of 20 with a 0.3 s pause between batches to respect rate limits.

**Output:** `numpy.ndarray` of shape `(N, 1536)`, dtype `float32`.

### Why embeddings?

Embeddings place semantically similar text close together in vector space. "RAG reduces hallucinations" and "RAG grounds answers in retrieved context" will have very similar vectors even though they share no words. This makes similarity search far more powerful than keyword matching.

---

## Stage 4 — Indexing (FAISS)

**Function:** `stage4_index(embeddings)`

Builds a FAISS index from the embedding matrix.

```python
# L2-normalise so dot product == cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalised = embeddings / np.maximum(norms, 1e-10)

index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(normalised)
```

### Index type: `IndexFlatIP`

| Property | Value |
|---|---|
| Type | Flat (brute-force exact search) |
| Distance metric | Inner product (= cosine similarity after L2-norm) |
| Approximation | None — exact results |
| Best for | Corpora up to ~100k vectors |

### FAISS Index Types at a Glance

| Index | Approximate? | Speed | Memory | Use case |
|---|---|---|---|---|
| `IndexFlatL2` | ❌ | slow | high | Small corpus, exact L2 |
| `IndexFlatIP` | ❌ | slow | high | Small corpus, cosine sim |
| `IndexIVFFlat` | ✅ | fast | medium | Medium corpus |
| `IndexHNSWFlat` | ✅ | very fast | high | Low-latency production |
| `IndexIVFPQ` | ✅ | fast | **low** | Large-scale production |

For production with millions of vectors, swap in `IndexIVFPQ`.

---

## Stage 5 — Retrieval

**Function:** `stage5_retrieve(query, index, normalised_embeddings, chunks, top_k=3)`

### Step 5a — Embed the query

The user's question is embedded with the **same model** used for indexing. Consistency is critical — mixing models breaks similarity.

```python
query_vec = get_embeddings([query])          # shape (1, 1536)
query_norm = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
```

### Step 5b — Search FAISS

```python
scores, indices = index.search(query_norm, top_k)
# scores : cosine similarity, range [-1, 1], higher = more relevant
# indices: integer positions in the chunks list
```

### Output

A ranked list of the top-k chunk dicts, each augmented with:
```python
{
  ...,
  "retrieval_score": 0.8921,   # cosine similarity
  "retrieval_rank":  1,
}
```

Sample output for the query *"What is RAG?"*:
```
  Rank    Score   Source                          Preview
  ──────────────────────────────────────────────────────────────
  1      0.9134   doc_rag_overview.txt            "Retrieval-Augmented Generation (RAG) is…"
  2      0.8612   doc_prompt_engineering.txt      "Prompt engineering in RAG involves…"
  3      0.8201   doc_embeddings.txt              "Embeddings are dense, fixed-length…"
```

---

## Stage 6 — Prompt Augmentation

**Function:** `stage6_build_prompt(query, retrieved_chunks)`

The retrieved chunks are injected into the prompt so the LLM has the exact facts it needs. Without this step, the model would rely only on its training weights — which may be outdated, generic, or missing domain knowledge entirely.

### Prompt structure

**System prompt** — strict grounding instruction:
```
You are a precise and helpful assistant.
Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, respond with:
'I don't have enough information to answer this question.'
Cite the source file name when possible.
```

**User prompt** — context block + question:
```
CONTEXT:

[Source: doc_rag_overview.txt | Chunk #0 | Similarity: 0.9134]
Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by...

---

[Source: doc_prompt_engineering.txt | Chunk #0 | Similarity: 0.8612]
Prompt engineering in RAG involves carefully constructing the final prompt...

---

QUESTION: What is RAG and why does it reduce hallucinations?
```

### Design choices

- **Source labels** on each chunk allow the LLM to cite the file it used.
- **Similarity scores** are included for transparency and debugging.
- **Temperature = 0.1** keeps the answer deterministic and factual.
- The system instruction prevents the model from going "off-script" and adding information not in the context.

---

## Stage 7 — LLM Generation

**Function:** `stage7_generate(system_prompt, user_prompt)`

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=0.1,
)
answer = response.choices[0].message.content.strip()
```

Token usage is printed after each call:
```
  Prompt tokens    : 712
  Completion tokens:  98
  Total tokens     : 810
```

---

## Final Answer

The answer is displayed alongside the sources that were used:

```
  Question : What is RAG and why does it reduce hallucinations?

  ────────────────────────────────────────────────────────────────────
  RAG (Retrieval-Augmented Generation) enhances LLM responses by
  injecting relevant external knowledge into the prompt at inference
  time. It reduces hallucinations because the LLM is instructed to
  answer using only the retrieved context, keeping responses grounded
  in real, verifiable information rather than learned statistical
  patterns. (Source: doc_rag_overview.txt)
  ────────────────────────────────────────────────────────────────────

  Sources used:
    • doc_rag_overview.txt       (chunk #0, similarity=0.9134)
    • doc_prompt_engineering.txt (chunk #0, similarity=0.8612)
    • doc_embeddings.txt         (chunk #0, similarity=0.8201)
```

---

## Full Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                                │
├────────┬────────────────────────┬──────────────────────────────┤
│ Stage  │ What                   │ How (this script)            │
├────────┼────────────────────────┼──────────────────────────────┤
│   1    │ Ingest documents       │ Normalise whitespace, keep   │
│        │                        │ paragraph separators         │
├────────┼────────────────────────┼──────────────────────────────┤
│   2    │ Chunk documents        │ 4 strategies shown;          │
│        │                        │ paragraph-based applied      │
├────────┼────────────────────────┼──────────────────────────────┤
│   3    │ Embed chunks           │ OpenRouter text-embedding-   │
│        │                        │ 3-small → (N × 1536) matrix  │
├────────┼────────────────────────┼──────────────────────────────┤
│   4    │ Index in FAISS         │ L2-norm + IndexFlatIP        │
│        │                        │ (exact cosine similarity)    │
├────────┼────────────────────────┼──────────────────────────────┤
│   5    │ Retrieve top-k chunks  │ Embed query → FAISS search   │
│        │                        │ → ranked chunks with scores  │
├────────┼────────────────────────┼──────────────────────────────┤
│   6    │ Augment prompt         │ System grounding + context   │
│        │                        │ block + user question        │
├────────┼────────────────────────┼──────────────────────────────┤
│   7    │ Generate answer        │ OpenRouter gpt-4o-mini,      │
│        │                        │ temperature=0.1              │
└────────┴────────────────────────┴──────────────────────────────┘
```

---

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (3 example queries)
python week3_rag_pipeline.py
```

**Dependencies added for Week 3:**

| Package | Purpose |
|---|---|
| `faiss-cpu>=1.12.0` | Local in-memory vector index |
| `tiktoken>=0.6.0` | Token counting for OpenAI models |

**Required `.env`:**
```
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Key Concepts to Remember

- **Ingest once, query many times.** The FAISS index is built once at startup. Only retrieval (Stage 5 onwards) runs per query.
- **Embedding model consistency.** Documents and queries must use the exact same embedding model or similarity scores are meaningless.
- **Chunk size vs. retrieval quality.** Smaller chunks → more precise retrieval. Larger chunks → more context per result. Start with 512 tokens and tune from there.
- **Top-k is a tradeoff.** More chunks = more context but also more noise and higher token cost. 3–5 is a good starting range.
- **Temperature for RAG.** Set low (0.0–0.2) for factual Q&A. Higher temperatures make the model less grounded in the retrieved context.
- **Source attribution.** Always label chunks with their origin. It makes debugging retrieval failures much easier and lets the LLM cite sources in its answer.
