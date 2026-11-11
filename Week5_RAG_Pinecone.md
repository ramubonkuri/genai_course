# Week 5 — RAG Pipeline with Pinecone + Streamlit

> **Script:** `Week5_RAG_Pinecone.py`
> **Models (via OpenRouter):** `openai/text-embedding-3-small` (1536-dim) · `openai/gpt-4o-mini`
> **Vector DB:** Pinecone (serverless, cosine similarity)
> **UI:** Streamlit — ingestion and Q&A

---

## Overview

Week 5 rebuilds the Week 3 RAG pipeline on a managed vector database and wraps it
in a browser UI. The pipeline stages are unchanged; only the storage and interface
layers are new:

```
Documents → Ingest → Chunk → Embed → Index (Pinecone) → Retrieve → Augment → Generate → Answer
```

Two entry points share the same stage functions:

| Mode | Command | Purpose |
|---|---|---|
| GUI | `streamlit run Week5_RAG_Pinecone.py` | Upload documents, create the index, ask questions |
| CLI | `python Week5_RAG_Pinecone.py <command>` | Headless index setup, ingestion, querying, stats |

The same file serves both. At startup, `st.runtime.exists()` decides whether to start
the Streamlit UI or parse command-line arguments, so CLI flags never collide with
Streamlit's own arguments.

---

## Configuration

```python
EMBEDDING_MODEL   = "openai/text-embedding-3-small"
EMBEDDING_DIM     = 1536
LLM_MODEL         = "openai/gpt-4o-mini"
TOP_K             = 3
EMBED_BATCH_SIZE  = 20      # chunks per embeddings request
UPSERT_BATCH_SIZE = 100     # vectors per Pinecone upsert request
```

Defaults are overridable through environment variables:

| Setting | Code default | Env var |
|---|---|---|
| Index name | `rag-week5` | `PINECONE_INDEX_NAME` |
| Namespace | `Default` | `PINECONE_NAMESPACE` |
| Cloud | `aws` | `PINECONE_CLOUD` |
| Region | `us-east-1` | `PINECONE_REGION` |

Required secrets:

```env
OPENROUTER_API_KEY=sk-or-v1-...
PINECONE_API_KEY=...
```

The OpenRouter SDK client is reused for both embeddings and chat completions. Both
clients are cached per API key so repeated Streamlit reruns do not create new HTTP
sessions.

> Namespaces are **case-sensitive** in Pinecone. `Default` and `default` are
> different partitions. The app passes the namespace explicitly on every upsert and
> query; its configured default is the name `Default`.

---

## Stage 1 — Document Ingestion

**Function:** `stage1_ingest(documents)`

Documents arrive as `{"source": str, "text": str}` from one of three sources:

| Source | Reader |
|---|---|
| Uploads (PDF / TXT / MD) | `read_uploaded_documents` |
| Built-in sample corpus | `SAMPLE_DOCUMENTS` |
| `./docs` folder PDFs | `read_pdf_files` |

Normalisation collapses line wrapping **inside** paragraphs while preserving the
blank-line separators between them:

```python
paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
clean_text = "\n\n".join(paragraphs)
```

Keeping the blank lines matters: paragraph-based chunking in Stage 2 depends on
them. Empty documents are dropped.

---

## Stage 2 — Chunking Strategies

**Class:** `ChunkingStrategy`  
**Function:** `stage2_chunk(documents, strategy="paragraph")`

Four strategies are available in code, GUI and CLI:

| Strategy | Key | Unit | Behaviour |
|---|---|---|---|
| Paragraph-based | `paragraph` | `\n\n` | Default; natural semantic units |
| Recursive | `recursive` | adaptive | `\n\n` → `. ` → ` `, 300-char cap |
| Sentence-based | `sentence` | sentences | 2 sentences per chunk |
| Fixed-size | `fixed` | characters | 200 chars, 40-char overlap |

### Recursive splitting

`recursive` tries separators from coarsest to finest and packs parts greedily until
`max_chars` would be exceeded. Any part that is itself larger than `max_chars` is
split **recursively**, so the size cap holds for every emitted chunk — not just at
the top level. A hard character slice is only used when no separator exists.

### Chunk records

Every chunk is stored as:

```python
{
  "source": "doc_chunking.txt",
  "chunk_index": 0,
  "text": "...",
  "strategy": "paragraph",
}
```

With paragraph chunking, each sample document is a single paragraph, so the
built-in corpus produces **6 chunks** (one per document). Selecting a different
strategy changes both chunk count and chunk boundaries.

---

## Stage 3 — Embedding

**Functions:** `get_embeddings(texts, batch_size=20)` · `stage3_embed(chunks)`

Each chunk is sent to OpenRouter's embeddings endpoint:

```
POST https://openrouter.ai/api/v1/embeddings
```

Requests are batched (20 chunks) with a 0.3 s pause between batches to stay within
rate limits. Output is a `numpy` array of shape `(N, 1536)`, dtype `float32`.

Embeddings place semantically similar text close together, so "vector database"
and "Pinecone namespace" can match even without shared keywords.

---

## Stage 4 — Indexing in Pinecone

**Functions:** `ensure_pinecone_index` · `stage4_upsert` · `index_stats`

### Creating the index

`ensure_pinecone_index` creates a serverless index if it does not exist:

```python
pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

It then waits (up to 120 s by default) for the index to report `ready`. Two guards
protect the pipeline from silent mistakes:

- **Dimension** must equal 1536, otherwise the index cannot store these embeddings.
- **Metric** must be `cosine`; retrieval scores are interpreted as cosine similarity.

Attempting to use an index with a different dimension or metric raises a clear
error rather than producing meaningless results.

### Vector records

Each chunk becomes one Pinecone vector:

```python
{
  "id": "doc_chunking.txt::chunk-0",
  "values": [0.0123, ...],          # 1536 floats
  "metadata": {
    "source": "doc_chunking.txt",
    "chunk_index": 0,
    "text": "...",
    "strategy": "paragraph",
  },
}
```

Design decisions:

- **Text lives in metadata.** Retrieval returns the chunk content directly, with no
  secondary database lookup.
- **Deterministic ids.** `make_chunk_id(source, chunk_index)` lowercases and
  slugifies the filename, then appends `::chunk-N`. Re-ingesting the same document
  overwrites its vectors instead of duplicating them.
- **Provenance is preserved.** Source, chunk index and strategy travel with every
  vector, which powers citations and debugging in the UI.

### Batching and visibility

Vectors are upserted in batches of 100. Serverless indexes are eventually
consistent, so `wait_for_vectors` polls the namespace count for up to 60 s before
the ingestion is reported complete.

### Clearing a namespace

With `clear_first=True` (the GUI default), the namespace is emptied before writing
so it contains exactly the documents just ingested:

```python
index.delete(delete_all=True, namespace=namespace)
```

A namespace that has never existed returns a `404 Namespace not found` error from
Pinecone. `clear_namespace` treats that as "already empty" and continues; other
errors are re-raised.

### Ingestion orchestration

`ingest_documents` runs Stages 1-4 in one call and reports progress:

| Progress | Step |
|---|---|
| 2% | Normalise documents |
| 8% | Chunk documents |
| 10-65% | Embed chunks |
| 68% | Create/connect index |
| 72% | Clear namespace (optional) |
| 75-95% | Upsert vectors |
| 97% | Wait until vectors are queryable |
| 100% | Done |

It returns counts, the chunk list, whether the index was created, whether the
vectors are queryable yet, and a fresh stats snapshot.

---

## Stage 5 — Retrieval

**Function:** `stage5_retrieve(query, index, namespace, top_k=3)`

```python
query_vec = get_embeddings([query])[0]

response = index.query(
    vector=query_vec.tolist(),
    top_k=top_k,
    namespace=namespace,
    include_metadata=True,
)
```

Matches are converted back into the pipeline's chunk shape with two additions:

```python
{
  ...,
  "retrieval_score": 0.7269,   # cosine similarity, higher is better
  "retrieval_rank":  1,
}
```

Pinecone normalises vectors internally for the cosine metric, so no manual L2
normalisation is needed (unlike the FAISS `IndexFlatIP` setup in Week 3).

If the namespace holds fewer vectors than `top_k`, Pinecone simply returns fewer
matches.

---

## Stage 6 — Prompt Augmentation

**Function:** `stage6_build_prompt(query, retrieved_chunks)`

**System prompt** — grounding instruction:

```
You are a precise and helpful assistant.
Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, respond with:
'I don't have enough information to answer this question.'
Cite the source file name when possible.
```

**User prompt** — labelled context blocks plus the question:

```
CONTEXT:

[Source: doc_vector_databases.txt | Chunk #0 | Similarity: 0.7269]
Vector databases are purpose-built storage systems...

---

[Source: doc_embeddings.txt | Chunk #0 | Similarity: 0.4134]
Embeddings are dense, fixed-length numeric vectors...

---

QUESTION: How does Pinecone store and search vectors?
```

Each block includes the source filename, chunk index and similarity score, so the
model can attribute claims and the developer can inspect retrieval quality.

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
```

The function returns `(answer, usage)` with prompt, completion and total token
counts. `temperature=0.1` keeps answers factual and close to the retrieved context.

---

## Streamlit GUI

### Sidebar

- **Connections** — shows whether `OPENROUTER_API_KEY` and `PINECONE_API_KEY` are
  configured. If a key is missing, a password field accepts one for the current
  session and sets it in the process environment.
- **Pinecone index** — index name, namespace, cloud and region inputs.
- **Create index** — creates the serverless index if missing, or validates the
  existing one. Reports whether it was created or already existed.
- **Refresh stats** — reads vector count, dimension and per-namespace counts.
- **Top K chunks** — slider from 1 to 10 (default 3).

### Ingest documents tab

1. Choose a source: **Upload files**, **Sample corpus**, or **docs/ folder (PDFs)**.
2. Choose a chunking strategy.
3. Keep **Clear namespace before ingest** checked to replace the namespace contents,
   or uncheck to merge with existing vectors.
4. Press **Ingest into Pinecone**.

During ingestion, a status panel and progress bar report each stage. On completion
the UI shows:

- document, chunk and vector counts;
- target index, namespace and strategy;
- total vectors in the index;
- a preview table of the first 10 chunks;
- a notice if Pinecone has not yet made all vectors queryable.

Uploads are parsed in memory: PDFs via `pypdf`, text and Markdown by UTF-8 decode.

### Ask questions tab

1. Enter a question and press **Generate answer**.
2. The app embeds the query, retrieves the top-k chunks from the selected namespace,
   builds the grounded prompt and calls the LLM.
3. Results are shown as:
   - the answer in Markdown;
   - token usage;
   - an expandable **Sources used** panel with score, rank and chunk text;
   - an expandable **Prompt debug** panel with both prompts.

If the namespace contains no vectors, the app stops with a clear message instead of
calling the LLM with empty context. The last answer is kept in session state so it
survives sidebar changes.

---

## CLI Reference

```bash
# Create the index (safe when it already exists)
python Week5_RAG_Pinecone.py create-index

# Destructive: delete and recreate the index
python Week5_RAG_Pinecone.py create-index --recreate

# Ingest the built-in corpus
python Week5_RAG_Pinecone.py ingest --source sample

# Ingest PDFs from ./docs with recursive chunking
python Week5_RAG_Pinecone.py ingest --source docs --path docs --strategy recursive

# Merge into a namespace instead of clearing it first
python Week5_RAG_Pinecone.py ingest --source sample --keep-existing

# Ask one question
python Week5_RAG_Pinecone.py ask "What is RAG?"

# Query a specific namespace with five chunks
python Week5_RAG_Pinecone.py ask "How does chunking work?" --namespace Default --top-k 5

# Index statistics
python Week5_RAG_Pinecone.py stats
```

Common flags on every command:

| Flag | Default |
|---|---|
| `--index` | `PINECONE_INDEX_NAME` or `rag-week5` |
| `--namespace` | `PINECONE_NAMESPACE` or `Default` |
| `--cloud` | `PINECONE_CLOUD` or `aws` |
| `--region` | `PINECONE_REGION` or `us-east-1` |

`ingest` also accepts `--source {sample,docs}`, `--path`, `--strategy` and
`--keep-existing`. `ask` accepts `--top-k`.

The CLI prints each stage's progress, the retrieval ranking table, the final answer,
token usage and the sources used.

---

## Data Model

### Chunk (in memory)

```python
{
  "source": "doc_rag_overview.txt",
  "chunk_index": 0,
  "text": "...",
  "strategy": "paragraph",
}
```

### Retrieved chunk

The chunk dict plus:

```python
{
  "id": "doc_rag_overview.txt::chunk-0",
  "retrieval_score": 0.7269,
  "retrieval_rank": 1,
}
```

### Vector id

```
<slugified source>::chunk-<index>
```

Ids are deterministic per source and chunk index. Consequences:

- Re-ingesting the same file overwrites its vectors.
- Renaming a file creates a new set of ids.
- Switching chunking strategy without clearing the namespace can leave stale
  vectors from the previous strategy behind — clear the namespace when you change
  strategy.

---

## Pinecone vs FAISS

| Aspect | Week 3 — FAISS | Week 5 — Pinecone |
|---|---|---|
| Deployment | In-process library | Managed serverless service |
| Persistence | Process lifetime only | Durable |
| Index types | Flat / IVF / HNSW / PQ | Managed serverless |
| Partitioning | None | Namespaces |
| Metadata | Manual bookkeeping | Stored and filterable per vector |
| Scaling | Manual | Managed |
| Cost | Free / local | Usage-based |
| Consistency | Immediate | Eventual (seconds) |
| Similarity | L2-normalise + `IndexFlatIP` | `metric="cosine"` |

---

## Operational Notes

- **Clearing is destructive.** "Clear namespace before ingest" and `delete_all=True`
  remove every vector in that namespace. Other namespaces are unaffected.
- **Namespaces are case-sensitive.** `Default` and `default` are different
  partitions. Pinecone's built-in unnamed namespace is the empty string; this app
  defaults to the explicit name `Default`. Leaving the namespace field blank targets
  the unnamed namespace, and changing only the capitalization targets a separate,
  empty partition.
- **Eventual consistency.** A query issued immediately after upsert may miss the
  newest vectors. The pipeline polls for up to 60 s and the UI warns when the index
  is still catching up.
- **Metadata size.** Chunk text is stored in vector metadata (Pinecone limit:
  40 KB per vector). Very large paragraph chunks can exceed this — use recursive
  chunking for long PDFs.
- **Batch limits.** Embeddings are sent 20 at a time with a short sleep; upserts are
  sent 100 at a time.
- **Index guards.** A dimension other than 1536 or a metric other than cosine is
  rejected with an explanatory error.
- **Free-tier limits.** Serverless accounts have index-count and usage quotas; index
  creation failures are surfaced in the UI and CLI.
- **Secrets.** Keys are read from `.env` or the environment. Sidebar-entered keys
  live only in the running process.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OPENROUTER_API_KEY is not set` | Missing key | Add it to `.env` or the sidebar |
| `PINECONE_API_KEY is not set` | Missing key | Add it to `.env` or the sidebar |
| `Index '...' does not exist` | Index not created | Run `create-index` or use the sidebar button |
| `Index '...' has dimension N` | Wrong index reused | Use an index created with 1536 dims |
| `uses metric '...'` | Index is not cosine | Create a new index with `metric="cosine"` |
| `[404] Namespace not found` | Empty namespace being cleared | Handled automatically by `clear_namespace` |
| No chunks retrieved | Namespace empty or wrong case | Ingest into that namespace; check spelling/case |
| PDF upload fails | `pypdf` missing or scanned PDF | `pip install pypdf`; use text-based PDFs |
| Streamlit ignores CLI flags | GUI mode active | Run CLI with `python`, GUI with `streamlit run` |

---

## Setup and Running

```bash
# Install dependencies
pip install -r requirements.txt

# GUI
streamlit run Week5_RAG_Pinecone.py

# CLI smoke check
python Week5_RAG_Pinecone.py stats
```

Week 5 dependencies:

| Package | Purpose |
|---|---|
| `pinecone>=10.0.0` | Managed vector database client |
| `streamlit>=1.30.0` | GUI |
| `pypdf>=4.0.0` | PDF text extraction |
| `openai>=1.0.0` | OpenRouter-compatible embeddings and chat client |
| `python-dotenv>=1.0.0` | `.env` loading |
| `numpy>=1.24.0` | Embedding matrices |

Required `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-...
PINECONE_API_KEY=...
# optional
PINECONE_INDEX_NAME=rag-week5
PINECONE_NAMESPACE=Default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

---

## Full Pipeline Summary

```
┌────────┬────────────────────────┬──────────────────────────────┐
│ Stage  │ What                   │ How (this script)            │
├────────┼────────────────────────┼──────────────────────────────┤
│   1    │ Ingest documents       │ Uploads / sample / docs PDFs │
│        │                        │ normalised, paragraph breaks │
│        │                        │ preserved                    │
├────────┼────────────────────────┼──────────────────────────────┤
│   2    │ Chunk documents        │ 4 strategies; paragraph used │
│        │                        │ by default                   │
├────────┼────────────────────────┼──────────────────────────────┤
│   3    │ Embed chunks           │ OpenRouter text-embedding-   │
│        │                        │ 3-small → (N × 1536)         │
├────────┼────────────────────────┼──────────────────────────────┤
│   4    │ Index in Pinecone      │ Serverless cosine index;     │
│        │                        │ metadata upsert, namespaces  │
├────────┼────────────────────────┼──────────────────────────────┤
│   5    │ Retrieve top-k chunks  │ Embed query → Pinecone query │
│        │                        │ → ranked chunks with scores  │
├────────┼────────────────────────┼──────────────────────────────┤
│   6    │ Augment prompt         │ System grounding + labelled  │
│        │                        │ context + question           │
├────────┼────────────────────────┼──────────────────────────────┤
│   7    │ Generate answer        │ OpenRouter gpt-4o-mini,      │
│        │                        │ temperature=0.1              │
└────────┴────────────────────────┴──────────────────────────────┘
```

---

## Key Concepts to Remember

- **The index is remote and persistent.** Unlike the in-memory FAISS index, vectors
  survive process restarts. Ingestion is a write to a shared service.
- **Namespaces isolate corpora.** One index can hold several document sets. Wrong
  namespace (including wrong case) means no retrieval, not an error.
- **Embedding consistency is mandatory.** Documents and queries must use the same
  model; changing it requires re-indexing.
- **Deterministic ids make ingestion idempotent.** Re-running an ingest overwrites
  the same chunk ids.
- **Clear before changing strategy.** New chunk boundaries mean new ids; old vectors
  would otherwise remain.
- **Top-k is a cost/quality tradeoff.** More chunks means more context, more noise
  and more prompt tokens. 3-5 is a sensible starting range.
- **Low temperature for factual Q&A.** 0.1 keeps the answer anchored to context.
- **Source metadata enables citations and debugging.** It is stored on the vector,
  so it comes back with every match.
