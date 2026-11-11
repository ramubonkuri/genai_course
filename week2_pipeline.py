"""
Week 2 — Complete Ingest & Embeddings Pipeline
================================================
This script demonstrates:
  1. Document loading and cleaning
  2. Recursive text chunking
  3. Embedding generation (OpenAI or local)
  4. Vector storage and search
  5. Retrieval quality evaluation

Requirements:
  pip install openai numpy
  # Optional for local embeddings: pip install sentence-transformers

Usage:
  export OPENAI_API_KEY="your-key"
  python week2_pipeline.py
"""

import os
import re
import json
import time
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
DOCS_DIR = "./sample_docs"          # Directory containing .txt and .md files
CHUNK_SIZE = 800                    # Target chunk size in characters
CHUNK_OVERLAP = 100                 # Overlap between chunks
TOP_K = 5                           # Number of results to retrieve
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
USE_LOCAL_EMBEDDINGS = False        # Set True to use sentence-transformers
LOCAL_MODEL = "all-MiniLM-L6-v2"    # Local model name

# Test queries with expected relevant chunk IDs (update after first run)
TEST_QUERIES = [
    {
        "query": "How do I reset my password?",
        "relevant_ids": []  # Fill after inspecting generated chunk IDs
    },
    {
        "query": "What is the refund policy for customers?",
        "relevant_ids": []
    },
    {
        "query": "What should I do during a data breach?",
        "relevant_ids": []
    },
    {
        "query": "How do I enroll in health benefits?",
        "relevant_ids": []
    }
]

# =============================================================================
# 1. DOCUMENT LOADING & CLEANING
# =============================================================================

def load_text_files(directory):
    """Load all .txt and .md files from a directory."""
    docs = []
    path = Path(directory)
    for ext in ["*.txt", "*.md"]:
        for file_path in path.glob(ext):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({
                "source": str(file_path),
                "content": content,
                "filename": file_path.name
            })
    return docs


def clean_text(text):
    """Basic text normalization."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[•·]', '-', text)
    return text.strip()


# =============================================================================
# 2. CHUNKING
# =============================================================================

def recursive_chunk(text, chunk_size=800, overlap=100):
    """
    Recursively split text into chunks by paragraphs, then sentences.
    Respects natural boundaries while targeting chunk_size.
    """
    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) < chunk_size:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_end = chunks[i-1][-overlap:]
                chunk = prev_end + " " + chunk
            overlapped.append(chunk)
        chunks = overlapped

    return chunks


# =============================================================================
# 3. EMBEDDINGS
# =============================================================================

def get_openai_client():
    """Initialize OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def embed_texts_openai(texts, model="text-embedding-3-small", batch_size=100):
    """Embed texts using OpenAI API."""
    client = get_openai_client()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def embed_texts_local(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Embed texts using local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings.tolist()


# =============================================================================
# 4. VECTOR STORE
# =============================================================================

class SimpleVectorStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self, dimension=None):
        self.vectors = []
        self.metadata = []
        self.dimension = dimension

    def add(self, vector, meta):
        v = np.array(vector, dtype=np.float32)
        if self.dimension is None:
            self.dimension = len(v)
        assert len(v) == self.dimension, f"Dimension mismatch: {len(v)} vs {self.dimension}"
        self.vectors.append(v)
        self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        q = np.array(query_vector, dtype=np.float32)
        if len(self.vectors) == 0:
            return []

        vectors = np.stack(self.vectors)
        norms = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(q)

        # Avoid division by zero
        valid = norms > 0
        similarities = np.zeros(len(vectors))
        similarities[valid] = np.dot(vectors[valid], q) / (norms[valid] * q_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "rank": len(results) + 1,
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx],
                "text": self.metadata[idx]["text"][:200] + "..."
            })
        return results

    def save(self, path):
        data = {
            "vectors": [v.tolist() for v in self.vectors],
            "metadata": self.metadata,
            "dimension": self.dimension
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Vector store saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        store = cls(dimension=data["dimension"])
        for v, m in zip(data["vectors"], data["metadata"]):
            store.add(v, m)
        print(f"Vector store loaded from {path} ({len(store.vectors)} vectors)")
        return store


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_retrieval(store, test_queries, embed_fn, top_k=5):
    """Evaluate retrieval quality using Precision@K and MRR."""
    results = []

    for test in test_queries:
        query_emb = embed_fn([test["query"]])[0]
        retrieved = store.search(query_emb, top_k=top_k)
        retrieved_ids = [r["metadata"]["chunk_id"] for r in retrieved]

        relevant = set(test["relevant_ids"])
        retrieved_set = set(retrieved_ids)

        precision_at_k = len(relevant & retrieved_set) / top_k if top_k > 0 else 0

        mrr = 0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant:
                mrr = 1 / (i + 1)
                break

        results.append({
            "query": test["query"],
            "precision@k": precision_at_k,
            "mrr": mrr,
            "retrieved_ids": retrieved_ids,
            "relevant_found": list(relevant & retrieved_set)
        })

    avg_precision = sum(r["precision@k"] for r in results) / len(results) if results else 0
    avg_mrr = sum(r["mrr"] for r in results) / len(results) if results else 0

    return {
        "per_query": results,
        "avg_precision@k": avg_precision,
        "avg_mrr": avg_mrr
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("WEEK 2 — INGEST & EMBEDDINGS PIPELINE")
    print("=" * 70)

    # --- Step 1: Load Documents ---
    print("\n[1/5] Loading documents...")
    docs = load_text_files(DOCS_DIR)
    if not docs:
        print(f"ERROR: No .txt or .md files found in {DOCS_DIR}")
        return

    for doc in docs:
        doc["content"] = clean_text(doc["content"])
        print(f"  ✓ {doc['filename']} ({len(doc['content'])} chars)")

    # --- Step 2: Chunk ---
    print(f"\n[2/5] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    for doc in docs:
        chunks = recursive_chunk(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": f"{doc['filename'].replace('.txt','').replace('.md','')}_chunk_{i}",
                "chunk_index": i,
                "doc_length": len(doc["content"])
            })
    print(f"  ✓ Created {len(all_chunks)} chunks from {len(docs)} documents")

    # Print sample chunk IDs for evaluation setup
    print("\n  Generated chunk IDs (for evaluation setup):")
    for chunk in all_chunks[:10]:
        print(f"    - {chunk['chunk_id']}")
    if len(all_chunks) > 10:
        print(f"    ... and {len(all_chunks) - 10} more")

    # --- Step 3: Embed ---
    print(f"\n[3/5] Generating embeddings...")
    texts = [c["text"] for c in all_chunks]

    if USE_LOCAL_EMBEDDINGS:
        print(f"  Using local model: {LOCAL_MODEL}")
        embeddings = embed_texts_local(texts, LOCAL_MODEL)
    else:
        print(f"  Using OpenAI model: {EMBEDDING_MODEL}")
        embeddings = embed_texts_openai(texts, EMBEDDING_MODEL)

    print(f"  ✓ Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # --- Step 4: Store ---
    print(f"\n[4/5] Storing in vector database...")
    store = SimpleVectorStore()
    for chunk, emb in zip(all_chunks, embeddings):
        store.add(emb, chunk)

    store.save("vector_store.json")
    print(f"  ✓ Stored {len(store.vectors)} vectors")

    # --- Step 5: Evaluate ---
    print(f"\n[5/5] Evaluating retrieval quality...")

    # Use same embed function for queries
    if USE_LOCAL_EMBEDDINGS:
        embed_fn = lambda texts: embed_texts_local(texts, LOCAL_MODEL)
    else:
        embed_fn = lambda texts: embed_texts_openai(texts, EMBEDDING_MODEL)

    # Check if test queries have relevant IDs set
    has_eval = any(len(t["relevant_ids"]) > 0 for t in TEST_QUERIES)

    if has_eval:
        eval_results = evaluate_retrieval(store, TEST_QUERIES, embed_fn, TOP_K)

        print("\n" + "=" * 70)
        print("RETRIEVAL EVALUATION REPORT")
        print("=" * 70)
        print(f"Average Precision@{TOP_K}: {eval_results['avg_precision@k']:.3f}")
        print(f"Average MRR:           {eval_results['avg_mrr']:.3f}")
        print()

        for r in eval_results["per_query"]:
            print(f"Query: {r['query']}")
            print(f"  Precision@{TOP_K}: {r['precision@k']:.3f} | MRR: {r['mrr']:.3f}")
            print(f"  Relevant found: {r['relevant_found']}")
            print(f"  Top retrieved:  {r['retrieved_ids'][:3]}")
            print()
    else:
        print("  ⚠ No relevant_ids configured in TEST_QUERIES.")
        print("    Run a manual search first, note the correct chunk IDs,")
        print("    then update TEST_QUERIES and re-run for evaluation.")

        # Demo search
        print("\n  Demo search results:")
        demo_query = "How do I reset my password?"
        demo_emb = embed_fn([demo_query])[0]
        demo_results = store.search(demo_emb, top_k=3)
        print(f"  Query: '{demo_query}'")
        for r in demo_results:
            print(f"    Rank {r['rank']} (score={r['score']:.4f}): {r['metadata']['chunk_id']}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Inspect chunk IDs above and update TEST_QUERIES relevant_ids")
    print("  2. Re-run to get quantitative evaluation metrics")
    print("  3. Experiment with CHUNK_SIZE and CHUNK_OVERLAP")
    print("  4. Try different embedding models")
    print("  5. Add more test queries for robust evaluation")


if __name__ == "__main__":
    main()
