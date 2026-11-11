# Week 2: Embeddings & Vector Databases

## 1. What is an Embedding?

An **embedding** is a way to convert text (or images, audio, etc.) into a list of numbers — a **vector** — that captures the *meaning* of that content.

> Think of it as translating human language into a language that math can understand.

### The Big Idea

Traditional NLP used keywords. Embeddings use **meaning**.

```
"dog"  →  [0.21, -0.45, 0.88, 0.12, ...]   # 384 numbers
"puppy" →  [0.19, -0.43, 0.85, 0.14, ...]   # very similar!
"cat"  →  [0.18, -0.40, 0.79, 0.20, ...]   # somewhat similar
"car"  →  [0.91,  0.12, -0.33, 0.67, ...]  # very different
```

Even though "dog" and "puppy" are different words, their vectors are close to each other because they mean similar things.

---

## 2. Embeddings in NLP — Step by Step

### Stage 1: Raw Text
```
"The bank can guarantee deposits will eventually cover future tuition costs."
```

### Stage 2: Tokenization
Break the sentence into tokens the model understands:
```
["The", "bank", "can", "guarantee", "deposits", "will", ...]
```

### Stage 3: Model Encodes Meaning
An embedding model (like `sentence-transformers/all-MiniLM-L6-v2`) reads the full sentence and produces a single vector:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

sentence = "The bank can guarantee deposits will eventually cover future tuition costs."
embedding = model.encode(sentence)

print(embedding.shape)  # (384,)
print(embedding[:5])    # [ 0.021, -0.134,  0.087,  0.302, -0.211 ]
```

The entire sentence is compressed into **one vector of 384 numbers** that captures the pattern and meaning.

### Why Does This Work?

The model was trained on billions of sentences. It learned that:
- "bank" near "deposits" → financial context
- "bank" near "river" → geographical context

The surrounding context shapes the numbers. **Same word, different meaning = different vector**.

---

## 3. Real-World Embedding Dimensions

Different models produce vectors of different sizes. Larger = more nuanced meaning.

| Model | Dimensions | Use Case |
|---|---|---|
| `all-MiniLM-L6-v2` | **384** | Fast, lightweight search |
| `all-mpnet-base-v2` | **768** | Balanced quality/speed |
| `text-embedding-ada-002` (OpenAI) | **1536** | High quality, general purpose |
| `text-embedding-3-large` (OpenAI) | **3072** | Best quality |
| Custom large models | **1024+** | Domain-specific tasks |

> The entire vector — all 384 or 1536 numbers together — captures the pattern in a meaningful way. No single number means anything alone.

---

## 4. RDBMS vs Vector Database

### Traditional RDBMS (Relational DB)

Stores structured data in rows and columns. Searches by **exact match or range**.

```sql
-- Products table
| id | name        | category    | price |
|----|-------------|-------------|-------|
|  1 | iPhone 15   | smartphone  |  999  |
|  2 | Galaxy S24  | smartphone  |  899  |
|  3 | MacBook Pro | laptop      | 1999  |

-- Keyword search (exact match)
SELECT * FROM products WHERE category = 'smartphone';
-- ✅ Finds iPhone 15, Galaxy S24
-- ❌ Misses "mobile phone", "cell phone", "handset"
```

**Problem:** You must know the exact word. Meaning doesn't matter.

---

### Vector Database

Stores vectors alongside documents. Searches by **meaning/similarity**.

```python
# Store documents as vectors
documents = [
    "iPhone 15 is Apple's latest smartphone",
    "Galaxy S24 is Samsung's flagship mobile phone",
    "MacBook Pro is a powerful laptop for developers",
    "AirPods Pro are wireless earbuds with noise cancellation",
]

# Each document gets embedded into a vector
vectors = model.encode(documents)
# vectors.shape = (4, 384)  →  4 documents, 384 dimensions each

# Now search with a query — no exact keyword needed!
query = "best handset for photography"   # ← "handset" doesn't appear anywhere!
query_vector = model.encode(query)

# Vector DB finds the most similar vectors
results = vector_db.search(query_vector, top_k=2)
# ✅ Returns: iPhone 15, Galaxy S24  (because they're semantically close)
```

**RDBMS vs Vector DB — Side by Side:**

| Feature | RDBMS (PostgreSQL, MySQL) | Vector DB (ChromaDB, Pinecone) |
|---|---|---|
| Data stored | Rows & columns | Vectors + metadata |
| Search type | Exact / keyword | Semantic / meaning |
| Query language | SQL | Similarity search |
| Finds synonyms? | ❌ No | ✅ Yes |
| Handles typos? | ❌ No | ✅ Mostly yes |
| Best for | Structured records | Unstructured text/images |
| Examples | MySQL, PostgreSQL | ChromaDB, Pinecone, FAISS |

---

## 5. A Vector is a List of Numbers Representing Meaning

```
Word:    "King"   →  [0.50,  0.80, -0.10,  0.30, ...]
Word:    "Queen"  →  [0.48,  0.75, -0.08,  0.32, ...]
Word:    "Man"    →  [0.30,  0.20,  0.10,  0.10, ...]
Word:    "Woman"  →  [0.28,  0.16,  0.12,  0.12, ...]

# Famous example:
King - Man + Woman ≈ Queen   ← the math actually works!
```

For AI, a vector represents a **pattern or meaning** extracted from data. The entire vector together is what matters — not any individual number.

---

## 6. Finding Similarity Between Vectors

Once text is converted to vectors, we need to measure how similar two vectors are. Three main methods:

### 6.1 Cosine Similarity

**Idea:** Measures the *angle* between two vectors. Direction matters, not magnitude.

```
        ↑
  B  /  |       angle between A and B is small  →  similar
    /   |
   / θ  |
  /_____|→  A
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Range: -1 (opposite) to +1 (identical)
# 1.0  = same direction = very similar
# 0.0  = perpendicular = unrelated
# -1.0 = opposite direction = antonyms

vec_dog   = model.encode("dog")
vec_puppy = model.encode("puppy")
vec_car   = model.encode("car")

print(cosine_similarity(vec_dog, vec_puppy))  # ~0.91  ✅ very similar
print(cosine_similarity(vec_dog, vec_car))    # ~0.18  ❌ not similar
```

> Best for: text similarity, semantic search (most common choice)

---

### 6.2 Euclidean Distance

**Idea:** The straight-line distance between two points in space.

```
  B •
     \
      \  distance
       \
        • A

Small distance = similar vectors
Large distance = different vectors
```

```python
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Range: 0 (identical) to ∞ (very different)

print(euclidean_distance(vec_dog, vec_puppy))  # ~0.4  ✅ close
print(euclidean_distance(vec_dog, vec_car))    # ~1.8  ❌ far apart
```

> Best for: image embeddings, cases where magnitude matters

---

### 6.3 Dot Product

**Idea:** Multiply corresponding numbers and sum them up. Large positive = strong alignment.

```python
def dot_product(a, b):
    return np.dot(a, b)
    # = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ...

# Large positive value = strong alignment (similar AND large vectors)
# Near zero          = unrelated
# Large negative     = opposite meaning

print(dot_product(vec_dog, vec_puppy))  # large positive ✅
print(dot_product(vec_dog, vec_car))    # small / near zero ❌
```

> Best for: recommendation systems, when vector magnitude carries information

---

### Similarity Methods — Comparison

| Method | What it measures | Best for | Sensitive to magnitude? |
|---|---|---|---|
| **Cosine Similarity** | Angle between vectors | Text search | ❌ No |
| **Euclidean Distance** | Straight-line distance | Images, spatial data | ✅ Yes |
| **Dot Product** | Directional alignment | Recommendations | ✅ Yes |

---

## 7. Vector Databases

A **vector database** is purpose-built to store vectors and run fast similarity searches across millions of them.

### Popular Options

| Database | Type | Best for |
|---|---|---|
| **ChromaDB** | Open source, embedded | Local development, prototyping |
| **Pinecone** | Managed cloud service | Production at scale |
| **FAISS** | In-memory library (Meta) | High-speed local search |
| **Weaviate** | Open source, self-hosted | Full-featured production use |
| **pgvector** | PostgreSQL extension | Teams already using Postgres |

### ChromaDB Example

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Setup
client = chromadb.Client()
collection = client.create_collection("my_docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add documents
docs = [
    "Python is great for data science and machine learning",
    "JavaScript is used for building web applications",
    "SQL is essential for querying relational databases",
    "Docker containers help deploy applications consistently",
]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# Search — no exact keyword needed!
query = "I want to learn AI and data analysis"
query_embedding = model.encode(query).tolist()

results = collection.query(query_embeddings=[query_embedding], n_results=2)
print(results["documents"])
# → ['Python is great for data science...']  ✅
```

---

## 8. How Vector Search Works — HNSW Algorithm

Vector databases don't compare your query against every single vector (that would be too slow for millions of records). They use **HNSW — Hierarchical Navigable Small World**.

### The Intuition

Think of it like a highway system:

```
Layer 2 (Highway):     A ————————————— E
                        \               \
Layer 1 (Roads):    A — B — C       D — E — F
                    |               |
Layer 0 (Streets):  A-B-C-D-E-F-G-H-I-J  (all nodes)
```

- **Start at the top layer** (few nodes, big jumps)
- **Navigate toward the query** at each layer
- **Drop down** to finer layers as you get closer
- **Final layer** does precise local search

### Why It's Fast

```
Brute force:    compare against ALL 1,000,000 vectors  →  slow 🐢
HNSW:           compare against ~log(1,000,000) ≈ 20 vectors  →  fast ⚡
```

HNSW trades a tiny bit of accuracy for massive speed gains — typically finds the **true nearest neighbor 99%+ of the time**.

```python
# FAISS example using HNSW index
import faiss
import numpy as np

d = 384          # vector dimensions
M = 16           # HNSW connections per node (higher = more accurate, more memory)

index = faiss.IndexHNSWFlat(d, M)

# Add 100,000 vectors
vectors = np.random.rand(100_000, d).astype("float32")
index.add(vectors)

# Search — returns top 5 nearest neighbors
query = np.random.rand(1, d).astype("float32")
distances, indices = index.search(query, k=5)

print(indices)    # [[ 42381, 7823, 91234, 55102, 3841 ]]  — found in microseconds ⚡
```

---

## 9. Full Pipeline — Text to Semantic Search

```
Raw Text
   │
   ▼
Embedding Model  (e.g., all-MiniLM-L6-v2)
   │
   ▼
Vector  [0.21, -0.45, 0.88, ...]  (384 numbers)
   │
   ▼
Vector Database  (ChromaDB / Pinecone / FAISS)
   │  stored with HNSW index
   ▼
Query comes in: "best phone for photos"
   │
   ▼
Query → Vector → HNSW search → Top K similar vectors
   │
   ▼
Return original documents ranked by similarity
```

---

## 10. Key Takeaways

| Concept | Summary |
|---|---|
| **Embedding** | Converts text → vector (list of numbers representing meaning) |
| **Dimensions** | 384, 768, 1024, 1536 depending on model — entire vector = the meaning |
| **vs Keywords** | Vectors find semantic matches; keywords need exact words |
| **Cosine Similarity** | Best for text; measures angle (direction matters) |
| **Euclidean Distance** | Straight-line distance; magnitude matters |
| **Dot Product** | Alignment strength; large positive = similar |
| **Vector DB** | Purpose-built store for vectors + fast similarity search |
| **HNSW** | Graph algorithm that makes vector search fast at scale |

