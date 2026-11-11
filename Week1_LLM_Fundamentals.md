
# Week 1 — LLM Fundamentals & Prompt Engineering


---

## 1. LLM Capabilities, Limitations, Hallucinations

### What LLMs Can Do Well
| Capability | Description | Example |
|-----------|-------------|---------|
| **Text Generation** | Produce coherent, context-aware prose | Draft emails, reports, creative writing |
| **Summarization** | Condense long documents into key points | TL;DR of research papers, meeting notes |
| **Translation** | Convert between languages with nuance | Technical docs, conversational text |
| **Code Generation** | Write, explain, and debug code | Python scripts, SQL queries, regex |
| **Classification** | Categorize text into predefined labels | Sentiment analysis, topic tagging |
| **Extraction** | Pull structured data from unstructured text | Named entities, dates, amounts |
| **Reasoning (Chain-of-Thought)** | Step-by-step logical deduction | Math problems, logic puzzles |

### Core Limitations
1. **No Real-Time Knowledge** — Training data has a cutoff date. GPT-4o knowledge cutoff: ~Oct 2023.
2. **No True Understanding** — Predicts next tokens; does not "know" facts in a human sense.
3. **Context Window Limits** — Even 128K tokens is finite. Long documents may be truncated.
4. **Deterministic Randomness** — Same prompt can yield different outputs (`temperature > 0`).
5. **No Memory Across Sessions** — Stateless by default; state must be managed externally.
6. **Tool Use Requires Orchestration** — Cannot browse, calculate precisely, or execute code without explicit tool integration.

### Hallucinations: Types & Mitigation
| Type | Description | Mitigation |
|------|-------------|------------|
| **Factual** | Invented facts, fake citations | Retrieval-Augmented Generation (RAG), grounding |
| **Logical** | Sound plausible but internally inconsistent | Chain-of-thought prompting, self-consistency |
| **Contextual** | Ignore instructions or contradict prior context | System prompts, few-shot examples |
| **Source** | Misattribute quotes or data | Citation tracking, source verification |

**Rule of thumb:** Never trust LLM output for critical facts without verification. Always treat it as a "helpful intern who sometimes lies confidently."

---

## 2. Cost / Latency Tradeoffs

### Key Variables
| Variable | Impact on Cost | Impact on Latency |
|----------|---------------|-------------------|
| **Model size** | Larger = more expensive | Larger = slower |
| **Input tokens** | Linear cost increase | Linear time increase |
| **Output tokens** | Linear cost increase | Linear time increase |
| **Temperature** | No cost impact | No direct impact |
| **Streaming** | No cost impact | Perceived latency ↓ |
| **Batching** | Potential discount | Higher individual latency |



### Decision Framework
```
High accuracy needed + budget allows     → GPT-4o / Claude 3.5 Sonnet
Balanced accuracy/cost                  → GPT-4o-mini / Claude 3 Haiku
High throughput, simple tasks           → GPT-4o-mini / local models (Llama 3.1 8B)
Latency-critical (chat UX)              → Streaming + smaller model + caching
```

---

## 3. Prompt Engineering Patterns

### Pattern 1: Structured Output (JSON Mode)
**Use case:** Extracting structured data from unstructured text.

```python
import openai
import json

client = openai.OpenAI(api_key="YOUR_API_KEY")

prompt = """
Extract the following from the customer support ticket:
- customer_name
- issue_category (Billing, Technical, Account, Other)
- urgency (Low, Medium, High)
- summary (max 20 words)

Ticket: "Hi, I have been charged twice for my subscription this month. 
I need this resolved ASAP or I will cancel. My account is under john.doe@email.com"

Respond ONLY with valid JSON. No markdown, no explanation.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.0
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))
```

### Pattern 2: Chain-of-Thought (CoT)
**Use case:** Complex reasoning where accuracy matters more than speed.

```python
prompt = """
You are a math tutor. Solve the following problem step by step.
Show your reasoning clearly before giving the final answer.

Problem: A train travels 120 km in 2 hours. It then slows down and 
travelsthe next 80 km in 2.5 hours. What is the average speed 
for the entire journey?

Step 1: Calculate total distance.
Step 2: Calculate total time.
Step 3: Apply average speed formula.
Final Answer: ___ km/h
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)
print(response.choices[0].message.content)
```

### Pattern 3: Few-Shot Classification
**Use case:** Teaching the model a custom classification task with examples.

```python
prompt = """
Classify customer feedback into: Praise, Complaint, Suggestion, Question.

Examples:
Feedback: "Your app crashed 3 times today. This is unacceptable."
Classification: Complaint

Feedback: "Could you add dark mode? It would help with eye strain."
Classification: Suggestion

Feedback: "The new dashboard is incredibly intuitive. Great job!"
Classification: Praise

Feedback: "How do I export my data to CSV?"
Classification: Question

Now classify:
Feedback: "I love the new feature but it loads too slowly on mobile."
Classification:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=10
)
print(response.choices[0].message.content.strip())
```

---

## 4. Measuring Latency & Cost Per Call

```python
import time
import openai

client = openai.OpenAI(api_key="YOUR_API_KEY")

# Pricing per 1M tokens (update as needed)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

def call_with_metrics(model, messages, temperature=0.7):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Cost in USD
    rates = PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens / 1_000_000 * rates["input"] + 
            output_tokens / 1_000_000 * rates["output"])

    return {
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6),
        "content": response.choices[0].message.content
    }

# Run experiment
messages = [{"role": "user", "content": "Explain quantum computing in 3 sentences."}]

for model in ["gpt-4o-mini", "gpt-4o"]:
    result = call_with_metrics(model, messages, temperature=0.5)
    print(f"\n{'='*50}")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_ms']} ms")
    print(f"Tokens: {result['total_tokens']} (in: {result['input_tokens']}, out: {result['output_tokens']})")
    print(f"Cost: ${result['cost_usd']}")
    print(f"Output: {result['content'][:100]}...")
```

---

## 5. Embeddings Concept & Similarity Search

### What Are Embeddings?
Embeddings are dense vector representations of text (or images, audio) where **semantically similar items are close together in vector space**.

- Input: "The cat sat on the mat" → Output: `[0.023, -0.156, 0.891, ...]` (1536 dims for text-embedding-3-small)
- Distance metric: **Cosine similarity** (1 = identical meaning, 0 = unrelated, -1 = opposite)

### Why Use Them?
1. **Semantic Search** — Find documents related to a query even without keyword overlap
2. **Clustering** — Group similar items automatically
3. **Classification** — Use vectors as features for ML models
4. **RAG (Retrieval-Augmented Generation)** — Feed relevant context to LLMs

### Hands-On: Basic Similarity Search
```python
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Sample documents
docs = [
    "Machine learning models require large amounts of training data.",
    "Deep neural networks can recognize patterns in images.",
    "The bakery on Main Street makes excellent sourdough bread.",
    "Convolutional networks are used for computer vision tasks.",
    "Fresh bread smells amazing in the morning."
]

# Compute embeddings
embeddings = [get_embedding(doc) for doc in docs]

# Query
query = "How do AI systems learn from images?"
query_emb = get_embedding(query)

# Rank by similarity
scores = [(doc, cosine_similarity(query_emb, emb)) for doc, emb in zip(docs, embeddings)]
scores.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{query}'\n")
for doc, score in scores:
    print(f"Score: {score:.4f} | {doc}")
```

---

## Deliverable: One-Page LLM Tradeoffs Summary

### LLM Tradeoffs Cheat Sheet

| Decision | Tradeoff | Recommendation |
|----------|----------|----------------|
| **Model Choice** | Accuracy vs Cost | Start with cheapest (GPT-4o-mini); upgrade only if quality insufficient |
| **Temperature** | Creativity vs Consistency | 0.0 for extraction/classification; 0.7 for creative tasks; 0.3-0.5 for general use |
| **Max Tokens** | Completeness vs Cost/Latency | Set explicitly; unused tokens do not cost output $ but affect latency |
| **Streaming** | Perceived speed vs Complexity | Enable for UX; disable if you need full response before processing |
| **JSON Mode** | Structure vs Flexibility | Use for all structured outputs; add retry logic for malformed JSON |
| **Caching** | Cost savings vs Staleness | Cache embeddings and common LLM responses; invalidate on data change |
| **Context Length** | Relevance vs Cost | Trim irrelevant history; use RAG instead of dumping full docs |

### Cost Estimation Formula
```
Cost = (input_tokens × input_rate + output_tokens × output_rate) / 1,000,000
```

### Latency Tips
- Use **streaming** for chat interfaces
- Use **smaller models** for simple tasks (routing, classification)
- **Batch** embedding requests (up to 2048 texts per call)
- **Cache** frequent queries at application layer
- Consider **async/parallel** calls for independent subtasks

### Prompt Template Checklist
- [ ] Clear role definition ("You are a...")
- [ ] Explicit output format (JSON, bullet points, etc.)
- [ ] Examples for complex tasks (few-shot)
- [ ] Constraints (max length, tone, what NOT to do)
- [ ] Error handling instructions ("If unsure, say I do not know")
