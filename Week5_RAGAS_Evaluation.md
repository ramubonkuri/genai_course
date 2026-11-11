# Week 5 Extension — RAG Evaluation with RAGAS

> **Script:** `Week5_RAGAS_Evaluation.py`  
> **Pipeline under test:** `Week5_RAG_Pinecone.py`  
> **Judge LLM (via OpenRouter):** `openai/gpt-4o-mini`  
> **Judge embeddings (via OpenRouter):** `openai/text-embedding-3-small` (1536-dim)  
> **Vector DB:** Pinecone (`rag-week5`, namespace `Default`)

---

## Overview

The Week 5 app answers questions with real Pinecone retrieval and real OpenRouter generation. This extension answers a different question: **how good are those answers?**

RAGAS (Retrieval-Augmented Generation Assessment) scores a RAG pipeline on four separable things:

1. Did retrieval find the right chunks?
2. Did the answer stick to those chunks?
3. Did the answer actually address the question?
4. Did the answer match a known-correct reference?

The evaluation script does **not** reimplement the pipeline. It loads `Week5_RAG_Pinecone.py` with `importlib` and calls the exact `stage5_retrieve`, `stage6_build_prompt`, and `stage7_generate` functions the GUI and CLI use. Whatever you measure is the production path.

```
Evaluation questions
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  A. CONNECT   Pinecone index + namespace                │
  │  B. RUN RAG   retrieve top-k → augment → generate       │
  │  C. SCORE     RAGAS dataset → evaluate() with judge LLM │
  │  D. REPORT    console table + CSV + JSON                │
  └─────────────────────────────────────────────────────────┘
```

---

## Configuration

```python
OPENROUTER_BASE_URL      = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL        = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL  = "openai/text-embedding-3-small"

METRIC_REGISTRY = {
    "faithfulness":       Faithfulness,
    "answer_relevancy":   AnswerRelevancy,
    "context_precision":  ContextPrecision,
    "context_recall":     ContextRecall,
    "answer_correctness": AnswerCorrectness,
}
REFERENCE_METRICS = {
    "context_precision", "context_recall", "answer_correctness"
}
```

Index and namespace resolve in this order:

1. CLI flag (`--index`, `--namespace`)
2. environment variable (`PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`)
3. constant from `Week5_RAG_Pinecone.py` (`rag-week5`, `Default`)

The target namespace **must already contain vectors**. The evaluator refuses to run against an empty namespace and prints the ingest command to fix it.

```bash
python Week5_RAG_Pinecone.py ingest --source sample --namespace Default
```

---

## Stage A — Connect

**Method:** `RAGASEvaluator.connect()`

```python
index = pipeline.connect_existing_index(self.index_name)
stats = pipeline.index_stats(index)
count = stats["namespaces"].get(self.namespace, 0)
if count == 0:
    raise RuntimeError(...)
```

The index object is cached on the evaluator, so a full run opens the Pinecone connection once and reuses it for every question.

This stage also records `namespace_size` for the run header:

```
  Index     : rag-week5
  Namespace : Default (36 vectors)
  Top K     : 3
```

---

## Stage B — Run the RAG Pipeline

**Method:** `RAGASEvaluator.answer_question(question)`  
**Method:** `RAGASEvaluator.build_dataset(items)`

For each question the evaluator calls the real Week 5 stages:

```python
retrieved = pipeline.stage5_retrieve(question, index,
                                     namespace=self.namespace,
                                     top_k=self.top_k)
system_prompt, user_prompt = pipeline.stage6_build_prompt(question, retrieved)
answer, usage = pipeline.stage7_generate(system_prompt, user_prompt)
```

The output is converted into a RAGAS `SingleTurnSample`:

```python
SingleTurnSample(
    user_input=question,
    retrieved_contexts=[chunk["text"] for chunk in retrieved],
    response=answer,
    reference=reference_or_None,
)
```

A parallel `trace` dict keeps the things RAGAS does not need but a human does:

```python
{
  "question": "...",
  "reference": "...",
  "answer": "...",
  "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
  "retrieved": [
      {"source": "doc_rag_overview.txt", "chunk_index": 0,
       "score": 0.9134, "text": "..."},
      ...
  ],
}
```

Console output per question:

```
  [1/3] What is RAG and why does it reduce hallucinations?
      retrieved 3 chunks · 651 tokens · answer: Retrieval-Augmented Generation (RAG) is a technique that enhances LLM …
```

The dataset is wrapped once:

```python
EvaluationDataset(samples=samples)
```

---

## Stage C — Score with RAGAS

**Method:** `RAGASEvaluator.evaluate(dataset, metric_names)`

### Judge model wiring

RAGAS is an LLM-as-judge framework: every metric prompts a model to decompose claims, generate questions, or compare answers. Both judge components go through OpenRouter using the same key as the app.

```python
chat = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0,
    timeout=60,
    max_retries=3,
)
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,   # OpenRouter expects raw strings
)

ragas_llm = LangchainLLMWrapper(chat)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
```

`check_embedding_ctx_length=False` is required: OpenRouter rejects the pre-tokenized integer arrays LangChain would otherwise send.

### The call

```python
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ...],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False,
    show_progress=True,
)
dataframe = result.to_pandas()
```

`raise_exceptions=False` means a single failed judge call turns that one metric cell into `NaN` instead of aborting the whole run. The summary counts those failures per metric so silent degradation is visible.

> **Why classic metric classes?** RAGAS 0.4 also ships `ragas.metrics.collections`, but `evaluate()` validates `isinstance(metric, Metric)` and rejects the collection variants. The legacy classes in `ragas.metrics` still work and are used here; their deprecation warning is filtered because it is noise for this version.

---

## The Five Metrics

| Metric | Needs reference? | What it measures | Judge signal |
|---|---|---|---|
| `faithfulness` | no | Are answer claims supported by retrieved context? | claim → context NLI |
| `answer_relevancy` | no | Does the answer address the question? | generated questions vs. original |
| `context_precision` | yes | Are relevant chunks ranked above noise? | chunk → reference relevance |
| `context_recall` | yes | Does context cover the reference answer? | reference claims → context |
| `answer_correctness` | yes | Final answer vs. known-correct answer | facts + semantic similarity |

### 1. Faithfulness — hallucination check

1. LLM splits the answer into atomic claims.
2. For each claim, the judge decides whether it is supported by `retrieved_contexts`.
3. Score = supported claims / total claims.

`1.0` means every statement traces back to retrieved text. Low faithfulness means the model added outside knowledge or invented facts. This is the metric most directly tied to RAG's purpose.

### 2. Answer Relevancy — did it answer the question?

1. The judge generates several candidate questions from the answer alone.
2. Each candidate is embedded and compared to the original question (cosine similarity).
3. Score = mean similarity.

A vague, padded, or evasive answer drifts away from the original question and scores lower. This metric uses the embedding model, not just the chat model.

> In a real run against OpenRouter, RAGAS asks for `n=3` candidate questions but the provider returns 1. RAGAS logs a warning and proceeds with what it got. The evaluator installs a logging filter to suppress that specific repeated warning; the score itself is unaffected.

### 3. Context Precision — retrieval ranking quality

Reference-based. The judge marks each retrieved chunk as relevant or not relevant to the reference answer, then computes a rank-weighted precision:

```
                Σ (precision@k × relevance@k)
precision = ───────────────────────────────────
                     number of relevant chunks
```

Because it is rank-weighted, a relevant chunk sitting at rank 1 scores better than the same chunk at rank 5. This is the metric that catches "the right document was retrieved, but buried under noise".

### 4. Context Recall — retrieval coverage

Reference-based. The reference answer is decomposed into claims, and each claim is checked against the concatenated retrieved contexts:

```
              supported reference claims
recall = ────────────────────────────────────
              total reference claims
```

Low recall means retrieval missed documents the correct answer depends on. No amount of prompt tuning fixes that — the fix is chunking, embedding model, or `top_k`.

### 5. Answer Correctness — agreement with ground truth

Reference-based, combining two signals:

```
correctness = 0.75 × factual_score + 0.25 × semantic_similarity
```

- **Factual score** — TP/FP/FN classification of statements against the reference, turned into an F1.
- **Semantic similarity** — embedding cosine similarity between answer and reference.

The defaults favour factual agreement over phrasing. Unlike faithfulness (which only checks the context), this metric checks whether the answer is actually *right*.

### Reference-dependent metrics are skipped safely

If any evaluation item has no `reference`, `_effective_metric_names()` drops `context_precision`, `context_recall`, and `answer_correctness` with a printed warning. If nothing remains, it raises a clear error.

```python
REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness"}
```

---

## Evaluation Dataset

Built-in questions target the sample corpus:

```python
DEFAULT_QUESTIONS = [
    {"question": "What is RAG and why does it reduce hallucinations?",
     "reference": "RAG retrieves the most semantically similar document chunks ..."},
    {"question": "Which chunking strategies are described and how do they differ?",
     "reference": "Fixed-size ... sentence-based ... paragraph-based ... recursive ..."},
    {"question": "How does Pinecone differ from FAISS as a vector store?",
     "reference": "Pinecone is a managed vector database ... FAISS is an in-process library ..."},
    {"question": "Which embedding model is used and what dimensionality does it produce?",
     "reference": "openai/text-embedding-3-small, 1536 dimensions."},
    {"question": "What three parts make up a well-designed RAG prompt?",
     "reference": "System instruction, labelled context block, user question."},
]
```

To evaluate a private document set, supply JSON:

```json
[
  {
    "question": "What is the refund window?",
    "reference": "Customers may request a refund within 30 days of purchase."
  },
  {
    "question": "Who approves expense reports?",
    "reference": "Direct managers approve expenses up to $500; finance approves above that."
  }
]
```

`reference` is optional per item. `question` is required. `--limit N` truncates the list for cheap smoke runs.

---

## The Evaluator Class

```python
evaluator = RAGASEvaluator(
    index_name="rag-week5",        # default: env or Week 5 constant
    namespace="Default",           # case-sensitive
    top_k=3,
    llm_model="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-small",
    metric_names=["faithfulness", "answer_relevancy"],
    show_progress=True,
)

report = evaluator.run(items)                  # A → B → C → D
paths  = evaluator.save_report(report)         # CSV + JSON
print(report.summary)                          # {"faithfulness": 0.95, ...}
print(report.dataframe)                        # per-sample pandas table
```

### Method map

| Method | Stage | Purpose |
|---|---|---|
| `connect()` | A | Open the index, verify the namespace is non-empty |
| `answer_question(q)` | B | Retrieve → augment → generate for one question |
| `build_dataset(items)` | B | Run the pipeline for all items, return dataset + traces |
| `evaluate(dataset, metrics)` | C | Wire judge models, call RAGAS, return result + DataFrame |
| `run(items)` | A–D | End-to-end orchestration, returns `EvaluationReport` |
| `save_report(report)` | D | Write timestamped CSV and JSON |

### `EvaluationReport`

```python
@dataclass
class EvaluationReport:
    metric_names: List[str]
    summary: Dict[str, Optional[float]]     # mean per metric
    failed_counts: Dict[str, int]           # NaN cells per metric
    records: List[Dict[str, Any]]           # per-question detail
    dataframe: pd.DataFrame                 # raw RAGAS table
    total_tokens: Optional[int]             # RAG pipeline tokens (not judge)
```

### A note on token accounting

RAGAS `EvaluationResult.total_tokens()` raises unless `evaluate()` was given a cost callback. Rather than add that plumbing, the report sums the token usage captured from the **RAG pipeline's own** generation calls:

```python
total = sum(int(trace["usage"].get("total_tokens") or 0) for trace in traces)
```

This is the number that matters for app cost. Judge-model tokens are separate and are visible in the OpenRouter dashboard.

---

## CLI

```bash
# Evaluate all 5 built-in questions against the configured namespace
python Week5_RAGAS_Evaluation.py

# Evaluate an explicit namespace
python Week5_RAGAS_Evaluation.py --namespace Default

# Cheap smoke test: first 3 questions, two metrics
python Week5_RAGAS_Evaluation.py --limit 3 --metrics faithfulness,answer_relevancy

# Custom question set, larger context window
python Week5_RAGAS_Evaluation.py --questions eval_questions.json --top-k 5

# Write reports somewhere else, hide the progress bar
python Week5_RAGAS_Evaluation.py --output-dir reports --no-progress

# Discover metric names
python Week5_RAGAS_Evaluation.py --list-metrics
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--questions` | built-in set | JSON list of `{question, reference}` |
| `--index` | env / `rag-week5` | Pinecone index name |
| `--namespace` | env / `Default` | Namespace to evaluate (case-sensitive) |
| `--top-k` | `3` | Chunks retrieved per question |
| `--limit` | none | Evaluate only the first N items |
| `--metrics` | all five | Comma-separated metric names |
| `--output-dir` | `ragas_reports` | Where CSV/JSON are written |
| `--no-progress` | off | Hide the RAGAS progress bar |
| `--list-metrics` | — | Print available metrics and exit |

Errors from `RuntimeError` / `ValueError` are printed as `ERROR: ...` with exit code 1, so the script is CI-friendly.

---

## Outputs

### Console summary

```
========================================================================
  RAGAS EVALUATION SUMMARY
========================================================================

  Metric                     Score  Failed
  ------------------------------------------
  faithfulness               0.842       0
  answer_relevancy           0.768       0
  context_precision          1.000       0
  context_recall             0.667       0
  answer_correctness         0.673       0

  RAG pipeline tokens used: 1,985
```

### CSV — `ragas_reports/ragas_report_<timestamp>.csv`

The raw `result.to_pandas()` table: `user_input`, `retrieved_contexts`, `response`, `reference`, and one column per metric.

### JSON — `ragas_reports/ragas_report_<timestamp>.json`

```json
{
  "generated_at": "2026-10-31T00:56:11",
  "index": "rag-week5",
  "namespace": "ragas-verify-...",
  "top_k": 3,
  "judge_llm": "openai/gpt-4o-mini",
  "embedding_model": "openai/text-embedding-3-small",
  "samples": 3,
  "metrics": {"faithfulness": 0.8417, "...": 0.0},
  "failed_counts": {"faithfulness": 0, "...": 0},
  "rag_pipeline_total_tokens": 1985,
  "records": [
    {
      "question": "...",
      "reference": "...",
      "response": "...",
      "sources": "doc_rag_overview.txt#0, ...",
      "retrieval_scores": "0.9134, 0.8612, 0.8201",
      "rag_total_tokens": 651,
      "retrieved_contexts": ["..."],
      "faithfulness": 0.9
    }
  ]
}
```

The JSON keeps per-question answers, references, sources, retrieval scores, and metric values together — enough to debug a regression without re-running the pipeline.

---

## Interpreting Scores

| Symptom | Likely cause | Fix |
|---|---|---|
| Low `faithfulness` | LLM added outside knowledge | stricter system prompt, lower temperature |
| Low `context_recall` | retriever missed the answer | better chunking, more `top_k`, different embedding model |
| Low `context_precision` | right chunk ranked below noise | reranking, higher-quality embeddings, metadata filters |
| Low `answer_relevancy` | answer is vague or off-topic | prompt clarity, `top_k` too low/high |
| High recall, low correctness | context present but answer wrong | prompt grounding, model quality |
| High correctness, low faithfulness | answer right but not grounded | may be memorized from training data — inspect contexts |

Guidelines for the 0–1 scale on this corpus:

- **> 0.85** — solid
- **0.7 – 0.85** — usable, worth tuning
- **< 0.7** — investigate retrieval and prompt before trusting the app

A single aggregate number is never the goal. Read the per-question records: a pipeline that scores 0.8 because 4 questions are perfect and 1 is zero has a different problem from one that scores 0.8 everywhere.

---

## Dependency Notes

```txt
ragas>=0.4.3
pandas>=2.0.0

langchain>=0.3.27,<0.4
langchain-core>=0.3.72,<0.4
langchain-community>=0.3.27,<0.4
langchain-openai>=0.3.20,<0.4
langchain-text-splitters>=0.3.9,<0.4
```

**Why pin LangChain to 0.3?** RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai` at import time. That module was removed in langchain-community 0.4, so installing RAGAS against the newest LangChain produces:

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

The 0.3 series still exposes it, and RAGAS runs cleanly. The pins are ranges, not exact versions, so security patches within 0.3 are still picked up.

Install/repair:

```bash
pip install -r requirements.txt
# or, explicitly
pip install "ragas>=0.4.3" "langchain>=0.3.27,<0.4" "langchain-core>=0.3.72,<0.4" \
            "langchain-community>=0.3.27,<0.4" "langchain-openai>=0.3.20,<0.4" \
            "langchain-text-splitters>=0.3.9,<0.4"
```

---

## Running a Safe Verification

Evaluation reads from a namespace; it never writes or deletes. To verify without touching the app's real data, use a temporary namespace:

```bash
NS="ragas-verify-$(date +%Y%m%d%H%M%S)"

python Week5_RAG_Pinecone.py ingest --source sample --namespace "$NS"
python Week5_RAGAS_Evaluation.py --namespace "$NS" --limit 3
```

Clean up with the same helper the app uses:

```python
import importlib.util
from dotenv import load_dotenv
load_dotenv(".env")

spec = importlib.util.spec_from_file_location("w5", "Week5_RAG_Pinecone.py")
w5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w5)

index = w5.connect_existing_index(w5.DEFAULT_INDEX_NAME)
w5.clear_namespace(index, "ragas-verify-...")
print(w5.index_stats(index))
```

Namespaces are **case-sensitive** on Pinecone: `Default` and `default` are different namespaces.

---

## Limitations

- **LLM-as-judge is not ground truth.** Different judge models produce different scores. Compare runs that use the same judge, prompt set, and metrics — not scores across frameworks.
- **Cost and latency.** Every metric makes judge calls. Five metrics × N questions means many API requests. Use `--limit` and `--metrics` while iterating.
- **Small samples are noisy.** Five questions give a directional signal, not a benchmark. Use 30–100 questions for meaningful comparisons.
- **Reference quality dominates reference metrics.** A sloppy reference answer caps `context_recall` and `answer_correctness`. Write references as complete, factual answers.
- **`answer_relevancy` is sensitive to OpenRouter's `n` handling.** Scores can drift slightly between runs even at temperature 0.
- **Pinecone eventual consistency.** If you ingest and immediately evaluate, a fresh vector may not be queryable yet. The Week 5 ingest path already waits for vector counts to settle.
- **Evaluation is passive.** It reports problems; it does not tune chunk size, embeddings, or prompts for you.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Namespace '...' in index '...' has no vectors` | Namespace not ingested | `python Week5_RAG_Pinecone.py ingest --source sample --namespace Default` |
| `OPENROUTER_API_KEY is not set` | Missing `.env` | add the key, or export it |
| `ModuleNotFoundError: ... vertexai` | LangChain 0.4 installed | reinstall the pinned 0.3 stack |
| `Unknown metric(s)` | Typo in `--metrics` | run `--list-metrics` |
| All metrics `NaN` | Judge calls failing | check OpenRouter credits and model access |
| Deprecation warnings at startup | RAGAS legacy metrics/wrappers | expected; filtered by the script |
| `result.total_tokens()` ValueError | No cost callback configured | not used; the script sums pipeline usage instead |

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAGAS EVALUATION PIPELINE                      │
├────────┬────────────────────────┬──────────────────────────────┤
│ Stage  │ What                   │ How (this script)            │
├────────┼────────────────────────┼──────────────────────────────┤
│   A    │ Connect                │ Real Pinecone index +        │
│        │                        │ namespace, non-empty check   │
├────────┼────────────────────────┼──────────────────────────────┤
│   B    │ Run the RAG pipeline   │ Week 5 stage5/6/7 per        │
│        │                        │ question → SingleTurnSample  │
├────────┼────────────────────────┼──────────────────────────────┤
│   C    │ Score with RAGAS       │ ChatOpenAI + OpenAIEmbeddings│
│        │                        │ via OpenRouter as judge      │
├────────┼────────────────────────┼──────────────────────────────┤
│   D    │ Report                 │ Console table + CSV + JSON   │
│        │                        │ with per-question traces     │
└────────┴────────────────────────┴──────────────────────────────┘
```

---

## Key Concepts to Remember

- **Evaluate the real pipeline.** Reusing the Week 5 stages means a score change reflects an actual app change, not a reimplementation drift.
- **Retrieval metrics and generation metrics are separate.** Low `context_recall` is a retrieval bug; low `faithfulness` is a generation bug. Fixing one does not fix the other.
- **References unlock half the metrics.** `context_precision`, `context_recall`, and `answer_correctness` need ground truth. Without references, you can still measure grounding and relevancy.
- **Faithfulness is the RAG-specific metric.** It is the closest measure of "did the answer come from the documents".
- **Same judge, same questions, same metrics — only then compare.** RAGAS scores are relative, not absolute.
- **Cost scales with samples × metrics.** Use `--limit` during development and a full run before claiming an improvement.
- **Keep the dependency pins.** RAGAS 0.4.3 and LangChain 0.4 do not import together; the 0.3 series is the working combination.
