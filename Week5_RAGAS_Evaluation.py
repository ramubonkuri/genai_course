"""
=============================================================================
  Week 5 Extension: RAG Evaluation with RAGAS
=============================================================================

Evaluates the Week 5 Pinecone RAG pipeline with RAGAS metrics, using the real
OpenRouter models and the real Pinecone index that the app already uses.

Pipeline under evaluation (reused from Week5_RAG_Pinecone.py):

  question → embed → Pinecone top-k → augment prompt → gpt-4o-mini → answer

For each evaluation question this script:

  A. CONNECT      : connect to the existing Pinecone index / namespace
  B. RUN RAG      : retrieve contexts and generate an answer (Stages 5-7)
  C. SCORE        : build a RAGAS dataset and compute metrics
  D. REPORT       : print a summary and save CSV + JSON reports

RAGAS metrics:

  faithfulness        answer claims supported by retrieved context
  answer_relevancy    how well the answer addresses the question (embeddings)
  context_precision   whether relevant chunks rank above irrelevant ones
  context_recall      whether retrieved context covers the reference answer
  answer_correctness  factual + semantic agreement with the reference

Example evaluation question:

"Which alumni studied at Osmania University and have Python experience?"

Your RAG system searches the alumni database and generates an answer.
1. Faithfulness
Question: Is the AI telling the truth based on the retrieved information?

Checks whether the AI's answer is supported by the information retrieved from the database.

Example:
- Retrieved data: Ravi studied at Osmania University and knows Python.
- AI answer: Ravi studied at Osmania University and knows Python.
Faithful
The answer matches the retrieved information.

If AI additionally says Ravi worked at Google, but the retrieved data doesn't mention Google, that's a faithfulness problem.

2. Answer Relevancy
Question: Is the AI answering what the user actually asked?

Checks whether the generated answer is relevant to the user's question.

Example:
- User: Which alumni studied at Osmania University?
- AI: Here are alumni who studied at Osmania University.
Relevant
If AI starts explaining the history of Osmania University, the answer is not relevant.


3. Context Precision
Question: Did the search retrieve the right information without too much unnecessary data?

Checks how much of the retrieved information is actually useful for answering the question.

Example:

User asks for alumni who studied at Osmania University and know Python.

Search retrieves:
- Ravi – Osmania University, Python.
- Kumar – Osmania University, Java.
- Suresh – Andhra University, Python.
- Krishna – Osmania University, Python.
Ravi and Krishna are relevant. The others are unnecessary for this particular query.

Higher context precision means fewer irrelevant search results.


4. Context Recall
Question: Did the search retrieve all the information needed to answer the question?

Checks whether the retrieval system found all the relevant information available in the database.

Example:

Your database has 10 alumni matching the question.

But your search retrieves only 6.

Your context recall is low because 4 relevant alumni were missed.

Higher context recall means fewer relevant records are missed.

5. Answer Correctness
Question: Is the final answer factually correct compared with the expected answer?

Checks whether the AI's final answer matches the ground-truth or reference answer.

Example:

Expected answer:
- Ravi
- Krishna
- Mahesh
AI answer:
- Ravi
- Krishna
The answer is partially correct but incomplete.

Answer correctness evaluates the final result against the expected answer.


context_precision, context_recall and answer_correctness require a reference
answer. If any evaluation item lacks one, these metrics are skipped.

Run:
  python Week5_RAGAS_Evaluation.py
  python Week5_RAGAS_Evaluation.py --questions eval_questions.json
  python Week5_RAGAS_Evaluation.py --namespace Default --top-k 5 --limit 3
  python Week5_RAGAS_Evaluation.py --metrics faithfulness,answer_relevancy
  python Week5_RAGAS_Evaluation.py --list-metrics

The target namespace must already contain ingested documents. Example:
  python Week5_RAG_Pinecone.py ingest --source sample --namespace Default

Required .env:
  OPENROUTER_API_KEY=...
  PINECONE_API_KEY=...

Install dependencies:
  pip install -r requirements.txt
=============================================================================
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

# RAGAS 0.4 still supports the classic metric classes, but warns about their
# future move to ragas.metrics.collections; evaluate() only accepts the
# classic Metric subclasses, so the deprecation notice is noise here.
warnings.filterwarnings(
    "ignore", message=r".*from 'ragas\.metrics' is deprecated.*"
)
# The LangChain wrappers are the only reliable path through OpenRouter, since
# RAGAS' modern factories build their own OpenAI client without a base_url.
warnings.filterwarnings(
    "ignore", message=r".*LangchainLLMWrapper is deprecated.*"
)
warnings.filterwarnings(
    "ignore", message=r".*LangchainEmbeddingsWrapper is deprecated.*"
)

# OpenRouter's gpt-4o-mini ignores RAGAS' request for n>1 candidates when the
# metric only needs one; RAGAS logs a warning for every such call.
class _QuietMultiGeneration(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "generations instead of requested" not in record.getMessage()


logging.getLogger("ragas").addFilter(_QuietMultiGeneration())

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
#DEFAULT_LLM_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

METRIC_REGISTRY = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "answer_correctness": AnswerCorrectness,
}
DEFAULT_METRIC_NAMES = list(METRIC_REGISTRY)
REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness"}

# Questions for the built-in sample corpus. Point --questions at your own JSON
# file when evaluating a private document set.
DEFAULT_QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "What is RAG and why does it reduce hallucinations?",
        "reference": (
            "RAG retrieves the most semantically similar document chunks from a "
            "vector database and injects them into the prompt at inference time. "
            "The LLM answers using the provided context instead of relying only on "
            "its training weights, which reduces hallucination and keeps answers "
            "grounded in real, up-to-date information."
        ),
    },
    {
        "question": "Which chunking strategies are described and how do they differ?",
        "reference": (
            "Fixed-size chunking splits every N characters with overlap; "
            "sentence-based chunking splits on sentence boundaries; paragraph-based "
            "chunking splits on double newlines into natural semantic units; and "
            "recursive splitting tries paragraph, sentence, then word separators "
            "while respecting a maximum chunk size."
        ),
    },
    {
        "question": "How does Pinecone differ from FAISS as a vector store?",
        "reference": (
            "Pinecone is a managed vector database that provides serverless indexes, "
            "namespaces and metadata filtering without infrastructure to operate. "
            "FAISS is an in-process library that stores vectors in local memory and "
            "requires manual operation for production scale."
        ),
    },
    {
        "question": "Which embedding model is used and what dimensionality does it produce?",
        "reference": (
            "The pipeline uses openai/text-embedding-3-small, which produces "
            "1536-dimensional vectors."
        ),
    },
    {
        "question": "What three parts make up a well-designed RAG prompt?",
        "reference": (
            "A system instruction telling the model to answer only from the provided "
            "context, a labelled context block containing the retrieved chunks, and "
            "the user's original question."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def load_pipeline_module() -> Any:
    """Load Week5_RAG_Pinecone.py once so its stages can be reused as-is."""
    module_path = BASE_DIR / "Week5_RAG_Pinecone.py"
    spec = importlib.util.spec_from_file_location("week5_rag_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the Week 5 pipeline from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_questions(path: Optional[str], limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Load evaluation items from JSON or fall back to the built-in set."""
    if path:
        """ Home work: load questions from a JSON file. Each item must be a dict with a non-empty 'question' field, and an optional 'reference' field. """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Questions file must contain a JSON list of objects.")
        items = []
        for entry in data:
            if not isinstance(entry, dict) or not str(entry.get("question", "")).strip():
                raise ValueError("Each item needs a non-empty 'question' field.")
            items.append(
                {
                    "question": str(entry["question"]).strip(),
                    "reference": str(entry.get("reference", "")).strip() or None,
                }
            )
    else:
        items = [dict(item) for item in DEFAULT_QUESTIONS]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        items = items[:limit]
    if not items:
        raise ValueError("No evaluation questions were provided.")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    metric_names: List[str]
    summary: Dict[str, Optional[float]]
    failed_counts: Dict[str, int]
    records: List[Dict[str, Any]]
    dataframe: pd.DataFrame
    total_tokens: Optional[int]


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Runs the Week 5 RAG pipeline and scores its output with RAGAS.

    The class deliberately reuses stage5_retrieve, stage6_build_prompt and
    stage7_generate from Week5_RAG_Pinecone.py, so evaluation measures the
    production pipeline rather than a reimplementation.
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        top_k: int = 3,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        metric_names: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> None:
        self.pipeline = load_pipeline_module()

        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", self.pipeline.DEFAULT_INDEX_NAME
        )
        self.namespace = namespace or os.getenv(
            "PINECONE_NAMESPACE", self.pipeline.DEFAULT_NAMESPACE
        )
        self.top_k = top_k
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.show_progress = show_progress

        self.metric_names = list(metric_names) if metric_names else list(DEFAULT_METRIC_NAMES)
        unknown = [name for name in self.metric_names if name not in METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {', '.join(unknown)}. "
                f"Choose from: {', '.join(METRIC_REGISTRY)}"
            )
        if not self.metric_names:
            raise ValueError("At least one metric is required.")

        self.index: Optional[Any] = None
        self.namespace_size: int = 0
        self.ragas_llm: Optional[LangchainLLMWrapper] = None
        self.ragas_embeddings: Optional[LangchainEmbeddingsWrapper] = None

    # ── Stage A: connection ──────────────────────────────────────────────

    def connect(self) -> Any:
        """Connect to Pinecone and refuse to evaluate an empty namespace."""
        if self.index is not None:
            return self.index

        index = self.pipeline.connect_existing_index(self.index_name)
        stats = self.pipeline.index_stats(index)
        count = stats["namespaces"].get(self.namespace, 0)
        if count == 0:
            raise RuntimeError(
                f"Namespace '{self.namespace}' in index '{self.index_name}' has no "
                "vectors. Ingest documents first, for example:\n"
                f"  python Week5_RAG_Pinecone.py ingest --source sample "
                f"--namespace {self.namespace}"
            )

        self.index = index
        self.namespace_size = count
        return index

    # ── RAGAS model wiring ───────────────────────────────────────────────

    def _ensure_ragas_models(self) -> None:
        if self.ragas_llm is not None and self.ragas_embeddings is not None:
            return

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or the environment."
            )

        chat = ChatOpenAI(
            model=self.llm_model,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.0,
            timeout=60,
            max_retries=3,
        )
        embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            # OpenRouter expects raw strings, not OpenAI token arrays.
            check_embedding_ctx_length=False,
        )

        self.ragas_llm = LangchainLLMWrapper(chat)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ── Stage B: run the RAG pipeline ────────────────────────────────────

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Run retrieval, prompt augmentation and generation for one question."""
        index = self.connect()

        retrieved = self.pipeline.stage5_retrieve(
            question,
            index,
            namespace=self.namespace,
            top_k=self.top_k,
        )
        system_prompt, user_prompt = self.pipeline.stage6_build_prompt(
            question, retrieved
        )
        answer, usage = self.pipeline.stage7_generate(system_prompt, user_prompt)

        return {"answer": answer, "usage": usage, "retrieved": retrieved}

    def build_dataset(
        self, items: List[Dict[str, str]]
    ) -> Tuple[EvaluationDataset, List[Dict[str, Any]]]:
        """Execute the RAG pipeline for every question and collect traces."""
        samples: List[SingleTurnSample] = []
        traces: List[Dict[str, Any]] = []

        for position, item in enumerate(items, start=1):
            question = item["question"]
            print(f"\n  [{position}/{len(items)}] {question}")

            output = self.answer_question(question)
            retrieved = output["retrieved"]

            print(f"      retrieved {len(retrieved)} chunks · "
                  f"{output['usage'].get('total_tokens', 0)} tokens · "
                  f"answer: {output['answer'][:70]}…")

            traces.append(
                {
                    "question": question,
                    "reference": item.get("reference"),
                    "answer": output["answer"],
                    "usage": output["usage"],
                    "retrieved": [
                        {
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "score": chunk["retrieval_score"],
                            "text": chunk["text"],
                        }
                        for chunk in retrieved
                    ],
                }
            )

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=[chunk["text"] for chunk in retrieved],
                    response=output["answer"],
                    reference=item.get("reference"),
                )
            )

        return EvaluationDataset(samples=samples), traces

    # ── Stage C: compute RAGAS metrics ───────────────────────────────────

    def _effective_metric_names(self, items: List[Dict[str, str]]) -> List[str]:
        names = list(self.metric_names)
        if any(not item.get("reference") for item in items):
            dropped = [name for name in names if name in REFERENCE_METRICS]
            if dropped:
                print(
                    f"\n  [!] Some questions have no reference answer — skipping: "
                    f"{', '.join(dropped)}"
                )
                names = [name for name in names if name not in REFERENCE_METRICS]
        if not names:
            raise ValueError(
                "All selected metrics require reference answers, but at least one "
                "question has no 'reference'. Add references or choose "
                "faithfulness / answer_relevancy."
            )
        return names

    def evaluate(
        self,
        dataset: EvaluationDataset,
        metric_names: List[str],
    ) -> Tuple[Any, pd.DataFrame]:
        """Run RAGAS with the real judge LLM and embedding model."""
        self._ensure_ragas_models()
        metrics = [METRIC_REGISTRY[name]() for name in metric_names]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            raise_exceptions=False,
            show_progress=self.show_progress,
        )
        return result, result.to_pandas()

    # ── Stage D: report ──────────────────────────────────────────────────

    @staticmethod
    def _rag_tokens(traces: List[Dict[str, Any]]) -> Optional[int]:
        """Total tokens spent by the RAG pipeline itself (not the RAGAS judge)."""
        total = sum(
            int(trace["usage"].get("total_tokens") or 0) for trace in traces
        )
        return total or None

    def _build_records(
        self,
        dataframe: pd.DataFrame,
        traces: List[Dict[str, Any]],
        metric_names: List[str],
    ) -> List[Dict[str, Any]]:
        records = []
        for position, trace in enumerate(traces):
            row = dataframe.iloc[position]
            record = {
                "question": trace["question"],
                "reference": trace["reference"],
                "response": trace["answer"],
                "sources": ", ".join(
                    f"{chunk['source']}#{chunk['chunk_index']}"
                    for chunk in trace["retrieved"]
                ),
                "retrieval_scores": ", ".join(
                    f"{chunk['score']:.4f}" for chunk in trace["retrieved"]
                ),
                "rag_total_tokens": trace["usage"].get("total_tokens"),
                "retrieved_contexts": [chunk["text"] for chunk in trace["retrieved"]],
            }
            for name in metric_names:
                value = row.get(name)
                record[name] = None if pd.isna(value) else round(float(value), 4)
            records.append(record)
        return records

    def run(self, items: List[Dict[str, str]]) -> EvaluationReport:
        """Full evaluation: connect, run the pipeline, score, summarise."""
        if not items:
            raise ValueError("No evaluation items were provided.")

        self.connect()
        metric_names = self._effective_metric_names(items)

        print(f"\n  Index     : {self.index_name}")
        print(f"  Namespace : {self.namespace} ({self.namespace_size:,} vectors)")
        print(f"  Top K     : {self.top_k}")
        print(f"  Metrics   : {', '.join(metric_names)}")
        print(f"  Samples   : {len(items)}")

        dataset, traces = self.build_dataset(items)
        result, dataframe = self.evaluate(dataset, metric_names)

        summary: Dict[str, Optional[float]] = {}
        failed_counts: Dict[str, int] = {}
        for name in metric_names:
            series = pd.to_numeric(dataframe.get(name), errors="coerce")
            mean = series.mean(skipna=True)
            summary[name] = None if pd.isna(mean) else round(float(mean), 4)
            failed_counts[name] = int(series.isna().sum())

        return EvaluationReport(
            metric_names=metric_names,
            summary=summary,
            failed_counts=failed_counts,
            records=self._build_records(dataframe, traces, metric_names),
            dataframe=dataframe,
            total_tokens=self._rag_tokens(traces),
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_dir: str = "ragas_reports",
    ) -> Dict[str, Path]:
        """Save the per-sample table (CSV) and full report (JSON)."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = out_dir / f"ragas_report_{timestamp}.csv"
        json_path = out_dir / f"ragas_report_{timestamp}.json"

        report.dataframe.to_csv(csv_path, index=False)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "index": self.index_name,
            "namespace": self.namespace,
            "top_k": self.top_k,
            "judge_llm": self.llm_model,
            "embedding_model": self.embedding_model,
            "samples": len(report.records),
            "metrics": report.summary,
            "failed_counts": report.failed_counts,
            "rag_pipeline_total_tokens": report.total_tokens,
            "records": report.records,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return {"csv": csv_path, "json": json_path}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(report: EvaluationReport) -> None:
    banner("RAGAS EVALUATION SUMMARY")
    print(f"\n  {'Metric':<24}{'Score':>8}{'Failed':>8}")
    print("  " + "-" * 42)
    for name in report.metric_names:
        score = report.summary.get(name)
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(f"  {name:<24}{score_text:>8}{report.failed_counts.get(name, 0):>8}")
    if report.total_tokens is not None:
        print(f"\n  RAG pipeline tokens used: {report.total_tokens:,}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Evaluate the Week 5 Pinecone RAG pipeline with RAGAS.",
    )
    parser.add_argument(
        "--questions",
        help="JSON file with [{'question': ..., 'reference': ...}] items.",
    )
    parser.add_argument(
        "--index",
        default=os.getenv("PINECONE_INDEX_NAME"),
        help="Pinecone index name (default: PINECONE_INDEX_NAME or rag-week5).",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE"),
        help="Pinecone namespace to evaluate (default: PINECONE_NAMESPACE or Default).",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRIC_NAMES),
        help=f"Comma-separated metrics. Available: {', '.join(METRIC_REGISTRY)}",
    )
    parser.add_argument("--output-dir", default="ragas_reports")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--list-metrics", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_metrics:
        print("Available metrics:")
        for name in METRIC_REGISTRY:
            needs_reference = " (needs reference)" if name in REFERENCE_METRICS else ""
            print(f"  - {name}{needs_reference}")
        return 0

    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    items = load_questions(args.questions, limit=args.limit)

    evaluator = RAGASEvaluator(
        index_name=args.index,
        namespace=args.namespace,
        top_k=args.top_k,
        metric_names=metric_names,
        show_progress=not args.no_progress,
    )

    banner("RAGAS EVALUATION — WEEK 5 PIPELINE")
    print("  Running the real RAG pipeline, then scoring with RAGAS.")

    report = evaluator.run(items)
    paths = evaluator.save_report(report, args.output_dir)

    print_summary(report)
    print(f"\n  CSV  : {paths['csv']}")
    print(f"  JSON : {paths['json']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
