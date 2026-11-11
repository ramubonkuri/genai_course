"""
Week 5: RAG + RAGAS Example (standalone)
==================================================

This file contains the whole small demo. It does NOT import any of the other
Week 5 Python files.

The workflow is:
    1. Read sample documents or files from a folder.
    2. Split the text into chunks.
    3. Turn chunks into vectors (embeddings) with OpenRouter.
    4. Store the vectors in Pinecone.
    5. Run a predefined set of questions against Pinecone.
    6. Give retrieved chunks to an LLM and generate answers.
    7. Use RAGAS to score the answers and save scores as a CSV file.

Create a .env file beside this script with:
    OPENROUTER_API_KEY=your_openrouter_key
    PINECONE_API_KEY=your_pinecone_key

Install dependencies with:
    pip install -r requirements.txt

Examples:
    python Week5_RAGAS_Simple.py --source sample
    python Week5_RAGAS_Simple.py --source docs --path docs

The script does not delete old vectors unless --clear-namespace is supplied.
The built-in evaluation questions and reference answers are written for the
sample corpus. Update them in this file if you evaluate different documents.
"""

# Python's built-in libraries: no separate installation needed.
import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# python-dotenv reads key/value pairs from .env into environment variables.
from dotenv import load_dotenv

# Locate .env relative to this script, not the terminal's current directory.
SCRIPT_FOLDER = Path(__file__).resolve().parent
load_dotenv(SCRIPT_FOLDER / ".env")


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# OpenRouter provides OpenAI-compatible embedding and chat endpoints.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "openai/gpt-4o-mini"

# Pinecone connection details are fixed defaults for this beginner example.
# Change these values here only if you want to use a different Pinecone setup.
INDEX_NAME = "rag-week5-simple"
NAMESPACE = "simple-demo"
CLOUD = "aws"
REGION = "us-east-1"

# Small defaults keep the demo easy to follow and the requests manageable.
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3
EMBEDDING_BATCH_SIZE = 20
PINECONE_UPSERT_BATCH_SIZE = 100

# A tiny built-in corpus makes it possible to try the workflow without preparing files.
SAMPLE_DOCUMENTS = [
    {
        "source": "rag_basics.txt",
        "text": (
            "Retrieval-Augmented Generation (RAG) improves an LLM response by "
            "retrieving relevant information from a knowledge base and adding it "
            "to the prompt. The model can use this supplied context instead of "
            "depending only on information learned during training."
        ),
    },
    {
        "source": "embeddings.txt",
        "text": (
            "An embedding is a list of numbers that represents the meaning of text. "
            "Texts with similar meanings usually have vectors that are close together. "
            "The text-embedding-3-small model produces vectors with 1536 dimensions."
        ),
    },
    {
        "source": "pinecone.txt",
        "text": (
            "Pinecone is a managed vector database. Applications store vectors in an "
            "index and search the index for vectors similar to a query. A namespace "
            "is a partition inside an index that can keep one collection of data separate."
        ),
    },
    {
        "source": "chunking.txt",
        "text": (
            "Chunking divides a long document into smaller pieces before embedding. "
            "A small overlap between neighboring chunks helps preserve context near "
            "the boundary. Chunking by paragraph often keeps related ideas together."
        ),
    },
]

# These are the fixed questions RAGAS will evaluate every time the script runs.
# Each reference is the expected answer used by RAGAS reference-based metrics.
# They match SAMPLE_DOCUMENTS; edit these items when using your own documents.
EVALUATION_QUESTIONS = [
    {
        "question": "What is RAG and how does it help an LLM?",
        "reference": (
            "RAG retrieves relevant information from a knowledge base and adds it "
            "to the prompt, so the LLM can use supplied context instead of relying "
            "only on information learned during training."
        ),
    },
    {
        "question": "What is an embedding, and how many dimensions does the model produce?",
        "reference": (
            "An embedding is a list of numbers representing the meaning of text. "
            "The text-embedding-3-small model produces 1536-dimensional vectors."
        ),
    },
    {
        "question": "What is Pinecone and what is a namespace used for?",
        "reference": (
            "Pinecone is a managed vector database used to store and search vectors. "
            "A namespace is a partition inside an index that keeps collections of data separate."
        ),
    },
    {
        "question": "Why are documents split into chunks, and why can chunks overlap?",
        "reference": (
            "Chunking divides long documents into smaller pieces before embedding. "
            "A small overlap between neighboring chunks helps preserve context near boundaries."
        ),
    },
]


# ---------------------------------------------------------------------------
# STEP 0: CHECK CONFIGURATION
# ---------------------------------------------------------------------------

def check_api_keys() -> None:
    """Stop with a helpful message when a required API key is missing."""
    missing_keys = []
    if not os.getenv("OPENROUTER_API_KEY"):
        missing_keys.append("OPENROUTER_API_KEY")
    if not os.getenv("PINECONE_API_KEY"):
        missing_keys.append("PINECONE_API_KEY")

    if missing_keys:
        names = " and ".join(missing_keys)
        raise RuntimeError(
            f"Missing {names}. Add them to {SCRIPT_FOLDER / '.env'} "
            "or set them in your terminal environment."
        )


# ---------------------------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# ---------------------------------------------------------------------------

def load_documents(source: str, folder: str) -> List[Dict[str, str]]:
    """Read the built-in sample text or PDF/TXT/Markdown files from a folder."""
    if source == "sample":
        # Return a new list so later code can safely work with it.
        return [dict(document) for document in SAMPLE_DOCUMENTS]

    input_folder = Path(folder)
    if not input_folder.is_dir():
        raise RuntimeError(f"Document folder does not exist: {input_folder}")

    documents = []
    for file_path in sorted(input_folder.iterdir()):
        if file_path.suffix.lower() == ".pdf":
            # pypdf extracts text from text-based PDFs. Scanned-image PDFs need OCR.
            try:
                from pypdf import PdfReader
            except ImportError as error:
                raise RuntimeError("Install PDF support with: pip install pypdf") from error

            reader = PdfReader(str(file_path))
            page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(page for page in page_texts if page)
        elif file_path.suffix.lower() in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            # Ignore other file types rather than trying to decode them as text.
            continue

        text = text.strip()
        if text:
            documents.append({"source": file_path.name, "text": text})

    return documents


# ---------------------------------------------------------------------------
# STEP 2: CLEAN AND CHUNK DOCUMENTS
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace so extracted text is easier to split and search."""
    return " ".join(text.split())


def split_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split long text into overlapping character chunks."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be zero or more, and smaller than chunk size.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # When possible, end at a space so the chunk does not cut a word in half.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by the chunk length, minus overlap. The max() prevents
        # an accidental infinite loop if the chosen boundary is very short.
        next_start = end - overlap
        start = max(start + 1, next_start)

    return chunks


def make_chunk_records(documents: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Turn documents into chunks and keep each chunk's filename and number."""
    records = []
    for document in documents:
        text = clean_text(document["text"])
        chunks = split_into_chunks(text)
        for chunk_number, chunk_text in enumerate(chunks):
            records.append(
                {
                    "source": document["source"],
                    "chunk_number": chunk_number,
                    "text": chunk_text,
                }
            )
    return records


# ---------------------------------------------------------------------------
# STEP 3: CREATE EMBEDDINGS
# ---------------------------------------------------------------------------

def embed_texts(openai_client, texts: List[str]) -> List[List[float]]:
    """Ask OpenRouter to turn each text chunk into a vector of numbers."""
    all_vectors = []

    # Send a small group of texts per request instead of one large request.
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        # The API returns one vector per text, in the same order as the input.
        all_vectors.extend(item.embedding for item in response.data)

    return all_vectors


# ---------------------------------------------------------------------------
# STEP 4: CREATE/CONNECT TO PINECONE AND STORE VECTORS
# ---------------------------------------------------------------------------

def connect_to_index(pinecone_client, index_name: str, cloud: str, region: str):
    """Create a cosine-similarity index when needed, then return its handle."""
    from pinecone import ServerlessSpec

    if not pinecone_client.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}' ...")
        pinecone_client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        # A new managed index takes a short time to become ready.
        for _ in range(60):
            description = pinecone_client.describe_index(index_name)
            if getattr(description.status, "ready", False):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Pinecone index did not become ready in time.")

    # Check that an existing index matches the vectors made by our model.
    description = pinecone_client.describe_index(index_name)
    if int(description.dimension) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Index dimension is {description.dimension}; expected {EMBEDDING_DIMENSION}. "
            "Choose another index name or use the matching embedding model."
        )
    if str(description.metric).lower() != "cosine":
        raise RuntimeError("This example expects a Pinecone index using cosine similarity.")

    return pinecone_client.Index(index_name)


def make_vector_id(source: str, chunk_number: int) -> str:
    """Create a repeatable ID so re-ingesting the same chunk overwrites it."""
    safe_source = re.sub(r"[^a-zA-Z0-9._-]+", "-", source).strip("-").lower()
    return f"{safe_source or 'document'}-chunk-{chunk_number}"


def namespace_vector_count(index, namespace: str) -> int:
    """Read the current vector count for one Pinecone namespace."""
    stats = index.describe_index_stats()
    namespace_info = (stats.namespaces or {}).get(namespace)
    if namespace_info is None:
        return 0
    return int(getattr(namespace_info, "vector_count", 0) or 0)


def ingest_into_pinecone(
    documents: List[Dict[str, str]],
    openai_client,
    pinecone_client,
    index_name: str,
    namespace: str,
    cloud: str,
    region: str,
    clear_namespace: bool,
):
    """Clean, chunk, embed, and upsert documents into Pinecone."""
    # Convert raw document text into small records and preserve source metadata.
    chunks = make_chunk_records(documents)
    if not chunks:
        raise RuntimeError("No text chunks were created from the selected documents.")
    print(f"Created {len(chunks)} text chunk(s).")

    # Create or connect to the target index, then optionally clear the namespace.
    index = connect_to_index(pinecone_client, index_name, cloud, region)
    count_before = namespace_vector_count(index, namespace)
    if clear_namespace:
        print(f"Clearing namespace '{namespace}' before ingestion...")
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as error:
            # A namespace that does not exist yet is already empty. Reraise other errors.
            if "not found" not in str(error).lower() and "404" not in str(error):
                raise
        count_before = 0

    # Ask the embedding model to produce one vector for every chunk.
    print(f"Creating embeddings with {EMBEDDING_MODEL}...")
    vectors = embed_texts(openai_client, [chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("The embedding service returned an unexpected number of vectors.")

    # Pinecone stores vector values together with metadata used for citations.
    pinecone_vectors = []
    for chunk, vector in zip(chunks, vectors):
        pinecone_vectors.append(
            {
                "id": make_vector_id(str(chunk["source"]), int(chunk["chunk_number"])),
                "values": vector,
                "metadata": {
                    "source": chunk["source"],
                    "chunk_number": chunk["chunk_number"],
                    "text": chunk["text"],
                },
            }
        )

    # Upsert means insert a vector, or replace it when that ID already exists.
    for start in range(0, len(pinecone_vectors), PINECONE_UPSERT_BATCH_SIZE):
        batch = pinecone_vectors[start : start + PINECONE_UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Uploaded {min(start + len(batch), len(pinecone_vectors))}/{len(pinecone_vectors)} vectors.")

    # Pinecone is eventually consistent: uploaded vectors may take a moment to
    # appear in search. If the namespace was cleared or empty, wait for the
    # vector count to reach the number we just uploaded. When merging into an
    # existing namespace, IDs may overwrite old vectors, so a simple count
    # cannot tell whether those updates are visible; use a short settling pause.
    if clear_namespace or count_before == 0:
        expected_count = len(pinecone_vectors)
        for _ in range(30):
            if namespace_vector_count(index, namespace) >= expected_count:
                break
            time.sleep(2)
    else:
        time.sleep(2)

    print(f"Ingestion complete: {len(documents)} document(s), {len(chunks)} chunk(s).")
    return index


# ---------------------------------------------------------------------------
# STEP 5 & 6: RETRIEVE CONTEXT AND GENERATE AN ANSWER
# ---------------------------------------------------------------------------

def answer_question(openai_client, index, namespace: str, question: str, top_k: int):
    """Search Pinecone, build a grounded prompt, and ask the chat model."""
    # Use the same embedding model as ingestion so vectors share one space.
    query_vector = embed_texts(openai_client, [question])[0]

    search_result = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    # Extract the text and provenance from the best-matching Pinecone records.
    retrieved_chunks = []
    for match in search_result.matches:
        metadata = match.metadata or {}
        retrieved_chunks.append(
            {
                "source": metadata.get("source", "unknown"),
                "chunk_number": int(metadata.get("chunk_number", -1)),
                "text": metadata.get("text", ""),
                "score": float(match.score),
            }
        )
    if not retrieved_chunks:
        raise RuntimeError("Pinecone found no chunks. Check the index and namespace.")

    # Build context with source labels, then instruct the model not to invent facts.
    context = "\n\n---\n\n".join(
        f"[Source: {item['source']} | Chunk: {item['chunk_number']}]\n{item['text']}"
        for item in retrieved_chunks
    )
    system_message = (
        "Answer using only the document context provided. If the context does not "
        "contain the answer, say that you do not have enough information. "
        "Mention the source filename when useful."
    )
    user_message = f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    # Generate the answer. A low temperature makes answers less random.
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    answer = (response.choices[0].message.content or "").strip()
    token_count = getattr(response.usage, "total_tokens", 0) or 0
    return answer, retrieved_chunks, int(token_count)


# ---------------------------------------------------------------------------
# STEP 7: EVALUATE THE ANSWER WITH RAGAS
# ---------------------------------------------------------------------------

def evaluate_with_ragas(
    evaluation_samples: List[object],
    openrouter_api_key: str,
):
    """Score all prepared question/answer/context samples with RAGAS."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    dataset = EvaluationDataset(samples=evaluation_samples)

    # All predefined questions have references, so run the complete metric set.
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
    ]
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    # RAGAS needs an LLM judge and embeddings. LangChain adapters let RAGAS use
    # OpenRouter's OpenAI-compatible API endpoint.
    judge_llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        timeout=60,
        max_retries=3,
    )
    judge_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        raise_exceptions=False,
        show_progress=True,
    )
    return result.to_pandas(), metric_names


def explain_metric_scores(score_table, metric_names: List[str]) -> None:
    """Print what each score measures and a rough interpretation of its mean."""
    import pandas as pd

    # These descriptions explain what a high or low value says about the system.
    explanations = {
        "faithfulness": (
            "Checks whether claims in the answer are supported by the retrieved text. "
            "Low values can indicate unsupported or hallucinated claims."
        ),
        "answer_relevancy": (
            "Checks whether the answer addresses the question. A low value can mean "
            "the answer is off-topic or does not answer the question directly."
        ),
        "context_precision": (
            "Checks whether the retrieved chunks are relevant, especially near the top "
            "of the results. Low values suggest retrieval returned unnecessary text."
        ),
        "context_recall": (
            "Checks whether the retrieved chunks contain the information needed for "
            "the reference answer. Low values suggest relevant information was missed."
        ),
        "answer_correctness": (
            "Compares the generated answer with the reference answer for factual and "
            "semantic agreement. Low values may mean the answer is wrong or incomplete."
        ),
    }

    print("\nHow to interpret the metrics")
    print("Scores are generally between 0 and 1; higher is better.")
    print("These rough bands are guidance, not universal pass/fail thresholds:")
    print("  0.80–1.00  Strong signal")
    print("  0.60–0.79  Mixed; review the answers and retrieved chunks")
    print("  0.00–0.59  Needs investigation")

    for metric_name in metric_names:
        if metric_name not in score_table:
            print(f"\n{metric_name}: no score column was returned.")
            continue

        # RAGAS can return an empty value if a metric could not be calculated.
        values = pd.to_numeric(score_table[metric_name], errors="coerce").dropna()
        if values.empty:
            print(f"\n{metric_name}: no valid scores were returned.")
            continue

        average = float(values.mean())
        if average >= 0.80:
            interpretation = "strong signal"
        elif average >= 0.60:
            interpretation = "mixed; review the individual results"
        else:
            interpretation = "needs investigation"

        print(f"\n{metric_name} — average {average:.3f} ({interpretation})")
        print(f"  {explanations[metric_name]}")
        print(f"  Valid scores: {len(values)} of {len(score_table)} questions.")


# ---------------------------------------------------------------------------
# MAIN: CONNECT THE STEPS IN ORDER
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete ingest → ask → evaluate workflow."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Pinecone and evaluate predefined questions with RAGAS."
    )
    parser.add_argument("--source", choices=["sample", "docs"], default="sample",
                        help="Use built-in sample text or files from a folder.")
    parser.add_argument("--path", default="docs", help="Folder with PDF/TXT/MD files.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="How many chunks to retrieve (default: 3).")
    parser.add_argument("--clear-namespace", action="store_true",
                        help="Delete all vectors in this namespace before ingesting.")
    parser.add_argument("--output-dir", default="ragas_reports",
                        help="Folder for the CSV evaluation report.")
    args = parser.parse_args()

    # Import external services only after argparse handles --help. This way
    # people can view usage even before installing the project's dependencies.
    try:
        from openai import OpenAI
        from pinecone import Pinecone
        from ragas import SingleTurnSample
    except ImportError as error:
        print("Install the project packages first: pip install -r requirements.txt", file=sys.stderr)
        print(f"Missing package details: {error}", file=sys.stderr)
        return 1

    check_api_keys()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    # Make the two API clients. OpenRouter handles both chat and embeddings;
    # Pinecone stores the embeddings for later similarity search.
    openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
    openai_client = OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL)
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    print(f"Pinecone index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Serverless location: {CLOUD}/{REGION}")

    # Step 1: get source text and show a small summary.
    documents = load_documents(args.source, args.path)
    if not documents:
        raise RuntimeError(f"No readable documents found in source '{args.source}'.")
    print(f"Loaded {len(documents)} document(s).")

    # Steps 2-4: chunk, embed, and upsert into Pinecone.
    if args.clear_namespace:
        print("WARNING: the selected Pinecone namespace will be cleared first.")
    index = ingest_into_pinecone(
        documents=documents,
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
        cloud=CLOUD,
        region=REGION,
        clear_namespace=args.clear_namespace,
    )

    # Steps 5-6: run every predefined question. For each one, search Pinecone,
    # generate an answer, and keep the question/context/answer/reference together.
    evaluation_samples = []
    for number, item in enumerate(EVALUATION_QUESTIONS, start=1):
        question = item["question"]
        print(f"\n[{number}/{len(EVALUATION_QUESTIONS)}] {question}")

        answer, retrieved_chunks, token_count = answer_question(
            openai_client=openai_client,
            index=index,
            namespace=NAMESPACE,
            question=question,
            top_k=args.top_k,
        )
        print(f"Answer: {answer}")
        print(f"Retrieved {len(retrieved_chunks)} chunk(s); used {token_count} tokens.")

        # RAGAS needs the question, retrieved passages, generated response, and
        # expected answer in order to calculate the evaluation metrics.
        evaluation_samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=[chunk["text"] for chunk in retrieved_chunks],
                response=answer,
                reference=item["reference"],
            )
        )

    # Step 7: RAGAS scores the complete set of predefined examples.
    print("\nRAGAS makes additional model/API calls to calculate evaluation scores.")

    try:
        score_table, metric_names = evaluate_with_ragas(
            evaluation_samples=evaluation_samples,
            openrouter_api_key=openrouter_api_key,
        )
    except ImportError as error:
        print("Install the RAGAS dependencies with: pip install -r requirements.txt", file=sys.stderr)
        print(f"Missing package details: {error}", file=sys.stderr)
        return 1

    print("\nRAGAS evaluation scores:")
    print(score_table[metric_names].to_string(index=False))
    explain_metric_scores(score_table, metric_names)

    # Save the per-question metric scores for later review.
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_folder / f"simple_ragas_{timestamp}.csv"
    score_table.to_csv(report_path, index=False)
    print(f"\nCSV report saved to: {report_path.resolve()}")
    return 0


# Run main only when this file is executed directly.
if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
