# Week 4: RAG Pipeline with GUI and Web API

This project turns the Week 3 command-line Retrieval-Augmented Generation (RAG) pipeline into a full application with:

- a Streamlit graphical user interface
- a FastAPI web API
- a shared in-memory RAG engine

The app ingests documents, splits them into chunks, embeds them using OpenRouter, indexes them with FAISS, retrieves the most relevant chunks for a query, and sends the context to an LLM for grounded answers.

## Overview

The application is built around a shared core engine called `RAGEngine`, which performs:

1. Ingestion
2. Chunking
3. Embedding
4. Vector indexing
5. Retrieval
6. Prompt augmentation
7. LLM answer generation

This engine is used by both:
- the Streamlit UI
- the FastAPI backend

## Features

- Chat-style GUI with Streamlit
- REST API with FastAPI
- Local FAISS vector database
- OpenRouter integration for embeddings and LLMs
- Sample corpus included
- Upload PDFs or text files
- Load PDFs from a `docs/` folder
- Source citation for retrieved chunks
- Token usage tracking
- Timing metrics for the pipeline

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages include:
- `streamlit`
- `fastapi`
- `uvicorn`
- `openai`
- `numpy`
- `python-dotenv`
- `faiss-cpu`
- `pypdf`
- `requests`

## Environment setup

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

You can also set the environment variable directly:

```bash
export OPENROUTER_API_KEY=your_api_key_here
```

## Run the Streamlit GUI

```bash
streamlit run Week4_RAG_GUI.py
```

This starts the browser-based RAG application.

## Run the FastAPI server

```bash
python Week4_RAG_GUI.py --api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

This opens the Swagger UI for the API.

## API endpoints

### Health check
```http
GET /health
```

### Stats
```http
GET /stats
```

### Ingest documents
```http
POST /ingest
```

Example request:

```json
{
  "use_sample": true,
  "pdf_folder": "docs",
  "extra_documents": [
    {
      "source": "custom.txt",
      "text": "This is some custom knowledge."
    }
  ]
}
```

### Ask a question
```http
POST /ask
```

Example request:

```json
{
  "query": "What is RAG?",
  "top_k": 3
}
```

## Default model configuration

The app uses:

- Embedding model: `openai/text-embedding-3-small`
- LLM model: `openai/gpt-4o-mini`

These can be changed in the app configuration or in the engine settings.

## How the RAG pipeline works

The app follows this flow:

1. Load documents
2. Clean and normalize text
3. Split into chunks
4. Convert chunks to embeddings
5. Store vectors in FAISS
6. Embed the user query
7. Retrieve the top matching chunks
8. Build a grounded prompt
9. Send to the LLM
10. Return the answer with source references

## Notes

- The vector index is stored in memory, not persisted to disk.
- The app is intended for learning and demo usage.
- The included sample corpus explains LLMs, RAG, embeddings, vector databases, chunking, and prompt engineering.
- PDF ingestion requires `pypdf` to be installed.

## Example question

After loading the sample corpus, you can ask:

```text
What is Retrieval-Augmented Generation?
```

The system will find the relevant chunks and answer using the retrieved context.

## Summary

This Week 4 project demonstrates how to turn a basic RAG prototype into a practical application with:

- an interactive chat interface
- a web API
- a reusable retrieval engine behind both
