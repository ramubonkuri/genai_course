"""
Week 1 — LLM Fundamentals & Prompt Engineering
================================================
Hands-on exercises covering:
  1. Basic API calls with different models
  2. Prompt engineering patterns (JSON, CoT, Few-shot)
  3. Latency & cost measurement
  4. Embeddings and similarity search

Requirements:
  pip install openai numpy

Usage:
  1) Create a .env file in the project root:
     OPENROUTER_API_KEY="your-key"
  2) Run:
     python week1_exercises.py

Optional:
  export OPENAI_API_KEY="your-key"  # fallback if OPENROUTER_API_KEY is not set
"""

import os
import time
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# SETUP
# =============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Load a local .env file if present (project root or current working directory)
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()


def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "genai_course",
        },
    )


client = get_client()

# Pricing per 1M tokens (update as needed)
PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
}


def display_model_name(model_name):
    return model_name.split("/", 1)[-1] if "/" in model_name else model_name

# =============================================================================
# EXERCISE 1: Basic API Calls & Model Comparison
# =============================================================================

def exercise_1_basic_calls():
    """Compare responses from different models for the same prompt."""
    print("=" * 70)
    print("EXERCISE 1: Basic API Calls & Model Comparison")
    print("=" * 70)
    
    prompt = "Explain what a vector database is, in 2 sentences."
    
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]

    for model in models:
        print(f"\n--- Model: {display_model_name(model)} ---")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        content = response.choices[0].message.content
        usage = response.usage
        
        print(f"Response: {content}")
        print(f"Tokens used: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})")


# =============================================================================
# EXERCISE 2: Prompt Pattern — Structured Output (JSON Mode)
# =============================================================================

def exercise_2_json_mode():
    """Extract structured data from unstructured text using JSON mode."""
    print("\n" + "=" * 70)
    print("EXERCISE 2: Structured Output (JSON Mode)")
    print("=" * 70)
    
    ticket = """Subject: Urgent - Double billing on my account
    
Hi support team,
    
I noticed that I was charged $49.99 twice on September 1st for my Pro subscription. 
This is the second time this has happened. I need a refund for the duplicate charge 
immediately or I will dispute this with my credit card company.
    
My account email is sarah.chen@company.com and my user ID is SC-88291.
    
Please resolve this ASAP.
    
Thanks,
Sarah"""
    
    prompt = f"""
Extract the following information from the customer support ticket below:
- customer_name: The name of the customer
- customer_email: The email address
- issue_category: One of [Billing, Technical, Account, Feature_Request, Complaint]
- urgency: One of [Low, Medium, High, Critical]
- summary: A 10-15 word summary of the issue
- refund_requested: true or false

Ticket:
{ticket}

Respond ONLY with valid JSON. No markdown formatting, no explanation.
"""
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    result = json.loads(response.choices[0].message.content)
    print("\nExtracted Data:")
    print(json.dumps(result, indent=2))
    
    # Validation check
    required_keys = ["customer_name", "customer_email", "issue_category", "urgency", "summary", "refund_requested"]
    missing = [k for k in required_keys if k not in result]
    if missing:
        print(f"\n⚠ Missing keys: {missing}")
    else:
        print("\n✓ All required fields present")


# =============================================================================
# EXERCISE 3: Prompt Pattern — Chain-of-Thought (CoT)
# =============================================================================

def exercise_3_chain_of_thought():
    """Use step-by-step reasoning for complex problems."""
    print("\n" + "=" * 70)
    print("EXERCISE 3: Chain-of-Thought Reasoning")
    print("=" * 70)
    
    # Without CoT
    problem = "A store offers a 20% discount on a $150 jacket, then charges 8% sales tax on the discounted price. What is the final price?"
    
    print("\n--- Without Chain-of-Thought ---")
    response_direct = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": f"Solve: {problem}"}],
        temperature=0.0,
        max_tokens=50
    )
    print(f"Answer: {response_direct.choices[0].message.content.strip()}")
    
    # With CoT
    print("\n--- With Chain-of-Thought ---")
    prompt_cot = f"""Solve this problem step by step:

{problem}

Step 1: Calculate the discount amount.
Step 2: Calculate the price after discount.
Step 3: Calculate the sales tax on the discounted price.
Step 4: Calculate the final price.

Final Answer: $"""
    
    response_cot = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_cot}],
        temperature=0.0,
        max_tokens=150
    )
    print(f"Answer: {response_cot.choices[0].message.content.strip()}")
    
    # Note: CoT usually improves accuracy on math/reasoning tasks
    print("\n💡 Observation: CoT prompts often yield more accurate results on reasoning tasks.")


# =============================================================================
# EXERCISE 4: Prompt Pattern — Few-Shot Classification
# =============================================================================

def exercise_4_few_shot():
    """Teach the model a custom classification task with examples."""
    print("\n" + "=" * 70)
    print("EXERCISE 4: Few-Shot Classification")
    print("=" * 70)
    
    prompt = """Classify customer feedback into one of: Praise, Complaint, Suggestion, Question.

Examples:
Feedback: "Your app crashed 3 times today. This is unacceptable."
Classification: Complaint

Feedback: "Could you add dark mode? It would help with eye strain."
Classification: Suggestion

Feedback: "The new dashboard is incredibly intuitive. Great job team!"
Classification: Praise

Feedback: "How do I export my data to CSV format?"
Classification: Question

Feedback: "I love the new feature but it loads too slowly on mobile."
Classification:"""
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    
    classification = response.choices[0].message.content.strip()
    print(f"\nFeedback: 'I love the new feature but it loads too slowly on mobile.'")
    print(f"Classification: {classification}")
    
    # Test with zero-shot for comparison
    print("\n--- Zero-shot (no examples) for comparison ---")
    zero_shot_prompt = "Classify this feedback into one of: Praise, Complaint, Suggestion, Question.\n\nFeedback: \"I love the new feature but it loads too slowly on mobile.\"\nClassification:"
    
    response_zs = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": zero_shot_prompt}],
        temperature=0.0,
        max_tokens=10
    )
    print(f"Zero-shot Classification: {response_zs.choices[0].message.content.strip()}")
    print("\n💡 Observation: Few-shot often improves consistency on nuanced classifications.")


# =============================================================================
# EXERCISE 5: Measuring Latency & Cost
# =============================================================================

def call_with_metrics(model, messages, temperature=0.7, max_tokens=150):
    """Make an API call and measure latency + cost."""
    start = time.perf_counter()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    
    rates = PRICING.get(model, {"input": 0, "output": 0})
    cost = (input_tokens / 1_000_000 * rates["input"] + 
            output_tokens / 1_000_000 * rates["output"])
    
    return {
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 8),
        "content": response.choices[0].message.content
    }


def exercise_5_latency_cost():
    """Compare latency and cost across models and prompt sizes."""
    print("\n" + "=" * 70)
    print("EXERCISE 5: Latency & Cost Measurement")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Short prompt",
            "messages": [{"role": "user", "content": "Say hello."}]
        },
        {
            "name": "Medium prompt",
            "messages": [{"role": "user", "content": "Write a 3-sentence summary of machine learning."}]
        },
        {
            "name": "Long prompt + system",
            "messages": [
                {"role": "system", "content": "You are a helpful technical assistant. Be concise."},
                {"role": "user", "content": "Explain the difference between supervised and unsupervised learning, including 2 examples of each. Keep it under 100 words."}
            ]
        }
    ]
    
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]

    print(f"\n{'Test Case':<20} {'Model':<15} {'Latency(ms)':<12} {'Tokens':<10} {'Cost(USD)':<12} {'Output'}")
    print("-" * 100)

    for case in test_cases:
        for model in models:
            result = call_with_metrics(model, case["messages"], temperature=0.5, max_tokens=150)
            output_preview = result["content"][:40].replace("\n", " ") + "..."
            print(f"{case['name']:<20} {display_model_name(model):<15} {result['latency_ms']:<12} {result['total_tokens']:<10} {result['cost_usd']:<12} {output_preview}")
    
    print("\n💡 Key Takeaways:")
    print("   - Larger models are slower and more expensive")
    print("   - Input tokens cost less than output tokens (for most models)")
    print("   - System prompts count toward input tokens")
    print("   - Temperature does not affect cost or latency")


# =============================================================================
# EXERCISE 6: Embeddings & Similarity Search
# =============================================================================

def get_embedding(text, model="openai/text-embedding-3-small"):
    """Get embedding for a single text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def exercise_6_embeddings():
    """Demonstrate embeddings and semantic similarity search."""
    print("\n" + "=" * 70)
    print("EXERCISE 6: Embeddings & Similarity Search")
    print("=" * 70)
    
    # Sample documents
    documents = [
        "Machine learning models require large amounts of training data to generalize well.",
        "Deep neural networks can recognize complex patterns in images and videos.",
        "The bakery on Main Street is famous for its sourdough bread and croissants.",
        "Convolutional neural networks are widely used for computer vision tasks.",
        "Fresh bread smells amazing in the morning when it comes out of the oven.",
        "Reinforcement learning agents learn by interacting with their environment.",
        "Natural language processing enables machines to understand human text.",
        "The pastry chef won an award for her chocolate éclairs last year."
    ]
    
    print("\nDocuments:")
    for i, doc in enumerate(documents):
        print(f"  [{i}] {doc}")
    
    # Compute embeddings
    print("\nComputing embeddings...")
    embeddings = [get_embedding(doc) for doc in documents]
    print(f"✓ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
    
    # Query
    queries = [
        "How do AI systems learn from images?",
        "Tell me about bread and baking.",
        "What is needed to train AI models?"
    ]
    
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: '{query}'")
        print('=' * 50)

        query_emb = get_embedding(query)
        
        # Rank documents by similarity
        scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (idx, score) in enumerate(scores[:3], 1):
            print(f"  Rank {rank} (score={score:.4f}): [{idx}] {documents[idx]}")
    
    # Demonstrate semantic vs keyword search
    print("\n" + "=" * 70)
    print("SEMANTIC vs KEYWORD SEARCH COMPARISON")
    print("=" * 70)
    
    query = "computer vision and image recognition"
    query_emb = get_embedding(query)
    
    print(f"\nQuery: '{query}'")
    print("\nSemantic Search (embeddings):")
    scores = [(i, cosine_similarity(query_emb, emb)) for i, emb in enumerate(embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, score) in enumerate(scores[:3], 1):
        print(f"  {rank}. [{idx}] (score={score:.4f}) {documents[idx]}")
    
    # Simple keyword match
    print("\nKeyword Search (exact match):")
    keywords = query.lower().split()
    keyword_scores = []
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        score = sum(1 for kw in keywords if kw in doc_lower)
        keyword_scores.append((i, score))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, score) in enumerate(keyword_scores[:3], 1):
        print(f"  {rank}. [{idx}] (matches={score}) {documents[idx]}")
    
    print("\n💡 Observation: Semantic search finds related concepts even without keyword overlap.")


# =============================================================================
# EXERCISE 7: Temperature Experiment
# =============================================================================

def exercise_7_temperature():
    """Show how temperature affects output variability."""
    print("\n" + "=" * 70)
    print("EXERCISE 7: Temperature Effect")
    print("=" * 70)
    
    prompt = "Give me a creative name for a coffee shop."
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Running 3 times at each temperature...\n")
    
    for temp in temperatures:
        print(f"--- Temperature = {temp} ---")
        for i in range(3):
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=20
            )
            name = response.choices[0].message.content.strip()
            print(f"  Run {i+1}: {name}")
        print()
    
    print("💡 Observation: Lower temperature = more deterministic/repeatable.")
    print("   Higher temperature = more creative/varied.")
    print("   Use 0.0 for structured outputs, 0.7-1.0 for creative tasks.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "🚀 " * 20)
    print("WEEK 1 — LLM FUNDAMENTALS & PROMPT ENGINEERING")
    print("🚀 " * 20 + "\n")
    
    exercises = [
        ("Basic API Calls", exercise_1_basic_calls),
        ("JSON Mode", exercise_2_json_mode),
        ("Chain-of-Thought", exercise_3_chain_of_thought),
        ("Few-Shot Classification", exercise_4_few_shot),
        ("Latency & Cost", exercise_5_latency_cost),
        ("Embeddings & Similarity", exercise_6_embeddings),
        ("Temperature Experiment", exercise_7_temperature),
    ]
    
    for name, func in exercises:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETE")
    print("=" * 70)
    print("\nDeliverables checklist:")
    print("  ✓ One-page LLM tradeoffs summary (see Week1_LLM_Fundamentals.md)")
    print("  ✓ 3 prompt templates (JSON extraction, CoT, Few-shot)")
    print("  ✓ Cost/latency measurement code")
    print("  ✓ Embedding similarity search demo")


if __name__ == "__main__":
    main()