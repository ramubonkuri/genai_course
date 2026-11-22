"""
Week 6 - STEP 6: Ragas evaluation for the LangGraph agent
==========================================================

This file contains ONLY the evaluation step. It:

    1. builds the same agent used in the lesson (web search + MCP tools),
    2. runs a fixed set of questions,
    3. collects every tool result as "retrieved context",
    4. asks Ragas to score the answers,
    5. saves the scores to a CSV file.

The agent itself lives in Week6_Agents.py.
Shared tools and settings live in Week6_Agent_Utils.py.
The MCP server and client live in Week6_MCP_Tools.py.

RUN
---
    python Week6_RAGAS_Evaluation.py

SETUP
-----
    pip install -r requirements.txt

Create a .env file beside this file:

    OPENROUTER_API_KEY=your_openrouter_key

Ragas uses the same OpenRouter key to run its judge model and embeddings.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from Week6_Agents import build_agent, invoke_graph
from Week6_Agent_Utils import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    check_api_key,
    print_trace,
    shorten,
    show_graph,
    step_banner,
    web_search,
)
from Week6_MCP_Tools import load_mcp_tools

# Fixed questions used for evaluation. A reference answer lets Ragas check the
# factual quality of the agent's answer, not just its style.
EVALUATION_QUESTIONS = [
    {
        "question": "What is the Model Context Protocol (MCP) and why is it useful for AI agents?",
        "reference": (
            "MCP is an open standard introduced by Anthropic that lets AI "
            "applications connect to external tools and data sources through a "
            "common protocol, so each tool does not need a custom integration for "
            "every model or app."
        ),
    },
    {
        "question": "What is LangGraph and what problem does it solve?",
        "reference": (
            "LangGraph is a framework for building stateful, multi-step LLM "
            "applications as graphs. It supports cycles, tool use, checkpoints, "
            "and human-in-the-loop control, which helps developers build reliable "
            "agents instead of one-shot prompt chains."
        ),
    },
]


def evaluate_with_ragas(samples: list):
    """Ask Ragas to score the collected agent answers."""
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness

    # Ragas needs its own "judge" model and embeddings. We reuse OpenRouter.
    judge_llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        timeout=90,
        max_retries=2,
    )
    judge_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
    )

    # Faithfulness         -> is the answer supported by the tool evidence?
    # AnswerRelevancy      -> does the answer actually address the question?
    # AnswerCorrectness    -> does it agree with the reference answer?
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
    ]

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        raise_exceptions=False,
        show_progress=True,
    )
    return result.to_pandas()


async def step_6_evaluations() -> None:
    """Run fixed questions through the agent, then score the answers with Ragas."""
    step_banner(
        6,
        "Evaluations with Ragas",
        """
Shipping an agent without measurement is guesswork. Ragas scores each answer
against the evidence the agent collected and, when a reference answer exists,
against that reference too.

We run the agent on fixed questions, keep every tool result as "retrieved
context", and let Ragas judge the answers.
""",
    )

    from ragas import SingleTurnSample

    # STEP 6 uses the full agent: web search + MCP tools.
    mcp_tools = await load_mcp_tools()
    app = build_agent(tools=[web_search, *mcp_tools])
    show_graph(app, "Step 6 - Ragas evaluation agent")
    samples = []

    for item in EVALUATION_QUESTIONS:
        question = item["question"]
        print(f"\nAgent question: {question}")

        result = await invoke_graph(app, [HumanMessage(content=question)])
        messages = result["messages"]
        print_trace(messages)

        answer_message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        answer = str(answer_message.content) if answer_message else ""

        # Tool results are the evidence the agent saw. Ragas calls these
        # "retrieved contexts".
        contexts = [
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage) and message.content
        ]

        print(f"Answer: {shorten(answer, 400)}")
        print(f"Evidence items (tool results): {len(contexts)}")

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=item["reference"],
            )
        )

    print("\nRagas makes extra model calls to judge these answers...")
    try:
        table = await asyncio.to_thread(evaluate_with_ragas, samples)
    except ImportError as error:
        print("Ragas packages are missing. Run: pip install -r requirements.txt")
        print(f"Details: {error}")
        return
    except Exception as error:
        print(f"Ragas evaluation could not finish: {error}")
        return

    metric_columns = [
        column
        for column in ["faithfulness", "answer_relevancy", "answer_correctness"]
        if column in table.columns
    ]
    print("\nRagas scores:")
    print(table[metric_columns].to_string(index=False))
    print("\nScores are usually between 0 and 1; higher is better.")
    print("  0.80-1.00 strong signal")
    print("  0.60-0.79 mixed; read the answers")
    print("  0.00-0.59 needs investigation")

    report_folder = Path(__file__).resolve().parent / "agent_reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_folder / f"agentic_ragas_{timestamp}.csv"
    table.to_csv(report_path, index=False)
    print(f"\nCSV report saved to: {report_path}")


async def main() -> int:
    """Run the Ragas evaluation step."""
    check_api_key()
    await step_6_evaluations()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except (RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
