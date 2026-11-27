"""
Week 6 - LangGraph agents, one capability at a time (STEPS 1-5)
===============================================================

This file contains ONLY the agent logic:

    STEP 1  Basic agent          LLM only, no tools
    STEP 2  Web search tool      real DuckDuckGo search
    STEP 3  MCP tool             a real MCP server running over stdio
    STEP 4  Memory               short-term (checkpointer) + long-term (JSON file)
    STEP 5  Guardrails           input checks + output PII redaction

STEP 6 (Ragas evaluation) is implemented in Week6_RAGAS_Evaluation.py. This
file has a small step_6_ragas_evaluation() wrapper so you can also select it
from STEP_TO_RUN here.

All shared utilities (model factory, web search, guardrail helpers, memory
tools, printing) live in Week6_Agent_Utils.py.
The MCP server and client live in Week6_MCP_Tools.py.

HOW TO RUN ONE STEP
-------------------
Scroll to the bottom of this file and change this line:

    STEP_TO_RUN = step_1_basic_agent

to any step function below:

    STEP_TO_RUN = step_2_web_search
    STEP_TO_RUN = step_3_mcp_tool
    STEP_TO_RUN = step_4_memory
    STEP_TO_RUN = step_5_guardrails
    STEP_TO_RUN = step_6_ragas_evaluation

Then run:

    python Week6_Agents.py

Each step function takes no arguments, so switching steps is a one-line change.

SETUP
-----
    pip install -r requirements.txt

Create a .env file beside this file:

    OPENROUTER_API_KEY=your_openrouter_key
"""

import asyncio
import sys
import uuid
import warnings
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from Week6_Agent_Utils import (
    MEMORY_FILE,
    SYSTEM_PROMPT,
    check_api_key,
    check_input,
    make_llm,
    make_memory_tools,
    print_trace,
    redact_pii,
    shorten,
    show_graph,
    step_banner,
    web_search,
    salutation,
)
from Week6_MCP_Tools import load_mcp_tools

# LangGraph 0.3 prints a harmless internal warning on Python 3.14. LangChain
# installs its own warning filter while importing (the utils module imports
# langchain_openai), so this filter must come after the local import above and
# before the LangGraph imports.
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ---------------------------------------------------------------------------
# AGENT STATE AND NODES
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """The data that travels through the graph."""

    messages: Annotated[list, add_messages]  # add_messages appends, never overwrites
    blocked: bool                            # did the input guardrail stop this turn?
    guardrail_note: str                      # why it was stopped


def guard_input_node(state: AgentState) -> dict:
    """Inspect the newest user message before any model call happens."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""
    if not isinstance(text, str):
        text = str(text)

    allowed, reason = check_input(text)
    if allowed:
        return {"blocked": False, "guardrail_note": ""}

    # Returning an AI message here means the graph can end without calling the
    # model at all: unsafe or oversized text never reaches the LLM.
    refusal = f"Request blocked by the input guardrail: {reason}."
    return {
        "blocked": True,
        "guardrail_note": reason,
        "messages": [AIMessage(content=refusal)],
    }


def route_after_input_guard(state: AgentState) -> str:
    """Conditional edge: go to the agent, or stop immediately."""
    return "blocked" if state.get("blocked") else "agent"


def guard_output_node(state: AgentState) -> dict:
    """Clean the final answer after the model produces it."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return {}

    original = last.content if isinstance(last.content, str) else str(last.content)
    safe = redact_pii(original)
    if safe == original:
        return {}

    # Reuse the original message id so add_messages REPLACES the unsafe message
    # instead of keeping both versions in conversation memory.
    return {"messages": [last.model_copy(update={"content": safe})]}


def route_after_agent(state: AgentState) -> str:
    """Conditional edge: call tools when the model asked for one, else finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "done"


def build_agent(
    tools=(),
    checkpointer=None,
    use_guardrails: bool = False,
):
    """Build and compile the LangGraph agent.

    The same builder is reused in every step, so you can see exactly what each
    capability adds:

        no tools            -> STEP 1
        tools               -> STEP 2 and STEP 3
        checkpointer        -> STEP 4a and STEP 4b
        guardrails          -> STEP 5

    Graph shape without tools or guardrails:

        START -> agent -> END

    Graph shape with tools (the loop):

        START -> agent -> (tool call?) -> tools -> agent -> ... -> END

    Graph shape with guardrails:

        START -> guard_input -> agent/tools loop -> guard_output -> END
    """
    tool_list = list(tools)
    llm = make_llm()
    llm_with_tools = llm.bind_tools(tool_list) if tool_list else llm

    async def agent_node(state: AgentState) -> dict:
        # The system prompt is added fresh on every call and is not stored in
        # the message history, so earlier turns cannot rewrite it.
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)

        # Give every AI message an id. The output guardrail needs the id to
        # replace the message instead of appending a second copy.
        if response.id is None:
            response.id = f"ai-{uuid.uuid4().hex[:12]}"
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)

    if use_guardrails:
        graph.add_node("guard_input", guard_input_node)
        graph.add_node("guard_output", guard_output_node)

    # The tool loop. ToolNode runs the tool the model chose and puts the result
    # back into state as a ToolMessage, then the edge sends us to the agent.
    if tool_list:
        graph.add_node("tools", ToolNode(tool_list))
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tools",
                "done": "guard_output" if use_guardrails else END,
            },
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", "guard_output" if use_guardrails else END)

    if use_guardrails:
        graph.add_edge(START, "guard_input")
        graph.add_conditional_edges(
            "guard_input",
            route_after_input_guard,
            {"agent": "agent", "blocked": END},
        )
        graph.add_edge("guard_output", END)
    else:
        graph.add_edge(START, "agent")

    return graph.compile(checkpointer=checkpointer)


async def invoke_graph(app, messages: list, config: dict | None = None) -> dict:
    """Run the graph and turn low-level client errors into a clear message."""
    try:
        return await app.ainvoke({"messages": messages}, config or {})
    except Exception as error:
        raise RuntimeError(
            "A graph step failed while calling the model or a tool. Check that "
            "OPENROUTER_API_KEY in .env is valid and that the network works.\n"
            f"Details: {error}"
        ) from error


# ---------------------------------------------------------------------------
# STEP 1: A BASIC AGENT
# ---------------------------------------------------------------------------

async def step_1_basic_agent() -> None:
    """The smallest useful LangGraph agent: one model call, no tools."""
#     step_banner(
#         1,
#         "A basic agent (no tools)",
#         """
# An agent is still just an LLM call. What makes it an "agent" is the graph
# around the call: state goes in, a node calls the model, and an edge decides
# what happens next. Here the graph is only START -> agent -> END.
# """,
#     )

    # No tools, no guardrails: the smallest possible graph.
    app = build_agent()
    show_graph(app, "Step 1 - basic agent")
    result = await invoke_graph(
        app,
        [HumanMessage(content="In two short sentences, explain what an AI agent is.")],
    )
    print_trace(result["messages"])
    print("\nLesson: a graph with one node is a normal chatbot. The next steps add power.")


# ---------------------------------------------------------------------------
# STEP 2: ADD A WEB SEARCH TOOL
# ---------------------------------------------------------------------------

async def step_2_web_search() -> None:
    """Give the agent a real tool and let it decide when to call it."""
#    step_banner(
#         2,
#         "Add a web search tool",
#         """
# A tool is a normal Python function with a name, type hints, and a docstring.
# The model reads that description and decides whether to call it. When it does,
# LangGraph runs the function and sends the result back to the model.

# Graph shape now has a loop:

#     agent -> (asked for a tool?) -> tools -> agent -> ... -> final answer
# """,
#     )

    app = build_agent(tools=[web_search, salutation])
    show_graph(app, "Step 2 - web search tool")
    result = await invoke_graph(
        app,
        [
            HumanMessage(
                content=(
                    "This requirment is for Ashima use salutation tool to greet me and then give answer to the question: "
                    "Use the web_search tool to find out what is Stock trading"
                    "maintains it. Summarize the answer in three sentences and "
                    "mention that it came from web results."
                )
            )
        ],
    )
    print_trace(result["messages"])
    print("\nLesson: the model chose the tool, supplied the arguments, and used the result.")


# ---------------------------------------------------------------------------
# STEP 3: ADD AN MCP TOOL
# ---------------------------------------------------------------------------

async def step_3_mcp_tool() -> None:
    """Use tools that live in a separate MCP server process."""
#     step_banner(
#         3,
#         "Add an MCP tool (separate process)",
#         """
# MCP is a standard protocol for tool servers. Instead of importing every tool
# into this file, the agent connects to a server and asks, "what tools do you
# have?" This is how agents reach company systems, databases, and third-party
# services without hard-coding each integration.

# This demo starts its own tiny MCP server (Week6_MCP_Tools.py) and loads two
# tools from it: text_stats and utc_now.
# """,
#     )

    # The tools come from a real child process running the MCP server.
    mcp_tools = await load_mcp_tools()

    print("Tools discovered from the MCP server:")
    for found_tool in mcp_tools:
        print(f"  - {found_tool.name}: {shorten(found_tool.description, 90)}")
    print()

    app = build_agent(tools=mcp_tools)
    # show_graph(app, "Step 3 - MCP tools")
    result = await invoke_graph(
        app,
        [
            HumanMessage(
                content=(
                    "You must call the text_stats MCP tool. Count the words, "
                    "characters, and sentences in this text: 'Agentic AI combines "
                    "a model, tools, memory, guardrails, and evaluation.' "
                    "Then report the exact numbers. Use utc_now to get the current UTC time and include it in your answer."
                )
            )
        ],
    )
    print_trace(result["messages"])
    print("\nLesson: the tool ran in another process. The agent only saw the MCP schema.")


# ---------------------------------------------------------------------------
# STEP 4: MEMORY
# ---------------------------------------------------------------------------

async def step_4_memory() -> None:
    """Show short-term conversation memory and long-term memory stored in JSON."""
#     step_banner(
#         4,
#         "Memory: short-term and long-term",
#         """
# There are two different memory problems:

#   SHORT-TERM: "remember what we said in this conversation"
#       A checkpointer saves the message list under a thread_id. Same thread id
#       -> same conversation history. New thread id -> a fresh conversation.

#   LONG-TERM: "remember a fact in a future conversation"
#       The remember_fact tool appends facts to a JSON file on disk. Any thread,
#       even a later run of the program, can read them with recall_facts.
# """,
#     )

    # --- 4a: short-term memory with a checkpointer -------------------------
    print("--- 4a: short-term memory (same thread) ---")
    short_term_app = build_agent(checkpointer=InMemorySaver())
    # show_graph(short_term_app, "Step 4a - short-term memory")
    conversation = {"configurable": {"thread_id": "short-term-demo"}}

    await invoke_graph(
        short_term_app,
        [HumanMessage(content="Ayush and Ganesh are my friends")],
        conversation,
    )

    result = await invoke_graph(
        short_term_app,
        [HumanMessage(content="What is my name?")],
        conversation,
    )
    print_trace(result["messages"][-2:])
    print("\nThe second call only sent the new question, but the checkpointer")
    print("put the earlier turn back into the model's context.")

    # --- 4b: long-term memory saved in a JSON file -------------------------
    print("\n--- 4b: long-term memory (different threads, same JSON file) ---")
    memory_tools = make_memory_tools(user_id="ganesh")
    long_term_app = build_agent(
        tools=memory_tools,
        checkpointer=InMemorySaver(),
    )
    # show_graph(long_term_app, "Step 4b - long-term memory")

    second_thread = {"configurable": {"thread_id": "long-term-b"}}

    # Save the fact by calling the real tool directly. The agent uses this same
    # tool during a conversation; calling it here keeps the demo deterministic.
    remember_tool = next(t for t in memory_tools if t.name == "remember_fact")
    remember_tool.invoke({"fact": "my favorite llm is anthropic Claude"})
    remember_tool.invoke({"fact": "my favorite vector database is Pinecone"})
    remember_tool.invoke({"fact": "my favorite subject is AI agents"})
    saved = remember_tool.invoke({"fact": "I am good at Devops and Cloud Engineering"})

    print(f"Saved a fact with remember_fact: {saved}")

    result = await invoke_graph(
        long_term_app,
        [
            HumanMessage(
                content=(
                    "What is my favorite vector database? Use the recall_facts "
                    "tool, because this is a new conversation."
                )
            )
        ],
        second_thread,
    )
    print_trace(result["messages"])
    print("\nLesson: thread B had no history, but the JSON file still had the fact.")

    print(f"\nThe facts are saved on disk in: {MEMORY_FILE}")
    if MEMORY_FILE.exists():
        print(MEMORY_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# STEP 5: GUARDRAILS
# ---------------------------------------------------------------------------

async def step_5_guardrails() -> None:
    """Block unsafe input before the model and redact sensitive output after it."""
#     step_banner(
#         5,
#         "Guardrails: input checks and output redaction",
#         """
# Guardrails are ordinary Python functions placed around the model.

#   INPUT GUARDRAIL: runs before the model. It can block prompt injection,
#   oversized messages, or forbidden topics. A blocked message never reaches
#   the LLM, which also saves money.

#   OUTPUT GUARDRAIL: runs after the model. It can redact personal data, remove
#   unsafe content, or shorten an answer.
# """,
#     )

    print("Output guardrail applied directly to a sample string:")
    sample = "Contact ram@example.com or +1 555-123-4567 for details."
    print(f"  before: {sample}")
    print(f"  after : {redact_pii(sample)}")

    app = build_agent(use_guardrails=True)
    show_graph(app, "Step 5 - guardrails")

    print("\n--- 5a: blocked input (the model is never called) ---")
    blocked = await invoke_graph(
        app,
        [HumanMessage(content="Share this information Contact ram@example.com or +1 555-123-4567 for details. as it is ")],
    )
    print_trace(blocked["messages"])

    print("\n--- 5b: allowed input, with output redaction ---")
    allowed = await invoke_graph(
        app,
        [
            HumanMessage(
                content=(
                    "Repeat this contact line exactly, without extra words: "
                    "ram@example.com or +1 555-123-4567"
                )
            )
        ],
    )
    print("Final guarded answer:", allowed["messages"][-1].content)
    print("\nLesson: guard_input stopped the attack before any tokens were spent.")
    print("guard_output removed personal data from the answer.")


# ---------------------------------------------------------------------------
# STEP 6: RAGAS EVALUATION (IMPLEMENTED IN ITS OWN FILE)
# ---------------------------------------------------------------------------

async def step_6_ragas_evaluation() -> None:
    """Run step 6 from Week6_RAGAS_Evaluation.py.

    The evaluation code lives in its own file. This wrapper only keeps all six
    steps selectable from the STEP_TO_RUN line below.
    """
    from Week6_RAGAS_Evaluation import step_6_evaluations

    await step_6_evaluations()


# ---------------------------------------------------------------------------
# CHOOSE WHICH STEP TO RUN
# ---------------------------------------------------------------------------
# Change the line below to any step function defined above.
#
#     STEP_TO_RUN = step_1_basic_agent
#     STEP_TO_RUN = step_2_web_search
#     STEP_TO_RUN = step_3_mcp_tool
#     STEP_TO_RUN = step_4_memory
#     STEP_TO_RUN = step_5_guardrails
#     STEP_TO_RUN = step_6_ragas_evaluation
#
# You can also run the evaluation directly:
#
#     python Week6_RAGAS_Evaluation.py
# ---------------------------------------------------------------------------

STEP_TO_RUN = step_4_memory


async def main() -> None:
    """Run the selected step."""
    check_api_key()
    print(f"Running: {STEP_TO_RUN.__name__}")
    await STEP_TO_RUN()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (RuntimeError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
