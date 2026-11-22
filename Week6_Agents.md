# Week 6 — Agentic AI with LangGraph, Step by Step

This lesson is split into four files so each concern stays readable:

| File | Purpose |
|---|---|
| `Week6_Agent_Utils.py` | shared settings, web search, memory tools, guardrails, printing helpers |
| `Week6_MCP_Tools.py` | MCP server (its own process) and the MCP client loader |
| `Week6_Agents.py` | LangGraph agent builder and **steps 1–5**, plus a step 6 wrapper |
| `Week6_RAGAS_Evaluation.py` | **step 6**, Ragas evaluation and CSV report |

The idea: an agent is an LLM plus tools, memory, guardrails, and evaluation,
running inside a loop. Each step adds one capability.

---

## What you need

```bash
pip install -r requirements.txt
```

Create `.env` beside the files:

```env
OPENROUTER_API_KEY=your_valid_openrouter_key
```

A missing or invalid key fails on the first model call with a clear error.

The script uses `baidu/ernie-5.5-flash-preview:free` through OpenRouter. This
free model supports tool calling, which steps 2-6 need. Change `CHAT_MODEL` in
`Week6_Agent_Utils.py` to use a different one.

---

## How to run each step

Open `Week6_Agents.py`. Near the bottom you will find:

```python
STEP_TO_RUN = step_1_basic_agent
```

Change that one line to choose a step:

| Step | Change the line to |
|---|---|
| 1 — Basic agent | `STEP_TO_RUN = step_1_basic_agent` |
| 2 — Web search | `STEP_TO_RUN = step_2_web_search` |
| 3 — MCP tool | `STEP_TO_RUN = step_3_mcp_tool` |
| 4 — Memory | `STEP_TO_RUN = step_4_memory` |
| 5 — Guardrails | `STEP_TO_RUN = step_5_guardrails` |
| 6 — Ragas (via wrapper) | `STEP_TO_RUN = step_6_ragas_evaluation` |

Then run:

```bash
python Week6_Agents.py
```

Step 6 is implemented in its own file. You can run it through the wrapper above
or directly:

```bash
python Week6_RAGAS_Evaluation.py
```

Every step function takes **no arguments**, so switching steps never requires
editing the function body.

### Viewing each graph

Every step draws the graph as ASCII art in the terminal before it calls the
model:

```text
--- Graph: Step 1 - basic agent ---
          +-----------+
          | __start__ |
          +-----------+
                 *
                 *
                 *
            +-------+
            | agent |
            +-------+
                 *
                 *
                 *
          +---------+
          | __end__ |
          +---------+
```

The picture is drawn by `grandalf`, which `requirements.txt` installs. If it is
missing, the step still runs; only the picture is skipped.

After the picture, the step prints the equivalent Mermaid source:

```text
--- Mermaid source (paste into https://mermaid.live) ---
graph TD;
	__start__ --> agent;
	agent --> __end__;
```

Finally the step renders that same Mermaid to a real diagram image, saves it to
`agent_reports/graphs/`, and opens it in the default viewer on macOS:

```text
--- Mermaid image saved: .../agent_reports/graphs/step-1-basic-agent.png ---
--- opened the image in the default viewer ---
```

A terminal cannot draw a Mermaid diagram itself, so the image is the actual
Mermaid chart. The render comes from the public mermaid.ink service; if the
network is unavailable, the step continues with the ASCII chart and the source
text. You can always paste the source into
[mermaid.live](https://mermaid.live) by hand.

The helper is `show_graph(app, title)` in `Week6_Agent_Utils.py`, so you can
call it on any compiled graph. The shapes printed by the lesson are:

| Steps | Graph shape |
|---|---|
| 1 | `START -> agent -> END` |
| 2, 3, 4b, 6 | `agent <-> tools` loop, then `END` |
| 4a | `START -> agent -> END` with a checkpointer |
| 5 | `guard_input -> agent -> guard_output -> END` |

---

## File 1 — `Week6_Agent_Utils.py`

This file is imported by the agent and Ragas modules. It contains:

### Settings and model

- OpenRouter base URL
- `CHAT_MODEL` and `EMBEDDING_MODEL`
- input/output size limits
- the shared system prompt
- `make_llm()` and `check_api_key()`

### Web search tool

```python
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the public web with DuckDuckGo..."""
```

A tool is a normal Python function with type hints and a docstring. The model
reads that description and decides when to call it.

### MCP

The MCP server and client are kept separate in `Week6_MCP_Tools.py` (File 2).

### Long-term memory tools

`make_memory_tools(store, user_id)` returns:

| Tool | Purpose |
|---|---|
| `remember_fact` | save a stable fact about the user |
| `recall_facts` | read all saved facts |

Data lives in a `namespace = ("memories", user_id)` inside a LangGraph store,
so a new conversation can still read facts saved by an older one.

### Guardrail helpers

- `check_input()` rejects empty, oversized, or prompt-injection-style messages.
- `redact_pii()` replaces email addresses and phone numbers.

These are plain functions; `Week6_Agents.py` wraps them in graph nodes.

### Console helpers

`step_banner()`, `shorten()`, and `print_trace()` keep the lesson output easy to
follow.

You can run this file directly only to see guidance:

```bash
python Week6_Agent_Utils.py
```

---

## File 2 — `Week6_MCP_Tools.py`

This file contains both halves of the MCP integration.

### MCP server

`run_mcp_server()` starts a real MCP server over stdio. It exposes:

| Tool | What it does |
|---|---|
| `text_stats` | counts characters, words, sentences |
| `utc_now` | returns the current UTC time |

Run the file by itself to start the server:

```bash
python Week6_MCP_Tools.py
```

It waits silently for MCP messages on stdin. Press `Ctrl+C` to stop it.

### MCP client

`load_mcp_tools()` starts `Week6_MCP_Tools.py` as a child process:

```text
Week6_Agents.py  --starts-->  python Week6_MCP_Tools.py
       |                                  |
       +---------- MCP over stdio ---------+
```

The adapter turns the MCP tools into normal LangChain tools, so the agent code
does not change. Each tool call opens a fresh MCP session.

Why this matters: a company can expose a database or internal API as an MCP
server, and any MCP-compatible agent can use it without custom integration code.

---

## File 3 — `Week6_Agents.py`

This file contains the graph and steps 1–5. It also has a tiny
`step_6_ragas_evaluation()` wrapper that calls the evaluation function in the
Ragas file, so all six steps can be selected from `STEP_TO_RUN`.

### Graph anatomy

- **State** — `AgentState`, mainly a message list
- **Node** — a Python function that reads state and returns updates
- **Edge** — the next node
- **Conditional edge** — a function that chooses the next node
- **Checkpointer** — saves state per `thread_id` (short-term memory)
- **Store** — saves facts across threads (long-term memory)

`build_agent()` is reused by every step. What changes is what you pass in:

```python
build_agent()                                      # step 1
build_agent(tools=[web_search])                    # step 2
build_agent(tools=mcp_tools)                       # step 3
build_agent(checkpointer=..., store=...)           # step 4
build_agent(use_guardrails=True)                   # step 5
```

### Step 1 — Basic agent

```text
START -> agent -> END
```

One node calls the model. With no tools and no guardrails, this is a normal
chatbot. The graph is what makes the next steps possible.

### Step 2 — Web search tool

```text
START -> agent -> (tool call?) -> tools -> agent -> ... -> END
```

The model returns `tool_calls` instead of a final answer. `route_after_agent`
sends control to `ToolNode`, which runs the Python function. The result becomes
a `ToolMessage`, and the agent gets another turn to produce the final answer.

This is the reason+act loop people call **ReAct**.

### Step 3 — MCP tool

The same graph, but the tools come from `load_mcp_tools()`, which starts a real
MCP server in a child process. The agent only sees the tool schemas and results.

Why this matters: a company can expose a database or internal API as an MCP
server, and any MCP-compatible agent can use it without custom integration code.

### Step 4 — Memory

Two different memory problems:

**Short-term — one conversation**

```python
checkpointer = InMemorySaver()
app = build_agent(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "short-term-demo"}}
```

Turn 1: "My name is Ram and I am learning LangGraph."

Turn 2: "What is my name, and what am I learning?"

Only the new question is sent, but the checkpointer restores the earlier
messages so the model can answer.

**Long-term — across conversations**

A store keeps facts under a namespace. Thread B starts with no history, but the
agent calls `recall_facts` and finds what an earlier conversation saved.

| | Checkpointer | Store |
|---|---|---|
| Scope | one `thread_id` | a namespace, such as one user |
| Holds | full message history | small facts or values |
| New conversation | starts fresh | still available |
| Demo class | `InMemorySaver` | `InMemoryStore` |

Both are in-memory for the lesson. Production systems use database backends.

### Step 5 — Guardrails

```text
START -> guard_input -> agent/tools loop -> guard_output -> END
```

**Input guardrail** runs before the model. A blocked message ends the graph at
`guard_input`, so the LLM is never called and no tokens are spent.

**Output guardrail** runs after the model. It redacts emails and phone numbers
and replaces the original message by ID, so unsafe text is not kept in history.

The rules are intentionally simple for teaching. Real systems also need
semantic checks, per-tool permissions, human approval for dangerous actions, and
logging.

---

## File 4 — `Week6_RAGAS_Evaluation.py`

This is step 6.

The flow:

1. load the MCP tools,
2. build the full agent (web search + MCP),
3. run two fixed questions,
4. collect every `ToolMessage` as retrieved context,
5. score the answers with Ragas,
6. save a CSV report.

Every sample contains:

```python
SingleTurnSample(
    user_input=question,
    retrieved_contexts=tool_results,
    response=agent_answer,
    reference=expected_answer,
)
```

Metrics:

| Metric | What it checks |
|---|---|
| `Faithfulness` | are the answer's claims supported by tool evidence? |
| `AnswerRelevancy` | does the answer actually address the question? |
| `AnswerCorrectness` | does it agree with the reference answer? |

Ragas uses an LLM judge and embeddings, so it makes additional OpenRouter calls.
Results are saved to:

```text
agent_reports/agentic_ragas_<timestamp>.csv
```

Reading guide: `0.80+` is strong, `0.60–0.79` is mixed, below `0.60` needs
investigation. Live web results change, so scores can vary between runs.

---

## Code map

| Concept | File | Where |
|---|---|---|
| Shared settings | Utils | top of file |
| Graph view | Utils | `show_graph` |
| Web search | Utils | `web_search` |
| MCP server | MCP | `run_mcp_server` |
| MCP client | MCP | `load_mcp_tools` |
| Memory tools | Utils | `make_memory_tools` |
| Guardrail helpers | Utils | `check_input`, `redact_pii` |
| Graph builder | Agents | `build_agent` |
| Step selection | Agents | `STEP_TO_RUN` |
| Steps 1–5 | Agents | `step_1_basic_agent` … `step_5_guardrails` |
| Ragas scoring | RAGAS | `evaluate_with_ragas` |
| Step 6 | RAGAS | `step_6_evaluations` |

---

## Troubleshooting

**`401 User not found`**

The OpenRouter key is missing, expired, or belongs to a deleted account. Create
a new key and update `.env`.

**`ModuleNotFoundError: langchain_core.language_models._compat_bridge`**

LangGraph 1.x was installed with LangChain 0.3.x. Reinstall the pinned versions:

```bash
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'mcp.server.fastmcp'`**

MCP 2.x is installed; this lesson uses the v1 `FastMCP` API:

```bash
pip install "mcp>=1.9.4,<2"
```

**MCP tools do not appear**

Check that the MCP server starts:

```bash
python Week6_MCP_Tools.py
```

It should wait silently for MCP messages. Press `Ctrl+C` to stop.

**Ragas scores are blank**

Confirm the OpenRouter key works. Ragas needs the same key for its judge model
and embeddings.

---

## What to try next

- Replace `InMemorySaver` with a SQLite or Postgres checkpointer.
- Add a human-approval node before tools that change data.
- Add another MCP server, such as a database or issue tracker.
- Expand the Ragas question set and track scores over time.
- Add `ContextPrecision` / `ContextRecall` once retrieval uses a fixed corpus.
