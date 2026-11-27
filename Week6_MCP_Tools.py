"""
Week 6 - MCP tool server and client
===================================

MCP (Model Context Protocol) is a standard way for an AI application to
discover and call tools that live in another program. This file is kept
separate from the agent and the utility modules on purpose:

    - the MCP SERVER runs as a small web server in its own process
    - the MCP CLIENT connects over HTTP and turns its tools into LangChain tools

The server is this same file. Running it starts the web server:

    python Week6_MCP_Tools.py

It listens on http://127.0.0.1:8000/mcp and prints a line saying so. Press
Ctrl+C to stop it. The agent normally starts it automatically through
load_mcp_tools(), so you only need to run it yourself if you want to watch it.

To check that the server and client both work, run the demo instead:

    python Week6_MCP_Tools.py --demo

That starts the server the same way the agent does, lists the tools it
discovers, and calls each one.

MCP TOOLS EXPOSED BY THIS SERVER
--------------------------------
    text_stats   count characters, words, and sentences
    utc_now      return the current Coordinated Universal Time

SETUP
-----
    pip install -r requirements.txt

The MCP packages are already listed there:

    fastmcp>=2.14,<3
    mcp>=1.9.4,<2
    langchain-mcp-adapters>=0.1.14,<0.2
"""

import asyncio
import atexit
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()

# The server listens on localhost only. HTTP means the tools are reachable as
# a small web service instead of through a private stdin/stdout pipe.
MCP_HOST = "127.0.0.1"
MCP_PORT = int(os.environ.get("WEEK6_MCP_PORT", "8000"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# Filled in only when this process had to start the server itself.
_server_process: subprocess.Popen | None = None


def port_is_open() -> bool:
    """Return True when something is already listening on the MCP port."""
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((MCP_HOST, MCP_PORT)) == 0


def _stop_server() -> None:
    """Stop the background server when the agent process exits."""
    if _server_process is None or _server_process.poll() is not None:
        return
    _server_process.terminate()
    try:
        _server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _server_process.kill()


def start_mcp_server() -> None:
    """Make sure the MCP web server is running, starting it if needed.

    The agent calls this before connecting. If you already ran
    `python Week6_MCP_Tools.py` in another terminal, nothing is started here
    and that server is reused.
    """
    global _server_process

    if port_is_open():
        return

    print(f"Starting the MCP server at {MCP_URL} ...")
    _server_process = subprocess.Popen(
        [sys.executable, str(SCRIPT_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_stop_server)

    deadline = time.time() + 20
    while time.time() < deadline:
        if port_is_open():
            print(f"MCP server is listening at {MCP_URL}")
            return
        if _server_process.poll() is not None:
            raise RuntimeError(
                "The MCP server exited during startup. Run "
                "`python Week6_MCP_Tools.py` in a terminal to see its error."
            )
        time.sleep(0.25)

    raise RuntimeError(f"The MCP server did not open {MCP_URL} within 20 seconds.")


def run_mcp_server() -> None:
    """Run this file as a small MCP tool server over Streamable HTTP.

    A real production server might expose a database, a ticketing system, or a
    company API. These two tools are deliberately tiny so the lesson is easy
    to follow.
    """
    # fastmcp pulls in authlib, which warns about its own deprecated jose
    # module on Python 3.14. authlib installs a "show always" policy when it
    # loads, so import it first and then quiet everything else.
    with warnings.catch_warnings():
        import authlib.deprecate  # noqa: F401  installs authlib's warning policy
        warnings.simplefilter("ignore")
        from fastmcp import FastMCP

    # FastMCP and the underlying MCP SDK log every request at INFO level.
    # WARNING keeps the lesson output clean.
    logging.getLogger("fastmcp").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)

    # The name is shown to clients when they connect.
    server = FastMCP("week6-local-tools")

    @server.tool()
    def text_stats(text: str) -> str:
        """Count characters, words, and sentences in the supplied text. asdf"""
        words = text.split()
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        return json.dumps(
            {
                "characters": len(text),
                "words": len(words),
                "sentences": len(sentences),
            }
        )

    @server.tool()
    def utc_now() -> str:
        """Return the current Coordinated Universal Time (UTC)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # "http" is FastMCP's Streamable HTTP transport: the tools are served at
    # MCP_URL like a normal web endpoint. The banner would be noise in the
    # lesson output, so it is turned off.
    server.run(
        transport="http",
        host=MCP_HOST,
        port=MCP_PORT,
        show_banner=False,
    )


async def run_demo() -> None:
    """Start the server if needed, connect over HTTP, and call its tools.

    This is the quickest way to prove the file runs. It uses the same client
    path as the agent, so a success here means step 3 will work too.
    """
    print("Connecting to the MCP web server...")
    tools = await load_mcp_tools()
    print(f"Client discovered {len(tools)} tools: {[t.name for t in tools]}")

    by_name = {tool.name: tool for tool in tools}
    stats = await by_name["text_stats"].ainvoke(
        {"text": "MCP is running. The tools work!"}
    )
    now = await by_name["utc_now"].ainvoke({})
    print(f"text_stats result: {stats}")
    print(f"utc_now result: {now}")
    print("MCP demo finished: server and client are both working.")


async def load_mcp_tools() -> list:
    """Connect to the local MCP web server and return its tools.

    The server is started automatically when it is not already running. The
    adapter creates a fresh MCP session for each tool call, so no manual
    connect/close code is needed here.
    """
    # The adapter imports LangGraph, which prints a LangChain deprecation
    # notice on Python 3.14. LangChain re-installs its warning policy partway
    # through that import, so the filter alone loses; silence the display for
    # this import block instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.showwarning = lambda *args, **kwargs: None
        from langchain_mcp_adapters.client import MultiServerMCPClient

    start_mcp_server()

    client = MultiServerMCPClient(
        {
            "week6_local": {
                "url": MCP_URL,
                "transport": "streamable_http",
            }
        }
    )
    return await client.get_tools()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        asyncio.run(run_demo())
    else:
        # Stay quiet when the agent starts this file in the background.
        interactive = sys.stdin.isatty()
        if interactive:
            print(
                f"Week 6 MCP server listening at {MCP_URL}. "
                "Press Ctrl+C to stop. Run with --demo to test the tools.",
                file=sys.stderr,
                flush=True,
            )
        try:
            run_mcp_server()
        except KeyboardInterrupt:
            pass
        if interactive:
            print("MCP server stopped.", file=sys.stderr)
