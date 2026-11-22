"""
Week 6 - MCP tool server and client
===================================

MCP (Model Context Protocol) is a standard way for an AI application to
discover and call tools that live in another program. This file is kept
separate from the agent and the utility modules on purpose:

    - the MCP SERVER runs in its own process
    - the MCP CLIENT connects to it and turns its tools into LangChain tools

The server is this same file: running it starts a small stdio MCP server.

    python Week6_MCP_Tools.py

It then waits silently for MCP messages on stdin. Press Ctrl+C to stop it.
The agent normally starts it automatically through load_mcp_tools().

MCP TOOLS EXPOSED BY THIS SERVER
--------------------------------
    text_stats   count characters, words, and sentences
    utc_now      return the current Coordinated Universal Time

SETUP
-----
    pip install -r requirements.txt

The MCP packages are already listed there:

    mcp>=1.9.4,<2
    langchain-mcp-adapters>=0.1.14,<0.2
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()


def run_mcp_server() -> None:
    """Run this file as a small MCP tool server over stdio.

    A real production server might expose a database, a ticketing system, or a
    company API. These two tools are deliberately tiny so the lesson is easy
    to follow.
    """
    from mcp.server.fastmcp import FastMCP

    # MCP logs every request at INFO level. WARNING keeps the lesson output clean.
    logging.getLogger("mcp").setLevel(logging.WARNING)

    # The name is shown to clients when they connect.
    server = FastMCP("week6-local-tools")

    @server.tool()
    def text_stats(text: str) -> str:
        """Count characters, words, and sentences in the supplied text."""
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

    # stdio means "talk over standard input/output". The parent process starts
    # this child process and speaks the MCP protocol to it.
    server.run(transport="stdio")


async def load_mcp_tools() -> list:
    """Start the MCP server as a child process and return its tools.

    The adapter creates a fresh MCP session for each tool call, so no manual
    connect/close code is needed here.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "week6_local": {
                "command": sys.executable,
                "args": [str(SCRIPT_PATH)],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


if __name__ == "__main__":
    run_mcp_server()
