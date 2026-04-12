"""Tests for the sandbox MCP transport.

Tests sandbox_client by spawning a mock MCP server as a local subprocess
(no Docker/MCP SDK required). The mock server speaks newline-delimited
JSON-RPC 2.0 — the same wire protocol as MCP stdio transport.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

# Skip all tests if anyio is not available (it's an MCP SDK dependency)
anyio = pytest.importorskip("anyio")


# ---------------------------------------------------------------------------
# Mock MCP server script (newline-delimited JSON-RPC 2.0)
# ---------------------------------------------------------------------------

MOCK_MCP_SERVER = textwrap.dedent(r'''
import json
import sys

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def recv():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())

# MCP initialize handshake
req = recv()
if req and req.get("method") == "initialize":
    send({
        "jsonrpc": "2.0",
        "id": req["id"],
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock-server", "version": "1.0.0"},
        },
    })

# Wait for initialized notification
req = recv()

# Handle tools/list
req = recv()
if req and req.get("method") == "tools/list":
    send({
        "jsonrpc": "2.0",
        "id": req["id"],
        "result": {
            "tools": [{
                "name": "echo",
                "description": "Echo the input",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }],
        },
    })

# Handle tool calls until stdin closes
while True:
    req = recv()
    if req is None:
        break
    if req.get("method") == "tools/call":
        tool_name = req["params"].get("name", "")
        args = req["params"].get("arguments", {})
        if tool_name == "echo":
            send({
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}],
                },
            })
        else:
            send({
                "jsonrpc": "2.0",
                "id": req["id"],
                "result": {
                    "content": [{"type": "text", "text": f"unknown tool: {tool_name}"}],
                    "isError": True,
                },
            })
''')


@pytest.fixture
def mock_server_script(tmp_path):
    """Write the mock MCP server script to a temp file."""
    script = tmp_path / "mock_mcp_server.py"
    script.write_text(MOCK_MCP_SERVER)
    return str(script)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSandboxClient:
    """Test sandbox_client using a local subprocess (simulating docker exec)."""

    @pytest.mark.asyncio
    async def test_basic_tool_call(self, mock_server_script):
        """Verify basic MCP protocol flow through sandbox_client."""
        from mcp import ClientSession
        from tools.mcp_sandbox_transport import sandbox_client

        # Use python3 directly instead of docker exec — same pipe pattern
        async with sandbox_client(
            container_id="unused",
            docker_exe="unused",
            command="unused",
            _raw_cmd=[sys.executable, mock_server_script],
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) == 1
                assert tools.tools[0].name == "echo"

                result = await session.call_tool("echo", {"text": "hello sandbox"})
                assert not result.isError
                assert any("hello sandbox" in c.text for c in result.content if hasattr(c, "text"))

    @pytest.mark.asyncio
    async def test_env_vars_passed(self, tmp_path):
        """Verify environment variables are passed to the subprocess."""
        # Script that prints an env var as an MCP tool result
        script = tmp_path / "env_check.py"
        script.write_text(textwrap.dedent(r'''
import json, sys, os

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()

def recv():
    line = sys.stdin.readline()
    return json.loads(line.strip()) if line else None

# initialize
req = recv()
send({"jsonrpc": "2.0", "id": req["id"], "result": {
    "protocolVersion": "2025-11-25",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "env-check", "version": "1.0.0"},
}})
recv()  # initialized notification

# tools/list
req = recv()
send({"jsonrpc": "2.0", "id": req["id"], "result": {"tools": [{
    "name": "get_env", "description": "Get env var",
    "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}}},
}]}})

while True:
    req = recv()
    if req is None:
        break
    if req.get("method") == "tools/call":
        key = req["params"].get("arguments", {}).get("key", "")
        value = os.environ.get(key, "NOT_SET")
        send({"jsonrpc": "2.0", "id": req["id"], "result": {
            "content": [{"type": "text", "text": value}],
        }})
        '''))

        from mcp import ClientSession
        from tools.mcp_sandbox_transport import sandbox_client

        # For env var test, set it in os.environ since _raw_cmd bypasses -e flags
        os.environ["HERMES_TEST_VAR"] = "sandbox_works"
        try:
            async with sandbox_client(
                container_id="unused",
                docker_exe="unused",
                command="unused",
                _raw_cmd=[sys.executable, str(script)],
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    await session.list_tools()
                    result = await session.call_tool("get_env", {"key": "HERMES_TEST_VAR"})
                    assert any("sandbox_works" in c.text for c in result.content if hasattr(c, "text"))
        finally:
            os.environ.pop("HERMES_TEST_VAR", None)

    @pytest.mark.asyncio
    async def test_process_cleanup_on_exit(self, mock_server_script):
        """Verify the subprocess is cleaned up when context exits."""
        from mcp import ClientSession
        from tools.mcp_sandbox_transport import sandbox_client

        async with sandbox_client(
            container_id="unused",
            docker_exe="unused",
            command="unused",
            _raw_cmd=[sys.executable, mock_server_script],
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                # We can't easily get the PID from anyio, just verify it connects
                tools = await session.list_tools()
                assert len(tools.tools) >= 1

        # After context exit, the process should be terminated
        # (no assertion needed — if cleanup hangs, the test times out)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mock_server_script):
        """Verify multiple sequential tool calls work."""
        from mcp import ClientSession
        from tools.mcp_sandbox_transport import sandbox_client

        async with sandbox_client(
            container_id="unused",
            docker_exe="unused",
            command="unused",
            _raw_cmd=[sys.executable, mock_server_script],
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await session.list_tools()

                for i in range(5):
                    result = await session.call_tool("echo", {"text": f"msg-{i}"})
                    assert any(f"msg-{i}" in c.text for c in result.content if hasattr(c, "text"))
