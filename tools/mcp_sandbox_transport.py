"""Sandbox MCP transport — run MCP servers inside Docker/Podman containers.

Mirrors ``mcp.client.stdio.stdio_client`` but uses ``docker exec -i`` to
communicate with an MCP server running inside an existing sandbox container.
The exec process's stdin/stdout IS the MCP server's stdin/stdout.

Usage::

    async with sandbox_client(container_id, docker_exe, command, args, env) as (
        read_stream, write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            ...

Config (config.yaml)::

    mcp_servers:
      my-server:
        command: python3
        args: ["-m", "my_mcp_server"]
        sandbox: true
        env:
          SSH_AUTH_SOCK: /run/hermes-creds/S.ssh-agent
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import asynccontextmanager

import anyio
from anyio.streams.text import TextReceiveStream

logger = logging.getLogger(__name__)

# Match MCP SDK's termination timeout
_PROCESS_TERMINATION_TIMEOUT = 2.0


@asynccontextmanager
async def sandbox_client(
    container_id: str,
    docker_exe: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    *,
    _raw_cmd: list[str] | None = None,
):
    """MCP client transport via ``docker exec -i`` into a sandbox container.

    Spawns ``docker exec -i [-e K=V ...] <container> <command> [args...]``
    and yields ``(read_stream, write_stream)`` compatible with
    ``mcp.ClientSession``.

    For testing without Docker, pass ``_raw_cmd`` to override the exec command.

    The streams carry ``SessionMessage`` objects using newline-delimited
    JSON-RPC 2.0 ��� the same wire protocol as MCP stdio transport.
    """
    from mcp import types
    from mcp.shared.message import SessionMessage

    read_stream_writer, read_stream = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream[
        SessionMessage
    ](0)

    # Build docker exec command (or use raw command for testing)
    if _raw_cmd is not None:
        exec_cmd = list(_raw_cmd)
    else:
        import re
        _ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        exec_cmd = [docker_exe, "exec", "-i"]
        if env:
            for key, value in sorted(env.items()):
                if not _ENV_KEY_RE.match(key):
                    logger.warning("sandbox-mcp: skipping invalid env key: %r", key)
                    continue
                exec_cmd.extend(["-e", f"{key}={value}"])
        exec_cmd.append("--")  # end of flags, prevents container_id/command flag injection
        exec_cmd.append(container_id)
        exec_cmd.append(command)
        if args:
            exec_cmd.extend(args)

    logger.info("sandbox-mcp: launching %s in container %s", command, container_id[:12])

    try:
        process = await anyio.open_process(
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        await read_stream.aclose()
        await write_stream.aclose()
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise

    async def stdout_reader():
        """Read newline-delimited JSON-RPC from the MCP server's stdout."""
        assert process.stdout, "Process missing stdout"
        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(process.stdout):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        if not line.strip():
                            continue  # skip blank lines between messages
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception:
                            logger.warning(
                                "sandbox-mcp: failed to parse from %s: %.200s",
                                container_id[:12], line,
                            )
                            continue
                        await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdin_writer():
        """Write newline-delimited JSON-RPC to the MCP server's stdin."""
        assert process.stdin, "Process missing stdin"
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json_str = session_message.message.model_dump_json(
                        by_alias=True, exclude_none=True,
                    )
                    await process.stdin.send((json_str + "\n").encode("utf-8"))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stderr_drain():
        """Drain stderr to prevent pipe buffer blocking docker exec."""
        if not process.stderr:
            return
        try:
            async for chunk in TextReceiveStream(process.stderr):
                for line in chunk.splitlines():
                    if line.strip():
                        logger.debug("sandbox-mcp [%s]: %s", container_id[:12], line.rstrip())
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            pass

    async with anyio.create_task_group() as tg, process:
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_drain)
        try:
            yield read_stream, write_stream
        finally:
            # MCP shutdown sequence: close stdin, wait, then terminate
            if process.stdin:
                try:
                    await process.stdin.aclose()
                except Exception:
                    pass

            try:
                with anyio.fail_after(_PROCESS_TERMINATION_TIMEOUT):
                    await process.wait()
            except TimeoutError:
                process.terminate()
                try:
                    with anyio.fail_after(_PROCESS_TERMINATION_TIMEOUT):
                        await process.wait()
                except (TimeoutError, ProcessLookupError):
                    process.kill()
            except ProcessLookupError:
                pass

            # Close all stream endpoints. Each may already be closed by
            # task group tasks (stdout_reader, stdin_writer) or by the
            # ClientSession teardown, so guard each individually.
            for stream in (read_stream, write_stream, read_stream_writer, write_stream_reader):
                try:
                    await stream.aclose()
                except Exception:
                    pass
