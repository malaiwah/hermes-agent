"""SSH connection manager for remote JSON-RPC handler.

Manages the lifecycle of SSH pipes to remote hosts. Each connection starts
the thin JSON-RPC handler on the remote side by piping its source code to
``python3 -c "..."``.  Communication uses Content-Length framed JSON-RPC 2.0
over the SSH pipe's stdin/stdout.

This module is stdlib-only so it can be tested without the MCP SDK.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Load the handler source at import time
_HANDLER_PATH = Path(__file__).parent / "handler.py"
_HANDLER_SOURCE: str | None = None


def _get_handler_source() -> str:
    """Load and cache the handler source code."""
    global _HANDLER_SOURCE
    if _HANDLER_SOURCE is None:
        _HANDLER_SOURCE = _HANDLER_PATH.read_text(encoding="utf-8")
    return _HANDLER_SOURCE


class RemoteConnection:
    """A persistent SSH pipe to a remote JSON-RPC handler.

    Usage::

        conn = RemoteConnection("myhost", "deploy")
        conn.connect()
        result = conn.call("read_file", {"path": "/etc/hostname"})
        conn.disconnect()
    """

    def __init__(
        self,
        host: str,
        user: str,
        port: int = 22,
        key_path: str = "",
        identifier: str = "",
    ):
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path
        self.identifier = identifier or host
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1

    @property
    def connected(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def connect(self, timeout: float = 30) -> dict:
        """Start SSH connection and launch handler on remote host.

        Returns the handler's ready notification.
        """
        if self.connected:
            raise RuntimeError(f"Already connected to {self.identifier}")

        handler_source = _get_handler_source()

        # Build SSH command that pipes the handler to python3 -c
        ssh_cmd = ["ssh"]
        ssh_cmd.extend(["-o", "BatchMode=yes"])
        ssh_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        ssh_cmd.extend(["-o", "ConnectTimeout=10"])
        ssh_cmd.extend(["-o", "ServerAliveInterval=15"])
        ssh_cmd.extend(["-o", "ServerAliveCountMax=3"])
        if self.port != 22:
            ssh_cmd.extend(["-p", str(self.port)])
        if self.key_path:
            ssh_cmd.extend(["-i", self.key_path])
        ssh_cmd.append(f"{self.user}@{self.host}")
        # Use python3 -c with the handler source
        ssh_cmd.extend(["python3", "-c", shlex.quote(handler_source)])

        logger.info(
            "remote-fs: connecting to %s@%s:%d (id=%s)",
            self.user, self.host, self.port, self.identifier,
        )

        self._proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the ready notification
        try:
            ready = self._recv(timeout=timeout)
        except Exception as e:
            self._kill()
            # Try to read stderr for diagnostics
            stderr = ""
            if self._proc and self._proc.stderr:
                try:
                    stderr = self._proc.stderr.read(2000).decode(errors="replace")
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to connect to {self.identifier}: {e}"
                + (f"\nSSH stderr: {stderr}" if stderr else "")
            ) from e

        if not ready or ready.get("method") != "ready":
            self._kill()
            raise RuntimeError(
                f"Unexpected first message from handler: {ready}"
            )

        logger.info(
            "remote-fs: connected to %s (handler pid=%s)",
            self.identifier, ready.get("params", {}).get("pid"),
        )
        return ready.get("params", {})

    def disconnect(self) -> None:
        """Close the SSH connection."""
        self._kill()
        logger.info("remote-fs: disconnected from %s", self.identifier)

    def call(self, method: str, params: dict | None = None, timeout: float = 60) -> dict:
        """Send a JSON-RPC request and return the result.

        Raises RuntimeError on transport errors, or returns the error dict
        for JSON-RPC level errors.
        """
        if not self.connected:
            raise RuntimeError(f"Not connected to {self.identifier}")

        with self._lock:
            req_id = self._next_id
            self._next_id += 1

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        with self._lock:
            self._send(msg)
            response = self._recv(timeout=timeout)

        if response is None:
            self._kill()
            raise RuntimeError(f"Connection to {self.identifier} lost")

        if "error" in response:
            return response  # caller handles JSON-RPC errors

        return response.get("result", {})

    # -- Wire protocol (Content-Length framed) --

    def _send(self, msg: dict) -> None:
        body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        try:
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._kill()
            raise RuntimeError(f"Pipe to {self.identifier} broken: {e}") from e

    def _recv(self, timeout: float = 60) -> dict | None:
        """Read one Content-Length framed message from stdout.

        Uses a background thread for the blocking read to support timeouts
        on platforms where select() on subprocess pipes is unreliable.
        """
        import queue
        import time

        if self._proc is None or self._proc.stdout is None:
            return None

        result_q: queue.Queue = queue.Queue()

        def _reader():
            try:
                stdout = self._proc.stdout
                # Read headers byte-by-byte until \r\n\r\n
                # Cap at 8KB to prevent DoS from malicious/broken handler
                MAX_HEADER_SIZE = 8192
                content_length = None
                header_buf = b""
                while True:
                    byte = stdout.read(1)
                    if not byte:
                        result_q.put(None)
                        return
                    header_buf += byte
                    if len(header_buf) > MAX_HEADER_SIZE:
                        result_q.put(None)
                        return
                    if header_buf.endswith(b"\r\n\r\n"):
                        for line in header_buf.decode("utf-8", errors="replace").split("\r\n"):
                            if line.lower().startswith("content-length:"):
                                content_length = int(line.split(":", 1)[1].strip())
                        break

                if content_length is None:
                    result_q.put(None)
                    return

                body = stdout.read(content_length)
                if len(body) < content_length:
                    result_q.put(None)
                    return
                result_q.put(json.loads(body.decode("utf-8")))
            except Exception:
                result_q.put(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        try:
            return result_q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"Timeout reading from {self.identifier}")

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None


class ConnectionManager:
    """Registry of active remote connections."""

    def __init__(self):
        self._connections: dict[str, RemoteConnection] = {}
        self._lock = threading.Lock()

    def connect(
        self,
        host: str,
        user: str,
        port: int = 22,
        key_path: str = "",
        identifier: str = "",
    ) -> dict:
        """Create and connect a new remote connection."""
        ident = identifier or host

        with self._lock:
            if ident in self._connections:
                existing = self._connections[ident]
                if existing.connected:
                    return {
                        "identifier": ident,
                        "status": "already_connected",
                        "host": existing.host,
                        "user": existing.user,
                    }
                # Dead connection — remove and reconnect
                del self._connections[ident]

            conn = RemoteConnection(host, user, port, key_path, ident)
            self._connections[ident] = conn

        info = conn.connect()
        return {
            "identifier": ident,
            "status": "connected",
            "host": host,
            "user": user,
            "handler_version": info.get("version", "unknown"),
            "handler_pid": info.get("pid"),
        }

    def disconnect(self, identifier: str) -> dict:
        with self._lock:
            conn = self._connections.pop(identifier, None)
        if conn is None:
            return {"identifier": identifier, "status": "not_found"}
        conn.disconnect()
        return {"identifier": identifier, "status": "disconnected"}

    def list_connections(self) -> list[dict]:
        with self._lock:
            result = []
            for ident, conn in self._connections.items():
                result.append({
                    "identifier": ident,
                    "host": conn.host,
                    "user": conn.user,
                    "port": conn.port,
                    "connected": conn.connected,
                })
            return result

    def get(self, identifier: str) -> RemoteConnection | None:
        with self._lock:
            return self._connections.get(identifier)

    def call(self, identifier: str, method: str, params: dict | None = None, timeout: float = 60) -> Any:
        """Route a call to a specific remote connection."""
        conn = self.get(identifier)
        if conn is None:
            raise RuntimeError(f"No connection with identifier '{identifier}'")
        if not conn.connected:
            raise RuntimeError(f"Connection '{identifier}' is not active")
        return conn.call(method, params, timeout=timeout)

    def disconnect_all(self) -> None:
        with self._lock:
            for conn in self._connections.values():
                try:
                    conn.disconnect()
                except Exception:
                    pass
            self._connections.clear()
