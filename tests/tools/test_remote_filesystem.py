"""Tests for the remote filesystem JSON-RPC handler and connection manager.

Tests the handler as a subprocess (no SSH required) and the ConnectionManager
routing layer. SSH-based integration tests are deferred to the crash dummy.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HANDLER_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "tools", "mcp_servers", "remote_filesystem", "handler.py",
)
HANDLER_PATH = os.path.normpath(HANDLER_PATH)


class HandlerClient:
    """Spawns the JSON-RPC handler as a subprocess for testing."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, HANDLER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._next_id = 0

    def send(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()
        return self._recv()

    def recv_notification(self) -> dict:
        """Read a notification (no id) from the handler."""
        return self._recv()

    def _recv(self) -> dict:
        content_length = None
        while True:
            line = self.proc.stdout.readline().decode("utf-8").strip()
            if not line:
                break
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
        assert content_length is not None, "No Content-Length header received"
        body = self.proc.stdout.read(content_length)
        return json.loads(body.decode("utf-8"))

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=5)


@pytest.fixture
def handler():
    """Provide a running handler subprocess."""
    client = HandlerClient()
    # Read and discard the ready notification
    ready = client.recv_notification()
    assert ready["method"] == "ready"
    assert "pid" in ready.get("params", {})
    yield client
    client.close()


@pytest.fixture
def tmp_file(tmp_path):
    """Create a temporary file with known content."""
    f = tmp_path / "test.txt"
    f.write_text("line one\nline two\nline three\nline four\nline five\n")
    return str(f)


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temporary directory with some files."""
    (tmp_path / "hello.py").write_text("# hello\nprint('hello world')\n")
    (tmp_path / "goodbye.py").write_text("# goodbye\nprint('goodbye world')\n")
    (tmp_path / "data.txt").write_text("some data here\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("# nested\nx = 42\n")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Handler: ready notification
# ---------------------------------------------------------------------------

class TestHandlerReady:
    def test_ready_on_start(self):
        client = HandlerClient()
        ready = client.recv_notification()
        assert ready["jsonrpc"] == "2.0"
        assert ready["method"] == "ready"
        assert "version" in ready["params"]
        assert "pid" in ready["params"]
        client.close()


# ---------------------------------------------------------------------------
# Handler: read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_read_file_basic(self, handler, tmp_file):
        resp = handler.send("read_file", {"path": tmp_file})
        result = resp["result"]
        assert result["total_lines"] == 5
        assert "line one" in result["content"]
        assert result["truncated"] is False

    def test_read_file_with_offset(self, handler, tmp_file):
        resp = handler.send("read_file", {"path": tmp_file, "offset": 3, "limit": 2})
        result = resp["result"]
        assert "line three" in result["content"]
        assert "line four" in result["content"]
        assert "line one" not in result["content"]

    def test_read_file_not_found(self, handler):
        resp = handler.send("read_file", {"path": "/tmp/nonexistent_hermes_test_xyz"})
        assert "error" in resp
        assert resp["error"]["code"] == -32001  # FILE_NOT_FOUND

    def test_read_file_line_numbers(self, handler, tmp_file):
        resp = handler.send("read_file", {"path": tmp_file, "offset": 2, "limit": 1})
        content = resp["result"]["content"]
        assert content.startswith("2\t")


# ---------------------------------------------------------------------------
# Handler: write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_write_file_basic(self, handler, tmp_path):
        path = str(tmp_path / "new_file.txt")
        resp = handler.send("write_file", {"path": path, "content": "hello\n"})
        result = resp["result"]
        assert result["bytes_written"] == 6
        assert os.path.isfile(path)
        assert open(path).read() == "hello\n"

    def test_write_file_creates_dirs(self, handler, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "deep.txt")
        resp = handler.send("write_file", {"path": path, "content": "deep\n"})
        result = resp["result"]
        assert result["dirs_created"] is True
        assert open(path).read() == "deep\n"

    def test_write_file_deny_ssh(self, handler):
        resp = handler.send("write_file", {"path": "~/.ssh/authorized_keys", "content": "evil"})
        assert "error" in resp
        assert resp["error"]["code"] == -32002  # PERMISSION_DENIED

    def test_write_file_deny_etc_shadow(self, handler):
        resp = handler.send("write_file", {"path": "/etc/shadow", "content": "evil"})
        assert "error" in resp
        assert resp["error"]["code"] == -32002


# ---------------------------------------------------------------------------
# Handler: patch
# ---------------------------------------------------------------------------

class TestPatch:
    def test_patch_basic(self, handler, tmp_file):
        resp = handler.send("patch", {
            "path": tmp_file,
            "old_string": "line two",
            "new_string": "LINE TWO",
        })
        result = resp["result"]
        assert result["success"] is True
        assert "LINE TWO" in result["diff"]
        assert "LINE TWO" in open(tmp_file).read()

    def test_patch_replace_all(self, handler, tmp_path):
        f = tmp_path / "multi.txt"
        f.write_text("aaa bbb aaa ccc aaa\n")
        resp = handler.send("patch", {
            "path": str(f),
            "old_string": "aaa",
            "new_string": "ZZZ",
            "replace_all": True,
        })
        result = resp["result"]
        assert result["success"] is True
        content = f.read_text()
        assert content.count("ZZZ") == 3
        assert "aaa" not in content

    def test_patch_not_found(self, handler, tmp_file):
        resp = handler.send("patch", {
            "path": tmp_file,
            "old_string": "NONEXISTENT_STRING",
            "new_string": "replacement",
        })
        result = resp["result"]
        assert result["success"] is False

    def test_patch_write_denied(self, handler):
        resp = handler.send("patch", {
            "path": "~/.ssh/config",
            "old_string": "Host",
            "new_string": "evil",
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32002


# ---------------------------------------------------------------------------
# Handler: stat
# ---------------------------------------------------------------------------

class TestStat:
    def test_stat_file(self, handler, tmp_file):
        resp = handler.send("stat", {"path": tmp_file})
        result = resp["result"]
        assert result["exists"] is True
        assert result["is_file"] is True
        assert result["is_dir"] is False
        assert result["size"] > 0

    def test_stat_directory(self, handler, tmp_dir):
        resp = handler.send("stat", {"path": tmp_dir})
        result = resp["result"]
        assert result["exists"] is True
        assert result["is_dir"] is True

    def test_stat_nonexistent(self, handler):
        resp = handler.send("stat", {"path": "/tmp/nonexistent_hermes_xyz"})
        result = resp["result"]
        assert result["exists"] is False


# ---------------------------------------------------------------------------
# Handler: search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_content(self, handler, tmp_dir):
        resp = handler.send("search", {"pattern": "hello", "path": tmp_dir})
        result = resp["result"]
        assert result["total_count"] >= 1
        assert any("hello" in m["content"] for m in result["matches"])

    def test_search_with_glob(self, handler, tmp_dir):
        resp = handler.send("search", {"pattern": "print", "path": tmp_dir, "file_glob": "*.py"})
        result = resp["result"]
        # Should find in .py files but not .txt
        for m in result["matches"]:
            assert m["path"].endswith(".py")

    def test_search_no_results(self, handler, tmp_dir):
        resp = handler.send("search", {"pattern": "ZZZZNOTFOUND", "path": tmp_dir})
        result = resp["result"]
        assert result["total_count"] == 0

    def test_search_invalid_regex(self, handler, tmp_dir):
        resp = handler.send("search", {"pattern": "[invalid", "path": tmp_dir})
        result = resp["result"]
        assert "error" in result


# ---------------------------------------------------------------------------
# Handler: execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_execute_basic(self, handler):
        resp = handler.send("execute", {"command": "echo hello_from_handler"})
        result = resp["result"]
        assert "hello_from_handler" in result["output"]
        assert result["returncode"] == 0

    def test_execute_with_cwd(self, handler, tmp_dir):
        resp = handler.send("execute", {"command": "ls *.py", "cwd": tmp_dir})
        result = resp["result"]
        assert "hello.py" in result["output"]

    def test_execute_nonzero_exit(self, handler):
        resp = handler.send("execute", {"command": "exit 42"})
        result = resp["result"]
        assert result["returncode"] == 42

    def test_execute_timeout(self, handler):
        resp = handler.send("execute", {"command": "sleep 10", "timeout": 1})
        result = resp["result"]
        assert result["returncode"] == 124


# ---------------------------------------------------------------------------
# Handler: unknown method
# ---------------------------------------------------------------------------

class TestUnknownMethod:
    def test_unknown_method(self, handler):
        resp = handler.send("nonexistent_method", {})
        assert "error" in resp
        assert resp["error"]["code"] == -32601  # METHOD_NOT_FOUND


# ---------------------------------------------------------------------------
# Handler: python3 -c delivery
# ---------------------------------------------------------------------------

class TestDeliveryMode:
    def test_handler_via_python3_c(self):
        """Verify the handler works when delivered as python3 -c (SSH simulation)."""
        with open(HANDLER_PATH) as f:
            source = f.read()

        proc = subprocess.Popen(
            [sys.executable, "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        client = HandlerClient.__new__(HandlerClient)
        client.proc = proc
        client._next_id = 0

        ready = client.recv_notification()
        assert ready["method"] == "ready"

        resp = client.send("execute", {"command": "echo delivery_works"})
        assert "delivery_works" in resp["result"]["output"]

        client.close()

    def test_handler_source_size(self):
        """Ensure handler fits in shell argument limits (< 128KB)."""
        with open(HANDLER_PATH) as f:
            source = f.read()
        assert len(source) < 128_000, f"Handler is {len(source)} bytes, too large for -c delivery"


# ---------------------------------------------------------------------------
# ConnectionManager (subprocess, no SSH)
# ---------------------------------------------------------------------------

class TestConnectionManager:
    def test_connect_call_disconnect(self, tmp_path):
        """Test ConnectionManager with a local subprocess (no SSH)."""
        from tools.mcp_servers.remote_filesystem.connection import RemoteConnection, ConnectionManager

        # Create a connection that uses subprocess instead of SSH
        conn = RemoteConnection.__new__(RemoteConnection)
        conn.host = "localhost"
        conn.user = "test"
        conn.port = 22
        conn.key_path = ""
        conn.identifier = "test-local"
        conn._lock = threading.Lock()
        conn._next_id = 1
        conn._proc = subprocess.Popen(
            [sys.executable, HANDLER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Read ready
        ready = conn._recv(timeout=5)
        assert ready is not None
        assert ready["method"] == "ready"

        # File roundtrip
        test_file = str(tmp_path / "mgr_test.txt")
        result = conn.call("write_file", {"path": test_file, "content": "mgr\n"})
        assert result["bytes_written"] == 4

        result = conn.call("read_file", {"path": test_file})
        assert "mgr" in result["content"]

        result = conn.call("patch", {"path": test_file, "old_string": "mgr", "new_string": "patched"})
        assert result["success"] is True

        result = conn.call("stat", {"path": test_file})
        assert result["exists"] is True

        # Disconnect
        conn.disconnect()
        assert not conn.connected
