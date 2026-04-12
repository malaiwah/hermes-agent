"""Integration tests for remote filesystem over SSH.

Requires SSH access to the test host. Run with:
    python3 -m pytest tests/tools/test_remote_filesystem_ssh.py -v -o addopts= -m integration

Set environment variables:
    REMOTE_HOST: hostname or IP (default: localhost)
    REMOTE_USER: SSH username (default: current user)
    REMOTE_PORT: SSH port (default: 22)
    REMOTE_KEY:  Path to SSH key (default: empty, uses ssh-agent)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from tools.mcp_servers.remote_filesystem.connection import ConnectionManager

REMOTE_HOST = os.environ.get("REMOTE_HOST", "10.15.0.151")
REMOTE_USER = os.environ.get("REMOTE_USER", "mbelleau")
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "22"))
REMOTE_KEY = os.environ.get("REMOTE_KEY", "")


@pytest.fixture(scope="module")
def manager():
    mgr = ConnectionManager()
    yield mgr
    mgr.disconnect_all()


@pytest.fixture(scope="module")
def conn_id(manager):
    """Connect once per test module."""
    result = manager.connect(
        host=REMOTE_HOST,
        user=REMOTE_USER,
        port=REMOTE_PORT,
        key_path=REMOTE_KEY,
        identifier="test-ssh",
    )
    assert result["status"] in ("connected", "already_connected"), f"Failed to connect: {result}"
    return "test-ssh"


@pytest.mark.integration
class TestRemoteSSH:
    def test_stat(self, manager, conn_id):
        result = manager.call(conn_id, "stat", {"path": "/etc/hostname"})
        assert result["exists"] is True
        assert result["is_file"] is True

    def test_read_file(self, manager, conn_id):
        result = manager.call(conn_id, "read_file", {"path": "/etc/hostname"})
        assert result["total_lines"] >= 1
        assert len(result["content"]) > 0

    def test_write_read_patch_roundtrip(self, manager, conn_id):
        test_path = "/tmp/hermes_remote_test.txt"

        # Write
        result = manager.call(conn_id, "write_file", {
            "path": test_path, "content": "hello remote\nline two\n",
        })
        assert result["bytes_written"] > 0

        # Read
        result = manager.call(conn_id, "read_file", {"path": test_path})
        assert "hello remote" in result["content"]

        # Patch
        result = manager.call(conn_id, "patch", {
            "path": test_path,
            "old_string": "hello remote",
            "new_string": "goodbye remote",
        })
        assert result["success"] is True
        assert "goodbye" in result.get("diff", "")

        # Verify
        result = manager.call(conn_id, "read_file", {"path": test_path})
        assert "goodbye remote" in result["content"]

        # Cleanup
        manager.call(conn_id, "execute", {"command": f"rm -f {test_path}"})

    def test_search(self, manager, conn_id):
        # Write a file to search
        manager.call(conn_id, "write_file", {
            "path": "/tmp/hermes_search_test.py",
            "content": "# test\nprint('FINDME_TOKEN')\n",
        })
        result = manager.call(conn_id, "search", {
            "pattern": "FINDME_TOKEN",
            "path": "/tmp",
            "file_glob": "hermes_search_test*",
        })
        assert result["total_count"] >= 1
        # Cleanup
        manager.call(conn_id, "execute", {"command": "rm -f /tmp/hermes_search_test.py"})

    def test_execute(self, manager, conn_id):
        result = manager.call(conn_id, "execute", {"command": "whoami && hostname"})
        assert result["returncode"] == 0
        assert len(result["output"].strip()) > 0

    def test_write_denied(self, manager, conn_id):
        result = manager.call(conn_id, "write_file", {
            "path": "~/.ssh/authorized_keys", "content": "evil",
        })
        # Should be a JSON-RPC error response
        assert "error" in result
        assert result["error"]["code"] == -32002

    def test_multiple_connections(self, manager):
        """Test that multiple connections to the same host work."""
        result = manager.connect(
            host=REMOTE_HOST, user=REMOTE_USER, port=REMOTE_PORT,
            key_path=REMOTE_KEY, identifier="test-ssh-2",
        )
        assert result["status"] in ("connected", "already_connected")

        r1 = manager.call("test-ssh", "execute", {"command": "echo conn1"})
        r2 = manager.call("test-ssh-2", "execute", {"command": "echo conn2"})
        assert "conn1" in r1["output"]
        assert "conn2" in r2["output"]

        manager.disconnect("test-ssh-2")
