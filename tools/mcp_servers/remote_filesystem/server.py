"""Remote Filesystem MCP Server.

Runs inside the sandbox container. Exposes tools for connecting to remote
hosts over SSH and performing file/terminal operations via a thin JSON-RPC
handler piped to the remote's Python interpreter.

Usage (as MCP server in hermes config.yaml):

    mcp_servers:
      remote-filesystem:
        command: python3
        args: ["-m", "tools.mcp_servers.remote_filesystem.server"]

Or standalone for testing:

    python3 -m tools.mcp_servers.remote_filesystem.server
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from tools.mcp_servers.remote_filesystem.connection import ConnectionManager

_manager = ConnectionManager()


def _format_error(response: dict) -> str:
    """Format a JSON-RPC error response as a readable string."""
    err = response.get("error", {})
    msg = f"Error {err.get('code', '?')}: {err.get('message', 'unknown')}"
    if err.get("data"):
        msg += f"\n{err['data']}"
    return msg


def _build_server() -> FastMCP:
    mcp = FastMCP(
        "remote-filesystem",
        instructions=(
            "Connect to remote hosts over SSH and perform file/terminal operations. "
            "All operations go through a lightweight JSON-RPC handler that runs on "
            "the remote host with zero installation required (just Python 3)."
        ),
    )

    # ── Connection management ──────────────────────────────────────

    @mcp.tool()
    def remote_connect(
        host: str,
        user: str,
        port: int = 22,
        key_path: str = "",
        identifier: str = "",
    ) -> str:
        """Connect to a remote host over SSH.

        Starts a persistent JSON-RPC handler on the remote host by piping
        a lightweight Python script over the SSH connection. No files are
        written on the remote system. Requires Python 3 on the remote host.

        Args:
            host: Hostname or IP address of the remote machine.
            user: SSH username.
            port: SSH port (default 22).
            key_path: Path to SSH private key (optional, uses ssh-agent if empty).
            identifier: Friendly name for this connection (default: hostname).
                       Use this to manage multiple connections.

        Returns:
            Connection status and identifier for use with other remote_* tools.
        """
        try:
            result = _manager.connect(host, user, port, key_path, identifier)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    @mcp.tool()
    def remote_disconnect(identifier: str) -> str:
        """Disconnect from a remote host.

        Args:
            identifier: The connection identifier returned by remote_connect.
        """
        result = _manager.disconnect(identifier)
        return json.dumps(result, indent=2)

    @mcp.tool()
    def remote_list() -> str:
        """List all active remote connections."""
        connections = _manager.list_connections()
        return json.dumps({"connections": connections}, indent=2)

    # ── File operations ────────────────────────────────────────────

    @mcp.tool()
    def remote_read_file(
        identifier: str,
        path: str,
        offset: int = 1,
        limit: int = 500,
    ) -> str:
        """Read a file on a remote host.

        Args:
            identifier: Connection identifier (from remote_connect).
            path: Absolute path to the file on the remote host.
            offset: Starting line number (1-based, default 1).
            limit: Maximum number of lines to read (default 500, max 2000).

        Returns:
            File content with line numbers, total line count, and file size.
        """
        result = _manager.call(identifier, "read_file", {
            "path": path, "offset": offset, "limit": limit,
        })
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    @mcp.tool()
    def remote_write_file(
        identifier: str,
        path: str,
        content: str,
    ) -> str:
        """Write content to a file on a remote host.

        Creates parent directories if needed. Protected paths (e.g. ~/.ssh)
        are blocked.

        Args:
            identifier: Connection identifier.
            path: Absolute path to write on the remote host.
            content: File content to write.
        """
        result = _manager.call(identifier, "write_file", {
            "path": path, "content": content,
        })
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    @mcp.tool()
    def remote_patch(
        identifier: str,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace text in a file on a remote host.

        Finds old_string in the file and replaces it with new_string.
        Returns a unified diff of the changes.

        Args:
            identifier: Connection identifier.
            path: Absolute path to the file.
            old_string: Text to find and replace.
            new_string: Replacement text.
            replace_all: If True, replace all occurrences (default: first only).
        """
        result = _manager.call(identifier, "patch", {
            "path": path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
        })
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    @mcp.tool()
    def remote_search(
        identifier: str,
        pattern: str,
        path: str = ".",
        file_glob: str = "",
        limit: int = 50,
    ) -> str:
        """Search for content in files on a remote host.

        Uses regex pattern matching. Searches recursively from the given path.

        Args:
            identifier: Connection identifier.
            pattern: Regular expression pattern to search for.
            path: Directory to search in (default: current directory).
            file_glob: Optional glob to filter filenames (e.g. "*.py").
            limit: Maximum matches to return (default 50, max 200).
        """
        params = {"pattern": pattern, "path": path, "limit": limit}
        if file_glob:
            params["file_glob"] = file_glob
        result = _manager.call(identifier, "search", params)
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    @mcp.tool()
    def remote_stat(identifier: str, path: str) -> str:
        """Get file/directory information on a remote host.

        Args:
            identifier: Connection identifier.
            path: Path to check on the remote host.

        Returns:
            File metadata: exists, size, mtime, is_dir, is_file, permissions.
        """
        result = _manager.call(identifier, "stat", {"path": path})
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

    @mcp.tool()
    def remote_execute(
        identifier: str,
        command: str,
        cwd: str = "",
        timeout: int = 60,
    ) -> str:
        """Execute a shell command on a remote host.

        Runs the command via bash on the remote system. Output is capped at 50KB.

        Args:
            identifier: Connection identifier.
            command: Shell command to execute.
            cwd: Working directory (optional).
            timeout: Command timeout in seconds (default 60, max 300).
        """
        params = {"command": command, "timeout": timeout}
        if cwd:
            params["cwd"] = cwd
        result = _manager.call(identifier, "execute", params)
        if isinstance(result, dict) and "error" in result:
            return _format_error(result)
        if isinstance(result, dict):
            output = result.get("output", "")
            rc = result.get("returncode", 0)
            if rc != 0:
                return f"{output}\n[exit code: {rc}]"
            return output
        return str(result)

    return mcp


def main():
    """Entry point for the MCP server."""
    if not HAS_MCP:
        print(
            "Error: MCP SDK not installed. Install with: pip install 'mcp>=1.2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    server = _build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
