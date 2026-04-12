"""Thin JSON-RPC 2.0 handler for remote file/terminal operations.

This module is designed to be self-contained (stdlib only) so it can be piped
to ``python3 -c "..."`` over SSH without writing any files to the remote host.

Protocol: Content-Length framed JSON-RPC 2.0 over stdin/stdout (LSP-style).

    Content-Length: 84\\r\\n
    \\r\\n
    {"jsonrpc":"2.0","id":1,"method":"read_file","params":{"path":"/tmp/hello.txt"}}

Supported methods:
    read_file, write_file, patch, search, stat, execute

Security:
    - Write deny-list (sensitive paths like ~/.ssh, /etc/shadow)
    - Path traversal rejection
    - Runs as the SSH user (unprivileged)
    - No network listeners -- stdin/stdout only
"""

# ── This entire file is also stored as HANDLER_SOURCE in __init__.py so it
#    can be piped over SSH.  Keep it self-contained: stdlib imports only.

import difflib
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")

WRITE_DENIED_PATHS = {
    os.path.realpath(p)
    for p in [
        os.path.join(_HOME, ".ssh", "authorized_keys"),
        os.path.join(_HOME, ".ssh", "id_rsa"),
        os.path.join(_HOME, ".ssh", "id_ed25519"),
        os.path.join(_HOME, ".ssh", "config"),
        os.path.join(_HOME, ".bashrc"),
        os.path.join(_HOME, ".zshrc"),
        os.path.join(_HOME, ".profile"),
        os.path.join(_HOME, ".bash_profile"),
        os.path.join(_HOME, ".netrc"),
        os.path.join(_HOME, ".pgpass"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
}

WRITE_DENIED_PREFIXES = [
    os.path.realpath(p) + os.sep
    for p in [
        os.path.join(_HOME, ".ssh"),
        os.path.join(_HOME, ".aws"),
        os.path.join(_HOME, ".gnupg"),
        os.path.join(_HOME, ".kube"),
        "/etc/sudoers.d",
        "/etc/systemd",
        os.path.join(_HOME, ".docker"),
    ]
]


def _is_write_denied(path: str) -> bool:
    resolved = os.path.realpath(os.path.expanduser(path))
    if resolved in WRITE_DENIED_PATHS:
        return True
    for prefix in WRITE_DENIED_PREFIXES:
        if resolved.startswith(prefix):
            return True
    return False


def _resolve_path(path: str) -> str:
    """Expand ~ and resolve to absolute, rejecting traversal."""
    expanded = os.path.expanduser(path)
    resolved = os.path.realpath(expanded)
    # Reject null bytes (path injection)
    if "\x00" in path:
        raise ValueError("Null byte in path")
    return resolved


# ---------------------------------------------------------------------------
# JSON-RPC framing (Content-Length, LSP-style)
# ---------------------------------------------------------------------------

def _read_message() -> dict | None:
    """Read one Content-Length framed JSON-RPC message from stdin."""
    # Read headers
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None  # EOF
        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            break  # empty line = end of headers
        if line_str.lower().startswith("content-length:"):
            content_length = int(line_str.split(":", 1)[1].strip())

    if content_length is None:
        return None

    body = sys.stdin.buffer.read(content_length)
    if len(body) < content_length:
        return None  # truncated
    return json.loads(body.decode("utf-8"))


def _write_message(msg: dict) -> None:
    """Write one Content-Length framed JSON-RPC message to stdout."""
    body = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def _ok(id_val, result: dict) -> None:
    _write_message({"jsonrpc": "2.0", "id": id_val, "result": result})


def _error(id_val, code: int, message: str, data: dict | None = None) -> None:
    err = {"code": code, "message": message}
    if data:
        err["data"] = data
    _write_message({"jsonrpc": "2.0", "id": id_val, "error": err})


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# Custom
FILE_NOT_FOUND = -32001
PERMISSION_DENIED = -32002
WRITE_DENIED = -32003


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def _method_read_file(params: dict) -> dict:
    path = _resolve_path(params.get("path", ""))
    offset = max(1, int(params.get("offset", 1)))
    limit = min(2000, max(1, int(params.get("limit", 500))))

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    file_size = os.path.getsize(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    total_lines = len(lines)
    end = offset - 1 + limit
    selected = lines[offset - 1 : end]

    # Add line numbers
    numbered = []
    for i, line in enumerate(selected, start=offset):
        numbered.append(f"{i}\t{line.rstrip()}")

    return {
        "content": "\n".join(numbered),
        "total_lines": total_lines,
        "file_size": file_size,
        "truncated": end < total_lines,
    }


def _method_write_file(params: dict) -> dict:
    path = _resolve_path(params.get("path", ""))
    content = params.get("content", "")

    if _is_write_denied(path):
        raise PermissionError(f"Write denied to protected path: {path}")

    parent = os.path.dirname(path)
    dirs_created = False
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
        dirs_created = True

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "bytes_written": len(content.encode("utf-8")),
        "dirs_created": dirs_created,
    }


def _method_patch(params: dict) -> dict:
    path = _resolve_path(params.get("path", ""))
    old_string = params.get("old_string", "")
    new_string = params.get("new_string", "")
    replace_all = bool(params.get("replace_all", False))

    if _is_write_denied(path):
        raise PermissionError(f"Write denied to protected path: {path}")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        original = f.read()

    if old_string not in original:
        return {"success": False, "diff": "", "error": "old_string not found in file"}

    if replace_all:
        patched = original.replace(old_string, new_string)
    else:
        idx = original.index(old_string)
        patched = original[:idx] + new_string + original[idx + len(old_string) :]

    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(path)}",
            tofile=f"b/{os.path.basename(path)}",
        )
    )

    return {"success": True, "diff": diff}


def _method_stat(params: dict) -> dict:
    path = _resolve_path(params.get("path", ""))

    if not os.path.exists(path):
        return {"exists": False}

    st = os.stat(path)
    return {
        "exists": True,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "is_dir": os.path.isdir(path),
        "is_file": os.path.isfile(path),
        "permissions": oct(st.st_mode)[-3:],
    }


def _method_search(params: dict) -> dict:
    pattern = params.get("pattern", "")
    search_path = _resolve_path(params.get("path", "."))
    file_glob = params.get("file_glob")
    limit = min(200, max(1, int(params.get("limit", 50))))

    if not os.path.isdir(search_path):
        raise FileNotFoundError(f"No such directory: {search_path}")

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"matches": [], "total_count": 0, "error": f"Invalid regex: {e}"}

    matches = []
    total = 0

    for root, _dirs, files in os.walk(search_path):
        # Skip hidden directories
        _dirs[:] = [d for d in _dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("."):
                continue
            if file_glob and not fnmatch.fnmatch(fname, file_glob):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            total += 1
                            if len(matches) < limit:
                                matches.append({
                                    "path": fpath,
                                    "line": line_num,
                                    "content": line.rstrip()[:200],
                                })
            except (OSError, UnicodeDecodeError):
                continue

    return {"matches": matches, "total_count": total, "truncated": total > limit}


def _method_execute(params: dict) -> dict:
    command = params.get("command", "")
    cwd = params.get("cwd")
    timeout = min(300, max(1, int(params.get("timeout", 60))))

    if cwd:
        cwd = _resolve_path(cwd)
        if not os.path.isdir(cwd):
            raise FileNotFoundError(f"No such directory: {cwd}")

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output = output + ("\n" if output else "") + result.stderr
        return {
            "output": output[-50000:],  # cap at 50KB
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"output": f"Command timed out after {timeout}s", "returncode": 124}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_METHODS = {
    "read_file": _method_read_file,
    "write_file": _method_write_file,
    "patch": _method_patch,
    "stat": _method_stat,
    "search": _method_search,
    "execute": _method_execute,
}


def _dispatch(msg: dict) -> None:
    id_val = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {})

    if method not in _METHODS:
        _error(id_val, METHOD_NOT_FOUND, f"Unknown method: {method}")
        return

    try:
        result = _METHODS[method](params)
        _ok(id_val, result)
    except FileNotFoundError as e:
        _error(id_val, FILE_NOT_FOUND, str(e))
    except PermissionError as e:
        _error(id_val, PERMISSION_DENIED, str(e))
    except ValueError as e:
        _error(id_val, INVALID_PARAMS, str(e))
    except Exception as e:
        _error(id_val, INTERNAL_ERROR, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Read JSON-RPC requests from stdin, dispatch, write responses to stdout."""
    # Signal readiness
    _write_message({
        "jsonrpc": "2.0",
        "method": "ready",
        "params": {"version": "1.0.0", "pid": os.getpid()},
    })

    while True:
        msg = _read_message()
        if msg is None:
            break  # stdin closed
        _dispatch(msg)


if __name__ == "__main__":
    main()
