# Remote Filesystem MCP Server

An MCP server that provides file and terminal operations on remote hosts over SSH.

## Architecture

```
gateway (hermes-angelos)
  └── MCP server: remote-filesystem (this)
        ├── SSH pipe to host-A → python3 -c "...thin handler..."
        └── SSH pipe to host-B → python3 -c "...thin handler..."
```

The server manages persistent SSH connections to remote hosts. Each connection
pipes a self-contained JSON-RPC handler (~12KB, stdlib-only Python) to the
remote's `python3` interpreter. No files are written on the remote host.

## Requirements

- `openssh-client` in the gateway container
- `python3` on each remote host (no other dependencies)
- SSH key access to remote hosts (via ssh-agent or key file)

## Configuration

Add to `config.yaml`:

```yaml
mcp_servers:
  remote-filesystem:
    command: python3
    args: ["-m", "tools.mcp_servers.remote_filesystem"]
    env:
      SSH_AUTH_SOCK: /run/hermes-creds/S.ssh-agent
    timeout: 120
    connect_timeout: 15
```

## Tools

### Connection Management
- `remote_connect(host, user, port, key_path, identifier)` — Connect to a remote host
- `remote_disconnect(identifier)` — Disconnect
- `remote_list()` — List active connections

### File Operations
- `remote_read_file(identifier, path, offset, limit)` — Read a file
- `remote_write_file(identifier, path, content)` — Write a file
- `remote_patch(identifier, path, old_string, new_string, replace_all)` — Patch a file
- `remote_search(identifier, pattern, path, file_glob, limit)` — Search in files
- `remote_stat(identifier, path)` — Get file metadata

### Terminal
- `remote_execute(identifier, command, cwd, timeout)` — Run a shell command

## Security

- SSH keys never leave the gateway container
- Write deny-list on remote (blocks ~/.ssh, /etc/shadow, etc.)
- Path traversal rejection
- No network listeners on remote — stdin/stdout only
