"""Remote Filesystem MCP Server.

An MCP server that manages SSH connections to remote hosts and provides
file/terminal operations through a thin JSON-RPC handler piped over SSH.

Architecture:
    gateway → sandbox container → [this MCP server] → SSH pipe → remote host
                                                        ↓
                                              python3 -c "...handler..."
                                                        ↓
                                              native file I/O on remote
"""
