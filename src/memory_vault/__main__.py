"""`python -m memory_vault` runs the server over stdio, for local MCP clients
(Claude Desktop config, MCP Inspector, Claude Code). Auth is not used over stdio:
the local process is trusted, so AUTH_MODE should be 'none'."""

from .config import get_settings
from .server import create_server

if __name__ == "__main__":
    create_server(get_settings()).run()  # default transport: stdio
