"""ASGI entry point:  uvicorn memory_vault.app:app --host 0.0.0.0 --port 8000

Lesson: the MCP endpoint is just an HTTP route (/mcp) on a normal ASGI app, so you
deploy it like any web service: behind TLS, with health checks, and with as many
replicas as you like, because the 2026-07-28 protocol carries no session state.
"""

from __future__ import annotations

import logging

from .config import get_settings
from .server import create_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()
mcp = create_server(settings)
app = mcp.http_app(
    path=settings.mcp_path,
    # Also keep legacy (2025-xx) clients stateless: no server memory between their
    # requests, so free-tier restarts and multiple instances never break a session.
    stateless_http=True,
)
