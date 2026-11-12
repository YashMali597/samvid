"""End-to-end over real HTTP (Streamable HTTP transport + auth), with uvicorn."""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from memory_vault.config import Settings
from memory_vault.embeddings import HashingEmbedder
from memory_vault.server import create_server

from .conftest import TEST_DB


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """Run an ASGI app in a background thread for the duration of a test."""

    def __init__(self, app):
        self.port = _free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 15
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def token_app(settings):
    verifier = StaticTokenVerifier(
        tokens={
            "alice-token": {"client_id": "test", "sub": "alice", "login": "alice", "scopes": []},
            "bob-token": {"client_id": "test", "sub": "bob", "login": "bob", "scopes": []},
        }
    )
    mcp = create_server(settings, auth=verifier, embedder=HashingEmbedder())
    return mcp.http_app(path="/mcp", stateless_http=True)


async def test_http_multi_user_isolation(token_app):
    with _Server(token_app) as base:
        url = f"{base}/mcp"
        async with Client(url, auth="alice-token") as alice:
            await alice.call_tool("remember", {"content": "Alice's locker code is 4417", "tags": ["gym"]})
            hits = await alice.call_tool("recall", {"query": "locker code"})
            assert len(hits.structured_content["result"]) == 1
        async with Client(url, auth="bob-token") as bob:
            hits = await bob.call_tool("recall", {"query": "locker code"})
            assert hits.structured_content["result"] == []
            stats = await bob.call_tool("vault_stats", {})
            assert stats.structured_content["total"] == 0

        async with httpx.AsyncClient() as http:
            # No token -> 401 with a pointer to the protected resource metadata
            r = await http.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert r.status_code == 401
            # Wrong token -> 401
            r = await http.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Authorization": "Bearer nope"},
            )
            assert r.status_code == 401
            # Public routes stay public
            assert (await http.get(f"{base}/health")).json()["status"] == "ok"
            assert (await http.get(f"{base}/ready")).json()["status"] == "ready"
            assert "Samvid" in (await http.get(f"{base}/")).text


async def test_github_oauth_discovery(tmp_path):
    """With AUTH_MODE=github the server publishes standard OAuth discovery documents.

    No real GitHub app is contacted: discovery is served locally.
    """
    port = _free_port()
    settings = Settings(
        database_url=TEST_DB,
        auth_mode="github",
        public_base_url=f"https://vault.example.com",
        github_client_id="Ov23liTEST",
        github_client_secret="test-secret",
        jwt_signing_key="test-signing-key-test-signing-key",
        storage_encryption_key="test-encryption-key",
    )
    app = create_server(settings, embedder=None).http_app(path="/mcp", stateless_http=True)
    with _Server(app) as base:
        async with httpx.AsyncClient(base_url=base) as http:
            r = await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert r.status_code == 401
            challenge = r.headers["www-authenticate"]
            assert "resource_metadata=" in challenge
            assert "/.well-known/oauth-protected-resource/mcp" in challenge

            prm = (await http.get("/.well-known/oauth-protected-resource/mcp")).json()
            assert prm["resource"] == "https://vault.example.com/mcp"
            assert prm["authorization_servers"]

            asm = (await http.get("/.well-known/oauth-authorization-server")).json()
            assert asm["issuer"].rstrip("/") == "https://vault.example.com"
            assert "S256" in asm["code_challenge_methods_supported"]  # PKCE
            assert asm.get("client_id_metadata_document_supported") is True  # CIMD
            assert asm["authorization_endpoint"].endswith("/authorize")
