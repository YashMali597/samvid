"""The MCP surface, through a real MCP client, in-memory (no network)."""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.exceptions import ToolError

from memory_vault.embeddings import HashingEmbedder
from memory_vault.server import CONFIRM_PHRASE, create_server


@pytest.fixture
def mcp(settings):
    return create_server(settings, auth=None, embedder=HashingEmbedder())


async def test_tool_catalog(mcp):
    async with Client(mcp) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert set(tools) == {
        "remember", "recall", "list_memories", "update_memory", "forget",
        "forget_everything", "vault_stats", "export_memories",
    }
    assert tools["recall"].annotations.read_only_hint is True
    assert tools["forget"].annotations.destructive_hint is True
    assert tools["remember"].output_schema["properties"]["duplicate"]["type"] == "boolean"
    # no tool lets the caller choose whose data to touch
    for t in tools.values():
        assert "user_id" not in t.input_schema.get("properties", {})


async def test_remember_recall_roundtrip(mcp):
    async with Client(mcp) as c:
        r = await c.call_tool(
            "remember",
            {"content": "Prefers concise answers with code", "kind": "preference", "tags": ["style"]},
        )
        assert r.structured_content["duplicate"] is False
        hits = await c.call_tool("recall", {"query": "concise answers"})
        assert hits.structured_content["result"][0]["content"].startswith("Prefers concise")
        profile = await c.read_resource("vault://profile")
        assert "Prefers concise answers" in profile[0].text
        mid = r.structured_content["memory"]["id"]
        one = await c.read_resource(f"vault://memories/{mid}")
        assert mid in one[0].text
        prompt = await c.get_prompt("use_my_memory", {"topic": "dinner ideas"})
        assert "recall" in prompt.messages[0].content.text


async def test_errors_are_actionable(mcp):
    async with Client(mcp) as c:
        with pytest.raises(ToolError, match="limit is 500"):
            await c.call_tool("remember", {"content": "x" * 501})
        with pytest.raises(ToolError):  # schema validation: importance out of range
            await c.call_tool("remember", {"content": "ok", "importance": 9})


async def _seed(c: Client, n: int = 3):
    for i in range(n):
        await c.call_tool("remember", {"content": f"memory {i}"})


async def test_forget_everything_confirm(mcp):
    async def yes(message, response_type, params, context):
        assert "delete all 3 memories" in message
        return ElicitResult(action="accept", content={"confirm": True})

    async with Client(mcp, elicitation_handler=yes) as c:
        await _seed(c)
        r = await c.call_tool("forget_everything", {})
        assert "Deleted 3" in r.content[0].text
        stats = await c.call_tool("vault_stats", {})
        assert stats.structured_content["total"] == 0


async def test_forget_everything_decline(mcp):
    async def no(message, response_type, params, context):
        return ElicitResult(action="decline")

    async with Client(mcp, elicitation_handler=no) as c:
        await _seed(c)
        r = await c.call_tool("forget_everything", {})
        assert "Cancelled" in r.content[0].text
        assert (await c.call_tool("vault_stats", {})).structured_content["total"] == 3


async def test_forget_everything_without_elicitation_support(mcp):
    async with Client(mcp) as c:  # no elicitation handler => capability not declared
        await _seed(c)
        r = await c.call_tool("forget_everything", {})
        assert "Nothing was deleted" in r.content[0].text
        with pytest.raises(ToolError, match="did not match"):
            await c.call_tool("forget_everything", {"confirm_phrase": "yes"})
        r = await c.call_tool("forget_everything", {"confirm_phrase": CONFIRM_PHRASE})
        assert "Deleted 3" in r.content[0].text
