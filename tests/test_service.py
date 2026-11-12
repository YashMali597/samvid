"""Business rules + SQL, tested together against real Postgres."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from memory_vault.service import MemoryService, VaultError, normalize_tags

from .conftest import make_user


async def test_remember_and_dedupe(service: MemoryService):
    u = await make_user(service, "u1")
    first = await service.remember(u, "Prefers   dark mode", "preference", ["UI", "#ui", "Dark Mode"])
    assert not first.duplicate
    assert first.memory.content == "Prefers dark mode"  # whitespace collapsed
    assert first.memory.tags == ["ui", "dark-mode"]  # normalized + deduped
    again = await service.remember(u, "prefers dark MODE", "preference")
    assert again.duplicate and again.memory.id == first.memory.id  # case-insensitive dedupe


async def test_validation(service: MemoryService):
    u = await make_user(service, "u1")
    with pytest.raises(VaultError, match="empty"):
        await service.remember(u, "   ")
    with pytest.raises(VaultError, match="limit is 500"):
        await service.remember(u, "x" * 501)
    with pytest.raises(VaultError, match="At most 10 tags"):
        normalize_tags([f"t{i}" for i in range(11)])


async def test_hybrid_recall_ranks_relevant_first(service: MemoryService):
    u = await make_user(service, "u1")
    await service.remember(u, "Is vegetarian and allergic to peanuts", "fact", ["food", "health"], 5)
    await service.remember(u, "Manager at work is Priya Raman", "person", ["work"])
    await service.remember(u, "Favourite programming language is Python", "preference", ["code"])
    hits = await service.recall(u, "any peanut allergies?")  # stemming + OR semantics
    assert hits and "peanuts" in hits[0].content
    hits = await service.recall(u, "Priya")
    assert hits[0].kind == "person"
    # filters
    assert await service.recall(u, "Python", kind="person") == []
    assert len(await service.recall(u, "Python", tags=["code"])) == 1


async def test_recall_handles_hostile_input(service: MemoryService):
    u = await make_user(service, "u1")
    await service.remember(u, "Uses 100% cotton shirts")
    # tsquery operators, LIKE wildcards and SQL fragments are all just data
    for q in ["100%", "'); DROP TABLE memories; --", "a & | ! (", "_"]:
        await service.recall(u, q)
    assert (await service.stats(u)).total == 1


async def test_update_and_forget(service: MemoryService):
    u = await make_user(service, "u1")
    m = (await service.remember(u, "Lives in Dallas", "fact")).memory
    updated = await service.update(u, m.id, content="Lives in Austin", importance=4)
    assert updated.content == "Lives in Austin" and updated.importance == 4 and updated.kind == "fact"
    assert (await service.recall(u, "Austin"))[0].id == m.id
    assert (await service.forget(u, m.id)).deleted
    assert not (await service.forget(u, m.id)).deleted  # idempotent
    with pytest.raises(VaultError, match="No memory"):
        await service.update(u, uuid4(), content="x")


async def test_update_to_duplicate_content_is_rejected(service: MemoryService):
    u = await make_user(service, "u1")
    await service.remember(u, "Likes tea")
    m = (await service.remember(u, "Likes coffee")).memory
    with pytest.raises(VaultError, match="already has exactly this content"):
        await service.update(u, m.id, content="likes TEA")


async def test_keyset_pagination(service: MemoryService):
    u = await make_user(service, "u1")
    for i in range(7):
        await service.remember(u, f"note number {i}")
    seen, cursor = [], None
    while True:
        page = await service.list(u, limit=3, cursor=cursor)
        seen += [m.content for m in page.items]
        if not (cursor := page.next_cursor):
            break
    assert len(seen) == 7 and len(set(seen)) == 7
    with pytest.raises(VaultError, match="Invalid cursor"):
        await service.list(u, cursor="garbage")


async def test_quota(service: MemoryService):
    u = await make_user(service, "u1")
    service.max_memories = 3
    for i in range(3):
        await service.remember(u, f"m{i}")
    with pytest.raises(VaultError, match="vault is full"):
        await service.remember(u, "one too many")


async def test_users_are_isolated(service: MemoryService):
    alice = await make_user(service, "github:1")
    bob = await make_user(service, "github:2")
    secret = (await service.remember(alice, "Alice's bank is First National")).memory
    assert await service.recall(bob, "bank") == []
    assert (await service.list(bob)).items == []
    with pytest.raises(VaultError):
        await service.get(bob, secret.id)
    assert not (await service.forget(bob, secret.id)).deleted
    assert (await service.stats(alice)).total == 1


async def test_rls_blocks_cross_tenant_access_even_without_where(service: MemoryService):
    """Defense in depth: a buggy query with NO user filter still sees only one tenant."""
    alice = await make_user(service, "github:1")
    bob = await make_user(service, "github:2")
    await service.remember(alice, "alice secret")
    await service.remember(bob, "bob secret")
    async with service.repo.user_tx(bob) as conn:
        rows = await conn.fetch("SELECT content FROM memories")  # forgot WHERE user_id!
        assert [r["content"] for r in rows] == ["bob secret"]
        # and Bob cannot write rows that claim to be Alice's
        with pytest.raises(Exception, match="row-level security"):
            await conn.execute(
                "INSERT INTO memories (user_id, content, content_hash) VALUES ('github:1', 'x', 'h')"
            )
    # without app.user_id set at all, nothing is visible
    async with service.repo.pool.acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM memories") == 0


async def test_concurrent_writes(service: MemoryService):
    u = await make_user(service, "u1")
    await asyncio.gather(*(service.remember(u, f"parallel {i}") for i in range(10)))
    await asyncio.gather(*(service.remember(u, "same thing") for _ in range(5)))
    assert (await service.stats(u)).total == 11
