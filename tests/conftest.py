"""Test fixtures. Tests run against a REAL Postgres with pgvector, as a non-superuser
role, so Row-Level Security is genuinely enforced (superusers bypass RLS).

  export TEST_DATABASE_URL=postgresql://vault_app@127.0.0.1:5433/vault_test

A free Neon branch works too: create a branch called `test` and use its URL.
"""

from __future__ import annotations

import os

import asyncpg
import pytest

from memory_vault.config import Settings
from memory_vault.db import MemoryRepository, create_pool, run_migrations
from memory_vault.embeddings import HashingEmbedder
from memory_vault.service import MemoryService

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql://vault_app@127.0.0.1:5433/vault_test")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(database_url=TEST_DB, max_memories_per_user=50, max_content_chars=500)


@pytest.fixture(scope="session")
async def pool(settings: Settings):
    pool = await create_pool(settings.asyncpg_dsn, 1, 5)
    await run_migrations(pool)
    yield pool
    await pool.close()


@pytest.fixture(autouse=True)
async def clean(pool: asyncpg.Pool):
    """Each test starts with an empty vault. TRUNCATE is not subject to RLS."""
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE memories, users CASCADE")
    yield


@pytest.fixture
def service(pool: asyncpg.Pool) -> MemoryService:
    return MemoryService(
        MemoryRepository(pool, min_similarity=0.2), HashingEmbedder(), max_memories=50, max_chars=500
    )


async def make_user(service: MemoryService, user_id: str) -> str:
    await service.repo.touch_user(user_id, user_id)
    return user_id
