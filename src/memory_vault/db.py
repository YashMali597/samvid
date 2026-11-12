"""Postgres access: connection pool, migrations, and the memory repository.

Lessons in this file:
  * One pool per process, created in the server lifespan (not per request).
  * Migrations are plain SQL files applied once, under an advisory lock, so two
    instances starting at the same time don't race each other.
  * Every user-scoped query runs inside ``user_tx()``, which sets ``app.user_id`` for the
    transaction. Row-Level Security policies use it, so tenant isolation is enforced by
    the database itself, not just by our WHERE clauses.
  * Only parameterized SQL ($1, $2 ...). User input is never formatted into SQL text.
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import resources
from typing import Any, AsyncIterator, Sequence
from uuid import UUID

import asyncpg

from .models import Memory, MemoryHit, MemoryPage

log = logging.getLogger(__name__)

_MIGRATION_LOCK_ID = 7_355_608  # any constant bigint, shared by all instances
_COLUMNS = "id, content, kind, tags, importance, created_at, updated_at"


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=15,
        # Serverless Postgres poolers (PgBouncer in transaction mode) can break
        # server-side prepared statement caching; disabling it is the safe default.
        statement_cache_size=0,
    )


async def run_migrations(pool: asyncpg.Pool) -> list[str]:
    """Apply any migrations/*.sql files that have not run yet. Returns names applied."""
    files = sorted(
        (f for f in resources.files("memory_vault.migrations").iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )
    applied_now: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_ID)
        try:
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                       name text PRIMARY KEY,
                       applied_at timestamptz NOT NULL DEFAULT now())"""
            )
            done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
            for f in files:
                if f.name in done:
                    continue
                log.info("applying migration %s", f.name)
                async with conn.transaction():
                    await conn.execute(f.read_text())
                    await conn.execute("INSERT INTO schema_migrations(name) VALUES ($1)", f.name)
                applied_now.append(f.name)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID)
    return applied_now


class DuplicateContentError(Exception):
    """Raised when an update would make two memories of one user identical."""


def _vec(embedding: Sequence[float] | None) -> str | None:
    """pgvector accepts the text form '[0.1,0.2,...]'; we cast with ::vector in SQL."""
    if embedding is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def _row_to_memory(row: asyncpg.Record) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        kind=row["kind"],
        tags=list(row["tags"]),
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def encode_cursor(created_at: datetime, memory_id: UUID) -> str:
    raw = f"{created_at.isoformat()}|{memory_id}".encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        created, mid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        return datetime.fromisoformat(created), UUID(mid)
    except Exception as exc:  # noqa: BLE001 - any malformed cursor is a client error
        raise ValueError("Invalid cursor") from exc


class _Params:
    """Collects query parameters and hands out $n placeholders in order."""

    def __init__(self) -> None:
        self.values: list[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"${len(self.values)}"

    @property
    def last(self) -> str:
        return f"${len(self.values)}"


class MemoryRepository:
    """All SQL lives here. The service layer above never sees a connection."""

    def __init__(self, pool: asyncpg.Pool, *, min_similarity: float = 0.55):
        self.pool = pool
        # Vector search always returns *something* (the nearest neighbours), even for an
        # unrelated query. A similarity floor turns "nearest" into "actually relevant".
        # 0.55 suits bge-small; calibrate per model on your own data.
        self.max_distance = 1.0 - min_similarity

    @asynccontextmanager
    async def user_tx(self, user_id: str) -> AsyncIterator[asyncpg.Connection]:
        """A transaction in which RLS only exposes ``user_id``'s rows."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # set_config(..., is_local => true) == SET LOCAL: reset at transaction end,
                # so a pooled connection never leaks one user's identity to the next request.
                await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
                yield conn

    # ------------------------------------------------------------------ users
    async def touch_user(self, user_id: str, display_name: str | None) -> None:
        async with self.user_tx(user_id) as conn:
            await conn.execute(
                """INSERT INTO users (id, display_name) VALUES ($1, $2)
                   ON CONFLICT (id) DO UPDATE
                   SET last_seen_at = now(),
                       display_name = COALESCE(EXCLUDED.display_name, users.display_name)""",
                user_id,
                display_name,
            )

    # ------------------------------------------------------------------ writes
    async def count(self, user_id: str) -> int:
        async with self.user_tx(user_id) as conn:
            return await conn.fetchval("SELECT count(*) FROM memories WHERE user_id = $1", user_id)

    async def insert(
        self,
        user_id: str,
        *,
        content: str,
        kind: str,
        tags: list[str],
        importance: int,
        content_hash: str,
        embedding: Sequence[float] | None,
    ) -> tuple[Memory, bool]:
        """Insert, or return the existing row if this exact content is already stored.

        Returns (memory, duplicate).
        """
        async with self.user_tx(user_id) as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO memories (user_id, content, kind, tags, importance, content_hash, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                    ON CONFLICT (user_id, content_hash) DO NOTHING
                    RETURNING {_COLUMNS}""",
                user_id, content, kind, tags, importance, content_hash, _vec(embedding),
            )
            if row is not None:
                return _row_to_memory(row), False
            row = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM memories WHERE user_id = $1 AND content_hash = $2",
                user_id, content_hash,
            )
            return _row_to_memory(row), True

    async def update(
        self,
        user_id: str,
        memory_id: UUID,
        *,
        content: str | None,
        kind: str | None,
        tags: list[str] | None,
        importance: int | None,
        content_hash: str | None,
        embedding: Sequence[float] | None,
        update_embedding: bool,
    ) -> Memory | None:
        sql = f"""UPDATE memories SET
                      content      = COALESCE($3, content),
                      kind         = COALESCE($4, kind),
                      tags         = COALESCE($5, tags),
                      importance   = COALESCE($6, importance),
                      content_hash = COALESCE($7, content_hash),
                      embedding    = CASE WHEN $9 THEN $8::vector ELSE embedding END,
                      updated_at   = now()
                  WHERE user_id = $1 AND id = $2
                  RETURNING {_COLUMNS}"""
        args = (user_id, memory_id, content, kind, tags, importance, content_hash,
                _vec(embedding), update_embedding)
        async with self.user_tx(user_id) as conn:
            try:
                row = await conn.fetchrow(sql, *args)
            except asyncpg.UniqueViolationError as exc:
                raise DuplicateContentError from exc
        return _row_to_memory(row) if row else None

    async def delete(self, user_id: str, memory_id: UUID) -> bool:
        async with self.user_tx(user_id) as conn:
            status = await conn.execute(
                "DELETE FROM memories WHERE user_id = $1 AND id = $2", user_id, memory_id
            )
            return status.endswith(" 1")

    async def delete_all(self, user_id: str) -> int:
        async with self.user_tx(user_id) as conn:
            status = await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
            return int(status.split()[-1])

    # ------------------------------------------------------------------ reads
    async def get(self, user_id: str, memory_id: UUID) -> Memory | None:
        async with self.user_tx(user_id) as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM memories WHERE user_id = $1 AND id = $2", user_id, memory_id
            )
            return _row_to_memory(row) if row else None

    async def list(
        self,
        user_id: str,
        *,
        kind: str | None,
        tag: str | None,
        limit: int,
        cursor: str | None,
        min_importance: int = 1,
    ) -> MemoryPage:
        """Keyset pagination: stable and fast at any depth (unlike OFFSET)."""
        after_ts, after_id = decode_cursor(cursor) if cursor else (None, None)
        async with self.user_tx(user_id) as conn:
            rows = await conn.fetch(
                f"""SELECT {_COLUMNS} FROM memories
                    WHERE user_id = $1
                      AND ($2::text IS NULL OR kind = $2)
                      AND ($3::text IS NULL OR tags @> ARRAY[$3::text])
                      AND importance >= $4
                      AND ($5::timestamptz IS NULL OR (created_at, id) < ($5, $6::uuid))
                    ORDER BY created_at DESC, id DESC
                    LIMIT $7""",
                user_id, kind, tag, min_importance, after_ts, after_id, limit + 1,
            )
        items = [_row_to_memory(r) for r in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            last = items[-1]
            next_cursor = encode_cursor(last.created_at, last.id)
        return MemoryPage(items=items, next_cursor=next_cursor)

    async def search(
        self,
        user_id: str,
        *,
        query: str,
        embedding: Sequence[float] | None,
        kind: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[MemoryHit]:
        """Hybrid search: keyword (full-text) + semantic (vector), fused with RRF.

        Reciprocal Rank Fusion scores each result as sum(1 / (60 + rank)) across the
        ranked lists it appears in. It needs no score normalization, which is why it
        is the usual default for combining keyword and vector rankings.
        """
        p = _Params()
        uid, k = p.add(user_id), p.add(max(limit * 4, 20))  # k = candidates per list
        where = (
            f"m.user_id = {uid}"
            f" AND ({p.add(kind)}::text IS NULL OR m.kind = {p.last})"
            f" AND ({p.add(tags or None)}::text[] IS NULL OR m.tags @> {p.last}::text[])"
        )
        tsq = p.add(query)
        like = p.add(query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))

        ranked_lists = [
            # 1) keyword search with Postgres full-text (stemming, stop words, ranking).
            #    plainto_tsquery ANDs the words ('peanut' & 'allergi'); for recall we want
            #    ANY word to match and let ts_rank_cd put multi-word matches first, so we
            #    turn & into |. The cast to tsquery does not re-parse user text.
            f"""SELECT m.id, row_number() OVER (ORDER BY ts_rank_cd(m.search_tsv, t.q) DESC) AS r
                FROM memories m,
                     (SELECT replace(plainto_tsquery('english', {tsq})::text, ' & ', ' | ')::tsquery AS q) t
                WHERE {where} AND m.search_tsv @@ t.q
                ORDER BY ts_rank_cd(m.search_tsv, t.q) DESC LIMIT {k}""",
            # 2) substring fallback: catches names and codes the stemmer mangles.
            #    Offset ranks by 10 so it never outranks a genuine full-text hit.
            f"""SELECT m.id, 10 + row_number() OVER (ORDER BY m.importance DESC, m.created_at DESC) AS r
                FROM memories m
                WHERE {where} AND m.content ILIKE '%' || {like} || '%'
                LIMIT {k}""",
        ]
        if embedding is not None:
            vec = p.add(_vec(embedding))
            # 3) semantic search: cosine distance on the HNSW index
            ranked_lists.append(
                f"""SELECT m.id, row_number() OVER (ORDER BY m.embedding <=> {vec}::vector) AS r
                    FROM memories m
                    WHERE {where} AND m.embedding IS NOT NULL
                      AND m.embedding <=> {vec}::vector <= {p.add(self.max_distance)}
                    ORDER BY m.embedding <=> {vec}::vector LIMIT {k}"""
            )
        union = " UNION ALL ".join(f"(SELECT id, r FROM ({q}) s{i})" for i, q in enumerate(ranked_lists))
        cols = ", ".join("m." + c.strip() for c in _COLUMNS.split(","))
        sql = f"""
            WITH fused AS (
                SELECT id, sum(1.0 / (60 + r)) AS score
                FROM ({union}) ranked
                GROUP BY id
            )
            SELECT {cols},
                   -- small boost so ties favour what the user marked important
                   f.score * (1 + 0.05 * m.importance) AS score
            FROM fused f JOIN memories m USING (id)
            ORDER BY score DESC
            LIMIT {p.add(limit)}
        """
        async with self.user_tx(user_id) as conn:
            rows = await conn.fetch(sql, *p.values)
        return [
            MemoryHit(**_row_to_memory(r).model_dump(), score=round(float(r["score"]), 6)) for r in rows
        ]

    async def stats(self, user_id: str) -> tuple[int, dict[str, int], list[tuple[str, int]]]:
        async with self.user_tx(user_id) as conn:
            kinds = await conn.fetch(
                "SELECT kind, count(*) AS n FROM memories WHERE user_id = $1 GROUP BY kind", user_id
            )
            tags = await conn.fetch(
                """SELECT t AS tag, count(*) AS n FROM memories, unnest(tags) AS t
                   WHERE user_id = $1 GROUP BY t ORDER BY n DESC, t LIMIT 15""",
                user_id,
            )
        by_kind = {r["kind"]: r["n"] for r in kinds}
        return sum(by_kind.values()), by_kind, [(r["tag"], r["n"]) for r in tags]

    async def export(self, user_id: str) -> list[Memory]:
        async with self.user_tx(user_id) as conn:
            rows = await conn.fetch(
                f"SELECT {_COLUMNS} FROM memories WHERE user_id = $1 ORDER BY created_at", user_id
            )
        return [_row_to_memory(r) for r in rows]
