"""Business rules, independent of MCP and of SQL.

Lesson: MCP tools should be thin adapters. Validation, normalization, quotas and
dedupe live here, so the same rules apply whether a call arrives via MCP, a REST
endpoint you add later, or a test.
"""

from __future__ import annotations

import hashlib
import re
from uuid import UUID

from .db import DuplicateContentError, MemoryRepository
from .embeddings import Embedder
from .models import (
    ForgetResult,
    Memory,
    MemoryHit,
    MemoryKind,
    MemoryPage,
    RememberResult,
    VaultStats,
)

_TAG_RE = re.compile(r"[^a-z0-9\-_]+")
MAX_TAGS = 10


class VaultError(ValueError):
    """An error whose message is safe and useful to show to the model and the user."""


def normalize_tags(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        t = _TAG_RE.sub("-", t.strip().lower().lstrip("#")).strip("-")[:40]
        if t and t not in out:
            out.append(t)
    if len(out) > MAX_TAGS:
        raise VaultError(f"At most {MAX_TAGS} tags per memory.")
    return out


def normalize_content(content: str) -> str:
    return " ".join(content.split())  # collapse whitespace


def content_hash(content: str) -> str:
    return hashlib.sha256(content.casefold().encode()).hexdigest()


class MemoryService:
    def __init__(
        self,
        repo: MemoryRepository,
        embedder: Embedder | None,
        *,
        max_memories: int,
        max_chars: int,
    ):
        self.repo = repo
        self.embedder = embedder
        self.max_memories = max_memories
        self.max_chars = max_chars

    # -------------------------------------------------------------- helpers
    def _clean(self, content: str) -> str:
        content = normalize_content(content)
        if not content:
            raise VaultError("Memory content cannot be empty.")
        if len(content) > self.max_chars:
            raise VaultError(
                f"Memory is {len(content)} characters; the limit is {self.max_chars}. "
                "Store a shorter summary, or split it into several memories."
            )
        return content

    async def _embed(self, text: str) -> list[float] | None:
        return await self.embedder.embed(text) if self.embedder else None

    # -------------------------------------------------------------- commands
    async def remember(
        self,
        user_id: str,
        content: str,
        kind: MemoryKind = "note",
        tags: list[str] | None = None,
        importance: int = 3,
    ) -> RememberResult:
        content = self._clean(content)
        tags = normalize_tags(tags)
        if not 1 <= importance <= 5:
            raise VaultError("importance must be between 1 and 5.")
        if await self.repo.count(user_id) >= self.max_memories:
            raise VaultError(
                f"Your vault is full ({self.max_memories} memories). "
                "Forget some old ones, or export and start fresh."
            )
        memory, duplicate = await self.repo.insert(
            user_id,
            content=content,
            kind=kind,
            tags=tags,
            importance=importance,
            content_hash=content_hash(content),
            embedding=await self._embed(content),
        )
        return RememberResult(memory=memory, duplicate=duplicate)

    async def update(
        self,
        user_id: str,
        memory_id: UUID,
        content: str | None = None,
        kind: MemoryKind | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
    ) -> Memory:
        new_content = self._clean(content) if content is not None else None
        if importance is not None and not 1 <= importance <= 5:
            raise VaultError("importance must be between 1 and 5.")
        try:
            memory = await self.repo.update(
                user_id,
                memory_id,
                content=new_content,
                kind=kind,
                tags=normalize_tags(tags) if tags is not None else None,
                importance=importance,
                content_hash=content_hash(new_content) if new_content else None,
                embedding=await self._embed(new_content) if new_content else None,
                update_embedding=new_content is not None and self.embedder is not None,
            )
        except DuplicateContentError as exc:
            raise VaultError("Another memory already has exactly this content.") from exc
        if memory is None:
            raise VaultError(f"No memory with id {memory_id}.")
        return memory

    async def forget(self, user_id: str, memory_id: UUID) -> ForgetResult:
        return ForgetResult(deleted=await self.repo.delete(user_id, memory_id), memory_id=memory_id)

    async def forget_everything(self, user_id: str) -> int:
        return await self.repo.delete_all(user_id)

    # -------------------------------------------------------------- queries
    async def recall(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        kind: MemoryKind | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryHit]:
        query = normalize_content(query)
        if not query:
            raise VaultError("Search query cannot be empty.")
        return await self.repo.search(
            user_id,
            query=query[:500],
            embedding=await self._embed(query),
            kind=kind,
            tags=normalize_tags(tags) or None,
            limit=max(1, min(limit, 25)),
        )

    async def get(self, user_id: str, memory_id: UUID) -> Memory:
        memory = await self.repo.get(user_id, memory_id)
        if memory is None:
            raise VaultError(f"No memory with id {memory_id}.")
        return memory

    async def list(
        self,
        user_id: str,
        kind: MemoryKind | None = None,
        tag: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        min_importance: int = 1,
    ) -> MemoryPage:
        tags = normalize_tags([tag]) if tag else []
        try:
            return await self.repo.list(
                user_id,
                kind=kind,
                tag=tags[0] if tags else None,
                limit=max(1, min(limit, 100)),
                cursor=cursor,
                min_importance=min_importance,
            )
        except ValueError as exc:
            raise VaultError(str(exc)) from exc

    async def stats(self, user_id: str) -> VaultStats:
        total, by_kind, top_tags = await self.repo.stats(user_id)
        return VaultStats(
            total=total,
            by_kind=by_kind,
            top_tags=top_tags,
            quota=self.max_memories,
            semantic_search=self.embedder is not None,
        )

    async def export(self, user_id: str) -> list[Memory]:
        return await self.repo.export(user_id)
