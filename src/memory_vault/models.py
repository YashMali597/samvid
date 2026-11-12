"""Domain models. These double as the tools' output schemas (MCP structured content).

Lesson: a typed return value gives the client an ``outputSchema`` and machine-readable
``structuredContent`` alongside the human-readable text. Models (and code) can rely on
the shape instead of parsing prose.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

MemoryKind = Literal["fact", "preference", "note", "task", "person", "project"]


class Memory(BaseModel):
    id: UUID
    content: str
    kind: MemoryKind
    tags: list[str]
    importance: int = Field(ge=1, le=5)
    created_at: datetime
    updated_at: datetime


class MemoryHit(Memory):
    score: float = Field(description="Relevance score; higher is more relevant.")


class RememberResult(BaseModel):
    memory: Memory
    duplicate: bool = Field(description="True if this exact memory already existed.")


class MemoryPage(BaseModel):
    items: list[Memory]
    next_cursor: str | None = Field(
        default=None, description="Pass back as `cursor` to get the next page."
    )


class ForgetResult(BaseModel):
    deleted: bool
    memory_id: UUID


class VaultStats(BaseModel):
    total: int
    by_kind: dict[str, int]
    top_tags: list[tuple[str, int]]
    quota: int
    semantic_search: bool
