"""Typed configuration, loaded from environment variables (or a .env file).

Lesson: keep every deploy-time knob in ONE validated object. The server fails fast at
startup with a clear message instead of failing on the first request in production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database -------------------------------------------------------------------
    database_url: str = Field(
        description="Postgres URL, e.g. the 'direct' connection string from Neon.",
    )
    db_pool_min: int = 1
    db_pool_max: int = 5  # free Postgres tiers have small connection limits

    # --- Public identity of this server ----------------------------------------------
    public_base_url: str = Field(
        default="http://localhost:8000",
        description="The URL people paste into their AI app, minus the /mcp path.",
    )
    mcp_path: str = "/mcp"

    # --- Auth -------------------------------------------------------------------------
    # none   = single-user local development (every request is user "local")
    # github = real multi-user OAuth 2.1 via a GitHub OAuth App (free)
    auth_mode: Literal["none", "github"] = "none"
    github_client_id: str | None = None
    github_client_secret: str | None = None
    jwt_signing_key: str | None = Field(
        default=None, description="Secret used to sign the tokens this server issues."
    )
    storage_encryption_key: str | None = Field(
        default=None, description="Secret used to encrypt OAuth state stored in Postgres."
    )

    # --- Search -----------------------------------------------------------------------
    embeddings: Literal["none", "fastembed"] = "none"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ~70 MB, CPU-friendly
    semantic_min_similarity: float = Field(default=0.55, ge=0, le=1)

    # --- Abuse protection (protects your free-tier quotas) ----------------------------
    max_memories_per_user: int = 5000
    max_content_chars: int = 4000
    rate_limit_rps: float = 5.0

    @model_validator(mode="after")
    def _check_auth(self) -> "Settings":
        if self.auth_mode == "github":
            missing = [
                name
                for name in (
                    "github_client_id",
                    "github_client_secret",
                    "jwt_signing_key",
                    "storage_encryption_key",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "AUTH_MODE=github needs these environment variables: "
                    + ", ".join(m.upper() for m in missing)
                )
            if not self.public_base_url.startswith("https://"):
                raise ValueError("AUTH_MODE=github requires an https:// PUBLIC_BASE_URL")
        self.public_base_url = self.public_base_url.rstrip("/")
        return self

    @property
    def asyncpg_dsn(self) -> str:
        """Hosted Postgres URLs often carry libpq-only parameters that asyncpg rejects.

        Neon, for example, adds ``channel_binding=require``. We keep ``sslmode`` (asyncpg
        understands it) and drop the rest.
        """
        parts = urlsplit(self.database_url)
        allowed = {"sslmode", "application_name", "options"}
        query = [(k, v) for k, v in parse_qsl(parts.query) if k in allowed]
        scheme = "postgresql" if parts.scheme.startswith("postgres") else parts.scheme
        return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), ""))


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
