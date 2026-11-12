"""Authentication: who is calling, and how they prove it.

Architecture (see Module 6 of the course):

  AI app (MCP client)  --OAuth 2.1 + PKCE-->  this server's /authorize, /token
                                                  |  (OAuth proxy)
                                                  v
                                         GitHub OAuth App (identity)

* This server is the OAuth *resource server* (it validates tokens on every request)
  and, via FastMCP's OAuth proxy, also presents an *authorization server* to MCP
  clients. It supports Client ID Metadata Documents (the 2026-07-28 default) and
  Dynamic Client Registration (deprecated, kept for older clients).
* GitHub only does identity. We never see the user's password, and we request the
  smallest scope that gives us a stable user id: ``read:user``.
* OAuth state (client registrations, upstream tokens) is stored in Postgres,
  encrypted, so redeploys and free-tier restarts don't log everyone out.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastmcp.server.auth import AuthProvider
from fastmcp.server.dependencies import get_access_token

from .config import Settings

LOCAL_USER_ID = "local"


@dataclass(frozen=True)
class CurrentUser:
    id: str
    display_name: str | None


def build_auth(settings: Settings) -> AuthProvider | None:
    if settings.auth_mode == "none":
        return None

    from fastmcp.server.auth.providers.github import GitHubProvider
    from key_value.aio.stores.postgresql import PostgreSQLStore
    from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

    storage = FernetEncryptionWrapper(
        PostgreSQLStore(url=settings.asyncpg_dsn, table_name="oauth_kv"),
        source_material=settings.storage_encryption_key,
        salt="memory-vault/oauth-state",
    )
    return GitHubProvider(
        client_id=settings.github_client_id,  # type: ignore[arg-type]
        client_secret=settings.github_client_secret,  # type: ignore[arg-type]
        base_url=settings.public_base_url,
        required_scopes=["read:user"],
        jwt_signing_key=settings.jwt_signing_key,
        client_storage=storage,
        # Show a consent screen naming the AI app that asked for access, so a
        # malicious client can't silently ride on an earlier approval (confused deputy).
        require_authorization_consent=True,
        cache_ttl_seconds=300,  # cache GitHub token checks; protects GitHub's rate limit
    )


def current_user(issuer: str | None) -> CurrentUser:
    """Resolve the caller from the verified access token.

    ``issuer`` is None when the server runs without auth (local, single-user).
    Never trust a user id passed as a tool argument.
    """
    if issuer is None:
        return CurrentUser(id=LOCAL_USER_ID, display_name="Local user")

    token = get_access_token()
    if token is None:
        # The auth middleware rejects unauthenticated HTTP requests before tools run,
        # so reaching here means a misconfiguration. Fail closed.
        raise PermissionError("Not authenticated.")
    claims = token.claims or {}
    subject = claims.get("sub") or getattr(token, "subject", None)
    if not subject:
        raise PermissionError("Token has no subject.")
    # Namespacing by provider keeps ids unique if you add Google login later.
    return CurrentUser(
        id=f"{issuer}:{subject}", display_name=claims.get("login") or claims.get("name")
    )
