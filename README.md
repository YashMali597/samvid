# Samvid

*Samvid* (संविद्, Sanskrit: "knowing together", shared knowledge and mutual understanding).

A personal, portable AI memory you own. It's a **remote MCP server**: connect it once to
Claude, ChatGPT, Cursor, Claude Code or any other MCP client, and they all share one
long-term memory of your preferences, people, projects and decisions. You can export or
delete everything at any time.

Built on free services only: Python + FastMCP 4 (MCP spec 2026-07-28), Postgres with
pgvector on Neon, GitHub OAuth for sign-in, and Render for hosting.

## What it can do

| Tool | What it does |
|---|---|
| `remember` | Save one fact, preference, person, project or task (deduplicated) |
| `recall` | Hybrid search: keyword + optional semantic, fused with Reciprocal Rank Fusion |
| `list_memories` | Browse newest-first, with keyset pagination |
| `update_memory` / `forget` | Correct or delete one memory |
| `forget_everything` | Wipe the vault after the user confirms in a dialog (multi round-trip request) |
| `vault_stats` / `export_memories` | Counts and quota, plus a full JSON export |

It also has two resources, `vault://profile` and `vault://memories/{id}`, and two prompts,
`use_my_memory` and `memory_checkup`.

## Architecture

```
AI app ──HTTPS /mcp (Streamable HTTP, stateless)──▶ server.py   tools, resources, prompts
         OAuth 2.1 + PKCE ◀──▶ auth.py (OAuth proxy) ──▶ GitHub (identity only)
                                    │
                               service.py   rules: validation, dedupe, quotas
                                    │
                                 db.py      SQL only; every query runs in a per-user
                                    │       transaction with Row-Level Security
                          Postgres + pgvector (Neon)
```

## Run locally

```bash
docker compose up -d                       # Postgres + pgvector (or use a free Neon branch)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn memory_vault.app:app --reload      # http://localhost:8000/mcp
fastmcp list http://localhost:8000/mcp     # see the tools
```

## Test

Create the non-superuser test role shown in `docker-compose.yml` first. Superusers
bypass Row-Level Security, so tests must not run as one.

```bash
export TEST_DATABASE_URL=postgresql://vault_app:app@localhost:5432/vault_test
pytest -q
```

## Deploy for free

The course (Step 13) walks through each step: create a Neon project, register a GitHub
OAuth App with callback `https://YOUR-APP.onrender.com/auth/callback`, deploy
`render.yaml` as a Render Blueprint, then add `https://YOUR-APP.onrender.com/mcp` as a
custom connector in Claude.
