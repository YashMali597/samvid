-- 001_init.sql: core schema for Memory Vault.
--
-- Design notes (see Module 5 of the course):
--  * Every row belongs to exactly one user (multi-tenant, shared schema).
--  * Row-Level Security (RLS) is defense in depth: even if application code forgets a
--    WHERE user_id = ..., Postgres refuses to return or modify another user's rows.
--  * Full-text search (tsvector + GIN) works with zero extra cost; the pgvector column
--    adds semantic search when embeddings are enabled.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id            text PRIMARY KEY,               -- e.g. 'github:583231' or 'local'
    display_name  text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content       text NOT NULL CHECK (char_length(content) BETWEEN 1 AND 8000),
    kind          text NOT NULL DEFAULT 'note'
                  CHECK (kind IN ('fact', 'preference', 'note', 'task', 'person', 'project')),
    tags          text[] NOT NULL DEFAULT '{}',
    importance    smallint NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    content_hash  text NOT NULL,                  -- sha256 of normalized content, for dedupe
    search_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding     vector(384),                    -- NULL when embeddings are disabled
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, content_hash)
);

-- Listing newest-first per user, and keyset pagination on (created_at, id).
CREATE INDEX IF NOT EXISTS memories_user_created_idx ON memories (user_id, created_at DESC, id DESC);
-- Keyword search.
CREATE INDEX IF NOT EXISTS memories_tsv_idx ON memories USING gin (search_tsv);
-- Tag filters: tags @> ARRAY['work'].
CREATE INDEX IF NOT EXISTS memories_tags_idx ON memories USING gin (tags);
-- Approximate nearest-neighbour search on cosine distance.
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------- Row-Level Security
-- The app sets `app.user_id` at the start of every transaction (SET LOCAL semantics).
-- FORCE makes the policy apply to the table owner too. NOTE: superusers always bypass
-- RLS, so run the app as a normal role (Neon's default role is not a superuser).

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS memories_owner ON memories;
CREATE POLICY memories_owner ON memories
    USING (user_id = current_setting('app.user_id', true))
    WITH CHECK (user_id = current_setting('app.user_id', true));

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS users_self ON users;
CREATE POLICY users_self ON users
    USING (id = current_setting('app.user_id', true))
    WITH CHECK (id = current_setting('app.user_id', true));
