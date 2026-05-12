-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── organisations ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS organisations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    created_by UUID,                          -- filled in after first super_admin exists
    is_active  BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── users ─────────────────────────────────────────────────────────────────────
CREATE TYPE user_role AS ENUM ('super_admin', 'org_admin', 'user');

CREATE TABLE IF NOT EXISTS users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email          VARCHAR(255) UNIQUE NOT NULL,
    password_hash  VARCHAR(255) NOT NULL,
    role           user_role NOT NULL DEFAULT 'user',
    org_id         UUID REFERENCES organisations(id) ON DELETE SET NULL,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE,
    invited_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at  TIMESTAMPTZ
);

-- ── indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_users_email   ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_org_role ON users(org_id, role);