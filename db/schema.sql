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

-- ── persons ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persons (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID REFERENCES organisations(id) ON DELETE CASCADE,
    added_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    full_name   VARCHAR(255) NOT NULL,
    employee_id VARCHAR(100),
    department  VARCHAR(255),
    gcs_path    VARCHAR(500),
    photo_count INTEGER NOT NULL DEFAULT 0,
    is_enrolled BOOLEAN NOT NULL DEFAULT FALSE,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── sessions ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id       UUID REFERENCES organisations(id) ON DELETE SET NULL,
    n_faces      INTEGER NOT NULL DEFAULT 0,
    n_identified INTEGER NOT NULL DEFAULT 0,
    elapsed_s    FLOAT NOT NULL DEFAULT 0.0,
    results_json JSONB,
    note         TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── audit_logs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id    UUID REFERENCES users(id) ON DELETE SET NULL,
    org_id      UUID REFERENCES organisations(id) ON DELETE SET NULL,
    action      VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_id   VARCHAR(255),
    detail      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_org_role ON users(org_id, role);
CREATE INDEX IF NOT EXISTS idx_sessions_org   ON sessions(org_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_persons_org    ON persons(org_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor    ON audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_created  ON audit_logs(created_at DESC);