-- ============================================================
-- VisionX — Multi-Tenant Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── master_roles ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS master_roles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_name   VARCHAR(100) UNIQUE NOT NULL,
    description VARCHAR(255),

    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by  UUID,
    created_ip  VARCHAR(45),
    updated_at  TIMESTAMPTZ,
    updated_by  UUID,
    updated_ip  VARCHAR(45)
);

-- ── master_actions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS master_actions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_name VARCHAR(100) UNIQUE NOT NULL,
    description VARCHAR(255),

    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted  BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by  UUID,
    created_ip  VARCHAR(45),
    updated_at  TIMESTAMPTZ,
    updated_by  UUID,
    updated_ip  VARCHAR(45)
);

-- ── organisations ─────────────────────────────────────────────────────────────
-- Created WITHOUT user FKs first (added later via ALTER TABLE)
CREATE TABLE IF NOT EXISTS organisations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,

    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID,
    created_ip VARCHAR(45),
    updated_at TIMESTAMPTZ,
    updated_by UUID,
    updated_ip VARCHAR(45),
    deleted_at TIMESTAMPTZ,
    deleted_by UUID,
    deleted_ip VARCHAR(45)
);

-- ── users ─────────────────────────────────────────────────────────────────────
-- Created WITH org FK but WITHOUT self-referencing user FKs (added later)
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        UUID        REFERENCES organisations(id) ON DELETE SET NULL,
    role          VARCHAR(50) NOT NULL DEFAULT 'member',

    full_name     VARCHAR(255) NOT NULL DEFAULT '',
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    contact       VARCHAR(20),
    dob           DATE,

    invited_by    UUID,
    last_login_at TIMESTAMPTZ,

    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID,
    created_ip VARCHAR(45),
    updated_at TIMESTAMPTZ,
    updated_by UUID,
    updated_ip VARCHAR(45),
    deleted_at TIMESTAMPTZ,
    deleted_by UUID,
    deleted_ip VARCHAR(45)
);

-- ── Now add self-referencing FKs on users ────────────────────────────────────
ALTER TABLE users
    ADD CONSTRAINT fk_users_invited_by  FOREIGN KEY (invited_by)  REFERENCES users(id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_users_created_by  FOREIGN KEY (created_by)  REFERENCES users(id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_users_updated_by  FOREIGN KEY (updated_by)  REFERENCES users(id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_users_deleted_by  FOREIGN KEY (deleted_by)  REFERENCES users(id) ON DELETE SET NULL;

-- ── Now add user FKs on organisations ────────────────────────────────────────
ALTER TABLE organisations
    ADD CONSTRAINT fk_org_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_org_updated_by FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_org_deleted_by FOREIGN KEY (deleted_by) REFERENCES users(id) ON DELETE SET NULL;

-- ── permissions ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS permissions (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id   UUID NOT NULL REFERENCES master_roles(id)   ON DELETE CASCADE,
    action_id UUID NOT NULL REFERENCES master_actions(id) ON DELETE CASCADE,
    UNIQUE (role_id, action_id),

    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID        REFERENCES users(id) ON DELETE SET NULL,
    created_ip VARCHAR(45)
);

-- ── password_reset_logs ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS password_reset_logs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reset_token  VARCHAR(500),
    token_expiry TIMESTAMPTZ,

    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_ip VARCHAR(45)
);

-- ── persons ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persons (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID REFERENCES organisations(id) ON DELETE CASCADE,
    full_name   VARCHAR(255) NOT NULL,
    employee_id VARCHAR(100),
    department  VARCHAR(255),
    gcs_path    VARCHAR(500),
    photo_count INTEGER     NOT NULL DEFAULT 0,
    is_enrolled BOOLEAN     NOT NULL DEFAULT FALSE,

    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    is_deleted BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by UUID        REFERENCES users(id) ON DELETE SET NULL,
    created_ip VARCHAR(45),
    updated_at TIMESTAMPTZ,
    updated_by UUID        REFERENCES users(id) ON DELETE SET NULL,
    updated_ip VARCHAR(45),
    deleted_at TIMESTAMPTZ,
    deleted_by UUID        REFERENCES users(id) ON DELETE SET NULL,
    deleted_ip VARCHAR(45)
);

-- ── sessions ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id)         ON DELETE CASCADE,
    org_id          UUID          REFERENCES organisations(id)  ON DELETE SET NULL,
    n_faces         INTEGER NOT NULL DEFAULT 0,
    n_identified    INTEGER NOT NULL DEFAULT 0,
    elapsed_s       FLOAT   NOT NULL DEFAULT 0.0,
    results_json    JSONB,
    annotated_image TEXT,
    note            TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_ip VARCHAR(45)
);

-- ── audit_logs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_logs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id    UUID REFERENCES users(id)         ON DELETE SET NULL,
    org_id      UUID REFERENCES organisations(id) ON DELETE SET NULL,
    action      VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_id   VARCHAR(255),
    detail      JSONB,
    ip_address  VARCHAR(45),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- SEED DATA
-- ============================================================

INSERT INTO master_roles (role_name, description) VALUES
    ('super_admin', 'Full access to all organisations and system settings'),
    ('org_admin',   'Manages users and data within their organisation'),
    ('member',      'Standard user with read and scan access')
ON CONFLICT (role_name) DO NOTHING;

INSERT INTO master_actions (action_name, description) VALUES
    ('READ',         'Can view and read data'),
    ('WRITE',        'Can create and update data'),
    ('DELETE',       'Can delete data'),
    ('INVITE_USER',  'Can invite new users'),
    ('MANAGE_ORG',   'Can manage organisation settings'),
    ('SCAN',         'Can run face recognition scans'),
    ('TRAIN_MODEL',  'Can trigger model retraining'),
    ('VIEW_REPORTS', 'Can view session reports and analytics')
ON CONFLICT (action_name) DO NOTHING;

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_users_email      ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_org_role   ON users(org_id, role);
CREATE INDEX IF NOT EXISTS idx_sessions_org     ON sessions(org_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user    ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_persons_org      ON persons(org_id);
CREATE INDEX IF NOT EXISTS idx_audit_actor      ON audit_logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_created    ON audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pwreset_user     ON password_reset_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_permissions_role ON permissions(role_id);