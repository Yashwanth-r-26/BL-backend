-- Interior Design AI -- Postgres schema
--
-- Kept alongside the Alembic migration on purpose. Alembic owns deployment;
-- this file is the readable statement of intent, and it is what you hand
-- someone who asks "what does the data actually look like".
--
-- Two invariants the schema enforces rather than merely documents:
--
--   1. scene_versions is append-only. There is no UPDATE path in the
--      repository, parent_version_id is ON DELETE RESTRICT, and the
--      (scene_id, version) unique constraint makes a duplicated version
--      number a database error rather than a silent fork.
--
--   2. quote_lines copy their prices instead of referencing price_current.
--      A foreign key to a mutable price table would make historical quotes
--      change when vendors move their rates, which defeats the point of
--      quoting at all.

BEGIN;

-- ---------------------------------------------------------------- projects

CREATE TABLE IF NOT EXISTS projects (
    id            VARCHAR(64)  PRIMARY KEY,
    name          VARCHAR(255) NOT NULL,
    client_name   VARCHAR(255),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------- scene_versions

CREATE TABLE IF NOT EXISTS scene_versions (
    version_id         VARCHAR(64) PRIMARY KEY,
    scene_id           VARCHAR(64) NOT NULL,
    parent_version_id  VARCHAR(64)
        REFERENCES scene_versions (version_id) ON DELETE RESTRICT,
    version            INTEGER     NOT NULL,
    project_id         VARCHAR(64)
        REFERENCES projects (id) ON DELETE SET NULL,
    -- Full serialised scene graph. JSONB rather than PostGIS: the geometry
    -- work happens in Shapely and CP-SAT, and we never query by shape, so a
    -- spatial extension would be a dependency bought for nothing.
    payload            JSONB       NOT NULL,
    notes              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_scene_version   UNIQUE (scene_id, version),
    CONSTRAINT ck_scene_version_positive CHECK (version > 0)
);

CREATE INDEX IF NOT EXISTS ix_scene_versions_scene_id
    ON scene_versions (scene_id);
CREATE INDEX IF NOT EXISTS ix_scene_versions_scene_version
    ON scene_versions (scene_id, version);
CREATE INDEX IF NOT EXISTS ix_scene_versions_created_at
    ON scene_versions (created_at);

-- ----------------------------------------------------------- price_history

-- Append-only. A correction is a new row with a later observed_at; the wrong
-- number stays visible, because a quote that used it must remain explainable.
CREATE TABLE IF NOT EXISTS price_history (
    id           SERIAL       PRIMARY KEY,
    sku          VARCHAR(128) NOT NULL,
    vendor       VARCHAR(128) NOT NULL,
    unit         VARCHAR(16)  NOT NULL,
    amount       NUMERIC(14,2) NOT NULL,
    currency     VARCHAR(8)   NOT NULL DEFAULT 'INR',
    observed_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    source       VARCHAR(255),
    recorded_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_price_non_negative CHECK (amount >= 0)
);

CREATE INDEX IF NOT EXISTS ix_price_history_sku
    ON price_history (sku);
CREATE INDEX IF NOT EXISTS ix_price_history_observed_at
    ON price_history (observed_at);
CREATE INDEX IF NOT EXISTS ix_price_history_sku_observed
    ON price_history (sku, observed_at);

-- ----------------------------------------------------------- price_current

-- A projection of price_history: latest observation per sku. Derived data --
-- droppable and rebuildable at any time via PriceRepository.rebuild_projection.
-- It exists so the hot path does one indexed lookup instead of a window
-- function over the entire log.
CREATE TABLE IF NOT EXISTS price_current (
    sku          VARCHAR(128) PRIMARY KEY,
    vendor       VARCHAR(128) NOT NULL,
    unit         VARCHAR(16)  NOT NULL,
    amount       NUMERIC(14,2) NOT NULL,
    currency     VARCHAR(8)   NOT NULL DEFAULT 'INR',
    observed_at  TIMESTAMPTZ  NOT NULL,
    history_id   INTEGER      NOT NULL
        REFERENCES price_history (id) ON DELETE RESTRICT,
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_current_non_negative CHECK (amount >= 0)
);

-- ------------------------------------------------------------------ quotes

CREATE TABLE IF NOT EXISTS quotes (
    id                VARCHAR(64) PRIMARY KEY,
    scene_id          VARCHAR(64) NOT NULL,
    -- RESTRICT, not CASCADE: a scene version with a quote against it can
    -- never be deleted, or the quote becomes unexplainable.
    scene_version_id  VARCHAR(64) NOT NULL
        REFERENCES scene_versions (version_id) ON DELETE RESTRICT,
    currency          VARCHAR(8)  NOT NULL DEFAULT 'INR',
    total             NUMERIC(14,2) NOT NULL,
    stale_total       NUMERIC(14,2) NOT NULL DEFAULT 0,
    is_complete       INTEGER     NOT NULL DEFAULT 0,
    warnings          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    generated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_quotes_scene_id
    ON quotes (scene_id);
CREATE INDEX IF NOT EXISTS ix_quotes_generated_at
    ON quotes (generated_at);

-- ------------------------------------------------------------- quote_lines

CREATE TABLE IF NOT EXISTS quote_lines (
    id                 SERIAL       PRIMARY KEY,
    quote_id           VARCHAR(64)  NOT NULL
        REFERENCES quotes (id) ON DELETE CASCADE,
    sku                VARCHAR(128) NOT NULL,
    description        VARCHAR(255) NOT NULL,
    quantity           NUMERIC(14,3) NOT NULL,
    unit               VARCHAR(16)  NOT NULL,
    -- The arithmetic that produced the quantity, in words. Without it a
    -- disputed line has nothing to point at.
    basis              TEXT         NOT NULL,
    room_id            VARCHAR(64),

    -- Price frozen at quote time. Copied, never joined.
    price_status       VARCHAR(16)  NOT NULL,
    vendor             VARCHAR(128),
    unit_price         NUMERIC(14,2),
    line_total         NUMERIC(14,2),
    price_observed_at  TIMESTAMPTZ,
    price_age_days     INTEGER,

    CONSTRAINT ck_quantity_non_negative CHECK (quantity >= 0)
);

CREATE INDEX IF NOT EXISTS ix_quote_lines_quote
    ON quote_lines (quote_id);

COMMIT;
