# Multi-stage build.
#
# ortools and shapely pull in a compiler toolchain and a large wheel set. Doing
# that in a builder stage and copying only the finished venv keeps the runtime
# image from carrying build-essential around forever.
#
# The image ships WITHOUT model weights and WITHOUT torch. The capability probe
# detects that and routes to CLOUD_API or MOCK. Baking weights in would add
# gigabytes to an image that, in every hosted deployment, calls an API anyway.

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# libgeos-c1v5 is shapely's runtime dependency; the -dev package and the
# compiler that needed it stay behind in the builder.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-c1v5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. The app writes nothing to disk -- images live in the
# database and renders are generated per request -- so it needs no writable
# volume, and giving it root would buy nothing.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app
# Migrations and the console are needed at runtime; the source itself is
# already installed into the venv.
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser alembic.ini ./alembic.ini
COPY --chown=appuser:appuser docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

USER appuser
EXPOSE 8000

# /health reports database reachability and whether the schema is applied, so
# a container that starts but cannot reach Postgres fails its check instead of
# silently accepting traffic it cannot serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" | grep -q '"status": *"ok"' || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]