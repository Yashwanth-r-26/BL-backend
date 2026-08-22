#!/bin/sh
# Apply migrations, then serve.
#
# Running `alembic upgrade head` on boot means a fresh database is usable
# without a manual step, and a deploy that adds a migration applies it before
# the new code that needs it starts taking requests. It is a no-op when the
# schema is already current.
#
# Set RUN_MIGRATIONS=0 to skip it -- appropriate when several replicas start
# together and you would rather migrate once, deliberately, from one place.
set -e

if [ "${RUN_MIGRATIONS:-1}" = "1" ] && [ -n "${DATABASE_URL}" ]; then
    echo "[entrypoint] applying migrations"
    alembic upgrade head
elif [ -z "${DATABASE_URL}" ]; then
    echo "[entrypoint] WARNING: DATABASE_URL is not set."
    echo "[entrypoint] The service will run on in-memory storage and LOSE all"
    echo "[entrypoint] products, prices and sessions when this container stops."
fi

exec uvicorn interior_ai.api.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-2}" \
    --proxy-headers \
    --forwarded-allow-ips '*'