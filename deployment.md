# Deploying to EasyPanel

The service is one container plus a Postgres database. It writes nothing to
disk — product images live in the database, renders are generated per request —
so there are no volumes to manage and it scales horizontally without
coordination.

---

## 1. Create the Postgres service

In your EasyPanel project: **+ Service → Postgres**.

| Field | Value |
|---|---|
| Service name | `interior-db` |
| Postgres version | 16 |
| Database | `interior_ai` |
| User | `interior` |
| Password | generate one |

EasyPanel puts services on a shared internal network, so the app reaches the
database by service name and the port never has to be exposed publicly. Leave
it internal — a database open to the internet is a liability with no upside
here.

The internal URL will be:

```
postgresql://interior:PASSWORD@interior-db:5432/interior_ai
```

> **Using Neon instead?** Skip this step and use the Neon connection string.
> The app adds `sslmode=require` and enables connection pre-ping automatically,
> which matters because Neon suspends idle computes and a pooled connection
> that worked a minute ago may be dead.

---

## 2. Create the app service

**+ Service → App**, then:

**Source** — point at your Git repository and branch. If the repo root is the
parent of `interior_ai/`, set **Build path** to `interior_ai` so the Dockerfile
is found.

**Build** — method **Dockerfile**, file `Dockerfile`. Nothing else to set; the
image builds itself.

**Environment** — paste this and fill in the two real values:

```env
DATABASE_URL=postgresql://interior:PASSWORD@interior-db:5432/interior_ai
GEMINI_API_KEY=your-key

GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_DETECT_MODEL=gemini-3.5-flash
GEMINI_IMAGE_MODEL=gemini-3.1-flash-image
GEMINI_EDIT_TIMEOUT_S=150

WEB_CONCURRENCY=2
```

**Domain** — add one and enable HTTPS. **This is not optional if the mobile app
uses device location**: browsers and both mobile platforms refuse geolocation
on an insecure origin. EasyPanel issues a Let's Encrypt certificate
automatically.

**Port** — `8000`.

Deploy.

---

## 3. Check it came up correctly

```bash
curl https://your-domain/health
```

```json
{
  "status": "ok",
  "persistent": true,
  "storage": "postgresql",
  "database": "reachable",
  "schema": "ready"
}
```

Read all five fields, not just `status`:

| Symptom | Cause | Fix |
|---|---|---|
| `"persistent": false` | `DATABASE_URL` not reaching the process | Check the env var name and redeploy |
| `"database": "unreachable"` | Wrong host, password, or the DB is not up | Confirm the service name matches the URL |
| `"schema": "missing tables…"` | Migrations did not run | See below |

Migrations run automatically on container start. If they were skipped, run once
from the service's terminal:

```bash
alembic upgrade head
```

---

## 4. Load the catalogue

From the service terminal in EasyPanel:

```bash
python -m interior_ai.db.build_catalogue --only sofa --limit 3
```

Check those three at `https://your-domain/admin` before committing to the full
run — each product is an image-generation call, so 136 items takes a while and
costs real money. When you are happy:

```bash
python -m interior_ai.db.build_catalogue
```

It is resumable. If it stops, run it again and it continues from where it got
to.

---

## 5. Scaling and operations

**Replicas.** The app holds no local state, so raising the replica count works
without further thought. Do set `RUN_MIGRATIONS=0` once you run more than one:
several containers racing to migrate the same database at once is asking for
trouble. Migrate deliberately from the terminal instead, then deploy.

**Workers.** `WEB_CONCURRENCY=2` suits a small instance. Requests spend most of
their time waiting on Gemini rather than using CPU, so more workers help more
than more cores.

**Memory.** Around 400 MB per worker — ortools and shapely are not small. Give
the service at least 1 GB.

**Timeouts.** An image edit can take 90 seconds. If EasyPanel's proxy cuts
requests shorter than that, raise its timeout to **180 seconds** or every swap
will fail at the gateway while the model is still working.

**Logs.** Every Gemini retry, fallback and failure is logged with its reason.
`503` in the logs means the model is busy, not that anything is wrong with the
request — the app already retries with backoff.

---

## 6. Locking it down

The API has **no authentication**. Anyone who can reach it can read your
catalogue, create sessions and spend your Gemini quota.

Before it is publicly reachable, do at least one of:

- Put it behind EasyPanel's basic-auth for internal use.
- Restrict it to your mobile app's traffic at the proxy.
- Add an API-key dependency in front of the router.

`/admin` deserves particular attention — it can add and deactivate products.

I would not treat this as a background task. An open endpoint that costs money
per call is the kind of thing that gets found.

---

## Local Docker

```bash
docker build -t interior-ai .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/interior_ai \
  -e GEMINI_API_KEY=your-key \
  interior-ai
```

Or with the bundled compose file, which brings its own Postgres:

```bash
docker compose up
```