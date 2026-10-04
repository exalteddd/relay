"""Per-user project storage.

A project is stored as one opaque JSON document. The graph the interface draws
is deeply nested and changes shape as the UI grows, so normalising it into
tables would buy nothing and cost a migration every time a node gains a field.
This layer therefore cares about exactly two things: who owns a document, and
when it last changed.

Two engines, one API:

    DATABASE_URL set  -> PostgreSQL (Render supplies this)
    DATABASE_URL unset -> SQLite file (local development)

Both are driven through the same `?`-placeholder SQL; `_q` rewrites it for
Postgres. Keeping the document in a TEXT column rather than `jsonb` is what
lets the two engines share one schema -- the server never queries inside a
document, so there is nothing to gain from a native JSON type.

Connections are opened per call rather than pooled. With gunicorn's 8 threads
that caps us at 8 concurrent connections, which sits well inside the free
tier's limit, and it means a dropped Postgres connection heals on the next
request instead of poisoning a long-lived pool.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A project document is a graph, not a file upload. 1 MB is far above any real
# graph and far below anything that could fill the free tier's 1 GB.
MAX_DOC_BYTES = 1_000_000

# How many projects one account may hold. Generous for real use, low enough
# that a scripted client cannot quietly consume the shared database.
MAX_PROJECTS_PER_USER = 50

_DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_POSTGRES = bool(_DATABASE_URL)

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
           login      TEXT PRIMARY KEY,
           created_at TEXT NOT NULL
       )""",
    """CREATE TABLE IF NOT EXISTS projects (
           id         TEXT PRIMARY KEY,
           owner      TEXT NOT NULL,
           doc        TEXT NOT NULL,
           created_at TEXT NOT NULL,
           updated_at TEXT NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS projects_owner_idx ON projects (owner)",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _q(sql: str) -> str:
    """Rewrite `?` placeholders to `%s` for psycopg; SQLite takes them as-is."""
    return sql.replace("?", "%s") if IS_POSTGRES else sql


@contextmanager
def _conn():
    """Yield a connection that commits on success and rolls back on error."""
    if IS_POSTGRES:
        import psycopg

        # Render hands out the historical `postgres://` scheme; psycopg 3 only
        # recognises `postgresql://`.
        dsn = _DATABASE_URL
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://"):]
        conn = psycopg.connect(dsn, connect_timeout=10)
    else:
        path = os.environ.get("RELAY_DB") or str(ROOT / "relay.db")
        conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    """Create the schema. Safe to call on every boot."""
    with _conn() as conn:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)


def engine_name() -> str:
    return "postgres" if IS_POSTGRES else "sqlite"


# --- users --------------------------------------------------------------
def ensure_user(login: str) -> bool:
    """Record a signed-in user. True if this is their first time here.

    The caller uses that flag to decide whether to seed a sample project, so
    the insert and the check have to be one statement -- two parallel sign-ins
    from the same account must not both believe they are first.
    """
    with _conn() as conn:
        cur = conn.cursor()
        if IS_POSTGRES:
            cur.execute(
                "INSERT INTO users (login, created_at) VALUES (%s, %s) "
                "ON CONFLICT (login) DO NOTHING RETURNING login",
                (login, _now()),
            )
            return cur.fetchone() is not None
        cur.execute(
            "INSERT OR IGNORE INTO users (login, created_at) VALUES (?, ?)",
            (login, _now()),
        )
        return cur.rowcount > 0


# --- projects -----------------------------------------------------------
def _summary(row) -> dict:
    """List entries carry only what the project switcher draws."""
    pid, doc, created_at, updated_at = row
    try:
        parsed = json.loads(doc)
    except (TypeError, ValueError):
        parsed = {}
    return {
        "id": pid,
        "name": parsed.get("name") or "Untitled project",
        "icon": parsed.get("icon") or "flask",
        "run": parsed.get("run") or "",
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def list_projects(owner: str) -> list[dict]:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            _q("SELECT id, doc, created_at, updated_at FROM projects "
               "WHERE owner = ? ORDER BY created_at"),
            (owner,),
        )
        return [_summary(r) for r in cur.fetchall()]


def count_projects(owner: str) -> int:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(_q("SELECT COUNT(*) FROM projects WHERE owner = ?"), (owner,))
        return int(cur.fetchone()[0])


def get_project(pid: str, owner: str) -> dict | None:
    """Fetch one document. Scoped by owner, so a guessed id reveals nothing."""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            _q("SELECT id, doc, created_at, updated_at FROM projects "
               "WHERE id = ? AND owner = ?"),
            (pid, owner),
        )
        row = cur.fetchone()
    if row is None:
        return None
    out = _summary(row)
    out["doc"] = json.loads(row[1])
    return out


def create_project(owner: str, doc: dict) -> dict:
    pid = uuid.uuid4().hex[:12]
    blob = json.dumps(doc, separators=(",", ":"))
    now = _now()
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            _q("INSERT INTO projects (id, owner, doc, created_at, updated_at) "
               "VALUES (?, ?, ?, ?, ?)"),
            (pid, owner, blob, now, now),
        )
    out = _summary((pid, blob, now, now))
    out["doc"] = doc
    return out


def update_project(pid: str, owner: str, doc: dict) -> bool:
    """Overwrite a document. False when it does not exist or is not theirs."""
    blob = json.dumps(doc, separators=(",", ":"))
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            _q("UPDATE projects SET doc = ?, updated_at = ? WHERE id = ? AND owner = ?"),
            (blob, _now(), pid, owner),
        )
        return cur.rowcount > 0


def delete_project(pid: str, owner: str) -> bool:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(_q("DELETE FROM projects WHERE id = ? AND owner = ?"), (pid, owner))
        return cur.rowcount > 0
