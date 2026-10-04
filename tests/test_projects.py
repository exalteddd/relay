"""Per-user project storage: ownership, seeding, and the anonymous path.

Runs against SQLite (no DATABASE_URL), which exercises the same SQL the
Postgres path uses -- `store._q` only rewrites the placeholder style.

    pytest tests/test_projects.py -q
"""

from __future__ import annotations

import os

import pytest

flask = pytest.importorskip("flask")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("RELAY_DB", str(tmp_path / "relay.db"))

    # store reads DATABASE_URL at import time, so reload it under the patched env
    import importlib

    from server import store as _store

    store = importlib.reload(_store)
    import server.projects as _projects

    projects = importlib.reload(_projects)
    store.init()

    app = flask.Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(projects.bp)

    @app.post("/_signin/<who>")
    def _signin(who):
        flask.session["user"] = who
        return {"ok": True}

    app.store = store
    return app


DOC = {"name": "NDM-1", "icon": "flask", "nodes": [{"id": "n1", "x": 1}], "edges": []}


def test_anonymous_is_read_only_and_cannot_write(client):
    with client.test_client() as c:
        body = c.get("/api/projects").get_json()
        assert body["user"] is None
        assert body["readOnly"] is True
        assert body["projects"] == []

        assert c.post("/api/projects", json={"doc": DOC}).status_code == 401
        assert c.put("/api/projects/x", json={"doc": DOC}).status_code == 401
        assert c.delete("/api/projects/x").status_code == 401


def test_is_new_account_fires_exactly_once(client):
    """The seed signal must come from the user insert, not from an empty list --
    otherwise deleting every project would make the sample reappear."""
    with client.test_client() as c:
        c.post("/_signin/alice")
        assert c.get("/api/projects").get_json()["isNewAccount"] is True
        assert c.get("/api/projects").get_json()["isNewAccount"] is False

        pid = c.post("/api/projects", json={"doc": DOC}).get_json()["id"]
        assert c.delete(f"/api/projects/{pid}").status_code == 200
        assert c.get("/api/projects").get_json()["isNewAccount"] is False


def test_crud_round_trip(client):
    with client.test_client() as c:
        c.post("/_signin/alice")
        created = c.post("/api/projects", json={"doc": DOC})
        assert created.status_code == 201
        pid = created.get_json()["id"]

        assert len(c.get("/api/projects").get_json()["projects"]) == 1
        assert c.get(f"/api/projects/{pid}").get_json()["doc"]["nodes"][0]["id"] == "n1"

        c.put(f"/api/projects/{pid}", json={"doc": dict(DOC, name="NDM-1 v2")})
        assert c.get(f"/api/projects/{pid}").get_json()["doc"]["name"] == "NDM-1 v2"

        assert c.delete(f"/api/projects/{pid}").status_code == 200
        assert c.get(f"/api/projects/{pid}").status_code == 404


def test_one_user_cannot_reach_another(client):
    with client.test_client() as c:
        c.post("/_signin/alice")
        pid = c.post("/api/projects", json={"doc": DOC}).get_json()["id"]

    with client.test_client() as c:
        c.post("/_signin/bob")
        assert c.get("/api/projects").get_json()["projects"] == []
        # 404 rather than 403: a guessed id must not confirm the project exists.
        assert c.get(f"/api/projects/{pid}").status_code == 404
        assert c.put(f"/api/projects/{pid}", json={"doc": DOC}).status_code == 404
        assert c.delete(f"/api/projects/{pid}").status_code == 404

    with client.test_client() as c:
        c.post("/_signin/alice")
        assert c.get(f"/api/projects/{pid}").get_json()["doc"]["name"] == "NDM-1"


def test_rejects_unnamed_and_oversized_documents(client):
    with client.test_client() as c:
        c.post("/_signin/alice")
        assert c.post("/api/projects", json={"doc": {"nodes": []}}).status_code == 400
        oversized = {"name": "big", "blob": "x" * (client.store.MAX_DOC_BYTES + 100)}
        assert c.post("/api/projects", json={"doc": oversized}).status_code == 400


def test_project_limit_is_enforced(client, monkeypatch):
    monkeypatch.setattr(client.store, "MAX_PROJECTS_PER_USER", 2)
    with client.test_client() as c:
        c.post("/_signin/alice")
        assert c.post("/api/projects", json={"doc": DOC}).status_code == 201
        assert c.post("/api/projects", json={"doc": DOC}).status_code == 201
        assert c.post("/api/projects", json={"doc": DOC}).status_code == 409
