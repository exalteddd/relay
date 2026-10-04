"""Project CRUD, scoped to the signed-in GitHub user.

Ownership rule, applied in one place: every read and write is filtered by the
session's login. There is no "look up by id, then check the owner" path, so a
guessed project id cannot leak a document or report that one exists.

Anonymous visitors are not an error case. They get `readOnly: true` and an
empty list, and the interface renders its built-in demo graph locally without
touching the database. That keeps the public demo working on a static host,
where there is no session at all.

Routes
------
    GET    /api/projects        list mine (anonymous: empty + readOnly)
    POST   /api/projects        create
    GET    /api/projects/<id>   fetch one
    PUT    /api/projects/<id>   overwrite (autosave)
    DELETE /api/projects/<id>   delete
"""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request, session

from server import store

bp = Blueprint("projects", __name__)


def _user() -> str | None:
    return session.get("user")


def _valid_doc(doc) -> str | None:
    """Return an error message, or None when the document is acceptable."""
    if not isinstance(doc, dict):
        return "project must be a JSON object"
    if not doc.get("name"):
        return "project needs a name"
    size = len(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    if size > store.MAX_DOC_BYTES:
        return f"project is too large ({size} bytes; limit {store.MAX_DOC_BYTES})"
    return None


@bp.get("/api/projects")
def list_projects():
    """List the caller's projects.

    `isNewAccount` is true exactly once per account, ever -- it comes from the
    insert in `ensure_user`, not from the list being empty. The interface uses
    it to seed one sample project on first sign-in, so a user who later deletes
    everything keeps an empty workspace instead of having the sample reappear.
    """
    user = _user()
    if user is None:
        return jsonify(user=None, readOnly=True, isNewAccount=False, projects=[])
    is_new = store.ensure_user(user)
    return jsonify(
        user=user,
        readOnly=False,
        isNewAccount=is_new,
        projects=store.list_projects(user),
    )


@bp.post("/api/projects")
def create_project():
    user = _user()
    if user is None:
        return jsonify(error="Sign in to create a project"), 401

    body = request.get_json(silent=True) or {}
    doc = body.get("doc") if isinstance(body.get("doc"), dict) else body
    problem = _valid_doc(doc)
    if problem:
        return jsonify(error=problem), 400

    if store.count_projects(user) >= store.MAX_PROJECTS_PER_USER:
        return jsonify(
            error=f"You have reached the limit of {store.MAX_PROJECTS_PER_USER} projects. "
                  "Delete one to make room."
        ), 409

    return jsonify(store.create_project(user, doc)), 201


@bp.get("/api/projects/<pid>")
def get_project(pid: str):
    user = _user()
    if user is None:
        return jsonify(error="Sign in to open a project"), 401
    found = store.get_project(pid, user)
    if found is None:
        return jsonify(error="No such project"), 404
    return jsonify(found)


@bp.put("/api/projects/<pid>")
def update_project(pid: str):
    user = _user()
    if user is None:
        return jsonify(error="Sign in to save"), 401

    body = request.get_json(silent=True) or {}
    doc = body.get("doc") if isinstance(body.get("doc"), dict) else body
    problem = _valid_doc(doc)
    if problem:
        return jsonify(error=problem), 400

    if not store.update_project(pid, user, doc):
        return jsonify(error="No such project"), 404
    return jsonify(ok=True, id=pid)


@bp.delete("/api/projects/<pid>")
def delete_project(pid: str):
    user = _user()
    if user is None:
        return jsonify(error="Sign in to delete a project"), 401
    if not store.delete_project(pid, user):
        return jsonify(error="No such project"), 404
    return jsonify(ok=True, id=pid)
