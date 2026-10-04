"""Relay backend: GitHub sign-in, and real runs streamed to the UI.

Viewing the interface is open to anyone. *Starting a run* costs compute, so it
requires a signed-in GitHub user (optionally one named in RELAY_ALLOWED_USERS).

Secrets
-------
Every secret is read from the environment and never leaves this process:

    SESSION_SECRET        signs the session cookie        (required)
    GITHUB_CLIENT_ID      OAuth app id                    (required for login)
    GITHUB_CLIENT_SECRET  OAuth app secret                (required for login)
    RELAY_ALLOWED_USERS   optional comma-separated GitHub logins; empty = any
    ANTHROPIC_API_KEY     reserved for the Omnigent route (not used yet)

The browser is never sent any of these. The GitHub access token is exchanged
server-side, read once to learn who signed in, and then dropped -- it is not
stored in the session and not returned by any endpoint. /api/config exposes
only booleans about how the server is configured, never a value.

Run locally:  python -m server.app
On Render:    gunicorn --threads 8 --timeout 300 server.app:app
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path

import requests
from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory, session

from server.pipeline_runner import DEFAULT_QUESTION, engine_available, run_events

# Load .env for local development. Real environments (Render) set these in the
# process environment, which already-set values take precedence over.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # optional: production does not need it
    pass

ROOT = Path(__file__).resolve().parent.parent
GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"

app = Flask(__name__, static_folder=None)

# --- configuration ------------------------------------------------------
# In production a missing SESSION_SECRET is fatal: falling back to a random
# value would silently log everyone out on each restart, and falling back to a
# constant would let anyone forge a session.
IS_DEV = os.environ.get("RELAY_ENV", "dev") == "dev"
_secret = os.environ.get("SESSION_SECRET")
if not _secret:
    if not IS_DEV:
        raise RuntimeError("SESSION_SECRET must be set when RELAY_ENV != dev")
    _secret = secrets.token_hex(32)  # ephemeral; dev only
app.secret_key = _secret

CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
ALLOWED = {u.strip().lower() for u in os.environ.get("RELAY_ALLOWED_USERS", "").split(",") if u.strip()}
AUTH_CONFIGURED = bool(CLIENT_ID and CLIENT_SECRET)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,    # JS cannot read it, so XSS cannot steal it
    SESSION_COOKIE_SAMESITE="Lax",   # survives the OAuth redirect, blocks cross-site POSTs
    SESSION_COOKIE_SECURE=not IS_DEV,  # HTTPS-only off localhost
)


# One run at a time. A screen peaks around 380MB, so two at once exceed the
# 512MB instance and the whole process is OOM-killed -- which would take the
# bystander's run down too. Refusing the second is the kinder failure, and it
# doubles as a cap on concurrent model spend.
# This guards a single process; more than one worker would need shared state.
_run_slot = threading.BoundedSemaphore(1)


def current_user():
    """The signed-in GitHub login, or None."""
    return session.get("user")


def dev_open_access() -> bool:
    """Local development with no OAuth app set up yet.

    Registering a GitHub OAuth app just to press Run on your own laptop is
    pointless friction, so that case is allowed. It requires all three of:
    dev mode, no OAuth configured, and a request from this machine. A
    deployment fails all three -- render.yaml sets RELAY_ENV=prod, real
    clients are not loopback, and configuring OAuth switches it off -- so
    this cannot leave a public instance open by accident.
    """
    if not (IS_DEV and not AUTH_CONFIGURED):
        return False
    return (request.remote_addr or "") in ("127.0.0.1", "::1")


def may_run(user: str | None) -> bool:
    """Runs need a signed-in user, and membership of the allowlist when set."""
    if dev_open_access():
        return True
    if user is None:
        return False
    return not ALLOWED or user.lower() in ALLOWED


# --- auth ---------------------------------------------------------------
@app.get("/auth/login")
def login():
    if not AUTH_CONFIGURED:
        return jsonify(error="GitHub OAuth is not configured on this server"), 503
    # CSRF: this value must come back unchanged on the callback.
    state = secrets.token_urlsafe(24)
    session["oauth_state"] = state
    session["post_login"] = request.args.get("next", "/")
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": request.url_root.rstrip("/") + "/auth/callback",
        "scope": "read:user",  # identity only; no repo or write access
        "state": state,
        "allow_signup": "true",
    }
    return redirect(GITHUB_AUTHORIZE + "?" + "&".join(f"{k}={requests.utils.quote(v)}" for k, v in params.items()))


@app.get("/auth/callback")
def callback():
    if not AUTH_CONFIGURED:
        abort(503)
    expected = session.pop("oauth_state", None)
    if not expected or request.args.get("state") != expected:
        return jsonify(error="Bad OAuth state"), 400
    code = request.args.get("code")
    if not code:
        return jsonify(error="No code returned"), 400

    token_res = requests.post(
        GITHUB_TOKEN,
        headers={"Accept": "application/json"},
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": request.url_root.rstrip("/") + "/auth/callback",
        },
        timeout=15,
    )
    token = token_res.json().get("access_token") if token_res.ok else None
    if not token:
        return jsonify(error="Token exchange failed"), 400

    who = requests.get(
        GITHUB_USER,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=15,
    )
    if not who.ok:
        return jsonify(error="Could not read GitHub profile"), 400
    profile = who.json()

    # The token has served its purpose. It is deliberately not persisted.
    session["user"] = profile.get("login")
    session["avatar"] = profile.get("avatar_url")
    return redirect(session.pop("post_login", "/") or "/")


@app.post("/auth/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/me")
def me():
    user = current_user()
    return jsonify(user=user, avatar=session.get("avatar"), mayRun=may_run(user),
                   devOpen=dev_open_access())


# --- status -------------------------------------------------------------
@app.get("/api/health")
def health():
    # `pipeline` tells the UI whether Run drives the real engine.
    return jsonify(ok=True, pipeline=engine_available())


@app.get("/api/config")
def config():
    """Non-secret facts about this deployment. Never exposes a key's value."""
    return jsonify(
        authConfigured=AUTH_CONFIGURED,
        authRequired=True,
        engine=engine_available(),
        allowlistActive=bool(ALLOWED),
    )


# --- runs ---------------------------------------------------------------
@app.post("/api/run")
def run():
    user = current_user()
    if not may_run(user):
        if user is None:
            return jsonify(error="Sign in with GitHub to start a run", loginUrl="/auth/login"), 401
        return jsonify(error=f"{user} is not on this server's allowlist"), 403

    body = request.get_json(silent=True) or {}
    question = (body.get("question") or DEFAULT_QUESTION).strip()[:500]
    aid = body.get("aid")
    try:
        aid = int(aid) if aid not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify(error="aid must be an integer PubChem assay id"), 400

    if not _run_slot.acquire(blocking=False):
        return jsonify(error="Another run is in progress on this server. "
                             "Runs are handled one at a time; try again in a moment."), 429

    def stream():
        try:
            for event in run_events(question=question, aid=aid):
                yield "data: " + json.dumps(event) + "\n\n"
        except GeneratorExit:  # client navigated away
            raise
        except Exception as exc:
            yield "data: " + json.dumps({"type": "error", "msg": f"{type(exc).__name__}: {exc}"}) + "\n\n"
        finally:
            # Released here, not in the view: the generator outlives the request.
            _run_slot.release()

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- static frontend ----------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    target = (ROOT / filename).resolve()
    if not str(target).startswith(str(ROOT)) or not target.is_file():
        abort(404)
    return send_from_directory(ROOT, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Relay on http://localhost:{port}  (engine: {engine_available()}, auth: {AUTH_CONFIGURED})")
    app.run(host="0.0.0.0", port=port, threaded=True)
