"""Relay backend: GitHub sign-in, and real runs streamed to the UI.

Viewing the interface is open to anyone. *Starting a run* costs compute, so it
requires a signed-in GitHub user named in RELAY_ALLOWED_USERS in production.

Secrets
-------
Every secret is read from the environment and never leaves this process:

    SESSION_SECRET        signs the session cookie        (required)
    GITHUB_CLIENT_ID      OAuth app id                    (required for login)
    GITHUB_CLIENT_SECRET  OAuth app secret                (required for login)
    RELAY_ALLOWED_USERS   comma-separated GitHub logins; required for production runs
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
import time
from pathlib import Path

import requests
from flask import Flask, Response, abort, jsonify, redirect, request, send_from_directory, session

from server import store
from server.pipeline_runner import DEFAULT_QUESTION, engine_available, run_events
from server.narrative import bp as narrative_bp
from server.projects import bp as projects_bp
from server.scaffold import bp as scaffold_bp
from server.research_runner import research_events, resolve_mode

# Load .env for local development. Real environments (Render) set these in the
# process environment, which already-set values take precedence over.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # optional: production does not need it
    pass

ROOT = Path(__file__).resolve().parent.parent
# Per-user saved graphs. A plain directory of JSON keyed by GitHub login:
# no database to run, and the only reader is the user who owns the file.
# Set RELAY_STATE_DIR to a mounted disk to make it survive a redeploy.
STATE_DIR = Path(os.environ.get("RELAY_STATE_DIR", ROOT / "user_state"))
GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"

app = Flask(__name__, static_folder=None)
# Render terminates TLS and forwards over http, so without this Flask believes
# it is serving http:// while the browser says https://. That mismatch breaks
# the origin check on every POST and makes the OAuth callback URL wrong.
if os.environ.get("RELAY_TRUST_PROXY", "1") != "0":
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

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
    # Large enough for a saved graph, small enough to stay a bound.
    MAX_CONTENT_LENGTH=256 * 1024,
    SESSION_COOKIE_HTTPONLY=True,    # JS cannot read it, so XSS cannot steal it
    SESSION_COOKIE_SAMESITE="Lax",   # survives the OAuth redirect, blocks cross-site POSTs
    SESSION_COOKIE_SECURE=not IS_DEV,  # HTTPS-only off localhost
)

app.register_blueprint(projects_bp)
app.register_blueprint(narrative_bp)
app.register_blueprint(scaffold_bp)

# Create the project tables once at boot. A failure here is reported rather
# than fatal: /api/health stays up so the cause is visible, and the project
# endpoints surface the real error instead of the service refusing to start.
STORAGE_READY = True
try:
    store.init()
except Exception as exc:  # noqa: BLE001 - surfaced through /api/config
    STORAGE_READY = False
    print(f"[relay] storage unavailable ({store.engine_name()}): {type(exc).__name__}: {exc}")


# One run at a time. A screen peaks around 380MB, so two at once exceed the
# 512MB instance and the whole process is OOM-killed -- which would take the
# bystander's run down too. Refusing the second is the kinder failure, and it
# doubles as a cap on concurrent model spend.
# This guards a single process; more than one worker would need shared state.
_run_slot = threading.BoundedSemaphore(1)
# Who holds it and since when, so a second person is told what they are
# waiting for rather than just being refused.
_run_holder = {"user": None, "started": 0.0}
# Nothing here runs for an hour. A slot still held past that is a leak, not
# work, so the next caller takes it rather than being refused forever.
RUN_SLOT_MAX_SECONDS = 3600


def _reclaim_stale_slot() -> bool:
    started = _run_holder.get("started") or 0.0
    if started and (time.time() - started) > RUN_SLOT_MAX_SECONDS:
        app.logger.warning("reclaiming a run slot held for %.0fs", time.time() - started)
        _run_holder.update(user=None, started=0.0)
        return True          # caller proceeds; the stale holder cannot release twice
    return False


@app.before_request
def check_request_origin():
    """Reject cross-site POSTs. Cookies alone do not stop a hostile sibling.

    Compared on host rather than full origin: behind a TLS-terminating proxy
    the scheme Flask sees need not match the one the browser sent, and a
    scheme mismatch is not a cross-site request. Refusals answer in JSON, so a
    caller expecting JSON gets a readable reason instead of an HTML page it
    cannot parse.
    """
    if request.method != "POST":
        return None
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return jsonify(error="Cross-site requests are not accepted"), 403
    origin = request.headers.get("Origin")
    if origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != urlsplit(request.host_url).netloc:
            return jsonify(error="Request origin does not match this server"), 403
    return None


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.path.startswith(("/api/", "/auth/")):
        response.headers["Cache-Control"] = "no-store"
    return response


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
    """Production runs require explicit allowlist membership."""
    if dev_open_access():
        return True
    if user is None:
        return False
    return user.lower() in ALLOWED or (IS_DEV and not ALLOWED)


# --- auth ---------------------------------------------------------------
@app.get("/auth/login")
def login():
    if not AUTH_CONFIGURED:
        return jsonify(error="GitHub OAuth is not configured on this server"), 503
    # CSRF: this value must come back unchanged on the callback.
    state = secrets.token_urlsafe(24)
    session["oauth_state"] = state
    next_path = request.args.get("next", "/")
    session["post_login"] = next_path if (next_path.startswith("/") and not next_path.startswith("//") and "\\" not in next_path and not any(ord(c) < 32 for c in next_path)) else "/"
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
        # "sqlite" in a real deployment means DATABASE_URL is missing and
        # projects are sitting on a disk that is wiped on the next deploy.
        storage=store.engine_name(),
        storageReady=STORAGE_READY,
    )


# --- per-user saved graphs ----------------------------------------------
def _state_file(user: str) -> Path:
    """One file per user. The name is derived, never taken from input."""
    safe = "".join(c for c in user.lower() if c.isalnum() or c in "-_")[:40]
    if not safe:
        raise ValueError("unusable username")
    return STATE_DIR / f"{safe}.json"


@app.get("/api/state")
def get_state():
    """The signed-in user's saved graphs, or nothing if they have none yet."""
    user = current_user()
    if not user:
        return jsonify(scope="local", state=None)
    try:
        f = _state_file(user)
        if f.is_file():
            return jsonify(scope="user", state=json.loads(f.read_text()))
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return jsonify(scope="user", state=None)


@app.put("/api/state")
def put_state():
    user = current_user()
    if not user:
        return jsonify(error="Sign in to save your work to this account"), 401
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object"), 400
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        f = _state_file(user)
        # Write then replace, so an interrupted save cannot truncate the file.
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(body))
        tmp.replace(f)
    except (OSError, ValueError) as exc:
        app.logger.warning("state save failed: %s", exc)
        return jsonify(error="Could not save"), 500
    return jsonify(ok=True, scope="user")


@app.delete("/api/state")
def delete_state():
    user = current_user()
    if not user:
        return jsonify(error="Not signed in"), 401
    try:
        _state_file(user).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    return jsonify(ok=True)


# --- question refinement ------------------------------------------------
@app.post("/api/refine-question")
def refine_question():
    """Sharpen a research question with one cheap model call.

    Gated like a run, because it spends money — just much less of it.
    """
    user = current_user()
    if not may_run(user):
        return jsonify(error="Sign in to use this"), 401 if user is None else 403
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="Nothing to sharpen"), 400
    question = question[:500]
    from brain import prompts
    from brain.config import load_config
    from brain.llm import LLM

    cfg = load_config()
    # Name the actual problem. "Could not reach the model" sent people looking
    # for a network fault when the usual cause is an unset key on this server.
    if cfg.provider == "openai" and not cfg.openai_api_key:
        return jsonify(error="No model key on this server. Set OPENAI_API_KEY and redeploy."), 503
    if cfg.provider == "anthropic" and not cfg.anthropic_api_key:
        return jsonify(error="No model key on this server. Set ANTHROPIC_API_KEY and redeploy."), 503
    try:
        out = LLM(cfg).json("refine", prompts.REFINE_QUESTION, f"Question: {question}",
                            fast=True, max_tokens=700, context={"question": question})
    except Exception as exc:
        app.logger.warning("refine failed: %s", exc)
        reason = str(exc)[:140] or exc.__class__.__name__
        return jsonify(error=f"The model call failed ({cfg.fast_model}): {reason}"), 502
    return jsonify(
        refined=str(out.get("refined") or question)[:500],
        tooBroad=bool(out.get("too_broad")),
        changes=[str(c)[:160] for c in (out.get("changes") or [])][:5],
        missing=[str(c)[:160] for c in (out.get("missing") or [])][:5],
    )


@app.get("/api/resolve")
def resolve():
    """Which engine would answer this question. One source of truth for the
    decision, so the interface does not keep a second copy of the rules."""
    q = (request.args.get("q") or "").strip()[:500]
    plan = resolve_mode(q)
    if plan["mode"] == "screen" and not engine_available():
        plan = {"mode": "literature", "experiment": None,
                "why": "the screening engine is not installed on this server"}
    return jsonify(plan)


# --- runs ---------------------------------------------------------------
@app.post("/api/run")
def run():
    user = current_user()
    if not may_run(user):
        if user is None:
            return jsonify(error="Sign in with GitHub to start a run", loginUrl="/auth/login"), 401
        return jsonify(error=f"{user} is not on this server's allowlist"), 403

    # The global cap is sized for saved graphs; a run request is a question
    # and a couple of fields, so hold it to the tighter bound it had before.
    if (request.content_length or 0) > 16 * 1024:
        return jsonify(error="Request too large"), 413
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict) or not isinstance(body.get("question", ""), str):
        return jsonify(error="Expected a JSON object with a text question"), 400
    question = (body.get("question") or DEFAULT_QUESTION).strip()[:500]
    aid = body.get("aid")
    try:
        aid = int(aid) if aid not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify(error="aid must be an integer PubChem assay id"), 400

    if not _run_slot.acquire(blocking=False):
        if not _reclaim_stale_slot():
            who = _run_holder["user"] or "someone"
            elapsed = int(time.time() - (_run_holder["started"] or time.time()))
            return jsonify(
                error=f"{who} is already running one. Runs go one at a time on this server.",
                busy=True, busyWith=who, elapsedSec=elapsed,
            ), 429
    _run_holder.update(user=user or "local", started=time.time())
    released = threading.Event()

    def release_slot():
        """Give the slot back exactly once, however the request ends."""
        if released.is_set():
            return
        released.set()
        _run_holder.update(user=None, started=0.0)
        _run_slot.release()

    # Which engine can answer this. Literature reasoning works for any
    # question; the compound screen only for the one thing it does.
    plan = resolve_mode(question)
    if body.get("mode") in ("screen", "literature"):
        plan = {**plan, "mode": body["mode"]}        # an explicit choice wins

    def stream():
        try:
            yield "data: " + json.dumps({"type": "plan", "data": plan}) + "\n\n"
            if plan["mode"] == "literature":
                from brain.config import load_config
                cfg = load_config()
                if cfg.provider == "openai" and not cfg.openai_api_key:
                    yield "data: " + json.dumps({"type": "error",
                        "msg": "No model key on this server. Set OPENAI_API_KEY and redeploy."}) + "\n\n"
                    return
                events = research_events(question, cfg)
            else:
                events = run_events(question=question, aid=aid)
            for event in events:
                yield "data: " + json.dumps(event) + "\n\n"
        except GeneratorExit:  # client navigated away
            raise
        except Exception as exc:
            app.logger.exception("Research run failed")
            yield "data: " + json.dumps({"type": "error", "msg": "Research run failed; check server logs."}) + "\n\n"
        finally:
            release_slot()

    response = Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    # The generator's finally only runs if the generator runs. A client that
    # disconnects before reading a byte leaves it untouched, and the slot was
    # then held forever -- every later run answered 429 with nothing running.
    # call_on_close fires whether or not anything was iterated.
    response.call_on_close(release_slot)
    return response


# --- static frontend ----------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    # Serve only deliberately published frontend files, never the repo root.
    if filename != "assets/relay-mark.svg":
        abort(404)
    target = ROOT / filename
    if target.is_symlink() or not target.resolve().is_relative_to(ROOT.resolve()):
        abort(404)
    return send_from_directory(ROOT, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Relay on http://localhost:{port}  (engine: {engine_available()}, auth: {AUTH_CONFIGURED})")
    # Localhost only by default: the dev server has no TLS and relaxed cookie
    # rules, so it should not be reachable from the rest of the network.
    # Set RELAY_HOST=0.0.0.0 deliberately if you need to reach it from a phone.
    host = os.environ.get("RELAY_HOST", "127.0.0.1")
    app.run(host=host, port=port, threaded=True)
