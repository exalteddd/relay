"""Run-trace plumbing so the Relay app can animate a REAL run.

Omnigent runs each Python tool in its own subprocess, so an in-memory trace won't
survive between tool calls. Instead every tool appends one JSON line to an events
file ($MEDLAB_TRACE_DIR/events.jsonl). `assemble()` folds those events (plus the
ranked array the Runner dumps) into the single trace.json the Relay UI loads.

Tracing is opt-in: if MEDLAB_TRACE_DIR is unset, emit() is a no-op, so the tools
behave identically when tracing is off.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def trace_dir() -> Path | None:
    d = os.getenv("MEDLAB_TRACE_DIR")
    return Path(d) if d else None


def emit(kind: str, agent: str = "", message: str = "", **data) -> None:
    """Append one event. Never raises — tracing must not break a tool."""
    d = trace_dir()
    if not d:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        rec = {"ts": round(time.time(), 3), "kind": kind, "agent": agent, "message": message, **data}
        with (d / "events.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def assemble(out_path: str | Path) -> dict:
    """Build trace.json for the Relay app from the events file + ranked dump."""
    d = trace_dir()
    events = []
    if d and (d / "events.jsonl").exists():
        for line in (d / "events.jsonl").read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))

    steps = [{"agent": e["agent"], "message": e["message"]} for e in events if e["kind"] == "step"]
    plan = next((e["plan"] for e in events if e["kind"] == "plan"), None)
    result = next((e["result"] for e in events if e["kind"] == "result"), None)
    question = next((e.get("question") for e in events if e.get("question")), None)

    ranked = []
    if d and (d / "ranked.json").exists():
        ranked = json.loads((d / "ranked.json").read_text())

    trace = {"question": question, "steps": steps, "plan": plan, "result": result,
             "ranked": ranked, "source": (result or {}).get("source", "")}
    Path(out_path).write_text(json.dumps(trace))
    return trace


def reset() -> None:
    d = trace_dir()
    if not d:
        return
    for name in ("events.jsonl", "ranked.json"):
        p = d / name
        if p.exists():
            p.unlink()
