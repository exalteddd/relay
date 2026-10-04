"""Omnigent guardrail policies for the lab.

Referenced from lab/config.yaml as dotted paths, e.g.
  function: {path: brain.policies.approval_gate, arguments: {...}}
A factory returns an evaluator `fn(event) -> {"result": ALLOW|ASK|DENY, "reason": ...}`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_ALLOW = {"result": "ALLOW"}


def _tool_args(event: dict[str, Any], names: set[str]) -> dict | None:
    if event.get("type") != "tool_call":
        return None
    data = event.get("data") or {}
    if data.get("name") not in names:
        return None
    args = data.get("arguments")
    return args if isinstance(args, dict) else {}


def approval_gate(tools: list[str], reason: str = "A scientist must approve this step.") -> Callable:
    """ASK a human before any of `tools` runs (e.g. approving the research plan)."""
    names = set(tools)

    def _evaluate(event: dict[str, Any]) -> dict[str, Any]:
        args = _tool_args(event, names)
        if args is None:
            return _ALLOW
        preview = ", ".join(f"{k}={str(v)[:200]}" for k, v in args.items() if k != "project")
        return {"result": "ASK", "reason": f"{reason}\n{preview}"}

    return _evaluate


def query_limit(max_queries: int = 10) -> Callable:
    """DENY literature searches that fan out to too many queries at once (API politeness + cost)."""

    def _evaluate(event: dict[str, Any]) -> dict[str, Any]:
        args = _tool_args(event, {"search_literature"})
        if args is None:
            return _ALLOW
        if len(args.get("queries") or []) > max_queries:
            return {"result": "DENY", "reason": f"At most {max_queries} queries per search call."}
        return _ALLOW

    return _evaluate
