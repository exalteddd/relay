"""Build a new project's research graph from the topic the user names.

A new project used to start as a copy of the bundled NDM-1 graph, which meant
a project called "Perovskite degradation" carried NDM-1's question, NDM-1's
hypotheses and NDM-1's evidence under a different title. The report then
rendered all of it faithfully. That is a worse failure than a hallucinating
model, because it is confidently wrong content with no model involved.

So the graph is scaffolded for the stated topic instead. Two rules shape what
the model is allowed to produce:

1. It writes into the *graph*, never into the paper. Everything it proposes
   lands on nodes the user can read, edit and delete before any of it becomes
   a claim in an exported document.
2. It proposes a plan, never a finding, and never a source. The result node
   is created empty and the evidence node is created with no citations --
   a scaffolded reference would be a fabricated one, since nothing has been
   read at scaffold time. Evidence arrives from the literature agent or from
   the user, both of which can be checked.

    POST /api/scaffold  {name, question}
      -> 200 {ok:true, doc:{...project graph...}}
      -> 503 {ok:false, reason}   not configured; caller uses a blank graph
"""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, request, session

bp = Blueprint("scaffold", __name__)

SYSTEM = """You design the opening research plan for a computational science project.

Given a topic and a research question, propose the scaffold of an investigation:
the objective, two genuinely competing hypotheses, a first experiment that
would discriminate between them, and sensible follow-up experiments.

Hard rules:
1. Propose a PLAN, never a RESULT. You have run nothing and read nothing. Do
   not state findings, numbers, accuracies, effect sizes or outcomes.
2. Do NOT cite anything. No papers, authors, DOIs, datasets-by-name you are
   not certain exist, or URLs. Sources are gathered separately and verified.
3. The two hypotheses must be genuinely competing explanations that a single
   experiment could distinguish -- not one hypothesis and a weaker restatement.
4. The experiment must name a concrete baseline and concrete negative controls
   appropriate to the method. A plan without controls is not a plan.
5. Stay in the domain of the question asked. If the question is about battery
   electrolytes, every field is about battery electrolytes.
6. Be specific and technical. Generic filler ("analyse the data", "evaluate
   performance") is a failed response.

Return STRICT JSON, no code fence, exactly these keys:

{
  "question": {
    "summary": "the research question, one sentence",
    "objective": "what a successful project produces",
    "scope": "what this does and explicitly does not cover",
    "outcome": "the measurable quantity that decides success",
    "constraints": "limits on method, data or validation",
    "data": "what data this would run on, described generically",
    "budget": "a rough compute/tool-call budget"
  },
  "hypotheses": [
    {"title": "Hypothesis A", "summary": "short label",
     "statement": "the full claim",
     "predictions": ["what must be observed if true", "..."],
     "uncertainty": "how confident, and why",
     "alt": "the competing explanation"},
    {"title": "Hypothesis B", "summary": "...", "statement": "...",
     "predictions": ["..."], "uncertainty": "...", "alt": "..."}
  ],
  "experiment": {
    "title": "Experiment 01", "summary": "short label",
    "tests": [
      {"name": "option considered", "sel": false, "why": "why it was weighed"},
      {"name": "option chosen", "sel": true, "why": "why this one discriminates"}
    ],
    "method": "the concrete method",
    "inputs": "what goes in",
    "baseline": "what it is compared against",
    "controls": "the negative controls and what each must show to pass",
    "params": "key parameters",
    "cost": "estimated tool calls",
    "approval": "Required before run."
  },
  "next": {
    "uncertainty": "the biggest open question once this runs",
    "tests": [
      {"name": "follow-up", "why": "what it would settle", "cost": "est."},
      {"name": "follow-up", "why": "...", "cost": "est."}
    ]
  }
}
"""

# Canvas positions mirror the bundled layout so a scaffolded graph opens
# readable rather than stacked at the origin.
LAYOUT = {
    "q1": (40, 300), "ev1": (300, 300), "hA": (560, 170), "hB": (560, 430),
    "exp": (830, 300), "res": (1100, 250), "next": (1380, 140),
}
EDGES = [
    {"from": "q1", "to": "ev1", "rel": "uses"},
    {"from": "ev1", "to": "hA", "rel": "supports"},
    {"from": "ev1", "to": "hB", "rel": "supports"},
    {"from": "hA", "to": "exp", "rel": "tests"},
    {"from": "hB", "to": "exp", "rel": "tests"},
    {"from": "exp", "to": "res", "rel": "produced"},
    {"from": "res", "to": "next", "rel": "uses"},
]


def _node(nid, ntype, title, summary, status, meta):
    x, y = LAYOUT[nid]
    return {"id": nid, "type": ntype, "title": title, "summary": summary,
            "status": status, "x": x, "y": y, "meta": meta or {}}


def _clean(v, limit=1200):
    """Strings only, trimmed. Keeps a verbose model from bloating a node."""
    return str(v).strip()[:limit] if v is not None else ""


def _clean_list(v, limit=10):
    if not isinstance(v, list):
        return []
    return [_clean(x, 400) for x in v[:limit] if _clean(x, 400)]


def build_doc(name: str, question: str, plan: dict) -> dict:
    """Map a validated plan onto the graph shape the interface renders."""
    q = plan.get("question") or {}
    hyps = [h for h in (plan.get("hypotheses") or []) if isinstance(h, dict)][:2]
    exp = plan.get("experiment") or {}
    nxt = plan.get("next") or {}

    nodes = [
        _node("q1", "question", "Research question",
              _clean(q.get("summary") or question, 300), "done",
              {k: _clean(q.get(k)) for k in
               ("objective", "scope", "outcome", "constraints", "data", "budget")
               if _clean(q.get(k))}),
        # Evidence starts empty on purpose: a scaffolded citation would be an
        # invented one. The node exists so there is somewhere for the
        # literature step, or the user, to put real sources.
        _node("ev1", "source", "Evidence", "No sources recorded yet", "planned",
              {"sources": [],
               "relevance": "Add sources here, or run the literature step. "
                            "Citations are never scaffolded — an invented reference "
                            "is worse than a missing one.",
               "limitations": "This project has no evidence base yet. Hypotheses below "
                              "are proposed starting points, not grounded claims."}),
    ]

    for nid, h in zip(("hA", "hB"), hyps):
        nodes.append(_node(
            nid, "hypothesis", _clean(h.get("title") or nid, 60),
            _clean(h.get("summary"), 160), "planned",
            {"statement": _clean(h.get("statement")),
             "predictions": _clean_list(h.get("predictions")),
             # supports/contradict stay empty: both cite evidence, and there is
             # none yet. They populate when real sources land on ev1.
             "supports": [], "contradict": [],
             "uncertainty": _clean(h.get("uncertainty"), 400),
             "alt": _clean(h.get("alt"), 400)}))

    tests = [t for t in (exp.get("tests") or []) if isinstance(t, dict)][:4]
    nodes.append(_node(
        "exp", "experiment", _clean(exp.get("title") or "Experiment 01", 60),
        _clean(exp.get("summary"), 160), "planned",
        {"tests": [{"name": _clean(t.get("name"), 120), "sel": bool(t.get("sel")),
                    "why": _clean(t.get("why"), 400)} for t in tests],
         "method": _clean(exp.get("method")), "inputs": _clean(exp.get("inputs")),
         "baseline": _clean(exp.get("baseline")), "controls": _clean(exp.get("controls")),
         "params": _clean(exp.get("params"), 300), "cost": _clean(exp.get("cost"), 120),
         "approval": _clean(exp.get("approval") or "Required before run.", 120)}))

    # The result node is created empty. A scaffold that pre-filled an
    # interpretation would be writing the conclusion before the experiment.
    nodes.append(_node("res", "result", "Results", "Not yet run", "planned", {}))

    ntests = [t for t in (nxt.get("tests") or []) if isinstance(t, dict)][:4]
    nodes.append(_node(
        "next", "next", "Next experiment", "Choose a test", "planned",
        {"uncertainty": _clean(nxt.get("uncertainty"), 400),
         "tests": [{"name": _clean(t.get("name"), 120), "why": _clean(t.get("why"), 400),
                    "cost": _clean(t.get("cost"), 80)} for t in ntests]}))

    return {
        "name": name, "icon": "flask", "run": "Experiment 01",
        "question": _clean(q.get("summary") or question, 300),
        "primaryExp": "exp", "resultNode": "res",
        "nodes": nodes, "edges": list(EDGES),
        "scaffolded": True,   # the interface flags this until a run replaces it
    }


@bp.post("/api/scaffold")
def scaffold():
    if session.get("user") is None:
        return jsonify(ok=False, reason="Sign in to scaffold a project"), 401

    body = request.get_json(silent=True) or {}
    name = _clean(body.get("name"), 120)
    question = _clean(body.get("question"), 500)
    if not name:
        return jsonify(ok=False, reason="A project name is required"), 400

    try:
        from brain.config import load_config
        from brain.llm import LLM, parse_json
    except ImportError as exc:
        return jsonify(ok=False, reason=f"LLM layer unavailable: {exc}"), 503
    try:
        cfg = load_config()
        llm = LLM(cfg)
    except Exception as exc:
        return jsonify(ok=False, reason=f"{type(exc).__name__}: {exc}"), 503

    prompt = (f"Topic: {name}\n"
              f"Research question: {question or '(not supplied — infer a precise one from the topic)'}\n\n"
              "Design the opening research plan.")
    try:
        plan = parse_json(llm.text(SYSTEM, prompt, max_tokens=2600))
    except Exception as exc:
        return jsonify(ok=False, reason=f"{type(exc).__name__}: {exc}"), 502

    if not isinstance(plan, dict) or not plan.get("question"):
        return jsonify(ok=False, reason="Model did not return a usable plan"), 502

    return jsonify(ok=True, doc=build_doc(name, question, plan),
                   model=cfg.model, provider=cfg.provider)
