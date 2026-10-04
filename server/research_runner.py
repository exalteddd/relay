"""Drive the general research pipeline and stream it to the interface.

`brain.run_research` already answers an arbitrary question: it scopes it,
searches four literature sources, screens what it finds, extracts claims whose
quotes it can verify against the abstract, maps the evidence, proposes
hypotheses and critiques them. It emits a progress event per stage. None of
that was reachable from the interface, which only ever spoke to the compound
screen.

This module is the adapter. It runs the pipeline on a worker thread, turns its
progress events into the events the UI already understands, and converts the
finished project into graph nodes so the run shows up as a graph rather than a
log that scrolls away.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

# Stage names the pipeline emits, mapped to something a reader recognises.
AGENT_LABEL = {
    "memory-agent": "Memory",
    "scoping-agent": "Scoping",
    "literature-agent": "Literature",
    "screening-agent": "Screening",
    "extraction-agent": "Extraction",
    "synthesis-agent": "Synthesis",
    "hypothesis-agent": "Hypothesizer",
    "critic-agent": "Critic",
    "brain": "Done",
}

# Questions the compound screen can actually execute, as opposed to merely
# read about. Everything else is literature-only, and says so.
TRIAGE_HINTS = (
    "compound", "inhibitor", "inhibit", "screen", "assay", "bioassay",
    "ligand", "docking", "molecule", "chemotype", "scaffold", "pubchem",
    "ic50", "potency", "hit rate", "shortlist",
)


def resolve_mode(question: str) -> dict:
    """Decide what this question can have done to it.

    Literature reasoning generalises to any question; execution does not. A
    question only runs the screen when the screen could actually answer it,
    and the caller is told which it got rather than left to infer.
    """
    q = (question or "").lower()
    hits = [w for w in TRIAGE_HINTS if w in q]
    if len(hits) >= 2:
        return {"mode": "screen", "experiment": "compound-triage", "why": "names a compound-screening task"}
    return {"mode": "literature", "experiment": None,
            "why": "no registered experiment matches this, so it is reviewed rather than executed"}


def _safe(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return default


def build_graph(proj_dir: Path, question: str) -> dict:
    """Turn a finished run into nodes and edges the canvas can show.

    One evidence node carrying the grounded claims, one node per surviving
    hypothesis, and a next-question node from the gaps. Claims are not given a
    node each: thirty-odd cards is a worse way to read an evidence base than
    one node that lists them.
    """
    def load(rel, fallback):
        try:
            return json.loads((proj_dir / rel).read_text())
        except Exception:
            return fallback

    claims = load("literature/claims.json", [])
    hyps = load("hypotheses/hypotheses.json", [])
    emap = load("literature/evidence_map.json", {})
    papers = load("literature/papers.json", [])

    nodes, edges = [], []
    qid = "rq"
    nodes.append({"id": qid, "type": "question", "title": "Research question",
                  "summary": question, "status": "done",
                  "meta": {"objective": question,
                           "outcome": _safe(emap, "summary", default="") or "",
                           "data": f"{len(papers)} papers kept from the literature sweep"}})

    if claims:
        ev = "ev"
        nodes.append({"id": ev, "type": "source", "title": "Evidence", "status": "done",
                      "summary": "", "meta": {
                          "sources": [{"src": f"{c.get('ref','?')} ({c.get('year','n.d.')})",
                                       "meta": c.get("strength", ""),
                                       "body": c.get("claim", ""),
                                       "url": ""} for c in claims[:24]],
                          "relevance": f"{len(claims)} claims extracted and quote-checked against their abstracts.",
                          "limitations": "Claims whose quote was not verbatim in the abstract were rejected before this point."}})
        edges.append({"from": qid, "to": ev, "rel": "uses"})
    else:
        ev = None

    for idx, h in enumerate(hyps[:4]):
        node, rejected = _hypothesis_node(h, f"Hypothesis {chr(65+idx)}")
        nodes.append(node)
        if ev:
            edges.append({"from": ev, "to": node["id"],
                          "rel": "contradicts" if rejected else "supports"})

    gaps = _safe(emap, "gaps", default=[]) or []
    if gaps:
        nodes.append({"id": "next", "type": "next", "title": "Next experiment",
                      "summary": _short(gaps[0].get("gap") if isinstance(gaps[0], dict) else str(gaps[0])),
                      "status": "planned",
                      "meta": {"changed": _safe(emap, "summary", default="") or "",
                               "uncertainty": "; ".join(
                                   (g.get("gap") if isinstance(g, dict) else str(g)) for g in gaps[:3]),
                               "tests": []}})
        for h in hyps[:2]:
            edges.append({"from": h.get("id"), "to": "next", "rel": "uses"})

    return {"nodes": nodes, "edges": edges}


def _question_node(question, emap=None, papers=0):
    return {"id": "rq", "type": "question", "title": "Research question",
            "summary": question, "status": "done",
            "meta": {"objective": question,
                     "outcome": _safe(emap or {}, "summary", default="") or "",
                     "data": f"{papers} papers kept from the literature sweep" if papers else ""}}


def _evidence_node(claims):
    return {"id": "ev", "type": "source", "title": "Evidence", "status": "done", "summary": "",
            "meta": {"sources": [{"src": f"{c.get('ref','?')} ({c.get('year','n.d.')})",
                                  "meta": c.get("strength", ""),
                                  "body": c.get("claim", ""), "url": ""} for c in claims[:24]],
                     "relevance": f"{len(claims)} claims extracted and quote-checked against their abstracts.",
                     "limitations": "Claims whose quote was not verbatim in the abstract were rejected before this point."}}


def _short(text, limit=78):
    """A one-line gist: first clause or sentence, trimmed on a word."""
    t = " ".join((text or "").split())
    for stop in (". ", "; "):
        if stop in t[:limit + 20]:
            t = t.split(stop, 1)[0]
            break
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "…"
    return t


def _hypothesis_node(h, label=None):
    review = h.get("review") or {}
    rejected = h.get("status") == "rejected"
    # A card reads as a label and a line, the way the worked example does.
    # Putting 46 characters of raw statement in the title wrapped every card
    # to four lines and made the column unreadable.
    return {"id": h.get("id") or f"h{id(h)}", "type": "hypothesis",
            "title": label or "Hypothesis",
            "summary": _short(h.get("statement")),
            "status": "failed" if rejected else "done",
            "meta": {"statement": h.get("statement", ""),
                     "predictions": [h["prediction"]] if h.get("prediction") else [],
                     "uncertainty": f"Reviewer confidence {h.get('confidence','?')}"
                                    + (f" — {review.get('verdict')}" if review.get("verdict") else ""),
                     "alt": "; ".join(review.get("issues", [])[:2]),
                     "contradict": review.get("issues", [])[:3]}}, rejected


# How much work a run does. The four strong-model calls -- scoping,
# synthesis, hypotheses, critique -- run in sequence because each needs the
# one before, so they set the floor on how fast a run can be. Quick shrinks
# what they read and how hard they think; thorough is the original settings.
DEPTHS = {
    "quick":    {"max_queries": 5, "prefilter": 36, "max_papers": 15,
                 "n_hypotheses": 5, "workers": 8, "effort": "low"},
    "thorough": {"max_queries": 8, "prefilter": 60, "max_papers": 25,
                 "n_hypotheses": 6, "workers": 8, "effort": None},
}


def research_events(question: str, cfg, depth: str = "quick"):
    """Run the pipeline on a thread and yield UI events as stages complete.

    run_research is synchronous and reports through a callback, so the work
    goes on a worker and the callback feeds a queue this generator drains.
    That way a stage that takes forty seconds still streams the eight that
    came before it.
    """
    from brain import pipeline as P

    d = DEPTHS.get(depth, DEPTHS["quick"])
    # Reasoning depth is the other half: these models spend most of a call
    # thinking, so lowering it on a quick run is worth more than trimming
    # the reading.
    if d["effort"] and not cfg.reasoning_effort:
        cfg.reasoning_effort = d["effort"]

    q: "queue.Queue[tuple]" = queue.Queue()
    result = {}

    def on_event(agent, msg):
        q.put(("act", agent, msg))

    def work():
        try:
            opts = P.Options(max_queries=d["max_queries"], prefilter=d["prefilter"],
                             max_papers=d["max_papers"], n_hypotheses=d["n_hypotheses"],
                             workers=d["workers"])
            proj = P.run_research(question, cfg, opts, on_event=on_event)
            result["dir"] = Path(proj.root)      # Project.path is a method; root is the Path
            result["proj"] = proj
        except Exception as exc:                     # reported, not swallowed
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            q.put(("done",))

    t = threading.Thread(target=work, daemon=True)
    t.start()

    yield {"type": "act", "agent": "Relay", "msg": f"Researching: {question[:90]}", "st": "done"}
    yield {"type": "status", "node": "exp", "status": "running"}

    # The graph is built as the run goes, not assembled at the end. A stage
    # that takes a minute should leave something on the canvas behind it,
    # otherwise the whole run looks like a frozen progress bar.
    import json as _json
    proj_dir = None
    sent = set()

    def read(rel, fallback):
        try:
            return _json.loads((proj_dir / rel).read_text())
        except Exception:
            return fallback

    while True:
        item = q.get()
        if item[0] == "done":
            break
        _, agent, msg = item
        yield {"type": "act", "agent": AGENT_LABEL.get(agent, agent), "msg": msg, "st": "done"}

        # The first event names the project, which is how we find its folder.
        if agent == "memory-agent" and proj_dir is None:
            slug = msg.split(":", 1)[0].strip()
            cand = Path(cfg.projects_dir) / slug
            if cand.exists():
                proj_dir = cand
                yield {"type": "graph", "replace": True,
                       "data": {"nodes": [_question_node(question)], "edges": []}}
                sent.add("rq")

        elif agent == "extraction-agent" and proj_dir and "ev" not in sent:
            claims = read("literature/claims.json", [])
            if claims:
                yield {"type": "graph", "data": {"nodes": [_evidence_node(claims)],
                                                 "edges": [{"from": "rq", "to": "ev", "rel": "uses"}]}}
                sent.add("ev")

        elif agent in ("hypothesis-agent", "critic-agent") and proj_dir:
            hyps = read("hypotheses/hypotheses.json", [])
            nodes, edges = [], []
            for idx, h in enumerate(hyps[:4]):
                node, rejected = _hypothesis_node(h, f"Hypothesis {chr(65+idx)}")
                if node["id"] in sent:
                    continue
                nodes.append(node); sent.add(node["id"])
                if "ev" in sent:
                    edges.append({"from": "ev", "to": node["id"],
                                  "rel": "contradicts" if rejected else "supports"})
            if nodes:
                yield {"type": "graph", "data": {"nodes": nodes, "edges": edges}}

    t.join(timeout=5)

    if result.get("error"):
        yield {"type": "status", "node": "exp", "status": "failed"}
        yield {"type": "error", "msg": result["error"]}
        return

    proj_dir = result.get("dir") or proj_dir
    if proj_dir is None or not proj_dir.exists():
        yield {"type": "error", "msg": "The run finished but its project folder could not be located."}
        return

    graph = build_graph(proj_dir, question)
    run = {}
    try:
        run = json.loads((proj_dir / "run.json").read_text())
    except Exception:
        pass

    yield {"type": "status", "node": "exp", "status": "done"}
    # One reconciling pass: the critic may have changed verdicts after the
    # nodes went out, and the next-question node comes from the evidence map.
    yield {"type": "graph", "replace": True, "data": graph}
    yield {"type": "done", "label": "research", "data": {
        "mode": "literature",
        "papers": run.get("papers_used"),
        "claims": run.get("claims"),
        "hypotheses": run.get("hypotheses"),
        "active": run.get("hypotheses_active"),
        "usage": run.get("llm_usage"),
        "sourceErrors": len(run.get("source_errors", []) or []),
    }}
