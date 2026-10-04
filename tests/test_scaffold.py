"""Scaffolding a new project's graph from the topic the user named.

The contract: the scaffolder proposes a PLAN onto editable nodes, and never
produces a finding or a citation. Those two prohibitions are what stop a new
project from carrying either invented evidence or a conclusion written before
the experiment.

    pytest tests/test_scaffold.py -q
"""

from __future__ import annotations

import json

import pytest

flask = pytest.importorskip("flask")

PLAN = {
    "question": {
        "summary": "Which electrolyte additives suppress dendrite growth in Li-metal cells?",
        "objective": "Rank candidate additives by predicted dendrite suppression.",
        "scope": "Simulation-based triage, not cell validation.",
        "outcome": "Ranking accuracy against measured cycling data.",
        "constraints": "No wet-lab cycling in-loop.",
        "data": "Published cycling datasets plus a labelled fixture.",
        "budget": "<= 40 tool calls",
    },
    "hypotheses": [
        {"title": "Hypothesis A", "summary": "Film-forming additives dominate",
         "statement": "Additives that form a stable SEI suppress dendrites.",
         "predictions": ["SEI-formers rank above the median"],
         "uncertainty": "Moderate.", "alt": "Transport-limited explanation."},
        {"title": "Hypothesis B", "summary": "Transport limits dominate",
         "statement": "Ionic transport uniformity governs dendrite onset.",
         "predictions": ["High-conductivity additives rank top"],
         "uncertainty": "Moderate.", "alt": "Film-forming explanation."},
    ],
    "experiment": {
        "title": "Experiment 01", "summary": "Ranking on cycling labels",
        "tests": [{"name": "Potency-only", "sel": False, "why": "baseline"},
                  {"name": "Transport-aware rank", "sel": True, "why": "separates the two"}],
        "method": "Gradient-boosted ranker on composition features.",
        "inputs": "Labelled additive set.", "baseline": "Random selection.",
        "controls": "Random control + y-scramble.", "params": "n=400",
        "cost": "~25 calls", "approval": "Required before run.",
    },
    "next": {"uncertainty": "Does it transfer to real cells?",
             "tests": [{"name": "Real-data run", "why": "measure on published cycling", "cost": "~30"}]},
}


class FakeLLM:
    reply = json.dumps(PLAN)

    def __init__(self, cfg):
        self.cfg = cfg

    def text(self, system, prompt, **kw):
        FakeLLM.last_system, FakeLLM.last_prompt = system, prompt
        return self.reply


class FakeCfg:
    provider, model = "openai", "gpt-test"


@pytest.fixture()
def app(monkeypatch):
    import brain.llm as real_llm

    monkeypatch.setattr(real_llm, "LLM", FakeLLM)
    monkeypatch.setattr("brain.config.load_config", lambda: FakeCfg())
    from server.scaffold import bp

    a = flask.Flask(__name__)
    a.secret_key = "test"
    a.register_blueprint(bp)
    return a


def _post(app, **body):
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        return c.post("/api/scaffold", json=body or {"name": "Li-metal dendrites",
                                                     "question": "Which additives suppress dendrites?"})


def test_requires_sign_in(app):
    with app.test_client() as c:
        assert c.post("/api/scaffold", json={"name": "x"}).status_code == 401


def test_requires_a_name(app):
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        assert c.post("/api/scaffold", json={"question": "q"}).status_code == 400


def test_graph_is_about_the_topic_not_the_demo(app):
    """The defect this endpoint exists to fix: a new project must carry none
    of the bundled NDM-1 content."""
    doc = _post(app).get_json()["doc"]
    blob = json.dumps(doc).lower()
    for leak in ("ndm-1", "ndm1", "carbapenem", "klebsiella", "metallo-", "zinc"):
        assert leak not in blob, f"demo content leaked into a new project: {leak}"
    assert "dendrite" in blob


def test_scaffold_never_invents_sources(app):
    """A citation proposed before anything was read is a fabricated one."""
    doc = _post(app).get_json()["doc"]
    ev = next(n for n in doc["nodes"] if n["type"] == "source")
    assert ev["meta"]["sources"] == []
    assert ev["status"] == "planned"


def test_scaffold_never_writes_a_result(app):
    """The conclusion must not be written before the experiment runs."""
    doc = _post(app).get_json()["doc"]
    res = next(n for n in doc["nodes"] if n["type"] == "result")
    assert res["meta"] == {}
    assert res["status"] == "planned"
    assert "results" not in doc
    # and hypothesis support must stay empty: it cites evidence that does not exist
    for h in [n for n in doc["nodes"] if n["type"] == "hypothesis"]:
        assert h["meta"]["supports"] == []
        assert h["meta"]["contradict"] == []


def test_graph_is_wired_for_a_run(app):
    doc = _post(app).get_json()["doc"]
    ids = {n["id"] for n in doc["nodes"]}
    assert {"q1", "ev1", "hA", "hB", "exp", "res", "next"} <= ids
    assert doc["primaryExp"] == "exp" and doc["resultNode"] == "res"
    for e in doc["edges"]:
        assert e["from"] in ids and e["to"] in ids
    assert doc["scaffolded"] is True


def test_prohibitions_reach_the_model(app):
    _post(app)
    sysmsg = FakeLLM.last_system.lower()
    assert "do not cite" in sysmsg
    assert "never a result" in sysmsg or "plan, never a result" in sysmsg


def test_unusable_plan_is_a_clean_failure(app, monkeypatch):
    FakeLLM.reply = '{"something": "else"}'
    try:
        r = _post(app)
        assert r.status_code == 502 and r.get_json()["ok"] is False
    finally:
        FakeLLM.reply = json.dumps(PLAN)
