"""The optional LLM-written report sections.

The contract that matters here is containment: the endpoint returns the three
named sections and nothing else, and every failure path leaves the caller free
to fall back to the report's assembled prose rather than returning a broken
document.

    pytest tests/test_narrative.py -q
"""

from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

PROJECT = {
    "name": "NDM-1",
    "question": "Which untested compounds inhibit NDM-1?",
    "nodes": [
        {"id": "q1", "type": "question", "title": "Research question", "x": 40, "y": 300,
         "meta": {"objective": "Rank untested compounds."}},
        {"id": "exp", "type": "experiment", "title": "Experiment 03", "x": 830, "y": 300,
         "meta": {"method": "Random forest on fingerprints.", "controls": "y-scramble"}},
    ],
    "edges": [{"from": "q1", "to": "exp", "rel": "tests"}],
}


@pytest.fixture()
def app():
    from server.narrative import bp

    a = flask.Flask(__name__)
    a.secret_key = "test"
    a.register_blueprint(bp)
    return a


class FakeLLM:
    """Stands in for brain.llm.LLM. `reply` is whatever the model 'returned'."""

    reply = '{"abstract":"A.","discussion":"D.","limitations":"L."}'

    def __init__(self, cfg):
        self.cfg = cfg

    def text(self, system, prompt, **kw):
        FakeLLM.last_prompt = prompt
        FakeLLM.last_system = system
        return self.reply


class FakeCfg:
    provider = "openai"
    model = "gpt-test"


def _patch(monkeypatch, reply=None):
    import brain.llm as real_llm

    if reply is not None:
        FakeLLM.reply = reply
    monkeypatch.setattr(real_llm, "LLM", FakeLLM)
    monkeypatch.setattr("brain.config.load_config", lambda: FakeCfg())


def test_requires_sign_in(app):
    with app.test_client() as c:
        r = c.post("/api/narrative", json={"project": PROJECT})
        assert r.status_code == 401
        assert r.get_json()["ok"] is False


def test_rejects_missing_project(app, monkeypatch):
    _patch(monkeypatch)
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        assert c.post("/api/narrative", json={}).status_code == 400


def test_happy_path_returns_three_sections(app, monkeypatch):
    _patch(monkeypatch)
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        body = c.post("/api/narrative", json={"project": PROJECT}).get_json()
        assert body["ok"] is True
        assert body["abstract"] == "A."
        assert body["discussion"] == "D."
        assert body["limitations"] == "L."
        assert body["provider"] == "openai"


def test_extra_keys_from_the_model_are_discarded(app, monkeypatch):
    """A model that decides to also return 'results' must not get them rendered."""
    _patch(monkeypatch, '{"abstract":"A.","discussion":"D.","limitations":"L.",'
                        '"results":"ROC-AUC 0.99","conclusion":"cures everything"}')
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        body = c.post("/api/narrative", json={"project": PROJECT}).get_json()
        # The response carries the three prose sections plus the claim-check
        # metadata -- and nothing the model invented beyond them.
        assert set(body) == {"ok", "abstract", "discussion", "limitations", "model",
                             "provider", "verified", "verifyError", "removed", "removedCount"}
        assert "0.99" not in str(body)
        assert "cures everything" not in str(body)


def test_unusable_model_output_is_a_clean_failure(app, monkeypatch):
    _patch(monkeypatch, "I'm afraid I can't do that.")
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        r = c.post("/api/narrative", json={"project": PROJECT})
        assert r.status_code == 502
        assert r.get_json()["ok"] is False


def test_prompt_carries_the_graph_but_not_layout(app, monkeypatch):
    """Coordinates are noise to the writer and cost tokens; science is kept."""
    _patch(monkeypatch)
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        c.post("/api/narrative", json={"project": PROJECT})
    prompt = FakeLLM.last_prompt
    assert "Random forest on fingerprints." in prompt
    assert "y-scramble" in prompt
    assert '"x":' not in prompt and '"y":' not in prompt
    # the anti-fabrication instruction must actually reach the model
    assert "do not" in FakeLLM.last_system.lower()


# --------------------------------------------------------------------------
# Claim check: the second, adversarial pass over the draft.
#
# Constraining generation is not the same as checking output. These tests pin
# the behaviour that matters: an unsupported sentence must leave the exported
# prose, and a check that did not run must never be reported as one that did.
# --------------------------------------------------------------------------

OVERSTATED = (
    '{"abstract":"The ranker reaches 1.2x enrichment. '
    'It substantially outperforms all published baselines and confirms binding.",'
    '"discussion":"Controls behaved as required.",'
    '"limitations":"Computational triage only."}'
)


class TwoCallLLM(FakeLLM):
    """First call drafts, second call fact-checks. Mirrors the real flow."""

    draft = OVERSTATED
    verdict = ('{"unsupported":[{"section":"abstract",'
               '"sentence":"It substantially outperforms all published baselines and confirms binding.",'
               '"why":"no baseline comparison or binding assay in the data"}]}')

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def text(self, system, prompt, **kw):
        TwoCallLLM.last_system = system
        self.calls += 1
        return TwoCallLLM.draft if self.calls == 1 else TwoCallLLM.verdict


def _patch_two_call(monkeypatch, verdict=None):
    import brain.llm as real_llm

    if verdict is not None:
        TwoCallLLM.verdict = verdict
    monkeypatch.setattr(real_llm, "LLM", TwoCallLLM)
    monkeypatch.setattr("brain.config.load_config", lambda: FakeCfg())


def _call(app):
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user"] = "alice"
        return c.post("/api/narrative", json={"project": PROJECT}).get_json()


def test_unsupported_sentence_is_removed_from_the_prose(app, monkeypatch):
    _patch_two_call(monkeypatch)
    body = _call(app)
    assert body["verified"] is True
    assert body["removedCount"] == 1
    # the overstatement is gone from what the report will print
    assert "substantially outperforms" not in body["abstract"]
    assert "confirms binding" not in body["abstract"]
    # the supported sentence beside it survives
    assert "1.2x enrichment" in body["abstract"]


def test_removals_are_reported_not_silently_dropped(app, monkeypatch):
    _patch_two_call(monkeypatch)
    body = _call(app)
    r = body["removed"][0]
    assert r["section"] == "abstract"
    assert "outperforms" in r["sentence"]
    assert r["why"]


def test_clean_draft_passes_through_untouched(app, monkeypatch):
    _patch_two_call(monkeypatch, '{"unsupported":[]}')
    body = _call(app)
    assert body["verified"] is True and body["removedCount"] == 0
    assert "Controls behaved as required." in body["discussion"]


def test_failed_check_is_declared_not_assumed(app, monkeypatch):
    """If the verifier breaks, the caller must learn the draft is unchecked."""
    _patch_two_call(monkeypatch, "not json at all")
    body = _call(app)
    assert body["ok"] is True
    assert body["verified"] is False
    assert body["verifyError"]
    # prose is still returned, but flagged as unverified rather than cleaned
    assert "substantially outperforms" in body["abstract"]


def test_verifier_is_given_the_data_and_the_prose(app, monkeypatch):
    _patch_two_call(monkeypatch)
    _call(app)
    sysmsg = TwoCallLLM.last_system.lower()
    assert "fact-check" in sysmsg or "adversarial" in sysmsg
    assert "do not" in sysmsg
