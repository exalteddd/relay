"""Offline tests: source parsers, dedupe, grounding, memory, and the full
pipeline with a mock LLM and fixture literature."""

from __future__ import annotations

import json

import pytest

from brain import lab_tools as lab
from brain.config import load_config
from brain.pipeline import Options, run_research
from brain.sources import Paper, merge, parse_arxiv, parse_europepmc, parse_openalex, parse_s2

OPENALEX = {"results": [{
    "id": "https://openalex.org/W123", "doi": "https://doi.org/10.1000/ABC.1",
    "title": "Gut microbiome diversity and sleep quality in adults",
    "publication_year": 2021, "cited_by_count": 42,
    "abstract_inverted_index": {"Higher": [0], "microbiome": [1], "diversity": [2], "predicted": [3],
                                "better": [4], "sleep": [5], "efficiency.": [6]},
    "authorships": [{"author": {"display_name": "A. Smith"}}],
    "primary_location": {"source": {"display_name": "Sleep"}, "landing_page_url": "https://x"}}]}

S2 = {"data": [{"paperId": "s2abc", "title": "Gut Microbiome Diversity and Sleep Quality in Adults.",
                "abstract": "A longer abstract from Semantic Scholar about microbiome diversity and sleep.",
                "year": 2021, "venue": "Sleep", "citationCount": 50,
                "externalIds": {"DOI": "10.1000/abc.1", "PubMed": "999"}, "url": "https://s2/abc",
                "authors": [{"name": "A. Smith"}]}]}

EPMC = {"resultList": {"result": [{"id": "111", "source": "MED", "pmid": "111", "doi": "10.2000/xyz",
                                   "title": "Butyrate producers and sleep.", "authorString": "Lee K, Park J.",
                                   "journalTitle": "Microbiome", "pubYear": "2020", "citedByCount": 7,
                                   "abstractText": "<h4>Background</h4>Butyrate <i>producers</i> matter."}]}}

ARXIV = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2101.00001v2</id><published>2021-01-01T00:00:00Z</published>
<title>Modeling  sleep and microbes</title><summary>We model sleep.</summary>
<author><name>B. Jones</name></author></entry></feed>"""


def test_parsers_and_merge():
    oa = parse_openalex(OPENALEX)[0]
    assert oa.abstract == "Higher microbiome diversity predicted better sleep efficiency."
    assert oa.doi == "10.1000/abc.1" and oa.id == "doi:10.1000/abc.1"
    s2 = parse_s2(S2)[0]
    assert s2.id == "doi:10.1000/abc.1"
    ep = parse_europepmc(EPMC)[0]
    assert ep.abstract == "Background Butyrate producers matter." and ep.authors == ["Lee K", "Park J"]
    ax = parse_arxiv(ARXIV)[0]
    assert ax.external_ids["arxiv"] == "2101.00001" and ax.title == "Modeling sleep and microbes"
    merged = merge([oa, s2, ep, ax])
    assert len(merged) == 3
    m = merged[0]
    assert m.citations == 50 and set(m.sources) == {"openalex", "semantic_scholar"}
    assert "Semantic Scholar" in m.abstract  # longer abstract wins


def test_grounding():
    abstract = "We found that X increases Y in mice (n=40). Effects were absent in rats."
    assert lab.is_grounded("X increases Y in mice", abstract)
    assert lab.is_grounded("effects  were ABSENT in rats.", abstract)
    assert not lab.is_grounded("X strongly increases Y", abstract)
    assert not lab.is_grounded("mice", abstract)  # too short to count as evidence


def _fixture_papers(topic: str, n: int = 14) -> list[Paper]:
    out = []
    for i in range(n):
        out.append(Paper(
            id=f"doi:10.9/{topic}.{i}", title=f"{topic.title()} study {i}: microbiome diversity and sleep",
            abstract=(f"This cohort study of {100 + i} adults examined gut microbiome diversity and sleep quality. "
                      f"Higher Shannon diversity was associated with longer total sleep time in participant group {i}. "
                      f"Butyrate-producing taxa correlated with fewer awakenings in {topic} analyses. "
                      "Limitations include the cross-sectional design and self-reported sleep."),
            year=2015 + i % 9, citations=i * 3, sources=["fixture"], url=f"https://doi.org/10.9/{topic}.{i}"))
    return out


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BRAIN_PROVIDER", "mock")
    cfg = load_config()
    lab._cfg = cfg
    yield cfg
    lab._cfg = None


def test_full_pipeline_and_memory(env, monkeypatch):
    monkeypatch.setattr(lab, "search_all", lambda q, c, s=None, n=20, **k: (_fixture_papers("alpha"), []))
    events = []
    proj = run_research("Does gut microbiome diversity affect sleep quality?", env,
                        Options(n_hypotheses=5), on_event=lambda a, m: events.append((a, m)))

    papers = proj.read_json("literature/papers.json")
    claims = proj.read_json("literature/claims.json")
    hyps = proj.read_json("hypotheses/hypotheses.json")
    assert 8 <= len(papers) <= 25
    # every saved claim is verbatim-grounded; the mock's fabricated claims were rejected
    by_ref = {p["ref"]: p for p in papers}
    assert claims and all(lab.is_grounded(c["quote"], by_ref[c["ref"]]["abstract"]) for c in claims)
    assert not any(c["claim"] == "fabricated claim" for c in claims)
    # hypotheses cite only real claims; critic verdicts applied; rejected kept as negative knowledge
    ids = {c["id"] for c in claims}
    assert all(set(h["supporting_claim_ids"]) <= ids for h in hyps)
    assert hyps[0]["unknown_citations"] == ["C999"] or any(h["unknown_citations"] for h in hyps)
    statuses = {h["status"] for h in hyps}
    assert {"proposed", "revised", "rejected"} <= statuses
    assert hyps[-1]["status"] == "rejected"
    # research record: files + one commit per agent
    for f in ("README.md", "plan.json", "notebook.md", "literature/evidence_map.md", "hypotheses/H01.md"):
        assert proj.path(f).exists(), f
    agents = {c["agent"] for c in proj.history()}
    assert {"scoping-agent", "literature-agent", "screening-agent", "extraction-agent", "synthesis-agent",
            "hypothesis-agent", "critic-agent"} <= agents

    # second project recalls the first project's knowledge
    monkeypatch.setattr(lab, "search_all", lambda q, c, s=None, n=20, **k: (_fixture_papers("beta"), []))
    started = lab.start_project("Can butyrate-producing microbiome taxa improve sleep?")
    assert started["memory"], "memory should recall claims/hypotheses from the first project"
    assert started["memory_size"]["projects"] == 2


def test_tool_contract_rejects_bad_quotes(env, monkeypatch):
    monkeypatch.setattr(lab, "search_all", lambda q, c, s=None, n=20, **k: (_fixture_papers("gamma"), []))
    slug = lab.start_project("Does microbiome diversity affect sleep?")["project"]
    lab.save_plan(slug, "q", ["sq"], ["microbiome sleep"])
    res = lab.search_literature(slug, ["microbiome sleep"])
    assert res["new_candidates"] == 14 and res["candidates"][0]["ref"] == "P001"
    again = lab.search_literature(slug, ["microbiome sleep"])  # no duplicate candidates
    assert again["new_candidates"] == 0
    sel = lab.save_screening(slug, [{"ref": c["ref"], "score": 3, "reason": "x"} for c in res["candidates"]]
                             + [{"ref": "P999", "score": 3}])
    assert sel["ignored_unknown_refs"] == ["P999"]
    ref = sel["selected"][0]
    abstract = lab.get_papers(slug, [ref])["papers"][0]["abstract"]
    good = abstract.split(". ")[1]
    out = lab.save_claims(slug, [
        {"ref": ref, "claim": "diversity ~ sleep time", "quote": good, "strength": "weak"},
        {"ref": ref, "claim": "made up", "quote": "Diversity causes sleep improvements in all humans"},
        {"ref": "P999", "claim": "wrong paper", "quote": good},
    ])
    assert out["accepted"] == ["C001"] and len(out["rejected"]) == 2
    hs = lab.save_hypotheses(slug, [{"statement": "H", "supporting_claim_ids": ["C001", "C404"]},
                                    {"statement": "no evidence", "supporting_claim_ids": []}])
    assert len(hs["warnings"]) == 2
    rank = lab.save_reviews(slug, [{"id": "H01", "verdict": "keep", "prior_confidence": 0.6},
                                   {"id": "H02", "verdict": "keep"}])["ranking"]
    assert [r["status"] for r in rank] == ["proposed", "rejected"]  # unsupported is auto-rejected
    status = lab.project_status(slug)
    assert status["claims"] == 1 and len(status["history"]) >= 6
    json.dumps(status)  # tool outputs must be JSON-serializable
