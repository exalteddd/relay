"""Deterministic lab tools shared by the Omnigent agents and the autopilot pipeline.

Agents do the reasoning; these functions do the things that must be exact:
searching the literature, checking that every claim is quoted verbatim from a
source, validating citations, and keeping the shared research record (project
repo + long-term memory). Every function takes and returns plain JSON-able data.
"""

from __future__ import annotations

import re

from .config import Config, load_config
from .memory import Brain
from .project import Project
from .rank import bm25_rank
from .sources import search_all

_cfg: Config | None = None


def cfg() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def brain() -> Brain:
    return Brain(cfg().db_path)


def _proj(project: str) -> Project:
    root = cfg().projects_dir / project
    if not root.exists():
        raise ValueError(f"unknown project '{project}'. Call start_project first.")
    return Project(root)


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def is_grounded(quote: str, abstract: str) -> bool:
    q = _norm(quote).strip(" .\"'")
    return len(q) >= 15 and q in _norm(abstract)


def _memory_brief(recalled: dict) -> list[str]:
    out = [f"prior hypothesis [{h['project']}, {h['status']}]: {h['statement']}" for h in recalled["hypotheses"][:5]]
    out += [f"prior finding [{c['project']}]: {c['text']} ({c['paper_title']})" for c in recalled["claims"][:8]]
    return out


# ------------------------------------------------------------------ project + memory

def start_project(question: str) -> dict:
    proj = Project.create(cfg().projects_dir, question)
    b = brain()
    b.add_project(proj.slug, question, proj.root)
    recalled = b.recall(question, exclude_project=proj.slug)
    proj.write_json("memory_recall.json", recalled)
    proj.log("brain", f"project created; recalled {len(recalled['claims'])} claims and "
                      f"{len(recalled['hypotheses'])} hypotheses from memory")
    proj.commit("brain", "start project")
    return {"project": proj.slug, "path": str(proj.root), "memory": _memory_brief(recalled),
            "memory_size": b.stats()}


def recall_memory(query: str, k: int = 8) -> dict:
    r = brain().recall(query, k=k)
    return {"papers": [{"id": p["id"], "title": p["title"], "year": p["year"]} for p in r["papers"]],
            "claims": [{"project": c["project"], "claim": c["text"], "paper": c["paper_title"]} for c in r["claims"]],
            "hypotheses": [{"project": h["project"], "status": h["status"], "confidence": h["confidence"],
                            "statement": h["statement"]} for h in r["hypotheses"]]}


def save_plan(project: str, refined_question: str, sub_questions: list[str], queries: list[str]) -> dict:
    proj = _proj(project)
    plan = {"refined_question": refined_question, "sub_questions": sub_questions, "queries": queries}
    proj.write_json("plan.json", plan)
    proj.log("scoping-agent", f"{len(sub_questions)} sub-questions, {len(queries)} queries")
    proj.commit("scoping-agent", "research plan")
    return {"saved": True, **plan}


def log_decision(project: str, agent: str, decision: str) -> dict:
    proj = _proj(project)
    proj.log(agent, decision)
    proj.commit(agent, decision[:72])
    return {"logged": True}


# ------------------------------------------------------------------ literature

def search_literature(project: str, queries: list[str], sources: list[str] | None = None,
                      per_query: int = 20, top_k: int = 60) -> dict:
    proj = _proj(project)
    plan = proj.read_json("plan.json", {}) or {}
    question = proj.read_text("question.md").split("\n", 2)[-1].strip()
    papers, errors = search_all(queries, cfg(), sources, per_query)
    b = brain()
    known = b.known_paper_ids()
    with_abs = [p for p in papers if len(p.abstract) >= 150]
    rank_q = " ".join([question, *plan.get("sub_questions", []), *queries])
    ranked = bm25_rank(with_abs, rank_q)[:top_k]
    existing = proj.read_json("literature/candidates.json", []) or []
    start = len(existing)
    seen = {c["id"] for c in existing}
    added = []
    for p, score in ranked:
        if p.id in seen:
            continue
        row = {**p.to_dict(), "ref": f"P{start + len(added) + 1:03d}", "bm25": round(score, 2),
               "in_memory": p.id in known}
        added.append(row)
        seen.add(p.id)
    proj.write_json("literature/candidates.json", existing + added)
    proj.log("literature-agent", f"{len(queries)} queries → {len(papers)} unique papers, "
                                 f"{len(added)} new candidates kept, {len(errors)} source errors")
    proj.commit("literature-agent", f"+{len(added)} candidate papers")
    return {"found": len(papers), "new_candidates": len(added), "total_candidates": start + len(added),
            "source_errors": errors[:5],
            "candidates": [{"ref": r["ref"], "title": r["title"], "year": r["year"], "citations": r["citations"],
                            "in_memory": r["in_memory"], "abstract": r["abstract"][:600]} for r in added]}


def get_papers(project: str, refs: list[str]) -> dict:
    proj = _proj(project)
    by_ref = {c["ref"]: c for c in proj.read_json("literature/candidates.json", [])}
    return {"papers": [{"ref": r, "title": by_ref[r]["title"], "year": by_ref[r]["year"],
                        "venue": by_ref[r]["venue"], "url": by_ref[r]["url"], "abstract": by_ref[r]["abstract"]}
                       for r in refs if r in by_ref],
            "unknown_refs": [r for r in refs if r not in by_ref]}


def save_screening(project: str, decisions: list[dict], max_papers: int = 25) -> dict:
    """decisions: [{ref, score 0-3, reason}]. Keeps score >= 2 (topped up to 8)."""
    proj = _proj(project)
    cands = {c["ref"]: c for c in proj.read_json("literature/candidates.json", [])}
    dec = {d["ref"]: d for d in decisions if d.get("ref") in cands}
    order = sorted(dec, key=lambda r: (-int(dec[r].get("score", 0)), r))
    keep = [r for r in order if int(dec[r].get("score", 0)) >= 2][:max_papers]
    if len(keep) < 8:
        keep += [r for r in order if r not in keep][:8 - len(keep)]
    prev = {p["ref"]: p for p in proj.read_json("literature/papers.json", [])}
    for r in keep:
        prev[r] = {**cands[r], "relevance": int(dec[r].get("score", 0)), "reason": dec[r].get("reason", "")}
    rows = sorted(prev.values(), key=lambda p: p["ref"])
    proj.write_json("literature/papers.json", rows)
    new = brain().upsert_papers(project, rows)
    proj.log("screening-agent", f"screened {len(dec)} papers, selected {len(keep)} ({new} new to memory)")
    proj.commit("screening-agent", f"selected {len(keep)} papers")
    return {"selected": keep, "total_selected": len(rows), "new_to_memory": new,
            "ignored_unknown_refs": [d.get("ref") for d in decisions if d.get("ref") not in cands]}


def save_claims(project: str, claims: list[dict]) -> dict:
    """claims: [{ref, claim, kind, direction, system, strength, quote}]. Quote must be verbatim."""
    proj = _proj(project)
    papers = {p["ref"]: p for p in proj.read_json("literature/papers.json", [])}
    existing = proj.read_json("literature/claims.json", []) or []
    accepted, rejected = [], []
    for c in claims:
        p = papers.get(c.get("ref", ""))
        if not p:
            rejected.append({"ref": c.get("ref"), "claim": c.get("claim"), "reason": "ref is not a selected paper"})
        elif not c.get("claim"):
            rejected.append({"ref": c.get("ref"), "reason": "empty claim"})
        elif not is_grounded(c.get("quote", ""), p["abstract"]):
            rejected.append({"ref": c["ref"], "claim": c["claim"],
                             "reason": "quote not found verbatim in the abstract"})
        else:
            cid = f"C{len(existing) + len(accepted) + 1:03d}"
            accepted.append({"id": cid, "paper_id": p["id"], "ref": p["ref"], "paper_title": p["title"],
                             "year": p["year"], "url": p["url"],
                             **{k: c.get(k, "") for k in ("claim", "kind", "direction", "system", "strength", "quote")}})
    allc = existing + accepted
    proj.write_json("literature/claims.json", allc)
    brain().add_claims(project, accepted)
    proj.log("extraction-agent", f"{len(accepted)} grounded claims saved, {len(rejected)} rejected")
    proj.commit("extraction-agent", f"+{len(accepted)} grounded claims")
    return {"accepted": [c["id"] for c in accepted], "rejected": rejected, "total_claims": len(allc)}


def list_claims(project: str) -> dict:
    claims = _proj(project).read_json("literature/claims.json", []) or []
    return {"claims": [{"id": c["id"], "ref": c["ref"], "year": c["year"], "strength": c["strength"],
                        "system": c["system"], "direction": c["direction"], "claim": c["claim"]} for c in claims]}


# ------------------------------------------------------------------ synthesis + hypotheses

def save_evidence_map(project: str, evidence_map: dict) -> dict:
    from .pipeline import render_evidence_map
    proj = _proj(project)
    claims = proj.read_json("literature/claims.json", []) or []
    ids = {c["id"] for c in claims}
    removed = 0
    for section in ("themes", "consensus", "contradictions", "gaps"):
        for item in evidence_map.get(section, []) or []:
            for key in ("claim_ids", "related_claim_ids"):
                if key in item:
                    before = len(item[key])
                    item[key] = [i for i in item[key] if i in ids]
                    removed += before - len(item[key])
    question = proj.read_text("question.md").split("\n", 2)[-1].strip()
    proj.write_json("literature/evidence_map.json", evidence_map)
    proj.write_text("literature/evidence_map.md", render_evidence_map(question, evidence_map, claims))
    proj.log("synthesis-agent", f"evidence map: {len(evidence_map.get('gaps', []))} gaps, "
                                f"{len(evidence_map.get('contradictions', []))} contradictions")
    proj.commit("synthesis-agent", "evidence map")
    return {"saved": True, "invalid_citations_removed": removed}


def save_hypotheses(project: str, hypotheses: list[dict]) -> dict:
    proj = _proj(project)
    ids = {c["id"] for c in proj.read_json("literature/claims.json", [])}
    out, warnings = [], []
    for i, h in enumerate(hypotheses):
        cited = h.get("supporting_claim_ids", []) or []
        h["supporting_claim_ids"] = [c for c in cited if c in ids]
        h["unknown_citations"] = [c for c in cited if c not in ids]
        h["id"] = f"H{i + 1:02d}"
        h["label"] = "agent-generated hypothesis (unvalidated)"
        h["status"] = "proposed"
        if h["unknown_citations"]:
            warnings.append(f"{h['id']}: dropped unknown citations {h['unknown_citations']}")
        if not h["supporting_claim_ids"]:
            warnings.append(f"{h['id']}: no valid supporting claims; critic should reject or it needs evidence")
        out.append(h)
    proj.write_json("hypotheses/hypotheses.json", out)
    proj.log("hypothesis-agent", f"proposed {len(out)} hypotheses")
    proj.commit("hypothesis-agent", f"{len(out)} candidate hypotheses")
    return {"ids": [h["id"] for h in out], "warnings": warnings}


def get_hypotheses_for_review(project: str) -> dict:
    proj = _proj(project)
    by_id = {c["id"]: c for c in proj.read_json("literature/claims.json", [])}
    hyps = proj.read_json("hypotheses/hypotheses.json", []) or []
    return {"hypotheses": [{"id": h["id"], "statement": h["statement"], "experiment": h.get("experiment"),
                            "novelty": h.get("novelty"), "feasibility": h.get("feasibility"),
                            "testability": h.get("testability"),
                            "cited_claims": [{"id": c, "claim": by_id[c]["claim"], "quote": by_id[c]["quote"],
                                              "strength": by_id[c]["strength"]} for c in h["supporting_claim_ids"]]}
                           for h in hyps]}


def save_reviews(project: str, reviews: list[dict]) -> dict:
    """reviews: [{id, verdict keep|revise|drop, issues, revised_statement, already_established,
    prior_confidence, comment}]. Applies verdicts, ranks, writes report + memory."""
    from .pipeline import render_hypothesis, render_report
    proj = _proj(project)
    claims = proj.read_json("literature/claims.json", []) or []
    by_id = {c["id"]: c for c in claims}
    hyps = proj.read_json("hypotheses/hypotheses.json", []) or []
    rv = {r.get("id"): r for r in reviews}
    for h in hyps:
        r = rv.get(h["id"], {})
        h["review"] = r
        try:
            h["confidence"] = float(r.get("prior_confidence", 0.5))
        except (TypeError, ValueError):
            h["confidence"] = 0.5
        verdict = r.get("verdict", "keep")
        if not h["supporting_claim_ids"]:
            verdict = "drop"
            r.setdefault("issues", []).append("no valid supporting claims")
        # The critic's novelty finding is a verdict, not a note: a hypothesis
        # the literature has already settled is not one worth testing.
        if r.get("already_established"):
            verdict = "drop"
            r.setdefault("issues", []).append("already established in the cited literature")
        if verdict == "drop":
            h["status"] = "rejected"
        elif verdict == "revise" and r.get("revised_statement"):
            h["original_statement"], h["statement"] = h["statement"], r["revised_statement"]
            h["status"] = "revised"
        else:
            h["status"] = "proposed"
        h["score"] = round(sum(float(h.get(k, 3) or 3) for k in ("novelty", "feasibility", "testability")) / 3, 2)
        # `score` is the proposer grading its own work; `confidence` is the
        # critic's independent read. Rank on both, so an unconvinced critic can
        # outweigh a self-flattering score instead of only breaking its ties.
        h["rank_score"] = round(0.5 * (h["score"] / 5.0) + 0.5 * h["confidence"], 3)
    hyps.sort(key=lambda h: (h["status"] == "rejected", -h["rank_score"], -h["score"]))
    proj.write_json("hypotheses/hypotheses.json", hyps)
    for h in hyps:
        proj.write_text(f"hypotheses/{h['id']}.md", render_hypothesis(h, by_id))
    b = brain()
    b.add_hypotheses(project, hyps)
    active = [h for h in hyps if h["status"] != "rejected"]
    papers = proj.read_json("literature/papers.json", []) or []
    cands = proj.read_json("literature/candidates.json", []) or []
    run = {"question": proj.read_text("question.md").split("\n", 2)[-1].strip(),
           "papers_found": len(cands), "papers_screened": len(cands), "papers_used": len(papers),
           "claims": len(claims), "hypotheses": len(hyps), "hypotheses_active": len(active), "memory": b.stats()}
    prev = proj.read_json("run.json", {}) or {}
    proj.write_json("run.json", {**prev, **run})
    proj.write_text("README.md", render_report(run["question"], proj.read_json("plan.json", {}) or {},
                                               proj.read_json("literature/evidence_map.json", {}) or {}, hyps,
                                               {**prev, **run}))
    proj.log("critic-agent", f"{len(active)} hypotheses survive review, {len(hyps) - len(active)} rejected")
    proj.commit("critic-agent", "reviewed hypotheses + report")
    return {"ranking": [{"id": h["id"], "status": h["status"], "score": h["score"],
                         "prior_confidence": h["confidence"], "statement": h["statement"]} for h in hyps]}


def project_status(project: str) -> dict:
    proj = _proj(project)
    count = lambda rel: len(proj.read_json(rel, []) or [])  # noqa: E731
    hyps = proj.read_json("hypotheses/hypotheses.json", []) or []
    return {"project": project, "plan": proj.read_json("plan.json"),
            "candidates": count("literature/candidates.json"), "selected_papers": count("literature/papers.json"),
            "claims": count("literature/claims.json"),
            "evidence_map": proj.path("literature/evidence_map.json").exists(),
            "hypotheses": [{"id": h["id"], "status": h["status"], "statement": h["statement"]} for h in hyps],
            "history": proj.history()[:15]}
