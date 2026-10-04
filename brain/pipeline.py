"""Autopilot pipeline + markdown renderers.

question -> memory recall -> literature -> claims -> evidence map -> hypotheses
-> critique. Uses the same lab tools as the Omnigent agents (brain/lab_tools.py).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import lab_tools as lab
from . import prompts
from .config import Config
from .llm import LLM
from .project import Project


@dataclass
class Options:
    sources: list[str] | None = None   # default: all four
    per_query: int = 20                # results per query per source
    max_queries: int = 8
    prefilter: int = 60                # papers the LLM screens
    max_papers: int = 25               # papers kept for evidence extraction
    n_hypotheses: int = 6
    workers: int = 4                   # parallel LLM calls


def _batches(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def run_research(question: str, cfg: Config, opts: Options | None = None, on_event=None) -> Project:
    """Autopilot mode: the same lab tools the Omnigent agents use, driven by
    direct LLM calls. Useful for batch runs, benchmarks and tests."""
    opts = opts or Options()
    lab._cfg = cfg
    llm = LLM(cfg)
    emit = on_event or (lambda agent, msg: None)

    # 1. project + memory recall
    started = lab.start_project(question)
    slug = started["project"]
    proj = Project(cfg.projects_dir / slug)
    memory_txt = "\n".join(f"- {m}" for m in started["memory"]) or "(nothing relevant in memory yet)"
    emit("memory-agent", f"{slug}: recalled {len(started['memory'])} related items from earlier projects")

    # 2. plan
    plan = llm.json("plan", prompts.SCOPING, f"Research question: {question}\n\nTeam memory:\n{memory_txt}",
                    context={"question": question})
    queries = [q for q in dict.fromkeys(plan.get("queries", [])) if q][:opts.max_queries] or [question]
    lab.save_plan(slug, plan.get("refined_question", question), plan.get("sub_questions", []), queries)
    emit("scoping-agent", f"{len(plan.get('sub_questions', []))} sub-questions, {len(queries)} search queries")

    # 3. search
    res = lab.search_literature(slug, queries, opts.sources, opts.per_query, opts.prefilter)
    emit("literature-agent", f"{res['found']} unique papers, {res['new_candidates']} candidates to screen, "
                             f"{len(res['source_errors'])} source errors")
    if not res["new_candidates"]:
        raise RuntimeError("No papers with abstracts found. Source errors:\n" + "\n".join(res["source_errors"]))
    cands = proj.read_json("literature/candidates.json", [])

    # 4. screen (parallel, fast model)
    def screen(batch):
        payload = [{"ref": c["ref"], "title": c["title"], "year": c["year"], "abstract": c["abstract"][:1200]}
                   for c in batch]
        return llm.json("screen", prompts.SCREENING,
                        f"Research question: {question}\n\nPapers:\n{json.dumps(payload, indent=1)}",
                        fast=True, context={"question": question, "papers": payload})

    with ThreadPoolExecutor(opts.workers) as pool:
        decisions = [s for r in pool.map(screen, _batches(cands, 15)) for s in r.get("scores", [])]
    sel = lab.save_screening(slug, decisions, opts.max_papers)
    emit("screening-agent", f"screened {len(cands)}, kept {sel['total_selected']} ({sel['new_to_memory']} new to memory)")

    # 5. extract grounded claims (parallel, fast model)
    papers = proj.read_json("literature/papers.json", [])

    def extract(batch):
        payload = [{"ref": p["ref"], "title": p["title"], "year": p["year"], "abstract": p["abstract"]} for p in batch]
        return llm.json("extract", prompts.EXTRACTION,
                        f"Research question: {question}\n\nPapers:\n{json.dumps(payload, indent=1)}",
                        fast=True, context={"question": question, "papers": payload})

    with ThreadPoolExecutor(opts.workers) as pool:
        raw_claims = [c for r in pool.map(extract, _batches(papers, 5)) for c in r.get("claims", [])]
    saved = lab.save_claims(slug, raw_claims)
    emit("extraction-agent", f"{len(saved['accepted'])} grounded claims; {len(saved['rejected'])} rejected "
                             f"(quote not verbatim in the abstract)")
    claims = lab.list_claims(slug)["claims"]
    if not claims:
        raise RuntimeError("No grounded claims could be extracted.")
    claim_lines = "\n".join(f"{c['id']} [{c['ref']}, {c['year']}, {c['strength']}, {c['system']}]: {c['claim']}"
                            for c in claims)

    # 6. evidence map
    emap = llm.json("synthesize", prompts.SYNTHESIS, f"Research question: {question}\n\nClaims:\n{claim_lines}",
                    max_tokens=6000, context={"question": question, "claims": claims})
    lab.save_evidence_map(slug, emap)
    emit("synthesis-agent", f"{len(emap.get('themes', []))} themes, {len(emap.get('contradictions', []))} "
                            f"contradictions, {len(emap.get('gaps', []))} gaps")

    # 7. hypotheses
    hres = llm.json("hypothesize", prompts.HYPOTHESIS,
                    f"Research question: {question}\n\nEvidence map:\n{json.dumps(emap, indent=1)}\n\n"
                    f"Claims:\n{claim_lines}\n\nTeam memory (earlier projects):\n{memory_txt}\n\n"
                    f"Propose {opts.n_hypotheses} hypotheses.",
                    max_tokens=8000, context={"question": question, "claims": claims, "n": opts.n_hypotheses})
    hs = lab.save_hypotheses(slug, hres.get("hypotheses", [])[:opts.n_hypotheses])
    emit("hypothesis-agent", f"proposed {len(hs['ids'])} hypotheses" +
         (f" ({len(hs['warnings'])} citation warnings)" if hs["warnings"] else ""))

    # 8. critique
    review_input = lab.get_hypotheses_for_review(slug)["hypotheses"]
    crit = llm.json("critique", prompts.CRITIC,
                    f"Research question: {question}\n\nHypotheses:\n{json.dumps(review_input, indent=1)}",
                    max_tokens=6000, context={"hypotheses": review_input})
    ranking = lab.save_reviews(slug, crit.get("reviews", []))["ranking"]
    active = [h for h in ranking if h["status"] != "rejected"]
    emit("critic-agent", f"{len(active)} hypotheses survive review, {len(ranking) - len(active)} rejected "
                         f"(kept as negative knowledge)")

    run = proj.read_json("run.json", {})
    run["llm_usage"] = llm.usage
    run["source_errors"] = res["source_errors"]
    proj.write_json("run.json", run)
    m = run["memory"]
    emit("brain", f"done. Memory: {m['papers']} papers, {m['claims']} claims, {m['hypotheses']} hypotheses, "
                  f"{m['projects']} projects")
    proj.commit("brain", "autopilot run complete")
    return proj


# ------------------------------------------------------------------ renderers

def _cite(ids: list[str], by_id: dict) -> str:
    return ", ".join(f"{i} ({by_id[i]['ref']})" for i in ids if i in by_id)


def render_evidence_map(question: str, emap: dict, claims: list[dict]) -> str:
    by_id = {c["id"]: c for c in claims}
    out = [f"# Evidence map\n\n**Question:** {question}\n\n{emap.get('summary', '')}\n"]
    out.append("## Themes")
    for t in emap.get("themes", []):
        out.append(f"- **{t.get('name')}**: {t.get('summary')} [{_cite(t.get('claim_ids', []), by_id)}]")
    out.append("\n## Consensus")
    for c in emap.get("consensus", []):
        out.append(f"- {c.get('statement')} [{_cite(c.get('claim_ids', []), by_id)}]")
    out.append("\n## Contradictions")
    for c in emap.get("contradictions", []):
        out.append(f"- {c.get('statement')} [{_cite(c.get('claim_ids', []), by_id)}]  \n"
                   f"  _Possible explanation:_ {c.get('possible_explanation', '')}")
    out.append("\n## Gaps")
    for g in emap.get("gaps", []):
        out.append(f"- **{g.get('gap')}** {g.get('why_it_matters', '')} "
                   f"[{_cite(g.get('related_claim_ids', []), by_id)}]")
    out.append("\n## Claims")
    for c in claims:
        out.append(f"- **{c['id']}** ({c['ref']}, {c['year']}, {c['strength']}): {c['claim']}  \n"
                   f"  > \"{c['quote']}\" ([{c['paper_title']}]({c['url']}))")
    return "\n".join(out) + "\n"


def render_hypothesis(h: dict, by_id: dict) -> str:
    exp = h.get("experiment") or {}
    r = h.get("review") or {}
    lines = [f"# {h['id']}: {h['statement']}\n",
             f"**Status:** {h['status']} | **Prior confidence:** {h['confidence']:.0%} | "
             f"novelty {h.get('novelty')}/5, feasibility {h.get('feasibility')}/5, "
             f"testability {h.get('testability')}/5\n"]
    if h.get("original_statement"):
        lines.append(f"_Original statement:_ {h['original_statement']}\n")
    lines += [f"## Rationale\n{h.get('rationale', '')}\n",
              f"**Gap addressed:** {h.get('gap_addressed', '')}\n",
              f"**Prediction:** {h.get('prediction', '')}\n",
              f"**Falsified if:** {h.get('falsification', '')}\n",
              f"## Proposed experiment ({exp.get('type', '?')})\n{exp.get('description', '')}\n\n"
              f"Data/tools: {exp.get('data_or_tools', '')}\n",
              "## Supporting evidence"]
    for cid in h.get("supporting_claim_ids", []):
        c = by_id[cid]
        lines.append(f"- **{cid}** {c['claim']}  \n  > \"{c['quote']}\" "
                     f"([{c['paper_title']}]({c['url']}), {c['year']})")
    lines.append(f"\n## Critic review: {r.get('verdict', 'n/a')}\n{r.get('comment', '')}")
    for issue in r.get("issues", []):
        lines.append(f"- {issue}")
    return "\n".join(lines) + "\n"


def render_report(question: str, plan: dict, emap: dict, hyps: list[dict], run: dict) -> str:
    lines = [f"# {plan.get('refined_question') or question}\n",
             f"Papers found {run['papers_found']} → screened {run['papers_screened']} → used "
             f"{run['papers_used']} · {run['claims']} grounded claims · "
             f"{run['hypotheses_active']}/{run['hypotheses']} hypotheses survived review\n",
             "## What the literature says\n", emap.get("summary", ""), "\n## Hypotheses\n",
             "| ID | Status | Score | Prior | Hypothesis |", "|---|---|---|---|---|"]
    for h in hyps:
        lines.append(f"| [{h['id']}](hypotheses/{h['id']}.md) | {h['status']} | {h['score']} | "
                     f"{h['confidence']:.0%} | {h['statement']} |")
    lines.append("\n## Gaps\n")
    lines += [f"- {g.get('gap')}" for g in emap.get("gaps", [])]
    lines.append("\nSee [evidence map](literature/evidence_map.md) and [lab notebook](notebook.md).")
    return "\n".join(lines) + "\n"
