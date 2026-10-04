"""Streamlit viewer for the lab's research record and memory.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from brain.config import load_config
from brain.memory import Brain
from brain.pipeline import Options, run_research
from brain.project import Project

st.set_page_config(page_title="Second Brain Lab", layout="wide")
cfg = load_config()
brain = Brain(cfg.db_path)

STATUS = {"proposed": "🟢 proposed", "revised": "🟡 revised", "rejected": "🔴 rejected"}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧠 Second Brain Lab")
    s = brain.stats()
    c1, c2 = st.columns(2)
    c1.metric("Projects", s["projects"])
    c2.metric("Papers", s["papers"])
    c1.metric("Claims", s["claims"])
    c2.metric("Hypotheses", s["hypotheses"])

    with st.expander("New research (autopilot)", expanded=not s["projects"]):
        q = st.text_area("Research question")
        n_h = st.slider("Hypotheses", 3, 10, 6)
        n_p = st.slider("Max papers", 8, 40, 25)
        if st.button("Run", type="primary", disabled=not q.strip()):
            with st.status("Agents at work…", expanded=True) as status:
                def ev(agent, msg):
                    status.write(f"**{agent}** · {msg}")
                try:
                    proj = run_research(q.strip(), cfg, Options(n_hypotheses=n_h, max_papers=n_p), on_event=ev)
                    status.update(label="Done", state="complete")
                    st.session_state["project"] = proj.slug
                    st.rerun()
                except Exception as e:  # surface the failure in the UI
                    status.update(label=f"Failed: {e}", state="error")

    projects = brain.projects()
    slugs = [p["slug"] for p in projects]
    if not slugs:
        st.info("No projects yet.")
        st.stop()
    default = slugs.index(st.session_state.get("project", slugs[0])) if st.session_state.get("project") in slugs else 0
    slug = st.selectbox("Project", slugs, index=default,
                        format_func=lambda s: next(p["question"][:60] for p in projects if p["slug"] == s))

proj = Project(cfg.projects_dir / slug)
plan = proj.read_json("plan.json", {}) or {}
claims = proj.read_json("literature/claims.json", []) or []
papers = proj.read_json("literature/papers.json", []) or []
hyps = proj.read_json("hypotheses/hypotheses.json", []) or []
emap = proj.read_json("literature/evidence_map.json", {}) or {}
by_claim = {c["id"]: c for c in claims}

st.header(plan.get("refined_question") or proj.read_text("question.md").split("\n", 2)[-1])
m = st.columns(5)
m[0].metric("Candidates", len(proj.read_json("literature/candidates.json", []) or []))
m[1].metric("Papers used", len(papers))
m[2].metric("Grounded claims", len(claims))
m[3].metric("Hypotheses", len(hyps))
m[4].metric("Survived critic", sum(h.get("status") != "rejected" for h in hyps))

tabs = st.tabs(["Hypotheses", "Evidence map", "Papers", "Research record", "Memory recall"])

with tabs[0]:
    if not hyps:
        st.info("No hypotheses yet.")
    for h in hyps:
        title = f"{h['id']} · {STATUS.get(h.get('status'), h.get('status'))} · score {h.get('score', '–')} · " \
                f"prior {h.get('confidence', 0):.0%} — {h['statement']}"
        with st.expander(title, expanded=h is hyps[0]):
            st.caption("Agent-generated hypothesis, not validated.")
            if h.get("original_statement"):
                st.markdown(f"~~{h['original_statement']}~~")
            st.markdown(f"**Rationale:** {h.get('rationale', '')}")
            a, b = st.columns(2)
            a.markdown(f"**Prediction:** {h.get('prediction', '')}")
            b.markdown(f"**Falsified if:** {h.get('falsification', '')}")
            exp = h.get("experiment") or {}
            st.markdown(f"**Proposed experiment ({exp.get('type', '?')}):** {exp.get('description', '')}  \n"
                        f"_Data/tools:_ {exp.get('data_or_tools', '')}")
            st.markdown("**Supporting evidence**")
            for cid in h.get("supporting_claim_ids", []):
                c = by_claim.get(cid)
                if c:
                    st.markdown(f"- **{cid}** {c['claim']}  \n  > “{c['quote']}”  \n  "
                                f"[{c['paper_title']}]({c['url']}) ({c['year']})")
            r = h.get("review") or {}
            if r:
                st.markdown(f"**Critic: {r.get('verdict', '')}** — {r.get('comment', '')}")
                for i in r.get("issues", []):
                    st.markdown(f"- {i}")

with tabs[1]:
    st.write(emap.get("summary", ""))
    for section, key in (("Contradictions", "contradictions"), ("Gaps", "gaps"), ("Consensus", "consensus"),
                         ("Themes", "themes")):
        st.subheader(section)
        for item in emap.get(key, []):
            text = item.get("statement") or item.get("gap") or f"**{item.get('name')}**: {item.get('summary')}"
            ids = item.get("claim_ids") or item.get("related_claim_ids") or []
            st.markdown(f"- {text} `{' '.join(ids)}`")
    st.subheader("Claims")
    st.dataframe([{k: c[k] for k in ("id", "ref", "year", "strength", "direction", "system", "claim")}
                  for c in claims], width="stretch", hide_index=True)

with tabs[2]:
    st.dataframe([{"ref": p["ref"], "relevance": p.get("relevance"), "year": p["year"], "title": p["title"],
                   "citations": p["citations"], "sources": ", ".join(p["sources"]), "url": p["url"]}
                  for p in papers], width="stretch", hide_index=True,
                 column_config={"url": st.column_config.LinkColumn()})

with tabs[3]:
    a, b = st.columns([2, 3])
    with a:
        st.subheader("Agent commits")
        for c in proj.history():
            st.markdown(f"`{c['hash']}` **{c['agent']}** · {c['message']}  \n<small>{c['when']}</small>",
                        unsafe_allow_html=True)
    with b:
        st.subheader("Lab notebook")
        st.markdown(proj.read_text("notebook.md"))

with tabs[4]:
    rq = st.text_input("Ask the brain what it already knows")
    if rq:
        r = brain.recall(rq)
        st.markdown("**Hypotheses**")
        for h in r["hypotheses"]:
            st.markdown(f"- [{h['project']}] ({h['status']}) {h['statement']}")
        st.markdown("**Claims**")
        for c in r["claims"]:
            st.markdown(f"- [{c['project']}] {c['text']} — _{c['paper_title']}_")
        st.markdown("**Papers**")
        for p in r["papers"]:
            st.markdown(f"- [{p['title']}]({p['url']}) ({p['year']})")
