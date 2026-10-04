"""Command line: python -m brain <command>"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load_config
from .memory import Brain
from .pipeline import Options, run_research
from .project import Project

COLORS = {"brain": "35", "memory-agent": "36", "scoping-agent": "34", "literature-agent": "33",
          "screening-agent": "33", "extraction-agent": "32", "synthesis-agent": "32",
          "hypothesis-agent": "95", "critic-agent": "31"}


def _event(agent: str, msg: str) -> None:
    c = COLORS.get(agent, "0")
    print(f"\033[{c}m{agent:>17}\033[0m  {msg}", flush=True)


def cmd_research(a, cfg):
    opts = Options(sources=a.sources.split(",") if a.sources else None, per_query=a.per_query,
                   max_papers=a.max_papers, n_hypotheses=a.hypotheses)
    t0 = time.time()
    proj = run_research(a.question, cfg, opts, on_event=_event)
    hyps = proj.read_json("hypotheses/hypotheses.json", [])
    print(f"\nDone in {time.time() - t0:.0f}s → {proj.root}\n")
    for h in hyps:
        mark = "✗" if h["status"] == "rejected" else "•"
        print(f" {mark} {h['id']} [{h['status']}, score {h['score']}, prior {h['confidence']:.0%}] {h['statement']}")
    print(f"\nReport: {proj.root / 'README.md'}")


def cmd_list(a, cfg):
    brain = Brain(cfg.db_path)
    for p in brain.projects():
        print(f"{p['slug']}\n    {p['question']}")
    print(f"\nMemory: {brain.stats()}")


def cmd_show(a, cfg):
    root = cfg.projects_dir / a.project
    if not root.exists():
        sys.exit(f"no project {a.project}")
    print(Project(root).read_text("README.md"))


def cmd_recall(a, cfg):
    r = Brain(cfg.db_path).recall(a.query, k=a.k)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print("Hypotheses:")
    for h in r["hypotheses"]:
        print(f"  [{h['project']}] ({h['status']}) {h['statement']}")
    print("Claims:")
    for c in r["claims"]:
        print(f"  [{c['project']}] {c['text']}  — {c['paper_title']}")
    print("Papers:")
    for p in r["papers"]:
        print(f"  {p['title']} ({p['year']})  {p['url']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="brain", description="Second Brain research memory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("research", help="research a question: literature → claims → hypotheses")
    r.add_argument("question")
    r.add_argument("--sources", help="comma list: openalex,semantic_scholar,arxiv,europepmc")
    r.add_argument("--per-query", type=int, default=20)
    r.add_argument("--max-papers", type=int, default=25)
    r.add_argument("--hypotheses", type=int, default=6)
    sub.add_parser("list", help="list projects and memory size")
    s = sub.add_parser("show", help="print a project report")
    s.add_argument("project")
    rc = sub.add_parser("recall", help="search everything the brain remembers")
    rc.add_argument("query")
    rc.add_argument("-k", type=int, default=8)
    rc.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    {"research": cmd_research, "list": cmd_list, "show": cmd_show, "recall": cmd_recall}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
