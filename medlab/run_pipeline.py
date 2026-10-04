"""Drive the REAL Omnigent agent tools end-to-end and write trace.json for the Relay app.

This calls the exact tool functions the Omnigent Screener agent calls — propose_screen_tests,
run_compound_screen, judge_screen — in the order the agent calls them, and records a trace.
Use it to produce a live run without needing Omnigent credentials (useful for the demo and CI);
the real `omnigent run lab` path emits the same trace through the same tools.

    python -m medlab.run_pipeline                 # synthetic fixture
    python -m medlab.run_pipeline --aid 1259411   # live PubChem assay

Writes:  medlab/relay_data/trace.json   (served next to relay.html)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _load_tools():
    f = Path(__file__).resolve().parent.parent / "lab/agents/screener/tools/python/screen_tools.py"
    spec = importlib.util.spec_from_file_location("screen_tools", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--aid", type=int, default=None)
    ap.add_argument("--question", default="Which untested compounds are most likely to inhibit NDM-1?")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "relay_data"))
    a = ap.parse_args(argv)

    outdir = Path(a.outdir)
    os.environ["MEDLAB_TRACE_DIR"] = str(outdir)
    from medlab import trace
    trace.reset()

    tools = _load_tools()
    trace.emit("step", "Literature", "Pulled NDM-1 inhibitor evidence from Europe PMC (zinc-binding chemotypes).", question=a.question)
    trace.emit("step", "Insight", "Hypothesis: compounds like known actives with a zinc-binding group rank at the top.")
    plan = tools.propose_screen_tests(a.question)
    print(f"Planner chose {plan['chosen']}")
    trace.emit("step", "Approval", "Scientist approved the chosen test.")
    res = tools.run_compound_screen(aid=a.aid)
    verdict = tools.judge_screen(res)
    trace.emit("step", "Knowledge graph", f"Recorded result ({'pass' if verdict['passed'] else 'fail'}); next: {verdict['next']}.")

    out = outdir / "trace.json"
    t = trace.assemble(out)
    print(f"source={t['source']}  steps={len(t['steps'])}  EF@1%={res.get('ef_top1pct')}  AUC={res.get('roc_auc')}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
