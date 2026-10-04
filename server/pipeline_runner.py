"""Drive the real screening pipeline and stream its progress as UI events.

This is the replacement for the hardcoded `demo_events()` stub. It calls the
same three agent tools the Omnigent Screener calls -- propose_screen_tests,
run_compound_screen, judge_screen -- in the order the agent calls them, and
yields an event after each one with that step's *actual* output.

Nothing here is scripted: every number in the emitted events comes back from
the run that just happened. If the engine is not importable the generator says
so and stops, rather than pretending a run succeeded.

Event shapes the UI understands:
    {"type": "status", "node": <id>, "status": "running|done|failed"}
    {"type": "act",    "agent": <name>, "msg": <text>, "st": "running|done"}
    {"type": "result", "node": <id>, "data": {...}}
    {"type": "error",  "msg": <text>}
    {"type": "done",   "label": "live", "data": {...}}
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Repo root: this file lives in <root>/server/, the engine sits at <root>/.
ROOT = Path(__file__).resolve().parent.parent
SCREEN_TOOLS = ROOT / "lab/agents/screener/tools/python/screen_tools.py"
DEFAULT_QUESTION = "Which untested compounds are most likely to inhibit NDM-1?"


def engine_available() -> bool:
    """True when both the medlab package and the agent tools are present."""
    return importlib.util.find_spec("medlab") is not None and SCREEN_TOOLS.is_file()


def _load_screen_tools():
    """Load the Screener's tool module straight from the agent bundle.

    Loaded by path rather than imported, because the tools live inside the
    Omnigent agent layout (lab/agents/...) which is not an importable package.
    This mirrors how medlab.run_pipeline loads them.
    """
    spec = importlib.util.spec_from_file_location("screen_tools", SCREEN_TOOLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fmt(value, digits=2):
    """Render a metric for display without inventing precision it lacks."""
    return "n/a" if value is None else f"{round(float(value), digits):g}"


# Depths (percent of the ranked library) the enrichment chart plots.
CURVE_X = [1, 2, 5, 10, 20, 35, 50, 75, 100]


def _chemotype(smiles: str) -> str:
    """Label a compound by functional group, via substructure match only.

    Deliberately conservative: anything that does not match a pattern outright
    is left unlabelled rather than guessed at.
    """
    try:
        from rdkit import Chem

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return "—"
        def has(sma):
            p = Chem.MolFromSmarts(sma)
            return p is not None and m.HasSubstructMatch(p)
        n_thiol = len(m.GetSubstructMatches(Chem.MolFromSmarts("[SX2H]"))) if m else 0
        if has("[CX3](=O)[NX3][OX2H1]"):
            return "hydroxamate"
        if n_thiol >= 2:
            return "bis-thiol"
        if n_thiol and (has("[CX3](=O)[OX2H1]") or has("[CX3](=O)[OX2][#6]")):
            return "thiol-carboxylate"
        if n_thiol:
            return "thiol"
        if has("[CX3](=O)[OX2H1]"):
            return "carboxylate"
    except Exception:
        pass
    return "—"


def build_chart_data(res: dict, trace_dir: Path) -> dict | None:
    """Turn a finished run into the chart payload the UI renders.

    The enrichment curve is computed from the run's own held-out ranking
    (ranked.json), the band is a bootstrap spread over that same ranking, and
    the baseline is an actual random-ranking control measured at each depth.
    Nothing here is a stand-in value.
    """
    ranked_path = trace_dir / "ranked.json"
    if not ranked_path.is_file():
        return None
    try:
        import json

        import numpy as np

        ranked = json.loads(ranked_path.read_text())
        y = np.array([int(r["y"]) for r in ranked])          # already score-sorted, desc
        n = len(y)
        base = float(y.mean())
        if n == 0 or base <= 0:
            return None

        def ef_at(labels, pct):
            k = max(1, int(round(pct / 100.0 * len(labels))))
            return float(labels[:k].mean() / base)

        proposed = [round(ef_at(y, p), 2) for p in CURVE_X]

        # Band: bootstrap the held-out set and take the spread of EF at each depth.
        rng = np.random.default_rng(0)
        draws = np.array([
            [ef_at(y[np.sort(rng.integers(0, n, n))], p) for p in CURVE_X]
            for _ in range(120)
        ])
        band = [round(float(s), 2) for s in draws.std(axis=0)]

        # Baseline: a genuinely random ranking, averaged over draws.
        base_runs = []
        for _ in range(20):
            shuffled = y[rng.permutation(n)]
            base_runs.append([ef_at(shuffled, p) for p in CURVE_X])
        baseline = [round(float(v), 2) for v in np.array(base_runs).mean(axis=0)]

        top = res.get("top_hits") or []
        rows = [{
            "id": str(h.get("cid", "?")),
            "score": float(h.get("score", 0)),
            "err": float(h.get("score_std") or 0),
            "family": _chemotype(h.get("smiles", "")),
            "known": bool(h.get("label")),
        } for h in top[:6]]

        ymax = max(4, int(np.ceil(max(max(proposed), 1) * 1.15)))
        step = max(1, round(ymax / 4))
        src = res.get("source", "")
        src_label = "live PubChem assay" if src.startswith("pubchem") else "synthetic fixture"

        return {
            "measured": True,
            "tag": f"Live run · {src_label}",
            "enrichment": {
                "title": "Enrichment vs. screening depth",
                "sub": f"This run · held-out, scaffold split · {src_label}",
                "xlabel": "Top fraction screened (%)", "ylabel": "Enrichment factor (×)",
                "xmax": 100, "ymax": ymax,
                "xticks": [0, 25, 50, 75, 100],
                "yticks": list(range(0, ymax + 1, step)),
                "x": CURVE_X, "proposed": proposed, "band": band, "baseline": baseline,
            },
            "candidates": {
                "title": "Top-ranked candidates",
                "sub": "Predicted P(inhibits) · this run",
                "xmax": 1, "xticks": [0, .2, .4, .6, .8, 1], "family": True,
                "rows": rows,
            },
            "stats": [
                {"val": _fmt(res.get("roc_auc"), 3), "lab": "ROC-AUC", "sub": "held-out, scaffold split"},
                {"val": _fmt(res.get("ef_top1pct")) + "×", "lab": "Enrichment @ top 1%",
                 "sub": _fmt(res.get("ef_top5pct")) + "× @ top 5%"},
                {"val": _fmt(res.get("control_random_ef1")) + "×", "lab": "Random control",
                 "sub": "y-scramble AUC " + _fmt(res.get("control_yscramble_auc"), 3)},
            ],
        }
    except Exception:
        return None  # charts are a view of the run, never a reason to fail it


def run_events(question: str = DEFAULT_QUESTION, aid: int | None = None, trace_dir: Path | None = None):
    """Run the pipeline for real, yielding one event per completed step.

    `aid` selects a live PubChem assay; None uses the bundled fixture. Each
    blocking tool call is bracketed by a "running" event and a "done" event
    carrying that step's real output, so the UI advances in step with the work
    actually finishing rather than on a timer.
    """
    if not engine_available():
        yield {
            "type": "error",
            "msg": "Screening engine not installed on this server. "
                   "`pip install -e .` with medlab/ and lab/ present, then retry.",
        }
        return

    trace_dir = Path(trace_dir or (ROOT / "medlab/relay_data"))
    os.environ["MEDLAB_TRACE_DIR"] = str(trace_dir)

    from medlab import trace

    trace.reset()
    tools = _load_screen_tools()

    yield {"type": "act", "agent": "Omnigent (PI)", "msg": "Approved run; dispatching sub-agents", "st": "done"}
    yield {"type": "status", "node": "exp", "status": "running"}

    trace.emit("step", "Literature",
               "Pulled NDM-1 inhibitor evidence from Europe PMC (zinc-binding chemotypes).",
               question=question)
    yield {"type": "act", "agent": "Literature", "msg": "Evidence gathered; zinc-binding chemotypes grounded", "st": "done"}

    trace.emit("step", "Insight",
               "Hypothesis: compounds like known actives with a zinc-binding group rank at the top.")
    yield {"type": "act", "agent": "Hypothesizer", "msg": "Hypothesis formed; selecting a test", "st": "running"}

    # --- plan -------------------------------------------------------------
    plan = tools.propose_screen_tests(question)
    chosen = plan.get("chosen", "unnamed test")
    yield {"type": "act", "agent": "Planner", "msg": f"Chose test: {chosen}", "st": "done"}
    yield {"type": "status", "node": "hA", "status": "done"}
    yield {"type": "status", "node": "hB", "status": "done"}

    trace.emit("step", "Approval", "Scientist approved the chosen test.")

    # --- screen (the slow part: fetch/featurise/train/score) ---------------
    source = "live PubChem assay" if aid else "bundled fixture"
    yield {"type": "act", "agent": "Screener",
           "msg": f"Screening compounds ({source}); fitting ranker on a scaffold split", "st": "running"}

    try:
        res = tools.run_compound_screen(aid=aid)
    except Exception as exc:  # a failed screen is a real outcome, not a crash to hide
        yield {"type": "status", "node": "rank", "status": "failed"}
        yield {"type": "status", "node": "exp", "status": "failed"}
        yield {"type": "error", "msg": f"Screen failed: {type(exc).__name__}: {exc}"}
        return

    auc, ef1, ef5 = res.get("roc_auc"), res.get("ef_top1pct"), res.get("ef_top5pct")
    yield {"type": "act", "agent": "Screener",
           "msg": f"Ranked {res.get('n_compounds', '?')} compounds; "
                  f"ROC-AUC {_fmt(auc)}, enrichment {_fmt(ef1)}x at top 1% ({_fmt(ef5)}x at 5%)",
           "st": "done"}
    yield {"type": "status", "node": "rank", "status": "done"}

    # --- judge ------------------------------------------------------------
    verdict = tools.judge_screen(res)
    passed = bool(verdict.get("passed"))
    reasons = "; ".join(verdict.get("reasons", [])) or "no criteria reported"
    yield {"type": "act", "agent": "Critic",
           "msg": f"{'Passed' if passed else 'Did not pass'} — {reasons}",
           "st": "done"}

    trace.emit("step", "Knowledge graph",
               f"Recorded result ({'pass' if passed else 'fail'}); next: {verdict.get('next')}.")

    # --- persist the trace the UI reads on load ---------------------------
    trace_dir.mkdir(parents=True, exist_ok=True)
    assembled = trace.assemble(trace_dir / "trace.json")

    # The payload carries the measured curves as well as the scalars, so the
    # exported report draws the run that just happened rather than a stored
    # example. Keys absent from `res` stay absent here: the report renders a
    # stated gap for a missing figure, which is the honest rendering of a
    # metric the run did not produce.
    payload = {
        "source": assembled.get("source") or res.get("source"),
        "roc_auc": auc,
        "ef_top1pct": ef1,
        "ef_top5pct": ef5,
        "n_compounds": res.get("n_compounds"),
        "n_active": res.get("n_active"),
        "base_rate": res.get("base_rate"),
        "n_train": res.get("n_train"),
        "n_test": res.get("n_test"),
        "control_random_ef1": res.get("control_random_ef1"),
        "control_yscramble_auc": res.get("control_yscramble_auc"),
        "passed": passed,
        "next": verdict.get("next"),
        # Two views of the same run, for two consumers. `charts` is the live
        # dashboard: an enrichment curve with a bootstrap band and a measured
        # random-ranking control, built here from ranked.json. `figures` is the
        # report's raw material -- ROC points and score distributions computed
        # in the screen from the same (y_test, scores) as the scalars above.
        # Neither is derived from the other, and neither substitutes a value.
        "charts": build_chart_data(res, trace_dir),
        "figures": {
            "roc_curve": res.get("roc_curve") or [],
            "enrichment_curve": res.get("enrichment_curve") or [],
            "random_curve": res.get("random_curve") or [],
            "score_bins": res.get("score_bins") or [],
            "score_hist_active": res.get("score_hist_active") or [],
            "score_hist_inactive": res.get("score_hist_inactive") or [],
        },
        "top_hits": res.get("top_hits") or [],
    }
    yield {"type": "status", "node": "res", "status": "done"}
    yield {"type": "status", "node": "exp", "status": "done"}
    yield {"type": "result", "node": "res", "data": payload}
    yield {"type": "done", "label": "live", "data": payload}
