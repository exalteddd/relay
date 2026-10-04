"""Tools for the NDM-1 Screener (Planner + Runner + Analyst over the medlab engine).

The heavy imports (rdkit, sklearn) are lazy, so the tool schema loads without them.
The engine itself lives in medlab/ and is unit-tested.
"""

try:  # the real decorator under Omnigent; a transparent no-op when driven directly
    from omnigent_client.tools import tool
except ModuleNotFoundError:
    def tool(fn=None, **_kw):
        def wrap(f):
            return f
        return wrap(fn) if callable(fn) else wrap


@tool(strict=False)
def propose_screen_tests(question: str) -> dict:
    """Propose two computational tests for an NDM-1 triage question and pick one,
    scoring each on learning, feasibility and cost (rank = learning * feasibility / cost).

    Args:
        question: The scientific question, e.g. "which untested compounds likely inhibit NDM-1".
    """
    t1 = {"id": "T1", "name": "Ligand-based ML on bioassay labels",
          "desc": "Morgan fingerprints + random forest trained on PubChem NDM-1 active/inactive "
                  "calls; rank an untested library. Scaffold-split so the model cannot memorize a series.",
          "learning": 5, "feasibility": 5, "cost": 1}
    t2 = {"id": "T2", "name": "Structure-based docking into the NDM-1 zinc site",
          "desc": "Dock candidates into an NDM-1 crystal structure (e.g. PDB 4EYL). Orthogonal "
                  "evidence, but metallo-enzyme zinc handling is slow and error-prone.",
          "learning": 3, "feasibility": 2, "cost": 4}
    for t in (t1, t2):
        t["rank"] = round(t["learning"] * t["feasibility"] / t["cost"], 2)
    chosen = max((t1, t2), key=lambda t: t["rank"])
    out = {"tests": [t1, t2], "chosen": chosen["id"],
           "rationale": f"{chosen['id']} wins on cost x feasibility and matches the measured "
                        "endpoint (assay labels); the other is kept as an orthogonal follow-up on the shortlist."}
    try:
        from medlab import trace
        trace.emit("step", "Planner", f"Proposed 2 tests; chose {chosen['id']} ({chosen['name']}).", question=question)
        trace.emit("plan", plan={"tests": [t1, t2], "chosen": chosen["id"], "rationale": out["rationale"]})
    except Exception:
        pass
    return out


@tool(strict=False)
def run_compound_screen(aid: int | None = None, max_compounds: int = 4000, seed: int = 0) -> dict:
    """Run the chosen triage: train on NDM-1 bioassay labels (PubChem AID if given and reachable,
    else a labeled synthetic smoke-test set) and rank a held-out library. Returns metrics and a shortlist.

    Args:
        aid: PubChem BioAssay ID for an NDM-1 inhibition assay, or null to use the offline fixture.
        max_compounds: cap on compounds fetched from PubChem.
        seed: random seed for reproducibility.
    """
    from medlab.data import load_pubchem_bioassay, make_synthetic_fixture
    from medlab.screen import run_screen
    from medlab import trace

    source = "synthetic_fixture"
    records = None
    if aid:
        try:
            records = load_pubchem_bioassay(aid, max_compounds=max_compounds)
            source = f"pubchem_aid_{aid}"
        except Exception:
            records = None
    if not records:
        records = make_synthetic_fixture()
    trace.emit("step", "Runner", f"Training on {len(records)} labeled compounds; ranking the library.")
    dump = None
    tdir = trace.trace_dir()
    if tdir:
        dump = str(tdir / "ranked.json")
    res = run_screen(records, seed=seed, dump_ranked_path=dump)
    out = {"source": source, **res.to_dict()}
    trace.emit("result", "Runner", "Ranking complete.", result=out)
    return out


@tool(strict=False)
def judge_screen(result: dict, min_auc: float = 0.7, min_ef_top1pct: float = 3.0) -> dict:
    """Judge a screen result against pre-set acceptance criteria and check the controls.

    Args:
        result: the dict returned by run_compound_screen.
        min_auc: minimum acceptable held-out ROC-AUC.
        min_ef_top1pct: minimum acceptable enrichment factor at the top 1%.
    """
    auc = result.get("roc_auc", 0) or 0
    ef = result.get("ef_top1pct", 0) or 0
    controls_ok = (result.get("control_random_ef1", 9) < 2.0
                   and result.get("control_yscramble_auc", 1) < 0.6)
    passed = auc >= min_auc and ef >= min_ef_top1pct and controls_ok
    try:
        from medlab import trace
        trace.emit("step", "Analysis", f"Judged: AUC {auc}, EF@1% {ef}, controls {'clean' if controls_ok else 'flagged'}.")
    except Exception:
        pass
    reasons = []
    if auc < min_auc:
        reasons.append(f"AUC {auc} below {min_auc}")
    if ef < min_ef_top1pct:
        reasons.append(f"EF@1% {ef} below {min_ef_top1pct}")
    if not controls_ok:
        reasons.append("controls failed (random or y-scramble not clean)")
    return {"passed": passed, "reasons": reasons or ["meets AUC, enrichment and control criteria"],
            "assays_saved_factor": ef,
            "next": ("cross-check the top shortlist with docking (T2) and propose wet-lab assays"
                     if passed else "revise features or widen the training set, then re-run")}
