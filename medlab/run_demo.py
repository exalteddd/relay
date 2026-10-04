"""End-to-end NDM-1 triage demo.

Tries real PubChem NDM-1 bioassay data; falls back to the labeled synthetic fixture
when PubChem is unreachable (e.g. a locked-down sandbox). Prints a report and writes
run.json.

    python -m medlab.run_demo --aid 1259411
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .data import load_pubchem_bioassay, make_synthetic_fixture
from .screen import run_screen


def main(argv=None):
    ap = argparse.ArgumentParser(description="NDM-1 in-silico triage demo")
    ap.add_argument("--aid", type=int, default=None, help="PubChem assay ID for NDM-1 inhibition")
    ap.add_argument("--max", type=int, default=4000, help="max compounds from PubChem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="run.json")
    a = ap.parse_args(argv)

    source = "synthetic_fixture"
    records = None
    t0 = time.time()
    if a.aid:
        try:
            print(f"Fetching PubChem assay AID {a.aid} ...", flush=True)
            records = load_pubchem_bioassay(a.aid, max_compounds=a.max)
            source = f"pubchem_aid_{a.aid}"
            print(f"  got {len(records)} labeled compounds")
        except Exception as e:
            print(f"  PubChem unavailable ({str(e)[:70]}); using synthetic fixture", flush=True)
    if not records:
        records = make_synthetic_fixture()
        print(f"Synthetic fixture: {len(records)} compounds "
              f"(SYNTHETIC smoke test, not NDM-1 data)")

    res = run_screen(records, seed=a.seed)
    elapsed = time.time() - t0

    print("\n" + "=" * 58)
    print(f"  NDM-1 triage result   [source: {source}]")
    print("=" * 58)
    print(f"  compounds screened     {res.n_compounds}")
    print(f"  active base rate       {res.base_rate:.1%}")
    print(f"  scaffold-split test    {res.n_test} held-out compounds")
    print(f"  held-out ROC-AUC       {res.roc_auc}")
    print(f"  enrichment @ top 1%    {res.ef_top1pct}x   (@5%: {res.ef_top5pct}x)")
    print(f"  --> ~{res.assays_saved_factor}x fewer assays per confirmed hit vs random")
    print(f"  CONTROL random rank    EF {res.control_random_ef1}x  (should be ~1)")
    print(f"  CONTROL y-scramble     AUC {res.control_yscramble_auc}  (should be ~0.5)")
    print(f"  wall-clock             {elapsed:.1f}s")
    for n in res.notes:
        print(f"  note: {n}")
    print("\n  top-ranked shortlist (first 5):")
    for h in res.top_hits[:5]:
        print(f"    cid {h['cid']}  score {h['score']}  {h['smiles'][:46]}")

    payload = {"source": source, "elapsed_s": round(elapsed, 1), **res.to_dict()}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
