"""Fast offline tests for the NDM-1 triage engine (no network, small model)."""

from medlab.chem import bemis_murcko_scaffold, featurize, morgan_fingerprint
from medlab.data import make_synthetic_fixture
from medlab.screen import enrichment_factor, run_screen, scaffold_split
import numpy as np


def test_fingerprint_and_scaffold():
    assert morgan_fingerprint("c1ccccc1C(=O)O").shape == (2048,)
    assert morgan_fingerprint("not_a_molecule") is None
    # thiomandelic-acid-like actives share a generic scaffold with benzoic acid (benzene ring)
    assert bemis_murcko_scaffold("O=C(O)C(S)Cc1ccccc1") != ""


def test_enrichment_math():
    # perfect ranking: all actives first -> EF = 1/base_rate
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # base rate 0.2
    perfect = np.array([9, 8, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    assert enrichment_factor(y, perfect, 0.2) == 5.0  # (2/2)/0.2
    assert enrichment_factor(y, np.zeros(10), 0.2) <= 5.0


def test_scaffold_split_disjoint():
    recs = make_synthetic_fixture(max_actives=60)
    smiles = [r["smiles"] for r in recs]
    labels = [r["label"] for r in recs]
    train, test = scaffold_split(smiles, labels, seed=0)
    assert set(train).isdisjoint(test)
    s_train = {bemis_murcko_scaffold(smiles[i]) or f"_{i}" for i in train}
    s_test = {bemis_murcko_scaffold(smiles[i]) or f"_{i}" for i in test}
    # named (ring) scaffolds must not straddle the split
    ring = {s for s in s_train if not s.startswith("_")}
    assert ring.isdisjoint({s for s in s_test if not s.startswith("_")})


def test_run_screen_learns_and_controls_behave():
    recs = make_synthetic_fixture(max_actives=80)
    res = run_screen(recs, seed=0, n_estimators=80)
    # the model learns real structure
    assert res.roc_auc > 0.65
    assert res.ef_top5pct > 1.5
    # controls: random ranking far below the model; y-scramble destroys the signal
    assert res.control_random_ef1 < res.ef_top1pct / 2
    assert res.control_yscramble_auc < res.roc_auc
    # output is JSON-serializable and carries a shortlist
    import json

    json.dumps(res.to_dict())
    assert len(res.top_hits) > 0
