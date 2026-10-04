"""The experiment: train a ligand-based classifier on known labels, rank untested
compounds, and measure how much better than random the top of the list is.

This is the "Experiment Runner" step. It reports:
  - ROC-AUC on a scaffold-disjoint held-out set
  - enrichment factor at 1% / 5% (actives in the top slice vs the base rate)
  - two controls: a random-ranking baseline and a y-scramble (labels shuffled)

Enrichment factor is the number the pitch is built on: EF = (hit rate in top k%) / (overall hit rate).
An EF of 10 means the shortlist is 10x richer in true actives than picking at random,
so you run ~10x fewer physical assays per confirmed hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .chem import bemis_murcko_scaffold, featurize


def scaffold_split(smiles: list[str], labels: list[int], test_frac: float = 0.3,
                   seed: int = 0) -> tuple[list[int], list[int]]:
    """Split indices so that no Bemis-Murcko scaffold appears in both train and test.

    Whole scaffold groups are assigned to test (largest-first, deterministic) until
    the test fraction is reached. Acyclic molecules ('' scaffold) each count as their
    own singleton group.
    """
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        scaf = bemis_murcko_scaffold(smi) or f"__singleton_{i}"
        groups.setdefault(scaf, []).append(i)
    ordered = sorted(groups.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    n_target = int(round(test_frac * len(smiles)))
    test: list[int] = []
    # interleave assignment a little so test isn't only the biggest families
    for grp in ordered:
        if len(test) < n_target:
            test.extend(grp)
    test_set = set(test)
    train = [i for i in range(len(smiles)) if i not in test_set]
    rng.shuffle(train)
    return train, sorted(test_set)


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    """Actives found in the top `frac` of the ranked list, relative to random."""
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-scores)
    top_hit_rate = y_true[order[:k]].mean()
    base_rate = y_true.mean()
    return float(top_hit_rate / base_rate) if base_rate > 0 else 0.0


@dataclass
class ScreenResult:
    n_compounds: int
    n_active: int
    base_rate: float
    n_train: int
    n_test: int
    roc_auc: float
    ef_top1pct: float
    ef_top5pct: float
    control_random_ef1: float
    control_yscramble_auc: float
    assays_saved_factor: float          # = ef_top1pct, reads as "x fewer assays per hit"
    top_hits: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # --- figure data -----------------------------------------------------
    # Measured curves for the exported report. These are computed from the
    # same held-out predictions as the scalars above, so a figure and the
    # number beside it can never disagree. Nothing here is modelled or
    # smoothed: a report must not draw a curve the run did not produce.
    roc_curve: list[list[float]] = field(default_factory=list)       # [[fpr, tpr], ...]
    enrichment_curve: list[list[float]] = field(default_factory=list)  # [[depth%, EF], ...]
    random_curve: list[list[float]] = field(default_factory=list)      # same depths, random ranking
    score_bins: list[float] = field(default_factory=list)              # histogram bin centres
    score_hist_active: list[int] = field(default_factory=list)
    score_hist_inactive: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def run_screen(records: list[dict], seed: int = 0, n_estimators: int = 300,
               dump_ranked_path: str | None = None) -> ScreenResult:
    """records: [{cid, smiles, label}]. Trains on a scaffold-disjoint train split and
    evaluates ranking on the held-out split (our stand-in for 'untested' compounds).

    If dump_ranked_path is set, writes the full held-out ranking as [{s,y}] (sorted by
    score, desc) there — used to drive the Relay app's live enrichment slider without
    bloating the tool's LLM-facing return value."""
    smiles = [r["smiles"] for r in records]
    labels = [int(r["label"]) for r in records]
    X, kept = featurize(smiles)
    notes = []
    if len(kept) < len(records):
        notes.append(f"{len(records) - len(kept)} compounds failed to parse and were dropped")
    smiles = [smiles[i] for i in kept]
    labels = [labels[i] for i in kept]
    recs = [records[i] for i in kept]
    y = np.array(labels)

    train, test = scaffold_split(smiles, labels, seed=seed)
    # guard: both splits need both classes
    if y[train].sum() in (0, len(train)) or y[test].sum() in (0, len(test)):
        notes.append("scaffold split left a split single-class; falling back to random split")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        train, test = sorted(idx[:cut]), sorted(idx[cut:])

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(X[train], y[train])
    scores = clf.predict_proba(X[test])[:, 1]
    y_test = y[test]

    auc = float(roc_auc_score(y_test, scores)) if len(set(y_test)) > 1 else float("nan")
    ef1 = enrichment_factor(y_test, scores, 0.01)
    ef5 = enrichment_factor(y_test, scores, 0.05)

    # control 1: random ranking, averaged over 20 draws (single draws are noisy)
    rng = np.random.default_rng(seed + 1)
    ef1_rand = float(np.mean([enrichment_factor(y_test, rng.random(len(y_test)), 0.01)
                              for _ in range(20)]))

    # control 2: y-scramble -- shuffle TRAIN labels, retrain; AUC should collapse to ~0.5
    y_scr = y[train].copy()
    rng.shuffle(y_scr)
    clf_scr = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf_scr.fit(X[train], y_scr)
    scores_scr = clf_scr.predict_proba(X[test])[:, 1]
    auc_scr = float(roc_auc_score(y_test, scores_scr)) if len(set(y_test)) > 1 else float("nan")

    # --- measured curves for the report -----------------------------------
    # Every point below comes from (y_test, scores). Depths are capped at the
    # test-set size, and a depth that would select fewer than one compound is
    # skipped rather than reported as zero.
    def _roc_points(y_true, sc, max_points: int = 120):
        from sklearn.metrics import roc_curve as _rc

        if len(set(y_true)) < 2:
            return []
        fpr, tpr, _ = _rc(y_true, sc)
        step = max(1, len(fpr) // max_points)
        pts = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)]
               for i in range(0, len(fpr), step)]
        if pts[-1] != [1.0, 1.0]:
            pts.append([1.0, 1.0])
        return pts

    depths = [d for d in (0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100)
              if int(round(d / 100 * len(y_test))) >= 1]
    ef_curve = [[d, round(enrichment_factor(y_test, scores, d / 100), 3)] for d in depths]
    rng_c = np.random.default_rng(seed + 2)
    rand_curve = [[d, round(float(np.mean([enrichment_factor(y_test, rng_c.random(len(y_test)), d / 100)
                                           for _ in range(10)])), 3)] for d in depths]

    n_bins = 20
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = [round(float((edges[i] + edges[i + 1]) / 2), 3) for i in range(n_bins)]
    hist_act = np.histogram(scores[y_test == 1], bins=edges)[0].tolist()
    hist_inact = np.histogram(scores[y_test == 0], bins=edges)[0].tolist()

    order = np.argsort(-scores)
    # Spread of the ensemble's own votes on each shortlisted compound — a real
    # measure of how much the forest disagrees with itself, not a placeholder.
    try:
        top_idx = order[:10]
        per_tree = np.array([t.predict_proba(X[test][top_idx])[:, 1] for t in clf.estimators_])
        top_std = per_tree.std(axis=0)
    except Exception:
        top_std = np.zeros(len(order[:10]))
    top_hits = [{"cid": recs[test[i]]["cid"], "smiles": recs[test[i]]["smiles"],
                 "score": round(float(scores[i]), 3), "label": int(y_test[i]),
                 "score_std": round(float(top_std[rank]), 3)}
                for rank, i in enumerate(order[:10])]
    if dump_ranked_path:
        import json as _json
        ranked = [{"s": round(float(scores[i]), 3), "y": int(y_test[i])} for i in order]
        with open(dump_ranked_path, "w") as _f:
            _json.dump(ranked, _f)

    return ScreenResult(
        n_compounds=len(y), n_active=int(y.sum()), base_rate=round(float(y_test.mean()), 4),
        n_train=len(train), n_test=len(test), roc_auc=round(auc, 3),
        ef_top1pct=round(ef1, 2), ef_top5pct=round(ef5, 2),
        control_random_ef1=round(ef1_rand, 2), control_yscramble_auc=round(auc_scr, 3),
        assays_saved_factor=round(ef1, 2), top_hits=top_hits, notes=notes,
        roc_curve=_roc_points(y_test, scores),
        enrichment_curve=ef_curve, random_curve=rand_curve,
        score_bins=centres, score_hist_active=hist_act, score_hist_inactive=hist_inact,
    )
