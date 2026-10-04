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

    order = np.argsort(-scores)
    top_hits = [{"cid": recs[test[i]]["cid"], "smiles": recs[test[i]]["smiles"],
                 "score": round(float(scores[i]), 3), "label": int(y_test[i])}
                for i in order[:10]]
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
    )
