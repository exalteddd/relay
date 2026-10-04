"""Compound + label sources for the NDM-1 triage.

Two sources, one interface (a list of {cid, smiles, label} dicts, label 1=active, 0=inactive):

  load_pubchem_bioassay(aid)  -- REAL NDM-1 labels from PubChem BioAssay (needs internet).
  make_synthetic_fixture()    -- a clearly-labeled SYNTHETIC smoke-test set so the
                                 pipeline runs and is testable offline. NOT NDM-1 data.

The synthetic labels come from a transparent chemical rule (presence of known
metal/zinc-binding groups -- thiol, carboxylate, hydroxamate, azole-thiol), which
is exactly the kind of signal a real metallo-beta-lactamase inhibitor screen rewards.
It tests that featurize -> scaffold-split -> learn -> rank works; it does not make a
scientific claim about any compound.
"""

from __future__ import annotations

import json
import random
import urllib.request

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 40


# ---------------------------------------------------------------- real data
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "medlab-ndm1/0.1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def load_pubchem_bioassay(aid: int, max_compounds: int | None = None) -> list[dict]:
    """Fetch active/inactive calls for a PubChem assay, then resolve CID -> SMILES.

    Works wherever PubChem is reachable. Raises on network failure so the caller can
    fall back to the fixture.
    """
    concise = _get(f"{PUBCHEM}/assay/aid/{aid}/concise/JSON")
    table = concise["Table"]
    cols = table["Columns"]["Column"]
    rows = table["Row"]
    try:
        cid_i = cols.index("CID")
        act_i = cols.index("Activity Outcome")
    except ValueError as e:
        raise RuntimeError(f"unexpected assay columns: {cols}") from e

    labels: dict[int, int] = {}
    for row in rows:
        cell = row["Cell"]
        cid, outcome = cell[cid_i], cell[act_i]
        if not cid or outcome not in ("Active", "Inactive"):
            continue
        labels[int(cid)] = 1 if outcome == "Active" else 0
    cids = list(labels)
    if max_compounds:
        cids = cids[:max_compounds]

    out = []
    for i in range(0, len(cids), 100):  # resolve SMILES in batches
        batch = cids[i : i + 100]
        ids = ",".join(map(str, batch))
        props = _get(f"{PUBCHEM}/compound/cid/{ids}/property/CanonicalSMILES/JSON")
        for p in props["PropertyTable"]["Properties"]:
            cid = int(p["CID"])
            smi = p.get("CanonicalSMILES")
            if smi:
                out.append({"cid": cid, "smiles": smi, "label": labels[cid]})
    return out


# ---------------------------------------------------------------- synthetic fixture
# Diverse drug-like ring systems, each with one substitution slot {R}.
_CORES = [
    "c1ccc({R})cc1", "c1ccc({R})nc1", "c1ccc2ccccc2c1{R}", "c1ccc2[nH]ccc2c1{R}",
    "c1ccc2ncccc2c1{R}", "C1CCN({R})CC1", "c1csc({R})c1", "c1cc({R})ccn1",
    "c1ccc(-c2ccc({R})cc2)cc1", "C1CCC({R})CC1", "c1cnc({R})nc1", "c1ccc({R})o1",
    "O=C(N{R})c1ccccc1", "c1ccc({R})cc1OC", "c1ccc({R})c(Cl)c1", "C1COC({R})CO1",
    "C1CCCCC1{R}", "C1CCCC1{R}", "C1CCCCCC1{R}", "c1ccc2occc2c1{R}",
    "c1ccc2sccc2c1{R}", "c1ccc2c(c1)cccc2{R}", "C1CCN2CCCC2C1{R}", "c1ccc2[nH]ncc2c1{R}",
    "c1cnc2[nH]ccc2c1{R}", "O=C1CCC(N1{R})=O", "c1ccoc1{R}", "C1CCOC1{R}",
    "c1cc2ccccc2cc1{R}", "C1CN({R})CCN1", "c1ccc2c(c1)OCCO2{R}", "c1cc({R})cc2ccccc12",
]
# Zinc-binding groups -> drive the (synthetic) "active" label. These mirror real
# metallo-beta-lactamase inhibitor chemotypes (thiols, carboxylates, hydroxamates).
_BINDERS = ["S", "C(=O)O", "C(=O)NO", "c1n[nH]c(S)n1", "CC(S)C(=O)O"]
_BINDER_TAILS = ["", "C", "CC", "Cc1ccccc1", "CCO", "CC(=O)O"]
# Neutral groups -> "inactive".
_NEUTRAL = ["C", "OC", "N", "F", "Cl", "C(C)C", "N(C)C", "CO", "C#N", "C(F)(F)F",
            "CC", "CCC", "Br", "OCC", "C(C)(C)C", "NC(C)=O", "S(C)(=O)=O", "OCCO"]


def _canon(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def make_synthetic_fixture(seed: int = 7, noise: float = 0.05,
                           target_active_frac: float = 0.05,
                           max_actives: int = 300) -> list[dict]:
    """Build a labeled, scaffold-diverse synthetic library that mimics a real primary
    screen: a large, dilute pool with a low active rate (~2%). NOT real assay data.

    Actives carry a zinc-binding motif; inactives are neutral decorations. Inactives
    are padded (two-substituent combos) until the active fraction hits the target, so
    enrichment is measured in the regime real high-throughput screens live in.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    actives: list[dict] = []
    inactives: list[dict] = []
    cid = [900_000_000]  # obviously-fake CID range

    def make(core: str, sub: str, label: int, bucket: list[dict]):
        smi = _canon(core.replace("{R}", sub))
        if not smi or smi in seen:
            return
        seen.add(smi)
        y = label if rng.random() > noise else 1 - label  # label noise
        bucket.append({"cid": cid[0], "smiles": smi, "label": y, "synthetic": True})
        cid[0] += 1

    for core in _CORES:
        for b in _BINDERS:
            for tail in _BINDER_TAILS:
                make(core, b + tail, 1, actives)
    for core in _CORES:                        # single-substituent inactives
        for n in _NEUTRAL:
            make(core, n, 0, inactives)
    # cap actives for a fast, realistically-sized demo library
    rng.shuffle(actives)
    actives = actives[:max_actives]

    # how many inactives we need to hit the target active fraction
    n_pos = sum(r["label"] for r in actives)
    n_total_target = int(round(n_pos / target_active_frac))
    n_inactive_keep = max(len(inactives), n_total_target - len(actives))

    # top up with random neutral alkyl/ether/halide chains (guaranteed-unique, clean)
    guard = 0
    while len(inactives) < n_inactive_keep and guard < n_inactive_keep * 40:
        guard += 1
        core = rng.choice(_CORES)
        k = rng.randint(1, 8)
        sub = "".join(rng.choice(["C", "C", "O", "N", "F", "Cl"]) for _ in range(k))
        if sub[0] in "ON":           # keep a carbon attachment to the ring
            sub = "C" + sub
        make(core, sub, 0, inactives)

    rng.shuffle(inactives)
    inactives = inactives[:n_inactive_keep]
    out = actives + inactives
    rng.shuffle(out)
    return out
