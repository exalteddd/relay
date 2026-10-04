"""Cheminformatics: fingerprints and Bemis-Murcko scaffolds.

Kept separate so it can be unit-tested without network or a model.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # silence parse warnings

FP_BITS = 2048
FP_RADIUS = 2


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """Morgan (ECFP4-like) bit vector as a float array, or None if unparseable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def bemis_murcko_scaffold(smiles: str) -> str:
    """Generic Bemis-Murcko scaffold SMILES; '' for acyclic/unparseable.

    Grouping train/test by this (never sharing a scaffold across the split) is the
    honesty control: the model cannot win by memorizing a chemical series.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic)
    except Exception:
        return ""


def featurize(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (X, kept_indices): fingerprint matrix and the indices that parsed."""
    rows, kept = [], []
    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi)
        if fp is not None:
            rows.append(fp)
            kept.append(i)
    if not rows:
        return np.zeros((0, FP_BITS), dtype=np.float32), []
    return np.vstack(rows), kept
