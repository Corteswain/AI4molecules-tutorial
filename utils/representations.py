"""Molecular representation functions for the AI4molecules tutorial.

Each function turns a list of SMILES strings into a 2D numpy array of
features (one row per molecule). They all share the same signature so they
can be swapped in and out via REPRESENTATIONS.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors


def _mols_from_smiles(smiles_list):
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
        mols.append(mol)
    return mols


def featurize_morgan_fingerprint(smiles_list, radius=2, n_bits=2048):
    """Morgan (circular) fingerprints: bit vectors encoding local substructures."""
    mols = _mols_from_smiles(smiles_list)
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [generator.GetFingerprintAsNumPy(mol) for mol in mols]
    return np.array(fps, dtype=np.float32)


def featurize_maccs_keys(smiles_list):
    """MACCS keys: a fixed 166-bit dictionary of predefined structural patterns."""
    mols = _mols_from_smiles(smiles_list)
    fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    return np.array(fps, dtype=np.float32)


# A small, interpretable set of whole-molecule physicochemical descriptors.
_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
}


def featurize_rdkit_descriptors(smiles_list):
    """A handful of global physicochemical descriptors (molecular weight, LogP, TPSA, ...)."""
    mols = _mols_from_smiles(smiles_list)
    rows = [[func(mol) for func in _DESCRIPTOR_FUNCS.values()] for mol in mols]
    return np.array(rows, dtype=np.float32)


featurize_rdkit_descriptors.feature_names = list(_DESCRIPTOR_FUNCS.keys())


REPRESENTATIONS = {
    "morgan": featurize_morgan_fingerprint,
    "maccs": featurize_maccs_keys,
    "rdkit_descriptors": featurize_rdkit_descriptors,
}
