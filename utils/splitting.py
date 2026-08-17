"""Dataset splitting functions for the AI4molecules tutorial.

Each function takes a DataFrame and returns (train_idx, test_idx): integer
positional indices into the DataFrame. Returning indices (rather than the
split DataFrames themselves) lets you compute a representation once for the
whole dataset and then slice both the DataFrame and the feature matrix the
same way. The functions differ in *how* molecules are assigned to train vs.
test, which changes how optimistic or realistic the resulting performance
estimate is.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans

from .representations import featurize_morgan_fingerprint


def random_split(df, smiles_col="smiles", test_size=0.2, seed=42):
    """Simple i.i.d. shuffle-and-split: no relationship between molecules is considered."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return np.sort(train_idx), np.sort(test_idx)


def _murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(df, smiles_col="smiles", test_size=0.2, seed=42):
    """Group molecules by Bemis-Murcko scaffold; whole scaffold groups go to train or test.

    This keeps near-identical molecules (same core, different substituents) on the
    same side of the split, giving a more honest estimate of generalization to new
    chemotypes than a random split.
    """
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col]):
        scaffolds.setdefault(_murcko_scaffold(smi), []).append(i)

    rng = np.random.RandomState(seed)
    groups = list(scaffolds.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # largest scaffold families stay intact in train

    n_test_target = int(len(df) * test_size)
    test_idx, train_idx = [], []
    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return np.sort(train_idx), np.sort(test_idx)


def cluster_split(df, smiles_col="smiles", test_size=0.2, n_clusters=20, seed=42):
    """Cluster molecules (Morgan fingerprints + KMeans), then hold out whole clusters.

    Like scaffold_split, this tests extrapolation to structurally distinct regions of
    chemical space rather than interpolation within a familiar neighborhood, but the
    notion of "similar" comes from fingerprint distance rather than an exact scaffold match.
    """
    X = featurize_morgan_fingerprint(df[smiles_col].tolist())
    k = max(1, min(n_clusters, len(df) // 5))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)

    rng = np.random.RandomState(seed)
    cluster_ids = list(range(k))
    rng.shuffle(cluster_ids)

    n_test_target = int(len(df) * test_size)
    test_idx, train_idx = [], []
    for cid in cluster_ids:
        members = np.where(labels == cid)[0].tolist()
        if len(test_idx) < n_test_target:
            test_idx.extend(members)
        else:
            train_idx.extend(members)
    return np.sort(train_idx), np.sort(test_idx)


SPLITTERS = {
    "random": random_split,
    "scaffold": scaffold_split,
    "cluster": cluster_split,
}
