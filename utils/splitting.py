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
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
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


def kmeans_split(df, smiles_col="smiles", test_size=0.2, n_clusters=20, seed=42, tolerance=0.05):
    """Cluster molecules (Morgan fingerprints + KMeans), then hold out whole clusters.

    Like scaffold_split, this tests extrapolation to structurally distinct regions of
    chemical space rather than interpolation within a familiar neighborhood, but the
    notion of "similar" comes from Euclidean distance between fingerprint vectors rather
    than an exact scaffold match. Because KMeans clusters on Euclidean distance while
    Butina (below) clusters on Tanimoto similarity, the two methods can disagree near
    cluster boundaries: two molecules "close" by one metric aren't guaranteed to be
    "close" by the other, so this split doesn't guarantee purity under, e.g., Butina
    clustering.
    """
    X = featurize_morgan_fingerprint(df[smiles_col].tolist())
    k = max(1, min(n_clusters, len(df) // 5))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    return _split_by_cluster_labels(labels, test_size, seed, tolerance=tolerance)


def butina_cluster_ids(smiles_list, cutoff=0.6, radius=2, n_bits=2048):
    """Cluster molecules by Tanimoto similarity of Morgan fingerprints (Butina algorithm).

    Returns one cluster id per molecule. `cutoff` is a Tanimoto *distance* threshold
    (1 - similarity): lower values require closer similarity to cluster together and
    yield more, smaller clusters.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [generator.GetFingerprint(m) for m in mols]

    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    cluster_id = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_id[m] = cid
    return cluster_id


def butina_split(df, smiles_col="smiles", test_size=0.2, cutoff=0.6, seed=42, tolerance=0.05):
    """Cluster molecules by Tanimoto similarity (Butina algorithm), then hold out whole clusters.

    Same idea as kmeans_split, but using the standard cheminformatics notion of
    molecular similarity (Tanimoto distance on Morgan fingerprints) instead of Euclidean
    distance, and without needing to pre-choose a number of clusters — Butina finds
    clusters directly from pairwise similarity.
    """
    labels = butina_cluster_ids(df[smiles_col].tolist(), cutoff=cutoff)
    return _split_by_cluster_labels(labels, test_size, seed, tolerance=tolerance)


def _split_by_cluster_labels(labels, test_size, seed, tolerance=0.05, max_consecutive_rejections=5, max_restarts=1000):
    """Greedily add whole clusters to the test set until its size lands within
    [test_size - tolerance, test_size + tolerance] of the dataset.

    Clusters are drawn in random order. A cluster that would push the test set *above*
    the tolerance band is rejected (skipped, left for train) rather than accepted
    outright — unlike a plain "stop once we've reached the target" rule, which lets a
    single large cluster massively overshoot the target in one step. If
    max_consecutive_rejections candidates in a row are all rejected (a sign the
    remaining clusters don't fit what's left of the band), the whole attempt is
    discarded and restarted from an empty test set with a fresh random order.
    """
    n = len(labels)
    lower = (test_size - tolerance) * n
    upper = (test_size + tolerance) * n

    cluster_ids_all = list(range(labels.max() + 1))
    members_by_cluster = {cid: np.where(labels == cid)[0].tolist() for cid in cluster_ids_all}

    rng = np.random.RandomState(seed)
    for _ in range(max_restarts):
        order = cluster_ids_all.copy()
        rng.shuffle(order)

        test_idx = []
        used = set()
        consecutive_rejections = 0
        for cid in order:
            new_size = len(test_idx) + len(members_by_cluster[cid])
            if new_size > upper:
                consecutive_rejections += 1
                if consecutive_rejections >= max_consecutive_rejections:
                    break  # stuck: abandon this attempt and reshuffle from scratch
                continue  # reject this cluster, try the next one

            consecutive_rejections = 0
            test_idx.extend(members_by_cluster[cid])
            used.add(cid)
            if len(test_idx) >= lower:
                train_idx = [i for cid2 in cluster_ids_all if cid2 not in used for i in members_by_cluster[cid2]]
                return np.sort(train_idx), np.sort(test_idx)
        # exhausted all clusters (or gave up after too many rejections) without landing
        # in the band: fall through and retry with a new shuffle

    raise RuntimeError(
        f"Could not land the test set within {tolerance:.0%} of test_size={test_size:.0%} "
        f"after {max_restarts} attempts. Try a larger `tolerance`, or a cutoff/n_clusters "
        "that produces less lopsided cluster sizes."
    )


SPLITTERS = {
    "random": random_split,
    "scaffold": scaffold_split,
    "kmeans": kmeans_split,
    "butina": butina_split,
}
