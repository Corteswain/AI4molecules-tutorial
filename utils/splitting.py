"""Dataset splitting functions for the AI4molecules tutorial.

Each function takes a DataFrame and returns (train_idx, val_idx, test_idx): integer
positional indices into the DataFrame. Returning indices (rather than the split
DataFrames themselves) lets you compute a representation once for the whole dataset and
then slice the DataFrame and the feature matrix the same way. `test_idx` is a *true
holdout*: nothing in this tutorial trains, tunes, or picks a checkpoint using it — it's
only ever touched once, at final evaluation. `val_idx` is what every model uses instead
for anything that needs feedback during development (Chemprop's early stopping, or your
own informal checks). The functions differ in *how* molecules are assigned to each set,
which changes how optimistic or realistic the resulting performance estimate is.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans

from .representations import featurize_morgan_fingerprint


def random_split(df, smiles_col="smiles", val_size=0.15, test_size=0.15, seed=42):
    """Simple i.i.d. shuffle-and-split: no relationship between molecules is considered."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    n_val = int(len(df) * val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(df, smiles_col="smiles", val_size=0.15, test_size=0.15, seed=42, tolerance=0.05):
    """Group molecules by Bemis-Murcko scaffold; whole scaffold groups go to train, val, or test.

    This keeps near-identical molecules (same core, different substituents) on the
    same side of the split, giving a more honest estimate of generalization to new
    chemotypes than a random split.
    """
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col]):
        scaffolds.setdefault(_murcko_scaffold(smi), []).append(i)

    labels = np.empty(len(df), dtype=int)
    for group_id, members in enumerate(scaffolds.values()):
        for i in members:
            labels[i] = group_id
    return _split_by_cluster_labels(labels, val_size, test_size, seed, tolerance=tolerance)


def kmeans_split(df, smiles_col="smiles", val_size=0.15, test_size=0.15, n_clusters=20, seed=42, tolerance=0.05):
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
    return _split_by_cluster_labels(labels, val_size, test_size, seed, tolerance=tolerance)


def butina_cluster_ids(smiles_list, cutoff=0.815, radius=2, n_bits=2048):
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


def butina_split(df, smiles_col="smiles", val_size=0.15, test_size=0.15, cutoff=0.815, seed=42, tolerance=0.05):
    """Cluster molecules by Tanimoto similarity (Butina algorithm), then hold out whole clusters.

    Same idea as kmeans_split, but using the standard cheminformatics notion of
    molecular similarity (Tanimoto distance on Morgan fingerprints) instead of Euclidean
    distance, and without needing to pre-choose a number of clusters — Butina finds
    clusters directly from pairwise similarity.
    """
    labels = butina_cluster_ids(df[smiles_col].tolist(), cutoff=cutoff)
    return _split_by_cluster_labels(labels, val_size, test_size, seed, tolerance=tolerance)


def _peel_holdout(labels, pool_idx, target_size, n_total, seed, tolerance, max_consecutive_rejections, max_restarts):
    """Select whole groups (per `labels`) from `pool_idx` to form a holdout set whose size,
    as a fraction of `n_total`, lands within [target_size - tolerance, target_size + tolerance].

    Groups are drawn in random order from the pool. A group that would push the holdout
    set *above* the tolerance band is rejected (skipped, left in the remaining pool)
    rather than accepted outright — unlike a plain "stop once we've reached the target"
    rule, which lets a single large group massively overshoot the target in one step. If
    max_consecutive_rejections candidates in a row are all rejected (a sign the remaining
    groups don't fit what's left of the band), the whole attempt is discarded and
    restarted from an empty holdout set with a fresh random order.

    Returns (remaining_pool_idx, holdout_idx).
    """
    lower = (target_size - tolerance) * n_total
    upper = (target_size + tolerance) * n_total

    groups = {}
    for i in pool_idx:
        groups.setdefault(labels[i], []).append(i)
    group_ids_all = list(groups.keys())

    rng = np.random.RandomState(seed)
    for _ in range(max_restarts):
        order = group_ids_all.copy()
        rng.shuffle(order)

        holdout = []
        used = set()
        consecutive_rejections = 0
        for gid in order:
            new_size = len(holdout) + len(groups[gid])
            if new_size > upper:
                consecutive_rejections += 1
                if consecutive_rejections >= max_consecutive_rejections:
                    break  # stuck: abandon this attempt and reshuffle from scratch
                continue  # reject this group, try the next one

            consecutive_rejections = 0
            holdout.extend(groups[gid])
            used.add(gid)
            if len(holdout) >= lower:
                remaining = [i for gid2 in group_ids_all if gid2 not in used for i in groups[gid2]]
                return np.array(sorted(remaining)), np.array(sorted(holdout))
        # exhausted the pool (or gave up after too many rejections) without landing in
        # the band: fall through and retry with a new shuffle

    raise RuntimeError(
        f"Could not land a holdout set within {tolerance:.0%} of target_size={target_size:.0%} "
        f"after {max_restarts} attempts. Try a larger `tolerance`, or a cutoff/n_clusters "
        "that produces less lopsided group sizes."
    )


def _split_by_cluster_labels(
    labels, val_size, test_size, seed, tolerance=0.05, max_consecutive_rejections=5, max_restarts=1000
):
    """Peel a test holdout, then a val holdout, off the full dataset (see _peel_holdout),
    leaving the remainder as train. Both val_size and test_size are fractions of the full
    dataset — sizes don't get renormalized against the shrinking pool — so, e.g.,
    val_size=test_size=0.15 always aims for roughly 70/15/15 train/val/test regardless of
    which is peeled first.
    """
    n = len(labels)
    kwargs = dict(
        n_total=n, tolerance=tolerance,
        max_consecutive_rejections=max_consecutive_rejections, max_restarts=max_restarts,
    )
    remaining_idx, test_idx = _peel_holdout(labels, np.arange(n), test_size, seed=seed, **kwargs)
    train_idx, val_idx = _peel_holdout(labels, remaining_idx, val_size, seed=seed + 1, **kwargs)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


SPLITTERS = {
    "random": random_split,
    "scaffold": scaffold_split,
    "kmeans": kmeans_split,
    "butina": butina_split,
}
