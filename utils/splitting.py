"""Dataset splitting functions for the AI4molecules tutorial.

Each function takes a DataFrame and returns (train_idx, val_idx): integer positional
indices into the DataFrame. Returning indices (rather than the split DataFrames
themselves) lets you compute a representation once for the whole dataset and then slice
the DataFrame and the feature matrix the same way. This is the *only* split the notebook
does — everything here (train and val) is data the tutorial actually looks at, including
for hyperparameter selection in Step 4. The real, external evaluation set
(data/real.csv, made by scripts/make_holdout.py) is never touched by any of this. The
functions below differ in *how* molecules are assigned to train vs. val, which changes
how optimistic or realistic val performance is as a stand-in for genuinely new molecules.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans

from .representations import featurize_morgan_fingerprint


def random_split(df, smiles_col="smiles", val_size=0.2, seed=42):
    """Simple i.i.d. shuffle-and-split: no relationship between molecules is considered."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_size)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return np.sort(train_idx), np.sort(val_idx)


def _murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(df, smiles_col="smiles", val_size=0.2, seed=42, tolerance=0.05):
    """Group molecules by Bemis-Murcko scaffold; whole scaffold groups go to train or val.

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
    return _split_by_cluster_labels(labels, val_size, seed, tolerance=tolerance)


def kmeans_split(df, smiles_col="smiles", val_size=0.2, n_clusters=20, seed=42, tolerance=0.05):
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
    return _split_by_cluster_labels(labels, val_size, seed, tolerance=tolerance)


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


def butina_split(df, smiles_col="smiles", val_size=0.2, cutoff=0.815, seed=42, tolerance=0.05):
    """Cluster molecules by Tanimoto similarity (Butina algorithm), then hold out whole clusters.

    Same idea as kmeans_split, but using the standard cheminformatics notion of
    molecular similarity (Tanimoto distance on Morgan fingerprints) instead of Euclidean
    distance, and without needing to pre-choose a number of clusters — Butina finds
    clusters directly from pairwise similarity.
    """
    labels = butina_cluster_ids(df[smiles_col].tolist(), cutoff=cutoff)
    return _split_by_cluster_labels(labels, val_size, seed, tolerance=tolerance)


def butina_holdout_split(df, smiles_col="smiles", test_size=0.15, seed=42, tolerance=0.05, cutoff=0.815):
    """Cluster molecules by Tanimoto similarity (Butina algorithm) and hold out whole
    clusters as a single *external* holdout.

    Unlike butina_split (used for the notebook's own train/val comparison), this returns
    (data_idx, holdout_idx) — meant for carving a fixed set-aside evaluation set out of
    the raw dataset *before* the tutorial starts (see scripts/make_holdout.py), kept
    completely separate from everything the notebook touches.
    """
    labels = butina_cluster_ids(df[smiles_col].tolist(), cutoff=cutoff)
    n = len(df)
    data_idx, holdout_idx = _peel_holdout(
        labels, np.arange(n), test_size, n_total=n, seed=seed,
        tolerance=tolerance, max_consecutive_rejections=5, max_restarts=1000,
    )
    return np.sort(data_idx), np.sort(holdout_idx)


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


def _split_by_cluster_labels(labels, val_size, seed, tolerance=0.05, max_consecutive_rejections=5, max_restarts=1000):
    """Peel a val holdout off the full dataset (see _peel_holdout), leaving the remainder as train."""
    n = len(labels)
    train_idx, val_idx = _peel_holdout(
        labels, np.arange(n), val_size, n_total=n, seed=seed,
        tolerance=tolerance, max_consecutive_rejections=max_consecutive_rejections, max_restarts=max_restarts,
    )
    return np.sort(train_idx), np.sort(val_idx)


SPLITTERS = {
    "random": random_split,
    "scaffold": scaffold_split,
    "kmeans": kmeans_split,
    "butina": butina_split,
}


def _group_labels(df, method, smiles_col="smiles", seed=42):
    """Per-molecule group id for `method`'s notion of "similar molecules" — the same grouping
    random_split/scaffold_split/kmeans_split/butina_split use to decide what can and can't be
    split across train/val. Used by nested_cv_splits, which needs the same grouping to build
    k-fold partitions instead of a single train/val split; computed independently here (rather
    than shared with the four functions above) so their exact, already-tested train/val output
    for a given seed can't shift as a side effect of this addition.
    """
    if method == "random":
        return np.arange(len(df))
    if method == "scaffold":
        scaffolds = {}
        for i, smi in enumerate(df[smiles_col]):
            scaffolds.setdefault(_murcko_scaffold(smi), []).append(i)
        labels = np.empty(len(df), dtype=int)
        for group_id, members in enumerate(scaffolds.values()):
            for i in members:
                labels[i] = group_id
        return labels
    if method == "kmeans":
        X = featurize_morgan_fingerprint(df[smiles_col].tolist())
        k = max(1, min(20, len(df) // 5))
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    if method == "butina":
        return butina_cluster_ids(df[smiles_col].tolist())
    raise ValueError(f"Unknown split method: {method!r}")


def _balanced_kfold_partition(labels, pool_idx, k, seed):
    """Partition pool_idx into k folds of roughly equal total size, keeping whole groups (per
    `labels`) together.

    Greedy longest-processing-time bin-packing: shuffle group order (so ties between
    equal-size groups aren't broken the same way every time), sort groups largest-first, and
    drop each one into whichever fold is currently smallest. Unlike _peel_holdout's
    reject/restart approach (built for a single holdout within a tolerance band), this always
    terminates in one pass and naturally balances all k folds at once.
    """
    groups = {}
    for i in pool_idx:
        groups.setdefault(labels[i], []).append(i)
    group_members = list(groups.values())

    rng = np.random.RandomState(seed)
    rng.shuffle(group_members)
    group_members.sort(key=len, reverse=True)

    folds = [[] for _ in range(k)]
    for members in group_members:
        target = min(range(k), key=lambda f: len(folds[f]))
        folds[target].extend(members)

    return [np.array(sorted(f)) for f in folds]


def nested_cv_splits(df, method, smiles_col="smiles", k_outer=5, k_inner=5, seed=42):
    """Build a k_outer x k_inner nested split for cross-validation, grouped the same way
    <method>_split groups molecules (whole scaffolds/clusters kept together; `random` treats
    every molecule as its own group).

    Returns a list of k_outer dicts: {"test_idx": array, "inner_splits": [(train_idx, val_idx), ...]}
    with k_inner (train_idx, val_idx) pairs per outer fold. Each outer test fold partitions the
    full dataset; each inner split further partitions the *other* k_outer - 1 folds (never the
    current test fold) into k_inner train/val pairs, the same way Step 1's single split is
    built — just repeated k_outer * k_inner times so each test fold gets k_inner independently
    trained models instead of one.
    """
    labels = _group_labels(df, method, smiles_col=smiles_col, seed=seed)
    n = len(df)
    outer_folds = _balanced_kfold_partition(labels, np.arange(n), k_outer, seed)

    result = []
    for i, test_idx in enumerate(outer_folds):
        test_set = set(test_idx.tolist())
        pool_idx = np.array([j for j in range(n) if j not in test_set])
        inner_folds = _balanced_kfold_partition(labels, pool_idx, k_inner, seed + i + 1)
        inner_splits = []
        for val_idx in inner_folds:
            val_set = set(val_idx.tolist())
            train_idx = np.array([j for j in pool_idx if j not in val_set])
            inner_splits.append((train_idx, val_idx))
        result.append({"test_idx": test_idx, "inner_splits": inner_splits})
    return result
