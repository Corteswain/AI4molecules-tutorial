"""Diagnostic plots for the AI4molecules tutorial.

Unlike representations.py / splitting.py / models.py, these functions render a plot
directly (they call plt.show()) rather than just returning data — they're meant to be
called as a single cell in the notebook right after a decision is made.
"""

import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .representations import featurize_morgan_fingerprint
from .splitting import butina_cluster_ids

# Shared palette (viridis) so train/val always render in the same two colors
# everywhere in the tutorial, not just within this module.
TRAIN_COLOR = plt.cm.viridis(0.15)
VAL_COLOR = plt.cm.viridis(0.85)


def low_dimensional_representation(df, train_idx, val_idx, smiles_col="smiles", cluster_cutoff=0.815, seed=7):
    """Visualize where train vs. val molecules fall in chemical space.

    Clusters the whole dataset with Butina clustering (Tanimoto similarity of Morgan
    fingerprints) as a structural reference — the same method used by butina_split
    — then shows train/val membership on t-SNE and PCA projections side by side. Also
    reports what fraction of Butina clusters end up "mixed" (containing both train and
    val molecules): a high mixed fraction means train and val are drawn from the same
    structural neighborhoods (as with a random split, or kmeans_split disagreeing
    with Butina near its own cluster boundaries); a low mixed fraction means val
    molecules sit in their own structurally distinct regions (as scaffold/cluster splits
    are designed to produce). Picking butina_split as the split method should
    drive this close to 0%, since it's the same clustering shown here.
    """
    smiles_list = df[smiles_col].tolist()
    X = featurize_morgan_fingerprint(smiles_list)

    cluster_id = butina_cluster_ids(smiles_list, cutoff=cluster_cutoff)
    n_clusters = cluster_id.max() + 1

    is_val = np.zeros(len(df), dtype=bool)
    is_val[val_idx] = True

    mixed = 0
    for cid in range(n_clusters):
        members_val = is_val[cluster_id == cid]
        if members_val.any() and not members_val.all():
            mixed += 1
    frac_mixed = mixed / n_clusters

    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    tsne_coords = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, title in [(axes[0], tsne_coords, "t-SNE"), (axes[1], pca_coords, "PCA")]:
        ax.scatter(coords[~is_val, 0], coords[~is_val, 1], s=14, alpha=0.6, label="train", color=TRAIN_COLOR)
        ax.scatter(coords[is_val, 0], coords[is_val, 1], s=14, alpha=0.8, label="val", color=VAL_COLOR)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Train/val split over chemical space  —  {n_clusters} Butina clusters "
        f"(cutoff={cluster_cutoff}), {frac_mixed:.0%} contain both train and val"
    )
    plt.tight_layout()
    plt.show()

    print(f"{mixed}/{n_clusters} Butina clusters ({frac_mixed:.0%}) contain both train and val molecules.")
    return {"n_clusters": int(n_clusters), "frac_mixed_clusters": float(frac_mixed)}


def nearest_neighbor_similarity(df, splits, smiles_col="smiles", k=5):
    """Box plot: for each split method, how Tanimoto-similar is each val molecule to its
    k-th nearest neighbor in that method's training set?

    A high k-th-neighbor similarity means val molecules typically have several close
    analogs already in train — an easy, interpolation-heavy validation. A low value means
    val molecules are genuinely unfamiliar relative to train — the split is testing
    extrapolation. Using the *k*-th neighbor rather than the 1st makes this a measure of
    how much of a val molecule's local neighborhood made it into train, not just whether
    a single near-duplicate slipped through.

    Args:
        df: the full DataFrame every split in `splits` was computed from.
        splits: dict mapping split method name -> (train_idx, val_idx), e.g. the
            `splits` dict built by running every split method in Step 1.
        smiles_col: SMILES column name.
        k: which nearest neighbor to report (default: 5th nearest).
    """
    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(m) for m in mols]

    labels = list(splits.keys())
    data = []
    for name in labels:
        train_idx, val_idx = splits[name]
        train_fps = [fps[i] for i in train_idx]
        kth_sims = []
        for i in val_idx:
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
            sims.sort(reverse=True)
            kth_sims.append(sims[min(k, len(sims)) - 1])
        data.append(kth_sims)

    colors = [plt.cm.viridis(x) for x in np.linspace(0.15, 0.85, len(labels))]
    positions = np.arange(1, len(labels) + 1)
    box_pos = positions - 0.15
    violin_pos = positions + 0.20

    fig, ax = plt.subplots(figsize=(9, 5))

    # Violin (KDE shape) to the right of each box — reveals double peaks, skew, etc.
    # that the box plot's five-number summary alone can't show.
    violin = ax.violinplot(data, positions=violin_pos, widths=0.35, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
        body.set_edgecolor("none")
    violin["cmedians"].set_color("black")

    # Box plot to the left, at its usual narrow width, for precise quartiles/outliers.
    box = ax.boxplot(data, positions=box_pos, widths=0.22, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Tanimoto similarity to {k}th-nearest train neighbor")
    ax.set_title(f"Val-set similarity to training set ({k}-NN Tanimoto)")
    plt.tight_layout()
    plt.show()

    stats = {name: {"median": float(np.median(vals)), "mean": float(np.mean(vals))} for name, vals in zip(labels, data)}
    for name, s in stats.items():
        print(f"{name:10s}: median={s['median']:.3f}  mean={s['mean']:.3f}")
    return stats
