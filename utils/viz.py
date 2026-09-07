"""Diagnostic plots for the AI4molecules tutorial.

Unlike representations.py / splitting.py / models.py, these functions render a plot
directly (they call plt.show()) rather than just returning data — they're meant to be
called as a single cell in the notebook right after a decision is made.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .representations import featurize_morgan_fingerprint
from .splitting import butina_cluster_ids

# Shared palette (viridis) so train/test always render in the same two colors
# everywhere in the tutorial, not just within this module.
TRAIN_COLOR = plt.cm.viridis(0.15)
TEST_COLOR = plt.cm.viridis(0.85)


def low_dimensional_representation(df, train_idx, test_idx, smiles_col="smiles", cluster_cutoff=0.6, seed=7):
    """Visualize where train vs. test molecules fall in chemical space.

    Clusters the whole dataset with Butina clustering (Tanimoto similarity of Morgan
    fingerprints) as a structural reference — the same method used by butina_split
    — then shows train/test membership on t-SNE and PCA projections side by side. Also
    reports what fraction of Butina clusters end up "mixed" (containing both train and
    test molecules): a high mixed fraction means train and test are drawn from the same
    structural neighborhoods (as with a random split, or kmeans_split disagreeing
    with Butina near its own cluster boundaries); a low mixed fraction means test
    molecules sit in their own structurally distinct regions (as scaffold/cluster splits
    are designed to produce). Picking butina_split as the split method should
    drive this close to 0%, since it's the same clustering shown here.
    """
    smiles_list = df[smiles_col].tolist()
    X = featurize_morgan_fingerprint(smiles_list)

    cluster_id = butina_cluster_ids(smiles_list, cutoff=cluster_cutoff)
    n_clusters = cluster_id.max() + 1

    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True

    mixed = 0
    for cid in range(n_clusters):
        members_test = is_test[cluster_id == cid]
        if members_test.any() and not members_test.all():
            mixed += 1
    frac_mixed = mixed / n_clusters

    pca_coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    tsne_coords = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, title in [(axes[0], tsne_coords, "t-SNE"), (axes[1], pca_coords, "PCA")]:
        ax.scatter(coords[~is_test, 0], coords[~is_test, 1], s=14, alpha=0.6, label="train", color=TRAIN_COLOR)
        ax.scatter(coords[is_test, 0], coords[is_test, 1], s=14, alpha=0.8, label="test", color=TEST_COLOR)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Train/test split over chemical space  —  {n_clusters} Butina clusters "
        f"(cutoff={cluster_cutoff}), {frac_mixed:.0%} contain both train and test"
    )
    plt.tight_layout()
    plt.show()

    print(f"{mixed}/{n_clusters} Butina clusters ({frac_mixed:.0%}) contain both train and test molecules.")
    return {"n_clusters": int(n_clusters), "frac_mixed_clusters": float(frac_mixed)}
