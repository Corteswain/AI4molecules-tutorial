"""Nested cross-validation for the AI4molecules tutorial (Step 5).

Step 5's 3x3 nested CV reuses the single hyperparameter config Step 4 already picked (via its
own quick search against the Step 1 train/val split) instead of tuning again here — a
deliberate simplification ("cheating" a little) that keeps this fast enough to just run
synchronously, right in Step 5, with no background job needed.
"""

from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from .models import MODEL_RUNNERS, run_chemprop
from .representations import REPRESENTATIONS
from .splitting import nested_cv_splits


def run_nested_cv(df, split_method, model, representation, best_params, target_col,
                   k_outer=3, k_inner=3, epochs=15, seed=42):
    """Train k_outer * k_inner models, one per (outer test fold, inner train/val split) pair,
    all using the same `best_params` (e.g. Step 4's already-chosen config) rather than
    re-tuning per fold.

    Returns (cv_results, ensemble):
      - cv_results: a list of {"outer_fold", "inner_split", "r2", "spearman"} dicts.
      - ensemble: a list of {"model", "work_dir"} dicts (exactly one of the two set,
        depending on `model`) — Step 6 applies these directly to data/real.csv.
    """
    nested = nested_cv_splits(df, method=split_method, k_outer=k_outer, k_inner=k_inner, seed=seed)
    cv_results, ensemble = [], []

    for outer_i, fold in enumerate(nested):
        test_idx = fold["test_idx"]
        outer_test_df = df.iloc[test_idx].reset_index(drop=True)
        y_outer_test = df[target_col].values[test_idx]
        if model != "chemprop":
            X_outer_test = REPRESENTATIONS[representation](outer_test_df["smiles"].tolist())

        for inner_j, (in_train_idx, in_val_idx) in enumerate(fold["inner_splits"]):
            in_train_df = df.iloc[in_train_idx].reset_index(drop=True)

            if model == "chemprop":
                # in_val_df still plays its normal early-stopping role in training.
                in_val_df = df.iloc[in_val_idx].reset_index(drop=True)
                best = run_chemprop(in_train_df, in_val_df, test_df=outer_test_df, target_col=target_col, epochs=epochs, **best_params)
                y_pred = best["y_pred"]
                ensemble.append({"model": None, "work_dir": best["work_dir"]})
            else:
                y_in_train = df[target_col].values[in_train_idx]
                X_in_train = REPRESENTATIONS[representation](in_train_df["smiles"].tolist())
                best = MODEL_RUNNERS[model](X_in_train, y_in_train, X_outer_test, y_outer_test, **best_params)
                y_pred = best["y_pred"]
                ensemble.append({"model": best["model"], "work_dir": None})

            r2 = r2_score(y_outer_test, y_pred)
            rho = spearmanr(y_outer_test, y_pred).correlation
            print(f"  outer fold {outer_i}, inner split {inner_j}: R2 = {r2:.3f}   spearman = {rho:.3f}")
            cv_results.append({"outer_fold": outer_i, "inner_split": inner_j, "r2": r2, "spearman": rho})

    return cv_results, ensemble
