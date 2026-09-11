"""Nested cross-validation for the AI4molecules tutorial (Step 5).

Step 5's 3x3 nested CV (9 trained models) is slow enough that running it synchronously would
eat into time better spent discussing Steps 4-5. NestedCVJob runs it on a background thread
instead, so it can be started as soon as Step 4 has a MODEL/SPLIT_METHOD/REPRESENTATION and a
hyperparameter grid to reuse, and finishes (or gets closer to finishing) while the tutorial
moves through Step 4's discussion and Step 5's intro. Step 5's own cell just calls .wait() —
if the background job is already done by then, that returns immediately; if not, it blocks
until it is, same as running it synchronously would have.

A thread (not a process) is used deliberately: the heavy work here is either a `chemprop`
subprocess call (which releases the GIL while the OS runs the subprocess, so the main thread
stays free) or a scikit-learn fit (which releases the GIL for its own C-level work), so a
background thread gets real concurrency without the complications of passing fitted models or
DataFrames across a process boundary.
"""

import threading

from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from .models import MODEL_RUNNERS, run_chemprop
from .representations import REPRESENTATIONS
from .splitting import nested_cv_splits


class NestedCVJob:
    """Runs a k_outer x k_inner nested CV (see splitting.nested_cv_splits) on a background
    thread, filling in .cv_results and .ensemble as models finish training.

    cv_results and ensemble are plain lists, appended to only from the background thread;
    reading them before .wait() returns can show a partial, still-growing result.
    """

    def __init__(
        self,
        df,
        split_method,
        model,
        representation,
        param_grid,
        target_col,
        k_outer=3,
        k_inner=3,
        epochs=15,
        seed=42,
    ):
        self.cv_results = []
        self.ensemble = []
        self._done = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(df, split_method, model, representation, param_grid, target_col, k_outer, k_inner, epochs, seed),
            daemon=True,
        )
        self._thread.start()

    def _run(self, df, split_method, model, representation, param_grid, target_col, k_outer, k_inner, epochs, seed):
        try:
            _run_nested_cv(
                df, split_method, model, representation, param_grid, target_col,
                k_outer, k_inner, epochs, seed, self.cv_results, self.ensemble,
            )
        except Exception as e:
            self._error = e
        finally:
            self._done.set()

    @property
    def is_done(self):
        return self._done.is_set()

    def wait(self):
        """Block until the background job finishes; re-raise any exception it hit."""
        if not self._done.is_set():
            print("Waiting for the background 3x3 cross-validation to finish...")
        self._thread.join()
        if self._error is not None:
            raise self._error
        print(f"Background CV done: {len(self.ensemble)} models trained.")


def _run_nested_cv(df, split_method, model, representation, param_grid, target_col,
                    k_outer, k_inner, epochs, seed, cv_results, ensemble):
    nested = nested_cv_splits(df, method=split_method, k_outer=k_outer, k_inner=k_inner, seed=seed)

    for outer_i, fold in enumerate(nested):
        test_idx = fold["test_idx"]
        outer_test_df = df.iloc[test_idx].reset_index(drop=True)
        y_outer_test = df[target_col].values[test_idx]
        if model != "chemprop":
            X_outer_test = REPRESENTATIONS[representation](outer_test_df["smiles"].tolist())

        for inner_j, (in_train_idx, in_val_idx) in enumerate(fold["inner_splits"]):
            in_train_df = df.iloc[in_train_idx].reset_index(drop=True)
            in_val_df = df.iloc[in_val_idx].reset_index(drop=True)
            y_in_train = df[target_col].values[in_train_idx]
            y_in_val = df[target_col].values[in_val_idx]

            if model == "chemprop":
                # Select the best config on inner-val only (same as Step 4)...
                trials = [(p, run_chemprop(in_train_df, in_val_df, target_col=target_col, epochs=epochs, **p)) for p in param_grid]
                best_params, _ = min(trials, key=lambda pr: pr[1]["rmse"])
                # ...then retrain with that config to get real predictions on the outer test fold.
                best = run_chemprop(in_train_df, in_val_df, test_df=outer_test_df, target_col=target_col, epochs=epochs, **best_params)
                y_pred = best["y_pred"]
                ensemble.append({"model": None, "work_dir": best["work_dir"]})
            else:
                X_in_train = REPRESENTATIONS[representation](in_train_df["smiles"].tolist())
                X_in_val = REPRESENTATIONS[representation](in_val_df["smiles"].tolist())
                trials = [(p, MODEL_RUNNERS[model](X_in_train, y_in_train, X_in_val, y_in_val, **p)) for p in param_grid]
                best_params, best = min(trials, key=lambda pr: pr[1]["rmse"])
                y_pred = best["model"].predict(X_outer_test)
                ensemble.append({"model": best["model"], "work_dir": None})

            r2 = r2_score(y_outer_test, y_pred)
            rho = spearmanr(y_outer_test, y_pred).correlation
            cv_results.append({"outer_fold": outer_i, "inner_split": inner_j, "best_params": best_params, "r2": r2, "spearman": rho})
