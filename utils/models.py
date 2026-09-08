"""Model-training functions for the AI4molecules tutorial.

Random Forest and XGBoost operate on hand-crafted feature vectors (X_train/X_test,
produced by a representation function from representations.py). Chemprop is different:
it is a message-passing neural network that learns its own representation directly
from SMILES, so it takes DataFrames with a SMILES column instead of feature arrays.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def _chemprop_executable():
    """Locate the chemprop console script, preferring the one next to the running interpreter.

    Falls back to PATH lookup for environments (e.g. some Colab setups) where the script
    isn't installed alongside sys.executable.
    """
    candidate = Path(sys.executable).parent / "chemprop"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("chemprop")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find the 'chemprop' command. Make sure it's installed "
        "(pip install chemprop) in the same environment as this Python interpreter."
    )


def evaluate_predictions(y_true, y_pred):
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def _val_test_result(model, y_val, y_pred_val, y_test, y_pred_test):
    """Bundle val + test predictions/metrics into one result dict.

    Top-level `y_pred`/`rmse`/`r2` are always the *test* (true holdout) numbers — the
    ones that count. `y_pred_val`/`val_rmse`/`val_r2` are there for comparison, e.g. to
    notice a model that looks great on val but falls apart on test.
    """
    test_metrics = evaluate_predictions(y_test, y_pred_test)
    val_metrics = evaluate_predictions(y_val, y_pred_val)
    return {
        "model": model,
        "y_pred": y_pred_test, "rmse": test_metrics["rmse"], "r2": test_metrics["r2"],
        "y_pred_val": y_pred_val, "val_rmse": val_metrics["rmse"], "val_r2": val_metrics["r2"],
    }


def run_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, seed=42, **kwargs):
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    return _val_test_result(model, y_val, model.predict(X_val), y_test, model.predict(X_test))


def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, seed=42, **kwargs):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed, n_jobs=-1, **kwargs
    )
    model.fit(X_train, y_train)
    return _val_test_result(model, y_val, model.predict(X_val), y_test, model.predict(X_test))


def run_chemprop(
    train_df,
    val_df,
    test_df,
    smiles_col="smiles",
    target_col="measured_log_solubility_mol_per_L",
    epochs=30,
    seed=42,
    work_dir=None,
):
    """Train a chemprop MPNN directly on SMILES and evaluate on val_df and test_df.

    Uses val_df — the same validation split every other model in this tutorial gets,
    from the splitting step, not a separately re-derived one — for early stopping /
    checkpoint selection, and keeps test_df fully held out.
    """
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="chemprop_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, test_path = work_dir / "train.csv", work_dir / "val.csv", work_dir / "test.csv"
    train_df[[smiles_col, target_col]].to_csv(train_path, index=False)
    val_df[[smiles_col, target_col]].to_csv(val_path, index=False)
    test_df[[smiles_col, target_col]].to_csv(test_path, index=False)

    ckpt_dir = work_dir / "checkpoint"
    cmd = [
        _chemprop_executable(), "train",
        "-i", str(train_path), str(val_path), str(test_path),
        "-t", "regression",
        "--target-columns", target_col,
        "-s", smiles_col,
        "--epochs", str(epochs),
        "-o", str(ckpt_dir),
        "--accelerator", "cpu",
        "-n", "0",
        "--pytorch-seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("chemprop training failed; see output above.")

    model_path = ckpt_dir / "model_0" / "best.pt"
    test_preds_path = ckpt_dir / "model_0" / "test_predictions.csv"
    y_pred_test = pd.read_csv(test_preds_path)[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    assert len(y_pred_test) == len(y_test), "chemprop returned a different number of predictions than test rows"

    # chemprop train only writes test_predictions.csv; get val predictions with a
    # separate predict call against the checkpoint it just saved.
    val_preds_path = work_dir / "val_predictions.csv"
    predict_cmd = [
        _chemprop_executable(), "predict",
        "-i", str(val_path),
        "--model-paths", str(model_path),
        "-o", str(val_preds_path),
        "-s", smiles_col,
        "--accelerator", "cpu",
        "-n", "0",
    ]
    result = subprocess.run(predict_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("chemprop predict (on val) failed; see output above.")
    y_pred_val = pd.read_csv(val_preds_path)[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    assert len(y_pred_val) == len(y_val), "chemprop returned a different number of predictions than val rows"

    return {
        **_val_test_result(None, y_val, y_pred_val, y_test, y_pred_test),
        "work_dir": str(work_dir),
    }


MODEL_RUNNERS = {
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
}
