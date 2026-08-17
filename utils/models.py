"""Model-training functions for the AI4molecules tutorial.

Random Forest and XGBoost operate on hand-crafted feature vectors (X_train/X_test,
produced by a representation function from representations.py). Chemprop is different:
it is a message-passing neural network that learns its own representation directly
from SMILES, so it takes DataFrames with a SMILES column instead of feature arrays.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def evaluate_predictions(y_true, y_pred):
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def run_random_forest(X_train, y_train, X_test, y_test, seed=42, **kwargs):
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model": model, "y_pred": y_pred, **evaluate_predictions(y_test, y_pred)}


def run_xgboost(X_train, y_train, X_test, y_test, seed=42, **kwargs):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=seed, n_jobs=-1, **kwargs
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {"model": model, "y_pred": y_pred, **evaluate_predictions(y_test, y_pred)}


def run_chemprop(
    train_df,
    test_df,
    smiles_col="smiles",
    target_col="measured_log_solubility_mol_per_L",
    epochs=30,
    seed=42,
    val_frac=0.1,
    work_dir=None,
):
    """Train a chemprop MPNN directly on SMILES and evaluate on test_df.

    Carves a small validation split out of train_df (used for early stopping /
    checkpoint selection) and keeps test_df fully held out, matching the
    train/test split chosen in the splitting step.
    """
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="chemprop_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    train_shuf = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(train_shuf) * val_frac))
    val_split = train_shuf.iloc[:n_val]
    train_split = train_shuf.iloc[n_val:]

    train_path, val_path, test_path = work_dir / "train.csv", work_dir / "val.csv", work_dir / "test.csv"
    train_split[[smiles_col, target_col]].to_csv(train_path, index=False)
    val_split[[smiles_col, target_col]].to_csv(val_path, index=False)
    test_df[[smiles_col, target_col]].to_csv(test_path, index=False)

    ckpt_dir = work_dir / "checkpoint"
    cmd = [
        "chemprop", "train",
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

    preds_path = ckpt_dir / "model_0" / "test_predictions.csv"
    preds = pd.read_csv(preds_path)
    y_pred = preds[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    assert len(y_pred) == len(y_test), "chemprop returned a different number of predictions than test rows"
    return {"model": None, "y_pred": y_pred, **evaluate_predictions(y_test, y_pred), "work_dir": str(work_dir)}


MODEL_RUNNERS = {
    "random_forest": run_random_forest,
    "xgboost": run_xgboost,
}
