"""Model-training functions for the AI4molecules tutorial.

Random Forest operates on hand-crafted feature vectors (X_train/X_val,
produced by a representation function from representations.py). Chemprop is different:
it is a message-passing neural network that learns its own representation directly
from SMILES, so it takes DataFrames with a SMILES column instead of feature arrays.

run_random_forest always trains on train and evaluates on val — that's it, no test set. Used
by Step 4's hyperparameter search. run_chemprop follows the same train/val-only contract by
default, but Step 5's nested cross-validation needs a genuine third set (an outer test fold),
so it optionally accepts one; see its docstring. Either way, the real evaluation set
(data/real.csv) is deliberately never touched by any of this code.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chemprop's default atom featurizer one-hots atomic number over ~37 elements (H through Kr,
# plus iodine) since it's built for arbitrary molecules. ESOL only ever contains C, O, N, Cl,
# S, F, Br, P, and I — chemprop's built-in "organic" scheme (H, B, C, N, O, F, Si, P, S, Cl,
# Br, I) already covers that as a subset, cutting the atom feature vector from 72 to 44 values
# without risking a crash on anything ESOL could actually contain. Must match between training
# and prediction — chemprop stores the trained weights but not this choice, so a mismatch
# fails loudly with a dimension error rather than silently giving wrong predictions.
ATOM_FEATURIZER_MODE = "organic"


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


def run_random_forest(X_train, y_train, X_val, y_val, seed=42, **kwargs):
    params = {"n_estimators": 300, **kwargs}
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return {"model": model, "y_pred": y_pred, **evaluate_predictions(y_val, y_pred)}


def run_chemprop(
    train_df,
    val_df,
    test_df=None,
    smiles_col="smiles",
    target_col="measured_log_solubility_mol_per_L",
    epochs=30,
    seed=42,
    work_dir=None,
    **hparams,
):
    """Train a chemprop MPNN directly on SMILES and evaluate on val_df (or test_df, if given).

    Extra keyword arguments are passed through as CLI flags for a quick hyperparameter
    search, e.g. run_chemprop(..., depth=2) adds `--depth 2`.

    val_df is always used for early stopping / checkpoint selection (chemprop's normal role
    for it). If test_df is given, it's passed as chemprop's genuine "test" input and the
    predictions this function returns are for test_df (used by Step 5's nested CV, which has
    a real outer test fold). If test_df is omitted (the default — what Steps 1-4 use, since
    they have no test set at all), val_df is passed again as chemprop's "test" input instead,
    so the predictions returned are for val_df — chemprop only writes predictions for
    whatever it's told is the test set.
    """
    eval_df = test_df if test_df is not None else val_df
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="chemprop_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path = work_dir / "train.csv", work_dir / "val.csv"
    train_df[[smiles_col, target_col]].to_csv(train_path, index=False)
    val_df[[smiles_col, target_col]].to_csv(val_path, index=False)
    if test_df is not None:
        test_path = work_dir / "test.csv"
        test_df[[smiles_col, target_col]].to_csv(test_path, index=False)
    else:
        test_path = val_path

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
        "--multi-hot-atom-featurizer-mode", ATOM_FEATURIZER_MODE,
    ]
    for key, value in hparams.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("chemprop training failed; see output above.")

    preds_path = ckpt_dir / "model_0" / "test_predictions.csv"
    y_pred = pd.read_csv(preds_path)[target_col].to_numpy()
    y_eval = eval_df[target_col].to_numpy()
    assert len(y_pred) == len(y_eval), "chemprop returned a different number of predictions than eval rows"
    return {"model": None, "y_pred": y_pred, **evaluate_predictions(y_eval, y_pred), "work_dir": str(work_dir)}


def predict_chemprop(work_dir, df, smiles_col="smiles"):
    """Predict with an already-trained chemprop checkpoint (the work_dir a run_chemprop call
    returned) on new molecules, without retraining.

    Used by Step 6 to apply Step 5's already-trained ensemble to data/real.csv.
    """
    work_dir = Path(work_dir)
    checkpoint_path = work_dir / "checkpoint" / "model_0" / "best.pt"
    input_path = work_dir / "predict_input.csv"
    output_path = work_dir / "predict_output.csv"
    df[[smiles_col]].to_csv(input_path, index=False)

    cmd = [
        _chemprop_executable(), "predict",
        "-i", str(input_path),
        "-s", smiles_col,
        "--model-paths", str(checkpoint_path),
        "-o", str(output_path),
        "--accelerator", "cpu",
        "--multi-hot-atom-featurizer-mode", ATOM_FEATURIZER_MODE,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError("chemprop predict failed; see output above.")

    # chemprop names the prediction column after whatever target the checkpoint was trained
    # on; read positionally instead of relying on that name.
    return pd.read_csv(output_path).iloc[:, -1].to_numpy()


MODEL_RUNNERS = {
    "random_forest": run_random_forest,
}
