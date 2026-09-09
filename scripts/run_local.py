"""Non-Colab, script version of the tutorial for local testing.

Runs the same Step 0-3 pipeline as notebooks/AI4molecules_tutorial.ipynb (split, model
choice, representation, train & evaluate) but as a plain script: no Jupyter needed, and
plots are saved as PNG files under outputs/ instead of being displayed inline.

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --split scaffold --model random_forest --representation maccs
    python scripts/run_local.py --model chemprop --epochs 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: save figures instead of trying to open a window
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.representations import REPRESENTATIONS
from utils.splitting import SPLITTERS
from utils.models import MODEL_RUNNERS, run_chemprop
from utils.viz import plot_split_diagnostics

TARGET = "measured_log_solubility_mol_per_L"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="random", choices=sorted(SPLITTERS))
    parser.add_argument("--model", default="random_forest", choices=sorted(MODEL_RUNNERS) + ["chemprop"])
    parser.add_argument("--representation", default="morgan", choices=sorted(REPRESENTATIONS))
    parser.add_argument("--epochs", type=int, default=30, help="chemprop training epochs")
    parser.add_argument("--outdir", default=str(REPO_ROOT / "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPO_ROOT / "data" / "esol.csv")
    print(f"Loaded {len(df)} molecules")

    # ---- Step 0: splitting ----
    split_fn = SPLITTERS[args.split]
    train_idx, test_idx = split_fn(df, test_size=0.2, seed=42)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = df[TARGET].values[train_idx], df[TARGET].values[test_idx]
    print(f"[Step 0] {args.split} split: {len(train_idx)} train / {len(test_idx)} test")

    plt.figure()
    plt.hist(y_train, bins=30, alpha=0.6, label="train", density=True)
    plt.hist(y_test, bins=30, alpha=0.6, label="test", density=True)
    plt.xlabel("measured log solubility (mol/L)")
    plt.ylabel("density")
    plt.legend()
    plt.title(f"Target distribution: {args.split} split")
    plt.savefig(outdir / "step0_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    split_stats = plot_split_diagnostics(df, train_idx, test_idx)
    plt.savefig(outdir / "step0_split_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Step 0] split diagnostics: {split_stats}")

    # ---- Step 1: model choice ----
    print(f"[Step 1] Selected model: {args.model}")
    if args.model == "chemprop":
        print("[Step 1] Chemprop learns its own representation — Step 2 (Representation) won't apply.")

    # ---- Step 2: representation ----
    if args.model == "chemprop":
        X_train = X_test = None
    else:
        featurize = REPRESENTATIONS[args.representation]
        X_train = featurize(train_df["smiles"].tolist())
        X_test = featurize(test_df["smiles"].tolist())
        print(f"[Step 2] {args.representation}: X_train.shape = {X_train.shape}")

    # ---- Step 3: train & evaluate ----
    if args.model == "chemprop":
        result = run_chemprop(train_df, test_df, target_col=TARGET, epochs=args.epochs)
        label = f"{args.model} ({args.split} split, learned representation)"
    else:
        result = MODEL_RUNNERS[args.model](X_train, y_train, X_test, y_test)
        label = f"{args.model} ({args.split} split, {args.representation} representation)"

    print(f"[Step 3] {label}")
    print(f"[Step 3] RMSE = {result['rmse']:.3f}   R2 = {result['r2']:.3f}")

    y_pred = result["y_pred"]
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("measured")
    plt.ylabel("predicted")
    plt.title(label)
    plt.savefig(outdir / "step3_parity_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots to {outdir}/")


if __name__ == "__main__":
    main()
