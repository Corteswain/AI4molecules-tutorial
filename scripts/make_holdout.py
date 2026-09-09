"""One-time data preparation: carve a fixed, external holdout set out of the raw ESOL
data using Butina clustering, so it can be handed to students at the end of the tutorial
as if it were new, never-seen data.

Splits data/esol.csv (raw, before any cleaning) into:
    data/data.csv   the working dataset — this is what the tutorial notebook loads and
                    runs its own cleaning pipeline on
    data/test.csv   the external holdout, deliberately left as-is (uncleaned) and never
                    loaded by the notebook — kept completely outside the tutorial until
                    you choose to hand it out

This only needs to be run once; data.csv/test.csv are committed to the repo, and that's
what the notebook actually uses from then on. Re-running would pick different molecules
(unless you keep --seed fixed), so don't re-run this casually once you've handed test.csv
out — that would let students' "unseen" molecules quietly change.

Usage:
    python scripts/make_holdout.py [--test-size 0.15] [--tolerance 0.05] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.splitting import butina_holdout_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-size", type=float, default=0.15, help="target fraction for test.csv")
    parser.add_argument("--tolerance", type=float, default=0.05, help="acceptable +/- band around --test-size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    esol_path = REPO_ROOT / "data" / "esol.csv"
    df = pd.read_csv(esol_path)
    print(f"Loaded {len(df)} molecules from {esol_path}")

    data_idx, holdout_idx = butina_holdout_split(
        df, test_size=args.test_size, tolerance=args.tolerance, seed=args.seed
    )

    data_df = df.iloc[data_idx].reset_index(drop=True)
    test_df = df.iloc[holdout_idx].reset_index(drop=True)

    data_path = REPO_ROOT / "data" / "data.csv"
    test_path = REPO_ROOT / "data" / "test.csv"
    data_df.to_csv(data_path, index=False)
    test_df.to_csv(test_path, index=False)

    n = len(df)
    print(f"data.csv: {len(data_df)} molecules ({len(data_df) / n:.1%}) -> {data_path}")
    print(f"test.csv: {len(test_df)} molecules ({len(test_df) / n:.1%}) -> {test_path}")


if __name__ == "__main__":
    main()
