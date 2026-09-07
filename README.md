# AI4molecules — Tutorial

A 75-minute, hands-on tutorial introducing machine learning for chemistry, designed for a
conference workshop. Runs entirely in [Google Colab](https://colab.research.google.com/) —
no local installation needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Corteswain/AI4molecules-tutorial/blob/main/notebooks/AI4molecules_tutorial.ipynb)

> **This repository must be public** for the badge above and the notebook's setup cell
> (which does `git clone`) to work for participants — Colab has no access to your GitHub
> credentials. Set visibility under repo **Settings → General → Danger Zone → Change visibility**
> before the workshop.

## What it covers

Participants train a solubility predictor (ESOL dataset, 1,128 molecules) and make three
modeling decisions in sequence, choosing between pre-built options at each step and
discussing the trade-offs as a group (each step ends with its own discussion prompts —
not saved up for the end):

0. **Dataset splitting** — random, scaffold-based, KMeans clustering, or Butina clustering.
   Comes first because it only depends on the SMILES and target, not on any later choice.
   Includes a diagnostic plot: Butina clustering (Tanimoto similarity) as a fixed structural
   reference, with train/test membership shown on t-SNE and PCA projections side by side.
1. **Model choice** — Random Forest / XGBoost on hand-crafted features, or Chemprop
   (a message-passing graph neural network) learning its own representation. Comes before
   Step 2 because it determines whether that step even applies.
2. **Representation** — Morgan fingerprints, MACCS keys, or RDKit physicochemical
   descriptors. Skipped entirely if Chemprop was picked in Step 1.
3. **Train & evaluate** — put the three choices together and see how the model did.

## Repository structure

```
notebooks/AI4molecules_tutorial.ipynb   the tutorial notebook (open this in Colab)
scripts/run_local.py                    plain-script version for local testing (no Jupyter needed)
utils/splitting.py                      train/test splitting functions (Step 0)
utils/models.py                         model training/evaluation functions (Steps 1 & 3)
utils/representations.py                featurization functions (Step 2)
utils/viz.py                            diagnostic plots (e.g. split visualization)
data/esol.csv                           ESOL solubility dataset
```

The notebook imports functions from `utils/` and lets participants pick between them by
setting a variable (e.g. `REPRESENTATION = "morgan"`) — implementation details stay out
of the notebook so the focus stays on the modeling decisions and discussion.

## Running it

**Participants:** click the "Open in Colab" badge above, then Runtime → Run all (or step
through cell by cell). The first cell installs dependencies and clones this repo — no
local setup required.

**Locally, in Jupyter:** clone the repo, `pip install -r requirements.txt`, and open
`notebooks/AI4molecules_tutorial.ipynb` from the repo root.

**Locally, as a plain script** (for quick testing without Jupyter — plots are saved as
PNGs under `outputs/` instead of shown inline):

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_local.py                                              # defaults: random / random_forest / morgan
python scripts/run_local.py --split scaffold --model xgboost --representation maccs
python scripts/run_local.py --model chemprop --epochs 20
```

## Dataset

[ESOL (Delaney, 2004)](https://pubs.acs.org/doi/10.1021/ci034243x): measured aqueous
solubility for 1,128 small organic molecules, sourced via
[MoleculeNet](https://moleculenet.org/)/DeepChem.
