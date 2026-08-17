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
discussing the trade-offs as a group:

1. **Representation** — Morgan fingerprints, MACCS keys, or RDKit physicochemical descriptors.
2. **Dataset splitting** — random, scaffold-based, or cluster-based.
3. **Model choice** — Random Forest / XGBoost on hand-crafted features, or Chemprop
   (a message-passing graph neural network) learning its own representation.

## Repository structure

```
notebooks/AI4molecules_tutorial.ipynb   the tutorial notebook (open this in Colab)
utils/representations.py                featurization functions (Step 1)
utils/splitting.py                      train/test splitting functions (Step 2)
utils/models.py                         model training/evaluation functions (Step 3)
data/esol.csv                           ESOL solubility dataset
```

The notebook imports functions from `utils/` and lets participants pick between them by
setting a variable (e.g. `REPRESENTATION = "morgan"`) — implementation details stay out
of the notebook so the focus stays on the modeling decisions and discussion.

## Running it

**Participants:** click the "Open in Colab" badge above, then Runtime → Run all (or step
through cell by cell). The first cell installs dependencies and clones this repo — no
local setup required.

**Locally:** clone the repo, `pip install rdkit scikit-learn xgboost chemprop pandas
matplotlib jupyter`, and open `notebooks/AI4molecules_tutorial.ipynb` from the repo root.

## Dataset

[ESOL (Delaney, 2004)](https://pubs.acs.org/doi/10.1021/ci034243x): measured aqueous
solubility for 1,128 small organic molecules, sourced via
[MoleculeNet](https://moleculenet.org/)/DeepChem.
