"""Data cleaning / standardization functions for the AI4molecules tutorial.

Each function takes a DataFrame and returns a new DataFrame with the SMILES column
cleaned in one specific way, printing a short summary of what it changed. They're meant
to be called one at a time in the notebook (rather than hidden behind a single
"clean_data()" call) so the individual steps of a standard cheminformatics curation
pipeline stay visible.

Order matters: run remove_invalid_smiles first (everything else assumes every SMILES
parses), and canonicalize_smiles right after it, before the rest — every function here
outputs Chem.MolToSmiles(), which is already canonical, so canonicalizing *after* the
other steps would just be a silent no-op.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

RDLogger.DisableLog("rdApp.*")


def remove_invalid_smiles(df, smiles_col="smiles"):
    """Drop rows whose SMILES RDKit cannot parse at all."""
    is_valid = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    n_removed = int((~is_valid).sum())
    print(f"remove_invalid_smiles: removed {n_removed} unparseable row(s), {int(is_valid.sum())} remain")
    return df[is_valid].reset_index(drop=True)


def canonicalize_smiles(df, smiles_col="smiles"):
    """Rewrite every SMILES into RDKit's canonical form, so two ways of writing the same
    molecule (e.g. "CCO" vs "OCC") always end up as an identical string."""
    df = df.copy()
    new_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df[smiles_col]]
    n_changed = sum(new != old for new, old in zip(new_smiles, df[smiles_col]))
    df[smiles_col] = new_smiles
    print(f"canonicalize_smiles: rewrote {n_changed} SMILES into canonical form")
    return df


def remove_salts(df, smiles_col="smiles"):
    """Strip counterions/salt fragments, keeping the parent molecule (e.g. "CCN.Cl" -> "CCN")."""
    remover = SaltRemover.SaltRemover()
    df = df.copy()
    new_smiles = []
    n_changed = 0
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        stripped_smi = Chem.MolToSmiles(remover.StripMol(mol))
        n_changed += stripped_smi != smi
        new_smiles.append(stripped_smi)
    df[smiles_col] = new_smiles
    print(f"remove_salts: stripped a salt/counterion from {n_changed} molecule(s)")
    return df


def remove_stereochemistry(df, smiles_col="smiles"):
    """Strip chiral tags and E/Z bond stereo (e.g. "C/C=C/C" -> "CC=CC").

    Reports tetrahedral (chiral-center) and E/Z (double-bond) stereochemistry
    separately, since they're different kinds of stereo information: how many
    molecules had exactly one tetrahedral stereocenter removed, how many had two or
    more, and how many had any E/Z double-bond stereochemistry removed.
    """
    df = df.copy()
    new_smiles = []
    n_one_chiral, n_multi_chiral, n_ez = 0, 0, 0
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False))
        has_ez = any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())

        Chem.RemoveStereochemistry(mol)
        new_smiles.append(Chem.MolToSmiles(mol))

        if n_chiral == 1:
            n_one_chiral += 1
        elif n_chiral >= 2:
            n_multi_chiral += 1
        if has_ez:
            n_ez += 1
    df[smiles_col] = new_smiles
    print(
        f"remove_stereochemistry: {n_one_chiral} molecule(s) had exactly 1 tetrahedral stereocenter removed, "
        f"{n_multi_chiral} molecule(s) had 2+ tetrahedral stereocenters removed, "
        f"{n_ez} molecule(s) had E/Z (double-bond) stereochemistry removed"
    )
    return df


def remove_duplicates(df, smiles_col="smiles", target_col=None):
    """Drop duplicate molecules (identical SMILES), keeping the first occurrence of each.

    Runs last, after canonicalize_smiles/remove_salts/remove_stereochemistry: those steps
    are exactly what can turn molecules that looked different into exact duplicates — two
    different spellings of the same compound, or two genuinely different stereoisomers that
    become identical once stereochemistry is stripped. If target_col is given, reports how
    many duplicate groups actually agree on the target value versus how many disagree — a
    disagreement is a real data-quality issue (measurement noise, or, as with stereoisomers
    collapsed by remove_stereochemistry, two different substances now sharing one SMILES)
    worth knowing about before it silently becomes label noise.
    """
    is_dup = df[smiles_col].duplicated(keep=False)
    n_groups = df.loc[is_dup, smiles_col].nunique()
    n_removed = int(df[smiles_col].duplicated(keep="first").sum())

    if target_col is not None and n_groups > 0:
        agree = int(df.loc[is_dup].groupby(smiles_col)[target_col].nunique().eq(1).sum())
        disagree = n_groups - agree
        print(
            f"remove_duplicates: removed {n_removed} duplicate row(s) ({n_groups} duplicate "
            f"molecule(s)) — {agree} group(s) agreed on '{target_col}', {disagree} group(s) disagreed"
        )
    else:
        print(f"remove_duplicates: removed {n_removed} duplicate row(s) ({n_groups} duplicate molecule(s))")

    return df.drop_duplicates(subset=smiles_col, keep="first").reset_index(drop=True)
