#!/usr/bin/env python3
"""Build a labelled activity dataset with scaffold-grouped five-fold assignment.

Training records come from ChEMBL and BindingDB; the external set comes from
NPASS and is held out. Exact canonical-structure overlaps with the external set
are removed from training before folds are assigned, so no structure can appear
on both sides.

A record is active when its relation and value decide the label at the
threshold, inactive when they decide the opposite, and uncertain otherwise;
uncertain records never enter binary training. A structure with both active and
inactive evidence is labelled conflicting and is also excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

FOLD_SEED = 23
UNIT_TO_UM = {"nm": 1e-3, "um": 1.0, "µm": 1.0, "umol/l": 1.0}
VALID_RELATIONS = ["=", "<", "<=", ">", ">="]


def canonical_parent(smiles: str) -> str | None:
    """Largest organic fragment, canonicalised; None when RDKit cannot parse."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        organic = [m for m in frags if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
        mol = max(organic, key=lambda m: m.GetNumHeavyAtoms()) if organic else mol
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) or None
    except Exception:
        return None


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or smiles


def record_label(value_um: float, relation: str, threshold_um: float) -> str:
    if relation in {"=", "<", "<="} and value_um <= threshold_um:
        return "active"
    if relation in {"=", ">", ">="} and value_um > threshold_um:
        return "inactive"
    return "uncertain"


def structure_status(labels) -> str:
    labels = set(labels)
    if "active" in labels and "inactive" in labels:
        return "conflicting"
    if "active" in labels:
        return "active"
    if "inactive" in labels:
        return "inactive"
    return "uncertain"


def clean_measurements(frame, smiles_col, value, relation, unit, source, source_id_col,
                       threshold_um):
    """Normalise one source to canonical SMILES, µM values and per-record labels."""
    unit = unit.str.lower().str.strip()
    keep = value.gt(0) & unit.isin(UNIT_TO_UM) & relation.isin(VALID_RELATIONS)
    out = frame.loc[keep].copy()
    out["canonical_smiles"] = out[smiles_col].map(canonical_parent)
    out = out.dropna(subset=["canonical_smiles"]).copy()
    out["value_uM"] = (value.loc[out.index].to_numpy()
                       * unit.loc[out.index].map(UNIT_TO_UM).to_numpy())
    out["relation"] = relation.loc[out.index].to_numpy()
    out["source"] = source
    out["source_id"] = out[source_id_col].astype(str)
    out["record_label"] = [record_label(v, r, threshold_um)
                           for v, r in zip(out.value_uM, out.relation)]
    return out


def aggregate_per_structure(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby("canonical_smiles", as_index=False).agg(
        measurement_count=("value_uM", "size"),
        median_value_uM=("value_uM", "median"),
        min_value_uM=("value_uM", "min"),
        max_value_uM=("value_uM", "max"),
        record_labels=("record_label", lambda x: ";".join(sorted(set(x)))),
        sources=("source", lambda x: ";".join(sorted(set(x)))),
        source_ids=("source_id", lambda x: ";".join(sorted(set(x)))),
        relations=("relation", lambda x: ";".join(sorted(set(x)))),
    )
    grouped["label_status"] = grouped.record_labels.map(lambda s: structure_status(s.split(";")))
    return grouped


def read_chembl(path: Path, endpoint: str, target_id: str | None, threshold_um: float):
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    relation = raw["Standard Relation"].str.replace("'", "", regex=False).str.strip()
    value = pd.to_numeric(raw["Standard Value"], errors="coerce")
    keep = raw["Standard Type"].str.upper().eq(endpoint.upper())
    if target_id:
        keep &= raw["Target ChEMBL ID"].eq(target_id)
    if endpoint.upper() == "EC50":
        # Agonist-direction endpoint: drop assays explicitly annotated as
        # inverse agonist or antagonist. Records with no direction field are
        # kept as EC50-response evidence, not as confirmed agonists.
        text = (raw.get("Assay Description", pd.Series("", index=raw.index)).fillna("") + " "
                + raw.get("Action Type", pd.Series("", index=raw.index)).fillna("")).str.lower()
        keep &= ~text.str.contains(r"inverse agonist|antagonist", regex=True, na=False)
    return clean_measurements(raw.loc[keep], "Smiles", value.loc[keep], relation.loc[keep],
                              raw["Standard Units"].loc[keep], "ChEMBL",
                              "Molecule ChEMBL ID", threshold_um)


def read_bindingdb(path: Path, endpoint: str, threshold_um: float):
    column = f"{endpoint.upper()} (nM)"
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False,
                      usecols=["Ligand SMILES", "BindingDB MonomerID", column])
    text = raw[column].str.strip()
    value = pd.to_numeric(text.str.replace(r"^[<>]=?", "", regex=True), errors="coerce")
    relation = text.str.extract(r"^(<=|>=|<|>)", expand=False).fillna("=")
    unit = pd.Series("nm", index=raw.index)
    return clean_measurements(raw, "Ligand SMILES", value, relation, unit,
                              "BindingDB", "BindingDB MonomerID", threshold_um)


def read_npass(activities: Path, structures: Path, target_id: str, endpoint: str,
               threshold_um: float):
    act = pd.read_csv(activities, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
    act = act[(act.target_id == target_id)
              & (act.activity_type.str.upper() == endpoint.upper())].copy()
    struct = pd.read_csv(structures, sep="\t", dtype=str, keep_default_na=False,
                         low_memory=False)
    # Restrict before the merge: NPASS repeats np_id in unrelated structure rows,
    # while the endpoint-associated ids are unique.
    struct = struct.loc[struct.np_id.isin(set(act.np_id)), ["np_id", "SMILES", "InChIKey"]].copy()
    act = act.merge(struct, on="np_id", how="left", validate="many_to_one").reset_index(drop=True)
    relation = act.activity_relation.fillna("").str.strip()
    value = pd.to_numeric(act.activity_value, errors="coerce")
    return clean_measurements(act, "SMILES", value, relation, act.activity_units,
                              "NPASS", "np_id", threshold_um)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="label prefix for compound ids, e.g. PPARG")
    ap.add_argument("--endpoint", required=True, choices=["EC50", "IC50"])
    ap.add_argument("--threshold-um", type=float, required=True)
    ap.add_argument("--chembl", type=Path, required=True)
    ap.add_argument("--bindingdb", type=Path, required=True)
    ap.add_argument("--npass-activities", type=Path, required=True)
    ap.add_argument("--npass-structures", type=Path, required=True)
    ap.add_argument("--npass-target-id", required=True, help="NPASS target id, e.g. NPT99")
    ap.add_argument("--chembl-target-id", default=None, help="e.g. CHEMBL235 for PPARG")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    threshold = float(args.threshold_um)

    chembl = read_chembl(args.chembl, args.endpoint, args.chembl_target_id, threshold)
    bindingdb = read_bindingdb(args.bindingdb, args.endpoint, threshold)
    columns = ["canonical_smiles", "value_uM", "relation", "source", "source_id", "record_label"]
    train_records = pd.concat([chembl[columns], bindingdb[columns]], ignore_index=True)

    train_structures = aggregate_per_structure(train_records)
    train_structures["scaffold"] = train_structures.canonical_smiles.map(murcko_scaffold)

    npass = read_npass(args.npass_activities, args.npass_structures, args.npass_target_id,
                       args.endpoint, threshold)
    external = aggregate_per_structure(npass)
    external["external_label"] = external.label_status.map({"active": 1, "inactive": 0})

    external_set = set(external.canonical_smiles)
    train_structures["exact_external_overlap"] = (
        train_structures.canonical_smiles.isin(external_set))

    training = train_structures[(~train_structures.exact_external_overlap)
                                & train_structures.label_status.isin(["active", "inactive"])].copy()
    training["activity_label"] = training.label_status.eq("active").astype(int)

    training["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, test_idx) in enumerate(
            splitter.split(training, training.activity_label, training.scaffold)):
        training.loc[training.index[test_idx], "fold"] = fold
    assert (training.fold >= 0).all(), "every training compound must receive a fold"
    assert training.groupby("scaffold").fold.nunique().max() == 1, "a scaffold spans two folds"

    training = training.sort_values(["fold", "activity_label", "canonical_smiles"]).reset_index(drop=True)
    training.insert(0, "compound_id",
                    [f"{args.target}_{args.endpoint}_{i + 1:05d}" for i in range(len(training))])
    training.to_csv(out / "training.csv", index=False)

    evaluable = external[external.label_status.isin(["active", "inactive"])].copy()
    evaluable = evaluable.sort_values(["label_status", "canonical_smiles"]).reset_index(drop=True)
    evaluable.insert(0, "compound_id",
                     [f"NPASS_{args.target}_{i + 1:04d}" for i in range(len(evaluable))])
    evaluable.to_csv(out / "external_npass.csv", index=False)

    summary = pd.DataFrame([{
        "target": args.target,
        "endpoint": args.endpoint,
        "threshold_uM": threshold,
        "chembl_measurements": len(chembl),
        "bindingdb_measurements": len(bindingdb),
        "training_structures_before_external_removal": len(train_structures),
        "training_external_overlap_removed": int(train_structures.exact_external_overlap.sum()),
        "training_N": len(training),
        "training_active": int(training.activity_label.sum()),
        "training_inactive": int((training.activity_label == 0).sum()),
        "training_scaffolds": training.scaffold.nunique(),
        "external_N": len(evaluable),
        "external_active": int(evaluable.external_label.sum()),
        "external_inactive": int((evaluable.external_label == 0).sum()),
        "external_excluded_uncertain_or_conflicting":
            int((~external.label_status.isin(["active", "inactive"])).sum()),
        "fold_seed": FOLD_SEED,
    }])
    summary.to_csv(out / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
