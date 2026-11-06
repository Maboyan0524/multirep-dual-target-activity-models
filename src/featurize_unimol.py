#!/usr/bin/env python3
"""Uni-Mol molecular representations.

Uni-Mol consumes 3D coordinates, so a single conformer is generated per
compound with ETKDGv3 at a fixed per-row seed and then MMFF optimised where
optimisation converges. Compounds whose embedding fails are written to a status
file and given a placeholder row, so the feature matrix stays aligned with the
input table; callers should drop those rows using the status file.

Run in an environment with unimol_tools installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from unimol_tools.predictor import UniMolRepr

CONFORMER_SEED_BASE = 20261006
BATCH = 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with compound_id and canonical_smiles")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--max-atoms", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    atoms, coordinates, status = [], [], []
    for position, row in data.iterrows():
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(row.canonical_smiles))
            params = AllChem.ETKDGv3()
            params.randomSeed = CONFORMER_SEED_BASE + position
            code = AllChem.EmbedMolecule(mol, params)
            if code:
                raise ValueError(f"ETKDGv3 returned {code}")
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass  # keep the unoptimised conformer rather than discard the compound
            atoms.append([a.GetSymbol() for a in mol.GetAtoms()])
            coordinates.append(mol.GetConformer().GetPositions().astype(np.float32))
            status.append({"compound_id": row.compound_id,
                           "canonical_smiles": row.canonical_smiles,
                           "success": True, "conformer_method": "ETKDGv3+MMFF",
                           "failure_reason": ""})
        except Exception as exc:
            atoms.append(["C"])
            coordinates.append(np.zeros((1, 3), dtype=np.float32))
            status.append({"compound_id": row.compound_id,
                           "canonical_smiles": row.canonical_smiles,
                           "success": False, "conformer_method": "placeholder",
                           "failure_reason": str(exc)})

    pd.DataFrame(status).to_csv(args.out / "conformer_generation_status.csv", index=False)

    model = UniMolRepr(data_type="molecule", remove_hs=False, use_cuda=not args.cpu,
                       batch_size=BATCH, max_atoms=args.max_atoms)
    vectors = []
    for start in range(0, len(atoms), BATCH):
        vectors.extend(model.get_repr({"atoms": atoms[start:start + BATCH],
                                       "coordinates": coordinates[start:start + BATCH]}))

    features = np.asarray(vectors, dtype=np.float32)
    np.save(args.out / "unimol_embeddings.npy", features)

    index_columns = [c for c in ("compound_id", "canonical_smiles", "fold", "activity_label")
                     if c in data.columns]
    data[index_columns].assign(
        embedding_row=np.arange(len(data)), embedding_dimension=features.shape[1],
    ).to_csv(args.out / "unimol_embedding_index.csv", index=False)
    print("Uni-Mol", features.shape)


if __name__ == "__main__":
    main()
