#!/usr/bin/env python3
"""Morgan fingerprints, radius 2, 2048 bits, no chirality and no feature invariants."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

RADIUS = 2
N_BITS = 2048


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with compound_id and canonical_smiles")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input)
    bits = []
    for smiles in data.canonical_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse: {smiles}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, RADIUS, nBits=N_BITS, useChirality=False, useFeatures=False)
        row = np.zeros(N_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        bits.append(row)

    features = np.asarray(bits, dtype=np.uint8)
    np.save(args.out / "morgan_radius2_2048.npy", features)

    index_columns = [c for c in ("compound_id", "canonical_smiles", "fold", "activity_label")
                     if c in data.columns]
    data[index_columns].assign(
        feature_row=np.arange(len(data)), radius=RADIUS, nBits=N_BITS,
        useChirality=False, useFeatures=False,
    ).to_csv(args.out / "morgan_feature_index.csv", index=False)
    print("Morgan", features.shape)


if __name__ == "__main__":
    main()
