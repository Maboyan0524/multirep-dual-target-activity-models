"""Loading cached representations and aligning them to a compound table.

Each featurise script writes a `.npy` matrix plus an index CSV mapping
canonical SMILES to a row in that matrix. Alignment goes through the SMILES so
a representation can be reused across datasets without assuming row order.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# representation -> (matrix filename, index filename, row column)
LAYOUT = {
    "Morgan": ("morgan_radius2_2048.npy", "morgan_feature_index.csv", "feature_row"),
    "UniMol": ("unimol_embeddings.npy", "unimol_embedding_index.csv", "embedding_row"),
    "NaFM": ("nafm_embeddings.npy", "nafm_embedding_index.csv", "embedding_row"),
    "MolFormer": ("molformer_pooler_embeddings.npy", "molformer_embedding_index.csv",
                  "embedding_row"),
}

REPRESENTATIONS = list(LAYOUT)


def representation_dir(root: Path, representation: str) -> Path:
    """features/<root>/<morgan|unimol|nafm|molformer>."""
    return root / representation.lower()


def load_features(master: pd.DataFrame, root: Path, representation: str) -> np.ndarray:
    """Return the feature matrix in the row order of `master`."""
    matrix_name, index_name, row_column = LAYOUT[representation]
    base = representation_dir(root, representation)
    matrix = np.load(base / matrix_name, mmap_mode="r")
    index = pd.read_csv(base / index_name).rename(columns={"SMILES": "canonical_smiles"})

    if index.canonical_smiles.duplicated().any():
        raise RuntimeError(f"duplicate feature keys in {base / index_name}")
    lookup = dict(zip(index.canonical_smiles, index[row_column]))

    missing = master.loc[~master.canonical_smiles.isin(lookup), "canonical_smiles"]
    if len(missing):
        raise RuntimeError(f"{len(missing)} compounds absent from {base / index_name}")

    rows = master.canonical_smiles.map(lookup).astype(int).to_numpy()
    if rows.min() < 0 or rows.max() >= len(matrix):
        raise RuntimeError("feature row index outside the cached matrix")
    return np.asarray(matrix[rows], dtype=np.float32)


def read_master(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path)
    required = {"canonical_smiles"}
    if not required <= set(master.columns):
        raise RuntimeError(f"{path} must contain {sorted(required)}")
    if master.canonical_smiles.duplicated().any():
        raise RuntimeError(f"duplicate canonical SMILES in {path}")
    return master
