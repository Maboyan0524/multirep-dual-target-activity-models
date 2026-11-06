#!/usr/bin/env python3
"""NaFM graph embeddings from the official pretrained checkpoint.

Atom and bond featurisation follows the NaFM reference implementation. The
embedding is the pretrained GNN readout with masking disabled.

Requires a local clone of NaFM-Official containing gnn/models/model.py and the
NaFM.ckpt checkpoint; neither is redistributed with this repository.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

EMBEDDING_DIM = 256
BATCH = 128

ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]
CHIRALITY_LIST = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                  Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_OTHER]
FORMAL_CHARGE = [-1, -2, 1, 2, 0]
BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_DIR_LIST = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
                 Chem.rdchem.BondDir.ENDDOWNRIGHT]


def install_lightning_shim() -> None:
    """NaFM's loader imports only rank_zero_warn from Lightning.

    Providing that one symbol avoids pulling in unrelated Lightning and
    torchvision dependencies. The NaFM architecture and checkpoint are untouched.
    """
    lightning = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    utilities.rank_zero_warn = lambda *a, **k: None
    lightning.utilities = utilities
    sys.modules.setdefault("pytorch_lightning", lightning)
    sys.modules.setdefault("pytorch_lightning.utilities", utilities)


def molecule_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse the SMILES")
    x = torch.tensor(
        [[ATOM_LIST.index(a.GetAtomicNum()),
          CHIRALITY_LIST.index(a.GetChiralTag()) if a.GetChiralTag() in CHIRALITY_LIST else 3,
          FORMAL_CHARGE.index(a.GetFormalCharge()) if a.GetFormalCharge() in FORMAL_CHARGE else 4]
         for a in mol.GetAtoms()], dtype=torch.long)
    edges = []
    for bond in mol.GetBonds():
        feature = [BOND_LIST.index(bond.GetBondType()), BOND_DIR_LIST.index(bond.GetBondDir())]
        edges += [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), feature),
                  (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), feature)]
    if not edges:
        raise ValueError("molecule has no bonds")
    edge_index = torch.tensor([(a, b) for a, b, _ in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([f for _, _, f in edges], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with compound_id and canonical_smiles")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nafm-repo", type=Path, required=True,
                    help="local NaFM-Official checkout")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to <nafm-repo>/NaFM.ckpt")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.nafm_repo.resolve()))
    install_lightning_shim()
    from gnn.models.model import load_pretrained_model

    checkpoint = args.checkpoint or (args.nafm_repo / "NaFM.ckpt")
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = load_pretrained_model(str(checkpoint), device=device).eval()

    data = pd.read_csv(args.input)
    graphs, rows, status = [], [], []
    for position, row in data.iterrows():
        try:
            graph = molecule_to_graph(row.canonical_smiles)
            graphs.append(graph)
            rows.append(position)
            status.append({"compound_id": row.compound_id,
                           "canonical_smiles": row.canonical_smiles, "success": True,
                           "failure_reason": "", "number_of_atoms": int(graph.num_nodes)})
        except Exception as exc:
            status.append({"compound_id": row.compound_id,
                           "canonical_smiles": row.canonical_smiles, "success": False,
                           "failure_reason": str(exc), "number_of_atoms": np.nan})

    features = np.zeros((len(data), EMBEDDING_DIM), dtype=np.float32)
    with torch.no_grad():
        for batch_index, batch in enumerate(DataLoader(graphs, batch_size=BATCH, shuffle=False)):
            target_rows = rows[batch_index * BATCH:(batch_index + 1) * BATCH]
            output = model.pretrain_gnn(batch.to(device), masked=False)
            features[target_rows] = output.detach().cpu().numpy()

    np.save(args.out / "nafm_embeddings.npy", features)
    pd.DataFrame(status).to_csv(args.out / "nafm_embedding_status.csv", index=False)

    index_columns = [c for c in ("compound_id", "canonical_smiles", "fold", "activity_label")
                     if c in data.columns]
    data[index_columns].assign(
        embedding_row=np.arange(len(data)), embedding_dimension=EMBEDDING_DIM,
    ).to_csv(args.out / "nafm_embedding_index.csv", index=False)
    print("NaFM", features.shape, device)


if __name__ == "__main__":
    main()
