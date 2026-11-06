#!/usr/bin/env python3
"""MolFormer pooled embeddings from ibm-research/MoLFormer-XL-both-10pct.

The pooler output is used directly as a 768-dimensional molecular
representation. The revision is pinned so the embedding is reproducible.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = "ibm-research/MoLFormer-XL-both-10pct"
REVISION = "compat-v4"
EMBEDDING_DIM = 768
MAX_LENGTH = 202
BATCH = 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with compound_id and canonical_smiles")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="HuggingFace cache holding the pinned revision")
    ap.add_argument("--allow-download", action="store_true",
                    help="fetch the weights instead of requiring a local cache")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir)
    # Import after HF_HOME is set so the cache location takes effect.
    import torch
    from transformers import AutoModel, AutoTokenizer

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        REPOSITORY, revision=REVISION, local_files_only=local_only, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        REPOSITORY, revision=REVISION, local_files_only=local_only,
        trust_remote_code=True).to(device).eval()

    data = pd.read_csv(args.input)
    parts = []
    with torch.no_grad():
        for start in range(0, len(data), BATCH):
            smiles = data.canonical_smiles.iloc[start:start + BATCH].tolist()
            encoded = tokenizer(smiles, padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt").to(device)
            pooled = model(**encoded).pooler_output
            parts.append(pooled.detach().cpu().numpy().astype(np.float32))

    features = np.vstack(parts)
    if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM or not np.isfinite(features).all():
        raise RuntimeError(f"unexpected embedding shape or non-finite values: {features.shape}")
    np.save(args.out / "molformer_pooler_embeddings.npy", features)

    index_columns = [c for c in ("compound_id", "canonical_smiles", "fold", "activity_label")
                     if c in data.columns]
    data[index_columns].assign(
        embedding_row=np.arange(len(data)), embedding_dimension=features.shape[1],
        repository=REPOSITORY, revision=REVISION, max_length=MAX_LENGTH,
    ).to_csv(args.out / "molformer_embedding_index.csv", index=False)
    print("MolFormer", features.shape, device)


if __name__ == "__main__":
    main()
