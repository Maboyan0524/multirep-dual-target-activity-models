#!/usr/bin/env python3
"""Score new compounds with one representation-classifier pair.

The classifier is refitted on the complete training set, and the decision
threshold is the one derived earlier from the out-of-fold predictions of the
same pair, so the threshold never sees the query compounds. Query compounds
must already be featurised with the same representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classifiers import active_probability, build_classifier, fit, mcc_optimal_threshold
from features_io import REPRESENTATIONS, load_features, read_master

SEED = 20261008


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, required=True, help="training.csv")
    ap.add_argument("--features", type=Path, required=True,
                    help="feature directory for the training set")
    ap.add_argument("--query", type=Path, required=True,
                    help="CSV with compound_id and canonical_smiles")
    ap.add_argument("--query-features", type=Path, required=True,
                    help="feature directory for the query compounds")
    ap.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold; recomputed from out-of-fold predictions "
                         "when omitted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    master = read_master(args.master)
    query = read_master(args.query)
    x_train = load_features(master, args.features, args.representation)
    x_query = load_features(query, args.query_features, args.representation)
    y_train = master.activity_label.to_numpy()

    threshold = args.threshold
    if threshold is None:
        # Reproduce the out-of-fold vector for this pair to recover its threshold.
        import numpy as np
        folds = master.fold.to_numpy()
        oof = np.full(len(y_train), np.nan)
        for fold in range(5):
            train = folds != fold
            model = build_classifier(args.classifier, SEED + fold, y_train[train])
            fit(model, args.classifier, x_train[train], y_train[train])
            oof[~train] = active_probability(model, x_train[~train])
        threshold = mcc_optimal_threshold(y_train, oof)

    model = build_classifier(args.classifier, SEED, y_train)
    fit(model, args.classifier, x_train, y_train)
    probability = active_probability(model, x_query)

    columns = [c for c in ("compound_id", "compound_name", "canonical_smiles")
               if c in query.columns]
    result = query[columns].copy()
    result["representation"] = args.representation
    result["classifier"] = args.classifier
    result["predicted_probability"] = probability
    result["threshold"] = threshold
    result["predicted_class"] = (probability >= threshold).astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"{args.representation}-{args.classifier} threshold={threshold:.4f} "
          f"n={len(result)} -> {args.out}")


if __name__ == "__main__":
    main()
